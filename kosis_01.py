import os, time, json, csv, sys, pathlib, itertools
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import yaml
from dotenv import load_dotenv

# ---------------------------
# 설정/유틸
# ---------------------------
load_dotenv()
API_KEY = os.getenv("KOSIS_API_KEY", "").strip()
DEFAULT_USER_ID = os.getenv("KOSIS_USER_ID", "").strip()
OUT_FORMATS = [x.strip() for x in os.getenv("OUT_FORMATS", "csv,parquet").split(",") if x.strip()]
RATE_LIMIT_PER_SEC = float(os.getenv("RATE_LIMIT_PER_SEC", "5"))

if not API_KEY:
    print("ERROR: KOSIS_API_KEY가 .env에 없습니다.")
    sys.exit(1)

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "kosis-pipeline/1.0"})

def rate_limit():
    time.sleep(1.0 / max(RATE_LIMIT_PER_SEC, 1e-6))

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# KOSIS 호출 빌더
# * KOSIS는 여러 엔드포인트 버전이 있는데,
#   - userStatsId 방식
#   - orgId+tblId 파라미터 방식
# 두 가지를 추상화해서 사용합니다.
# ---------------------------

def call_user_stats(user_id: str, user_stats_name: str, period: str, last_n: int = 0) -> List[Dict[str, Any]]:
    """
    사용자정의 통계 호출.
    - 일반적으로 KOSIS OpenAPI 문서의 'statisticsData.do?method=getList' + userStatsId= "USER/NAME/PERIOD"
    - 일부 계정/설정에 따라 도메인/파라미터가 다를 수 있어 endpoint를 변수화.
    """
    # 엔드포인트(필요시 조직 정책에 맞춰 https로 변경)
    endpoint = "https://kosis.kr/openapi/statisticsData.do"
    user_id_final = user_id or DEFAULT_USER_ID
    userStatsId = f"{user_id_final}/{user_stats_name}/{period}"

    params = {
        "method": "getList",
        "apiKey": API_KEY,
        "format": "json",
        "jsonVD": "Y",
        "userStatsId": userStatsId,
    }
    if last_n and last_n > 0:
        params["newEstPrdCnt"] = str(last_n)  # 최신 N개만

    rate_limit()
    resp = SESSION.get(endpoint, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("errCd"):
        raise RuntimeError(f"KOSIS error {data.get('errCd')}: {data.get('errMsg')}")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response format (userStatsId).")
    return data

def call_table(org_id: str, table_id: str, prdSe: str, startPrdDe: str, endPrdDe: str,
               itms: Optional[List[str]] = None,
               objL1: Optional[List[str]] = None,
               objL2: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    orgId/tblId 직접 호출.
    - KOSIS OpenAPI의 파라미터 엔드포인트(통상 statisticsData.do / statisticsParameterData.do 등)를 통합적으로 빌드.
    - 표/항목/분류 차원은 기관/표에 따라 다르므로 config에서 전달받아 조합.
    """
    endpoint = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    params = {
        "method": "getList",
        "apiKey": API_KEY,
        "format": "json",
        "jsonVD": "Y",
        "orgId": org_id,
        "tblId": table_id,
        "prdSe": prdSe,            # A/Q/M 등
        "startPrdDe": startPrdDe,  # 2000, 2000Q1, 200001 등
        "endPrdDe": endPrdDe,
    }

    # 항목/분류 차원(필요한 키 이름은 표에 따라 다를 수 있음)
    # 일반적으로: itmId(항목), objL1/objL2(지역/분류 레벨) 등
    # 여러 값일 경우 콤마 연결
    if itms:
        params["itmId"] = ",".join(itms)
    if objL1:
        params["objL1"] = ",".join(objL1)
    if objL2:
        params["objL2"] = ",".join(objL2)

    rate_limit()
    resp = SESSION.get(endpoint, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("errCd"):
        raise RuntimeError(f"KOSIS error {data.get('errCd')}: {data.get('errMsg')}")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response format (table).")
    return data

def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # 일반적으로 KOSIS 반환 컬럼 예시: "PRD_DE", "C1_NM", "ITM_NM", "DT" ...
    # 숫자 변환
    for c in ("DT", "dt", "DATA_VALUE"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 중복 제거
    df = df.drop_duplicates()
    return df

def save_outputs(df: pd.DataFrame, out_base: pathlib.Path):
    if df.empty:
        print(f"[WARN] 빈 데이터라 저장 건너뜀: {out_base}")
        return
    # 표준 컬럼명 소문자화(선택)
    df.columns = [c.strip() for c in df.columns]
    # 저장
    if "csv" in OUT_FORMATS:
        df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    if "parquet" in OUT_FORMATS:
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    print(f"[OK] Saved -> {out_base.with_suffix('.csv') if 'csv' in OUT_FORMATS else out_base.with_suffix('.parquet')}")

def run_job(job: Dict[str, Any]):
    mode = job["mode"]
    params = job["params"]
    category = job["category"]

    out_dir = DATA_DIR / category
    ensure_dir(out_dir)

    ts = now_ts()
    out_stem = f"{category}_{ts}"

    if mode == "userStatsId":
        rows = call_user_stats(
            user_id=params.get("user_id", DEFAULT_USER_ID),
            user_stats_name=params["user_stats_name"],
            period=params.get("period", "A"),
            last_n=int(params.get("last_n", 0)),
        )
        df = to_dataframe(rows)
        save_outputs(df, out_dir / out_stem)

    elif mode == "table":
        rows = call_table(
            org_id=params["org_id"],
            table_id=params["table_id"],
            prdSe=params["prdSe"],
            startPrdDe=params["startPrdDe"],
            endPrdDe=params["endPrdDe"],
            itms=params.get("itms") or [],
            objL1=params.get("objL1") or [],
            objL2=params.get("objL2") or [],
        )
        df = to_dataframe(rows)
        save_outputs(df, out_dir / out_stem)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    cfg_path = BASE_DIR / "kosis_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    jobs = cfg.get("jobs", [])
    if not jobs:
        print("config.yaml에 jobs가 비어 있습니다.")
        return

    for i, job in enumerate(jobs, 1):
        try:
            print(f"=== [{i}/{len(jobs)}] {job['category']} 시작 ===")
            run_job(job)
        except Exception as e:
            print(f"[ERROR] {job.get('category','?')} 실패: {e}")

if __name__ == "__main__":
    main()
