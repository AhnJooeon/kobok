import pandas as pd
import re
import calendar
from typing import List, Optional

# 사용하면 될듯함

# ---------------- Utils ----------------
def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def parse_kor_period(col_name: str) -> pd.Timestamp:
    """
    '2025년03월말', '2025년 3월말', '2025년 03월 말' -> 2025-03-31
    """
    s = str(col_name).strip()
    # 숫자+공백 허용, 한 자리/두 자리 월 허용
    m = re.fullmatch(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*말", s)
    if not m:
        # 보조 패턴: 'YYYY.MM월말' 같은 변형
        m2 = re.fullmatch(r"(\d{4})[.\-/ ](\d{1,2})\s*월\s*말", s)
        if not m2:
            return pd.NaT
        y, mm = int(m2.group(1)), int(m2.group(2))
    else:
        y, mm = int(m.group(1)), int(m.group(2))
    if not (1 <= mm <= 12):
        return pd.NaT
    return pd.Timestamp(year=y, month=mm, day=_last_day(y, mm))

def _is_time_col(name: str) -> bool:
    return pd.notna(parse_kor_period(name))

def _clean_money(x: str) -> Optional[float]:
    """
    금액 문자열 표준화:
    - 쉼표·공백 제거
    - 괄호표기 (123) -> -123
    - 유니코드 마이너스(−), '△' -> 음수
    - 퍼센트 기호 제거
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()

    if s == "" or s.upper() in {"N/A", "NA", "NULL", "-"}:
        return None

    neg = False
    # 괄호표기 음수
    if re.match(r"^\(.*\)$", s):
        neg = True
        s = s.strip("()")

    # 유니코드 마이너스, '△' 처리
    s = s.replace("−", "-").replace("–", "-").replace("△", "-")

    # 퍼센트 제거
    s = s.replace("%", "")

    # 쉼표·공백 제거
    s = s.replace(",", "").replace(" ", "").replace("\t", "")

    # 남은 +- 2개 이상이면 한번 정리
    s = re.sub(r"\++", "+", s)
    s = re.sub(r"-+", "-", s)

    try:
        v = float(s)
    except Exception:
        return None
    return -v if neg and v >= 0 else v

def _flatten_cols(df: pd.DataFrame, sep="__") -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join([str(p) for p in tup if p is not None and str(p) != ""]) for tup in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

# -------------- Main Reader --------------
def read_income_stmt(path: str):
    """
    - 헤더 행 자동 탐지(은행명/구분/코드가 들어있는 행)
    - 'Unnamed' 헤더/병합셀 대응
    - 'YYYY년MM월말' 등 다양한 변형 허용
    - 출력:
        tidy(세로) : [시점, 코드, 구분, 금액]
        wide_by_code : index=시점, columns=코드
        wide_multi   : index=시점, columns=(코드,구분) -> 평탄화하여 단일 헤더 보장
    """
    # 1) 헤더 없이 읽어서 탐지
    df0 = pd.read_excel(path, sheet_name=0, header=None, dtype=str)
    df0 = df0.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))


    # 헤더 후보 탐색 (상단 30행 정도)
    header_idx = None
    keyset = {"은행명", "구분", "코드"}
    for i in range(min(30, len(df0))):
        vals = [str(v).strip() for v in df0.iloc[i].tolist()]
        if keyset.issubset(set(vals)):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("헤더 행(은행명/구분/코드 포함)을 찾지 못했습니다.")

    # 2) 헤더 승격
    header_raw = [str(x).strip() if pd.notna(x) else "" for x in df0.iloc[header_idx].tolist()]
    # Unnamed 제거
    header = [("" if re.match(r"^Unnamed", h, flags=re.I) else h) for h in header_raw]
    df = df0.iloc[header_idx + 1:].copy()
    df.columns = header

    # 3) 필수 컬럼 존재성/이름 위치 확인 (앞 3열 가정 금지)
    def _must_col(name_list: List[str]) -> str:
        for nm in name_list:
            if nm in df.columns:
                return nm
        raise ValueError(f"필수 컬럼이 없습니다: {name_list}")

    col_bank = _must_col(["은행명"])
    col_kind = _must_col(["구분"])
    col_code = _must_col(["코드"])

    # 4) 유효 행 필터: 코드 비어있지 않은 것만
    df = df[~df[col_code].isna() & (df[col_code].astype(str).str.strip() != "")].copy()

    # 5) 시간 컬럼 전수 검사 (모든 컬럼 이름에서 패턴 검사)
    all_cols = list(df.columns)
    time_cols = [c for c in all_cols if _is_time_col(c)]
    if not time_cols:
        raise ValueError("시점 컬럼(YYYY년MM월말 패턴)들을 찾지 못했습니다.")

    # 6) 세로형 변환
    tidy = (
        df.melt(id_vars=[col_kind, col_code], value_vars=time_cols,
                var_name="시점_표기", value_name="금액_raw")
          .assign(시점=lambda d: d["시점_표기"].apply(parse_kor_period))
          .drop(columns=["시점_표기"])
    )
    # 금액 숫자화(강화)
    tidy["금액"] = tidy["금액_raw"].apply(_clean_money)
    tidy = tidy.drop(columns=["금액_raw"])
    tidy = tidy.dropna(subset=["시점"]).sort_values(["시점", col_code]).reset_index(drop=True)

    # 7) 가로형
    wide_by_code = tidy.pivot_table(index="시점", columns=col_code, values="금액", aggfunc="sum")
    wide_multi   = tidy.pivot_table(index="시점", columns=[col_code, col_kind], values="금액", aggfunc="sum")

    # 8) 컬럼 평탄화(멀티헤더 방지)
    wide_by_code = _flatten_cols(wide_by_code.reset_index())
    wide_multi   = _flatten_cols(wide_multi.reset_index(), sep="__")

    # 컬럼명 예쁘게(선택): '코드__구분' → '코드|구분'
    wide_multi.columns = [c.replace("__", "|") for c in wide_multi.columns]

    # 9) 반환(헤더 한 줄 보장)
    return tidy.rename(columns={col_kind: "구분", col_code: "코드"}), wide_by_code, wide_multi

# ----------------- Example -----------------
if __name__ == "__main__":
    tidy, wide_code, wide_multi = read_income_stmt("./bss_result/재무현황/연결손익계산서_20250820.xlsx")
    tidy.to_csv("손익_세로형_tidy.csv", index=False, encoding="utf-8-sig")
    wide_code.to_csv("손익_가로형_코드열.csv", index=False, encoding="utf-8-sig")
    wide_multi.to_csv("손익_가로형_코드_구분열.csv", index=False, encoding="utf-8-sig")
    print("OK")
