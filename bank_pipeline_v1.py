# -*- coding: utf-8 -*-
"""
bank_pipeline_v1.py
-------------------
여러 형태(영업점포 현황, 자동화기기 설치현황 등)의 은행 엑셀 데이터를
일관된 규칙으로 파싱하여 tidy / wide_code / wide_multi를 생성하고,
파일명에서 "_" 앞부분(prefix)을 결과 파일명에 사용해 저장.

사용 예시:
    from bank_pipeline_v1 import read_one, save_one, process_many
    tidy, wide_code, wide_multi = read_one("영업점포 현황_20250820.xlsx", sheet="auto")
    save_one("자동화기기 설치현황_20250820.xlsx", outdir="out", sheet="auto")
    process_many("data/*.xlsx", outdir="out", sheet="auto")
"""

import os, re, math, calendar, glob
from typing import Optional, List
import pandas as pd

# ───────────── helpers ─────────────
def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def _clean_str(x):
    return x.replace("\xa0"," ").strip() if isinstance(x, str) else x

def _clean_money(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A","NA","NULL","-"}:
        return math.nan
    neg = False
    if re.match(r"^\(.*\)$", s):
        neg, s = True, s.strip("()")
    s = (s.replace("−","-").replace("–","-").replace("△","-")
           .replace("%","").replace(",", "").replace(" ", "").replace("\t", ""))
    try:
        v = float(s)
    except:
        return math.nan
    return -v if neg and v >= 0 else v

def _parse_period_to_month_end(col) -> pd.Timestamp:
    # 1) 날짜/숫자 → Timestamp 시도
    try:
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts.year, ts.month, _last_day(ts.year, ts.month))
    except Exception:
        pass
    # 2) 문자열 패턴
    s = str(col).replace("\xa0"," ").strip()
    m = re.fullmatch(r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?:말|말일|현재|기준)?", s)
    if m:
        y, mm = int(m.group("y")), int(m.group("m"))
        if 1 <= mm <= 12:
            return pd.Timestamp(y, mm, _last_day(y, mm))
    m2 = re.fullmatch(r"(?P<y>\d{4})[.\-/\s](?P<m>\d{1,2})", s)
    if m2:
        y, mm = int(m2.group("y")), int(m2.group("m"))
        if 1 <= mm <= 12:
            return pd.Timestamp(y, mm, _last_day(y, mm))
    m3 = re.fullmatch(r"(?P<y>\d{4})(?P<m>\d{2})", s)
    if m3:
        y, mm = int(m3.group("y")), int(m3.group("m"))
        if 1 <= mm <= 12:
            return pd.Timestamp(y, mm, _last_day(y, mm))
    return pd.NaT

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    bank_syn = {"은행명","은행","기관명","금융기관","금융기관명","금융회사","금융회사명","기관"}
    gubun_syn = {"구분","분류","항목","항목명","유형","대분류","중분류","소분류"}
    code_syn  = {"코드","code","항목코드","계정코드","분류코드"}
    bank_syn_l = {k.lower() for k in bank_syn}
    gubun_syn_l= {k.lower() for k in gubun_syn}
    code_syn_l = {k.lower() for k in code_syn}
    new_cols = []
    for c in df.columns:
        raw = str(c)
        s = raw.replace("\xa0", " ")
        key = re.sub(r"\s+", "", s).lower()
        if key in bank_syn_l:  new_cols.append("은행명")
        elif key in gubun_syn_l: new_cols.append("구분")
        elif key in code_syn_l:  new_cols.append("코드")
        else: new_cols.append(raw)
    df.columns = new_cols
    if "기관명" in df.columns and "은행명" not in df.columns:
        df = df.rename(columns={"기관명":"은행명"})
    return df

def _is_total_label(x) -> bool:
    if x is None or (isinstance(x, float) and math.isnan(x)): return False
    s = str(x).replace("\xa0"," ").strip()
    return bool(re.search(r"(?:^|\s)(합계|총계|소계|전체)(?:\s|$)", s))

def _detect_time_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.notna(_parse_period_to_month_end(c))]

def _detect_header_row(path: str, sheet=0, probe: int = 12) -> int:
    """상단 probe행까지 훑어 코드 계열 헤더가 보이는 행을 선택. 실패 시 1행."""
    peek = pd.read_excel(path, sheet_name=sheet, header=None, nrows=probe, dtype=str)
    peek = peek.apply(lambda col: col.map(_clean_str))
    code_syn = {"코드","code","항목코드","계정코드","분류코드"}
    syn_l = {s.lower() for s in code_syn}
    for i in range(len(peek)):
        vals = [str(v) for v in list(peek.iloc[i].values)]
        norm = {re.sub(r"\s+","", v).lower() for v in vals if v not in {"", "nan", "None"}}
        if norm & syn_l:
            return i
    return 1

def _detect_sheet_auto(path: str) -> int:
    try:
        xl = pd.ExcelFile(path)
        for idx, name in enumerate(xl.sheet_names):
            try:
                h = _detect_header_row(path, sheet=idx)
                df_head = pd.read_excel(path, sheet_name=idx, header=h, nrows=3, dtype=str)
                df_head = df_head.apply(lambda col: col.map(_clean_str))
                df_head = _normalize_headers(df_head)
                if "코드" in df_head.columns and _detect_time_cols(df_head):
                    return idx
            except Exception:
                continue
        return 0
    except Exception:
        return 0

# ───────────── core readers ─────────────
def read_one(path: str, sheet: Optional[int] = "auto"):
    """
    단일 파일 읽어 tidy / wide_code / wide_multi 반환
    - sheet='auto'면 자동 탐지, 아니면 정수 인덱스 사용
    """
    sidx = _detect_sheet_auto(path) if sheet == "auto" else int(sheet)
    h = _detect_header_row(path, sheet=sidx)
    df = pd.read_excel(path, sheet_name=sidx, header=h, dtype=str)
    df = df.apply(lambda col: col.map(_clean_str))
    df = _normalize_headers(df)

    # 필수 컬럼 보정
    if "코드" not in df.columns:
        raise KeyError(f"[{os.path.basename(path)}] '코드' 컬럼 없음")
    for req in ["은행명","구분"]:
        if req not in df.columns:
            df[req] = None

    time_cols = _detect_time_cols(df)
    if not time_cols:
        raise ValueError(f"[{os.path.basename(path)}] 시점 컬럼 없음")

    # 은행명 전략 자동 선택
    codeset = set(df["코드"].dropna().unique().tolist())
    if {"A1","A2"} & codeset:
        # 영업점 전용: A2 앵커 로직
        anchor = df["은행명"].where(df["코드"]=="A2")
        bank_ff = anchor.ffill()
        bank_bf = anchor.bfill()
        domestic = {"A1","A11","A12"}
        foreign  = {"A2","A21","A22","A23","A"}
        bank_assigned = pd.Series(index=df.index, dtype=object)
        bank_assigned[df["코드"].isin(domestic)] = bank_bf[df["코드"].isin(domestic)]
        bank_assigned[df["코드"].isin(foreign)]  = bank_ff[df["코드"].isin(foreign)]
        df["은행명"] = bank_assigned
    else:
        # 일반(자동화기기 등): 상하 전파
        df["은행명"] = df["은행명"].ffill().bfill()

    # 총계 플래그
    df["is_total"] = df["구분"].map(_is_total_label) | df["은행명"].map(_is_total_label)

    # tidy
    id_vars = ["은행명","구분","코드","is_total"]
    tidy = df.melt(id_vars=id_vars, value_vars=time_cols,
                   var_name="시점_표기", value_name="금액_raw")
    tidy["시점"] = tidy["시점_표기"].map(_parse_period_to_month_end)
    tidy["금액"] = tidy["금액_raw"].map(_clean_money)
    tidy = tidy.drop(columns=["시점_표기","금액_raw"]).dropna(subset=["시점"]).reset_index(drop=True)

    # (시점, 은행명, 코드) 중복 → 마지막 유효값
    tidy["_rn"] = range(len(tidy))
    def _pick_last_nonnull(group):
        g = group.sort_values("_rn")
        vals = g["금액"].dropna()
        return vals.iloc[-1] if not vals.empty else g["금액"].iloc[-1]
    dedup = (tidy.groupby(["시점","은행명","코드"], as_index=False)
                  .apply(lambda d: pd.Series({"금액": _pick_last_nonnull(d)}))
                  .reset_index(drop=True))

    # wide
    wide_code  = dedup.pivot(index=["시점","은행명"], columns="코드", values="금액").reset_index()
    wide_multi = (tidy[tidy["is_total"]]
                  .groupby(["시점","코드"], as_index=False)["금액"].sum()
                  .pivot(index="시점", columns="코드", values="금액").reset_index())

    return tidy.drop(columns=["_rn"], errors="ignore"), wide_code, wide_multi

def save_one(path: str, outdir: str, sheet: Optional[int] = "auto"):
    """
    단일 파일을 읽어 <prefix>_tidy.csv / <prefix>_wide_code.csv / <prefix>_wide_multi.csv 저장
    prefix = 파일명에서 '_' 앞부분 (없으면 전체 파일명)
    """
    os.makedirs(outdir, exist_ok=True)
    tidy, wide_code, wide_multi = read_one(path, sheet=sheet)
    base = os.path.splitext(os.path.basename(path))[0]
    prefix = base.split("_")[0] if "_" in base else base
    tidy.to_csv(os.path.join(outdir, f"{prefix}_tidy.csv"), index=False)
    wide_code.to_csv(os.path.join(outdir, f"{prefix}_wide_code.csv"), index=False)
    wide_multi.to_csv(os.path.join(outdir, f"{prefix}_wide_multi.csv"), index=False)
    return (os.path.join(outdir, f"{prefix}_tidy.csv"),
            os.path.join(outdir, f"{prefix}_wide_code.csv"),
            os.path.join(outdir, f"{prefix}_wide_multi.csv"))

def process_many(pattern: str, outdir: str, sheet: Optional[int] = "auto"):
    """
    글롭 패턴으로 여러 파일을 받아 각 파일별로 저장.
    반환: [(원본파일, tidy_path, wide_code_path, wide_multi_path), ...]
    """
    paths = sorted(glob.glob(pattern))
    os.makedirs(outdir, exist_ok=True)
    out = []
    for p in paths:
        try:
            paths_ = save_one(p, outdir=outdir, sheet=sheet)
            out.append((p, ) + paths_)
        except Exception as e:
            out.append((p, f"ERROR: {e}", "", ""))
    return out
