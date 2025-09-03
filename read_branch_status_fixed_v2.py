
# -*- coding: utf-8 -*-
"""
영업점포 현황(엑셀) → tidy / wide_code / wide_multi 변환 (v2: 가변 '구분' 지원)
- '구분' 컬럼명이 파일마다 다를 수 있어 자동 탐지하거나 명시적으로 지정 가능
- 은행명: A2(국외) 행을 앵커로 사용하여 코드별로 ffill/bfill 배치
  * A1/A11/A12(국내/지점/출장소) → 다음 A2 은행명(bfill)
  * A2/A21/A22/A23/A(국외/세부/합계) → 현재 A2 은행명(ffill)
- 시점 컬럼 파싱: 'YYYY년MM월(말)'/YYYYMM/YYYY-MM/엑셀 날짜 → 월말 anchor
- 숫자 클린, 중복 해소(원본 순서 기준 마지막 유효값), wide 생성
"""

import pandas as pd, re, math, calendar
from typing import Tuple, Optional, List

# -----------------------------
# Helpers
# -----------------------------
def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def _parse_kor_period(col) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts.year, ts.month, _last_day(ts.year, ts.month))
    except Exception:
        pass
    s = str(col).replace("\xa0", " ").strip()
    for pat in [
        r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*말",
        r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월",
        r"(?P<y>\d{4})(?P<m>\d{2})",
        r"(?P<y>\d{4})-(?P<m>\d{1,2})",
    ]:
        m = re.fullmatch(pat, s)
        if m:
            y, mm = int(m.group("y")), int(m.group("m"))
            if 1 <= mm <= 12:
                return pd.Timestamp(y, mm, _last_day(y, mm))
    return pd.NaT

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

def _detect_header_row_single(path: str, sheet=0, probe=30) -> int:
    peek = pd.read_excel(path, sheet_name=sheet, header=None, nrows=probe, dtype=str)
    peek = peek.apply(lambda col: col.map(_clean_str))
    for i in range(len(peek)):
        vals = set(v for v in peek.iloc[i].astype(str) if v not in {"", "nan", "None"})
        norm = {re.sub(r"\s+","", v) for v in vals}
        if {"은행명","구분","코드"}.issubset(norm) or {"기관명","구분","코드"}.issubset(norm):
            return i
    return 1

def _normalize_header_names(df: pd.DataFrame) -> pd.DataFrame:
    bank_syn = {"은행명","은행","기관명","금융기관","금융기관명","금융회사","금융회사명","기관"}
    # '구분' 동의어 확장
    gubun_syn = {"구분","분류","분류1","분류2","항목","항목명","유형","대분류","중분류","소분류","세부","세목","구간","지역구분","구성"}
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
    return df

def _is_total_label(x) -> bool:
    if x is None or (isinstance(x, float) and math.isnan(x)): return False
    s = str(x).replace("\xa0"," ").strip()
    return bool(re.search(r"(?:^|\s)(합계|총계|소계|전체)(?:\s|$)", s))

def _choose_group_col(df: pd.DataFrame, time_cols: List[str]) -> Optional[str]:
    """'구분'이 없을 때, 가장 그럴듯한 범주형 컬럼을 자동 선택."""
    if "구분" in df.columns:
        return "구분"
    # 후보: object형 & 시간이 아니며 & 은행명/코드가 아니고 & 고유값 비율이 낮은 컬럼
    n = len(df)
    candidates = []
    for c in df.columns:
        if c in ("은행명","코드"): 
            continue
        if c in time_cols: 
            continue
        if df[c].dtype != object:
            continue
        uniq = df[c].dropna().nunique()
        ratio = (uniq / max(n,1))
        if 0 < uniq < min(50, n) and ratio < 0.6:
            candidates.append((c, uniq, ratio))
    # 가장 고유값이 적은(범주형일 가능성 높은) 것 선택
    candidates.sort(key=lambda x: (x[1], x[2]))
    return candidates[0][0] if candidates else None

# -----------------------------
# Main
# -----------------------------
def read_branch_status(path: str, sheet: int = 0, group_col: Optional[str] = "auto") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    반환:
      tidy        : 세로형 (은행명/[구분]/코드/시점/금액)
      wide_code   : (시점, 은행명) 가로형
      wide_multi  : (시점) 가로형(총계)
    인자:
      group_col: '구분' 컬럼명 지정 or "auto" (기본). None이면 구분 없이 처리.
    """
    # 1) 로드 & 정리
    h = _detect_header_row_single(path, sheet=sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=h, dtype=str)
    df = df.apply(lambda col: col.map(_clean_str))
    df = _normalize_header_names(df)

    if "은행명" not in df.columns or "코드" not in df.columns:
        missing = [c for c in ["은행명","코드"] if c not in df.columns]
        raise KeyError(f"필수 컬럼 누락: {missing}")

    # 2) 은행명 배치 (A2 앵커)
    codes = df["코드"]
    anchor = df["은행명"].where(codes=="A2")
    bank_ff = anchor.ffill()
    bank_bf = anchor.bfill()
    domestic_codes = {"A1","A11","A12"}
    foreign_codes  = {"A2","A21","A22","A23","A"}
    bank_assigned = pd.Series(index=df.index, dtype=object)
    bank_assigned[codes.isin(domestic_codes)] = bank_bf[codes.isin(domestic_codes)]
    bank_assigned[codes.isin(foreign_codes)]  = bank_ff[codes.isin(foreign_codes)]
    df["은행명"] = bank_assigned

    # 3) 시점 컬럼
    time_cols = [c for c in df.columns if pd.notna(_parse_kor_period(c))]
    if not time_cols:
        raise ValueError("시점 컬럼(엑셀 날짜 또는 'YYYY년MM월(말)'/YYYYMM/YYYY-MM) 없음")

    # 4) group_col 자동/명시 처리
    if group_col == "auto":
        gcol = _choose_group_col(df, time_cols)
    else:
        gcol = group_col if (group_col in df.columns) else None

    # 5) tidy 생성
    id_vars = ["은행명","코드"]
    if gcol is not None:
        id_vars.insert(1, gcol)  # 은행명, 구분, 코드 순서
    tidy = df.melt(id_vars=id_vars, value_vars=time_cols,
                   var_name="시점_표기", value_name="금액_raw")
    tidy["금액"] = tidy["금액_raw"].map(_clean_money)
    tidy["시점"] = tidy["시점_표기"].map(_parse_kor_period)
    tidy = tidy.drop(columns=["시점_표기","금액_raw"]).dropna(subset=["시점"]).reset_index(drop=True)

    # 6) 총계 플래그: gcol과 은행명 모두 검사 (있을 때만)
    if gcol is not None:
        tidy["is_total"] = tidy[gcol].map(_is_total_label) | tidy["은행명"].map(_is_total_label)
    else:
        tidy["is_total"] = tidy["은행명"].map(_is_total_label)

    # 7) 중복 해소
    group_keys = ["시점","은행명","코드"]
    tidy["_rn"] = range(len(tidy))
    def _pick_last_nonnull(group):
        g = group.sort_values("_rn")
        vals = g["금액"].dropna()
        return vals.iloc[-1] if not vals.empty else g["금액"].iloc[-1]
    dedup = (tidy.groupby(group_keys, as_index=False)
                  .apply(lambda d: pd.Series({"금액": _pick_last_nonnull(d)}))
                  .reset_index(drop=True))

    # 8) wide 생성
    wide_code  = dedup.pivot(index=["시점","은행명"], columns="코드", values="금액").reset_index()
    wide_multi = (dedup.groupby(["시점","코드"], as_index=False)["금액"].sum()
                       .pivot(index=["시점"], columns="코드", values="금액")
                       .reset_index())

    # tidy 컬럼 순서 정리
    base_cols = ["은행명"]
    if gcol is not None: base_cols.append(gcol)
    base_cols += ["코드","시점","금액","is_total"]
    tidy = tidy[base_cols + ([c for c in tidy.columns if c not in set(base_cols + ["_rn"])])]
    return tidy.drop(columns=["_rn"], errors="ignore"), wide_code, wide_multi
