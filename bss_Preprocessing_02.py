
# -*- coding: utf-8 -*-
"""
영업점포 현황(엑셀) → tidy / wide_code / wide_multi 변환
- 헤더 자동 감지 (이 파일은 보통 1행이 헤더)
- 은행명: A2(국외) 행을 앵커로 사용하여 코드별로 ffill/bfill 배치
  * A1/A11/A12(국내/지점/출장소) → 다음 A2 은행명(bfill)
  * A2/A21/A22/A23/A(국외/세부/합계) → 현재 A2 은행명(ffill)
- 시점 컬럼 파싱: 'YYYY년MM월(말)'/YYYYMM/YYYY-MM/엑셀 날짜 → 월말 anchor
- 숫자 클린: 콤마/괄호음수/특수 마이너스/공백/퍼센트
- (시점, 은행명, 코드) 중복은 "원본 순서 기준 마지막 유효값" 채택
- wide_code: (시점, 은행명)×코드, wide_multi: 시점×코드(총계)
"""

import pandas as pd, re, math, calendar
from typing import Tuple

# -----------------------------
# Helpers
# -----------------------------
def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def _parse_kor_period(col) -> pd.Timestamp:
    """엑셀 날짜/문자 모두 처리. 월말 anchor로 변환."""
    # 1) 엑셀/숫자형 날짜
    try:
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts.year, ts.month, _last_day(ts.year, ts.month))
    except Exception:
        pass
    # 2) 문자열 패턴
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
    """'1,234', '(123)', '−5', '△7', ' 8 % ' 등 숫자화."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A","NA","NULL","-"}:
        return math.nan
    neg = False
    if re.match(r"^\(.*\)$", s):  # 괄호 음수
        neg, s = True, s.strip("()")
    s = (s.replace("−","-").replace("–","-").replace("△","-")
           .replace("%","").replace(",", "").replace(" ", "").replace("\t", ""))
    try:
        v = float(s)
    except:
        return math.nan
    return -v if neg and v >= 0 else v

def _detect_header_row_single(path: str, sheet=0, probe=30) -> int:
    """일반화된 헤더 행 탐지. (없으면 1행으로 fallback)"""
    peek = pd.read_excel(path, sheet_name=sheet, header=None, nrows=probe, dtype=str)
    peek = peek.apply(lambda col: col.map(_clean_str))
    for i in range(len(peek)):
        vals = set(v for v in peek.iloc[i].astype(str) if v not in {"", "nan", "None"})
        norm = {re.sub(r"\s+","", v) for v in vals}
        if {"은행명","구분","코드"}.issubset(norm) or {"기관명","구분","코드"}.issubset(norm):
            return i
    return 1

def _normalize_header_names(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 공백/nbsp/동의어 보정 → '은행명','구분','코드'로 통일."""
    bank_syn = {"은행명","은행","기관명","금융기관","금융기관명","금융회사","금융회사명","기관"}
    gubun_syn = {"구분","분류","항목","항목명","계정","계정명"}
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
    """합계/총계/소계/전체만 총계로 본다(토큰 매칭)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    s = str(x).replace("\xa0"," ").strip()
    return bool(re.search(r"(?:^|\s)(합계|총계|소계|전체)(?:\s|$)", s))

# -----------------------------
# Main
# -----------------------------
def read_branch_status(path: str, sheet: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    반환:
      tidy        : 세로형 (은행명/구분/코드/시점/금액)
      wide_code   : (시점, 은행명) 가로형
      wide_multi  : (시점) 가로형(총계)
    """
    # 1) 로드 & 정리
    h = _detect_header_row_single(path, sheet=sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=h, dtype=str)
    df = df.apply(lambda col: col.map(_clean_str))
    df = _normalize_header_names(df)

    # 2) 은행명 배치: A2 앵커 사용 (핵심 로직)
    if "은행명" not in df.columns or "코드" not in df.columns or "구분" not in df.columns:
        missing = [c for c in ["은행명","코드","구분"] if c not in df.columns]
        raise KeyError(f"필수 컬럼 누락: {missing}")
    codes = df["코드"]
    anchor = df["은행명"].where(codes=="A2")  # 은행명은 A2 행에만 명시
    bank_ff = anchor.ffill()   # A2 포함 이후 블록
    bank_bf = anchor.bfill()   # A2 이전 블록(국내)

    domestic_codes = {"A1","A11","A12"}
    foreign_codes  = {"A2","A21","A22","A23","A"}

    bank_assigned = pd.Series(index=df.index, dtype=object)
    mask_dom = codes.isin(domestic_codes)
    mask_for = codes.isin(foreign_codes)
    bank_assigned[mask_dom] = bank_bf[mask_dom]
    bank_assigned[mask_for] = bank_ff[mask_for]
    df["은행명"] = bank_assigned  # 교체

    # 3) 시점 컬럼 추출
    time_cols = [c for c in df.columns if pd.notna(_parse_kor_period(c))]
    if not time_cols:
        raise ValueError("시점 컬럼(엑셀 날짜 또는 'YYYY년MM월(말)'/YYYYMM/YYYY-MM) 없음")

    # 4) tidy 생성
    tidy = df.melt(id_vars=["은행명","구분","코드"], value_vars=time_cols,
                   var_name="시점_표기", value_name="금액_raw")
    tidy["금액"] = tidy["금액_raw"].map(_clean_money)
    tidy["시점"] = tidy["시점_표기"].map(_parse_kor_period)
    tidy = tidy.drop(columns=["시점_표기","금액_raw"]).dropna(subset=["시점"]).reset_index(drop=True)

    # 5) 중복 해소: (시점, 은행명, 코드) 기준 마지막 유효값
    tidy["_rn"] = range(len(tidy))
    def _pick_last_nonnull(group):
        g = group.sort_values("_rn")
        vals = g["금액"].dropna()
        return vals.iloc[-1] if not vals.empty else g["금액"].iloc[-1]
    dedup = (tidy.groupby(["시점","은행명","코드"], as_index=False)
                  .apply(lambda d: pd.Series({"금액": _pick_last_nonnull(d)}))
                  .reset_index(drop=True))

    # 6) wide 생성
    wide_code  = dedup.pivot(index=["시점","은행명"], columns="코드", values="금액").reset_index()
    wide_multi = (dedup.groupby(["시점","코드"], as_index=False)["금액"].sum()
                       .pivot(index=["시점"], columns="코드", values="금액")
                       .reset_index())

    return tidy.drop(columns=["_rn"], errors="ignore"), wide_code, wide_multi


# -----------------------------
# (선택) 실행 예시
# -----------------------------
if __name__ == "__main__":
    path = "./bss_result/일반현황/영업점포 현황_20250820.xlsx"
    tidy, wide_code, wide_multi = read_branch_status(path)
    tidy.to_csv("./_parsed_out/branch_tidy.csv", index=False, encoding="utf-8-sig")
    wide_code.to_csv("./_parsed_out/branch_wide_code.csv", index=False, encoding="utf-8-sig")
    wide_multi.to_csv("./_parsed_out/branch_wide_multi.csv", index=False, encoding="utf-8-sig")
