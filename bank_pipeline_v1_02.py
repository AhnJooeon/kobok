import pandas as pd, re, math, calendar

# ────────── helpers (이름 유지) ──────────
def _last_day(y, m):
    return calendar.monthrange(y, m)[1]

def _parse_kor_period(col):
    """
    엑셀 날짜, 'YYYY년MM월(말/현재/기준)', 'YYYY.MM', 'YYYY-MM', 'YYYYMM' → 월말 Timestamp
    """
    # 1) 날짜형 먼저
    try:
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts.year, ts.month, _last_day(ts.year, ts.month))
    except Exception:
        pass
    # 2) 문자열 패턴
    s = str(col).replace("\xa0", " ").strip()
    for pat in [
        r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?:말|말일|현재|기준)?",
        r"(?P<y>\d{4})[.\-/\s](?P<m>\d{1,2})",
        r"(?P<y>\d{4})(?P<m>\d{2})",
    ]:
        m = re.fullmatch(pat, s)
        if m:
            y, mm = int(m.group("y")), int(m.group("m"))
            if 1 <= mm <= 12:
                return pd.Timestamp(y, mm, _last_day(y, mm))
    return pd.NaT

def _clean_str(x):
    return x.replace("\xa0", " ").strip() if isinstance(x, str) else x

def _clean_money(x):
    """
    '1,234', '(123)', '−5', '△7', '8 %' 등 숫자화. 실패 시 NaN
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A", "NA", "NULL", "-"}:
        return math.nan
    neg = False
    if re.match(r"^\(.*\)$", s):  # (123) → -123
        neg, s = True, s.strip("()")
    s = (s.replace("−", "-").replace("–", "-").replace("△", "-")
           .replace("%", "").replace(",", "").replace(" ", "").replace("\t", ""))
    try:
        v = float(s)
    except:
        return math.nan
    return -v if neg and v >= 0 else v

# ────────── header detect (이름 유지, 로직 강화) ──────────
def _detect_header_row_single(path, sheet=0, probe=30):
    """
    상단 probe행 스캔해서 헤더 후보를 스코어링:
      - 날짜처럼 보이는 셀 수(가중 2)
      - '코드' 동의어 존재(가중 2)
      - '은행명/기관명' 계열 존재(가중 1)
    최고점 행을 헤더로 채택.
    """
    peek = pd.read_excel(path, sheet_name=sheet, header=None, nrows=probe, dtype=str)
    peek = peek.apply(lambda col: col.map(_clean_str))
    code_syn = {"코드", "code", "항목코드", "계정코드", "분류코드"}
    bank_syn = {"은행명", "기관명", "금융기관", "금융기관명", "금융회사", "금융회사명", "기관"}
    best_i, best_score = 1, -1
    for i in range(len(peek)):
        vals = [str(v) for v in list(peek.iloc[i].values)]
        norm = {re.sub(r"\s+", "", v).lower() for v in vals if v not in {"", "nan", "None"}}
        has_code = any(c.lower() in norm for c in code_syn)
        has_bank = any(b.lower() in norm for b in bank_syn)
        time_like = sum(pd.notna(_parse_kor_period(v)) for v in vals)
        score = time_like * 2 + (2 if has_code else 0) + (1 if has_bank else 0)
        if score > best_score:
            best_score, best_i = score, i
    return best_i

# ────────── main (이름/시그니처 유지) ──────────
def read_branch_status(path: str, sheet: int = 0):
    """
    반환(동일):
      tidy        : 전체 세로형 (은행명/구분/코드/시점/금액/is_total)
      wide_code   : (시점[, 은행명]) × 코드  — 총계 제외
      wide_multi  : (시점) × 코드          — 총계만
    """
    # 1) 헤더 감지 & 로드
    h = _detect_header_row_single(path, sheet=sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=h, dtype=str)
    df = df.apply(lambda col: col.map(_clean_str))

    # 2) 컬럼 정규화/보정
    if "기관명" in df.columns and "은행명" not in df.columns:
        df = df.rename(columns={"기관명": "은행명"})
    # '코드'가 없으면 '구분'을 임시코드로 사용(임직원 등 호환)
    if "코드" not in df.columns:
        if "구분" in df.columns:
            df["코드"] = df["구분"]
        else:
            # 라벨 컬럼이 전혀 없을 때 대비: 첫 비-시점 컬럼을 코드로
            first_non_time = next((c for c in df.columns if pd.isna(_parse_kor_period(c))), None)
            df["코드"] = first_non_time if first_non_time is not None else "항목"

    # 3) 시점 컬럼 선정
    time_cols = [c for c in df.columns if pd.notna(_parse_kor_period(c))]
    if not time_cols:
        raise ValueError("시점 컬럼(엑셀 날짜 또는 YYYY년MM월/ YYYY.MM/ YYYY-MM/ YYYYMM) 없음")

    # 4) 은행명 채우기 전략
    #    - 영업점 계열: A2 행에 실제 은행명이 있으면 'A2-앵커' 전파
    #    - 그 외: 비결측 기반 ffill/bfill (임직원/자동화기기 등)
    if "은행명" in df.columns:
        codeset = set(df["코드"].dropna().unique().tolist()) if "코드" in df.columns else set()
        if {"A1", "A2"} & codeset and df.loc[df["코드"] == "A2", "은행명"].notna().any():
            anchor = df["은행명"].where(df["코드"] == "A2")
            bank_ff = anchor.ffill()
            bank_bf = anchor.bfill()
            domestic = {"A1", "A11", "A12"}
            foreign  = {"A2", "A21", "A22", "A23", "A"}
            bank_assigned = pd.Series(index=df.index, dtype=object)
            bank_assigned[df["코드"].isin(domestic)] = bank_bf[df["코드"].isin(domestic)]
            bank_assigned[df["코드"].isin(foreign)]  = bank_ff[df["코드"].isin(foreign)]
            df["은행명"] = bank_assigned
        else:
            # A2에 은행명이 없거나 영업점 패턴이 아니면: 근접 비결측 전파
            df["은행명"] = df["은행명"].where(df["은행명"].notna()).ffill().bfill()

    # 5) 총계 플래그 (과도하지 않게 — 정확 일치 위주)
    total_tokens = {"합계", "총계", "소계"}
    def _is_total_row(r):
        for key in ["은행명", "구분", "코드"]:
            if key in r and r[key] is not None and str(r[key]).strip() in total_tokens:
                return True
        return False
    df["is_total"] = df.apply(_is_total_row, axis=1)

    # 6) 세로화
    id_vars = [c for c in ["은행명", "구분", "코드", "is_total"] if c in df.columns]
    tidy = (
        df.melt(id_vars=id_vars, value_vars=time_cols,
                var_name="시점_표기", value_name="금액_raw")
          .assign(시점=lambda d: d["시점_표기"].map(_parse_kor_period))
    )
    tidy["금액"] = tidy["금액_raw"].map(_clean_money)
    tidy = (
        tidy.drop(columns=["시점_표기", "금액_raw"])
            .dropna(subset=["시점"])
            .sort_values(["시점"] + [c for c in ["은행명", "구분", "코드"] if c in tidy.columns])
            .reset_index(drop=True)
    )

    # 7) wide_code: 총계 제외, 은행명 있으면 (시점, 은행명), 없으면 (시점)
    tidy_bank = tidy[~tidy["is_total"]].copy() if "is_total" in tidy.columns else tidy.copy()
    idx_cols = ["시점"] + (["은행명"] if "은행명" in tidy_bank.columns else [])
    wide_code = (
        tidy_bank.pivot_table(index=idx_cols, columns="코드", values="금액", aggfunc="first")
                 .reset_index()
    )
    wide_code.columns = [str(c) for c in wide_code.columns]

    # 8) wide_multi: 총계만 — 은행 축 없음 (없으면 빈 프레임)
    if "is_total" in tidy.columns and tidy["is_total"].any():
        tidy_tot = tidy[tidy["is_total"]].copy()
        wide_multi = (
            tidy_tot.pivot_table(index=["시점"], columns="코드", values="금액", aggfunc="first")
                    .reset_index()
        )
        wide_multi.columns = [str(c) for c in wide_multi.columns]
    else:
        wide_multi = pd.DataFrame(columns=["시점"])

    return tidy, wide_code, wide_multi
