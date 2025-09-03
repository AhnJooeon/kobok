# -*- coding: utf-8 -*-
import pandas as pd
import re
from pathlib import Path

# =========================
# 설정
# =========================
INPUT_DIR  = Path("./bss_result/재무현황")   # 원본 폴더
OUTPUT_DIR = Path("output_folder/per_file")  # 결과 폴더
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_ROW = 1  # 0행 버리고 "1행을 헤더"(0-index)
CREATE_TIMESLICES_IF_NO_DATE = True  # 날짜 헤더가 없으면 '시점001..' 생성

ID_COLS = {"은행명", "구분", "코드"}  # 식별자 후보

# =========================
# 공통 유틸
# =========================
def clean_str_series(s: pd.Series) -> pd.Series:
    """None/NaN/<NA>/공백 -> '' 로 통일 (불리언 모호성 제거용)"""
    def conv(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        x = str(v).strip()
        xl = x.lower()
        return "" if xl in {"", "nan", "none", "<na>"} else x
    return s.astype("object").map(conv)

def is_date_like_token(x: str) -> bool:
    return bool(re.match(r"^\s*(?:19|20)\d{2}", str(x or "")))

def read_excel_raw(path: Path) -> pd.DataFrame:
    """확장자에 맞춰 엑셀을 헤더 없이 raw로 읽기"""
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xlsm", ".xltx"]:
        return pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
    elif suf == ".xls":
        # xlrd 1.2.0 필요. 없으면 에러 메시지 친절히.
        try:
            return pd.read_excel(path, header=None, dtype=str, engine="xlrd")
        except Exception as e:
            raise RuntimeError(f".xls 읽기 실패 (xlrd 1.2.0 필요): {e}")
    else:
        raise ValueError(f"지원하지 않는 확장자: {path.name}")

# =========================
# 1) 1행 헤더 기본 리더
# =========================
def read_excel_header1(path: Path) -> pd.DataFrame:
    df0 = read_excel_raw(path)
    if df0.shape[0] <= HEADER_ROW:
        raise ValueError("데이터 행이 부족하여 1행을 헤더로 사용할 수 없습니다.")
    header = pd.Series(df0.iloc[HEADER_ROW].values.flatten(), dtype="string")\
               .str.replace(r"^\s*nan\s*$", "", regex=True).str.strip()
    df = df0.iloc[HEADER_ROW+1:].reset_index(drop=True)

    # 열 개수 보정
    if len(header) < df.shape[1]:
        header = list(header) + [f"col_extra_{i}" for i in range(df.shape[1]-len(header))]
    elif len(header) > df.shape[1]:
        header = list(header)[:df.shape[1]]
    df.columns = header

    # 올블랭크 열 제거
    stripped = df.apply(lambda s: clean_str_series(s))
    all_blank_or_na = (stripped.eq("")).all(axis=0)
    return df.loc[:, ~all_blank_or_na]

# =========================
# 2) 2줄 헤더 + 같은 날짜 반복 → 첫 서브만 유지
# =========================
def read_excel_header1_or_2sub_first(path: Path) -> pd.DataFrame:
    df0 = read_excel_raw(path)
    if df0.shape[0] <= HEADER_ROW:
        raise ValueError("데이터 행이 부족하여 1행을 헤더로 사용할 수 없습니다.")

    # 상단 헤더 + 가로 ffill(병합셀 대응)
    top = df0.iloc[HEADER_ROW].astype(str).str.strip().tolist()
    if len(top) < df0.shape[1]:
        top += [""] * (df0.shape[1] - len(top))
    last = ""
    for i, v in enumerate(top):
        if v == "" or re.fullmatch(r"\s*nan\s*", v, flags=re.I):
            top[i] = last
        else:
            last = v

    has_second = df0.shape[0] > HEADER_ROW + 1
    sub = df0.iloc[HEADER_ROW+1].astype(str).str.strip().tolist() if has_second else []
    if len(sub) < df0.shape[1]:
        sub += [""] * (df0.shape[1] - len(sub))

    # 같은 날짜가 2번 이상 반복되는지 확인
    date_tokens = [t for t in top if is_date_like_token(t)]
    use_two = has_second and len(date_tokens) >= 2 and len(set(date_tokens)) < len(date_tokens)

    if not use_two:
        # 단일 헤더 처리
        return read_excel_header1(path)

    keep_idx, new_header, seen_dates = [], [], set()
    ncol = df0.shape[1]
    for j in range(ncol):
        t = re.sub(r"^\s*nan\s*$", "", str(top[j] if j < len(top) else ""), flags=re.I).strip()
        s = re.sub(r"^\s*nan\s*$", "", str(sub[j] if j < len(sub) else ""), flags=re.I).strip()
        if is_date_like_token(t):
            if t not in seen_dates:
                keep_idx.append(j)
                new_header.append(t)  # 날짜만 사용
                seen_dates.add(t)
            # 같은 날짜의 두 번째 서브는 버림
        else:
            name = t or s or f"col{j+1}"
            keep_idx.append(j)
            new_header.append(name)

    df = df0.iloc[HEADER_ROW+2:, keep_idx].reset_index(drop=True)
    df.columns = new_header

    # 올블랭크 열 제거
    stripped = df.apply(lambda s: clean_str_series(s))
    all_blank_or_na = (stripped.eq("")).all(axis=0)
    return df.loc[:, ~all_blank_or_na]

# =========================
# 3) 구분 패턴 탐지
# =========================
def detect_first_cycle(codes: list[str]) -> list[str]:
    if not codes:
        return []
    compact = [codes[0]] + [c for i, c in enumerate(codes[1:], 1) if c != codes[i-1]]
    pattern = []
    for c in compact:
        if not pattern:
            pattern.append(c); continue
        if c == pattern[0]:
            break
        if c not in pattern:
            pattern.append(c)
    return pattern if pattern else [compact[0]]

# =========================
# 4) 은행명 채우기 (구분 패턴 블록 × 은행명 유니크 등장순)
# =========================
def fill_bank_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    if "구분" not in df.columns or "은행명" not in df.columns:
        # 필수 컬럼이 없으면 그대로 반환 (파이프라인 유지)
        return df.copy()

    out = df.copy()
    out["구분"]   = clean_str_series(out["구분"])
    out["은행명"] = clean_str_series(out["은행명"])

    codes_nonempty = [c for c in out["구분"].tolist() if c != ""]
    if not codes_nonempty:
        return out

    pattern   = detect_first_cycle(codes_nonempty)
    start_code = pattern[0] if pattern else codes_nonempty[0]

    start_mask = out["구분"].apply(lambda x: x == start_code)   # 순수 bool
    block_id   = start_mask.astype(int).cumsum()

    # 파일 내 은행명 유니크 등장순
    seen, bank_order = set(), []
    for x in out["은행명"]:
        if x and x not in seen:
            seen.add(x); bank_order.append(x)
    name_iter = iter(bank_order)

    # 블록별로 '빈칸'만 채움
    for bid in sorted(block_id.unique()):
        fill_name = next(name_iter, None)
        if fill_name is None:
            break
        mask_block = (block_id == bid)
        mask_empty = out["은행명"].eq("")  # 문자열 비교이므로 NA 모호성 없음
        out.loc[mask_block & mask_empty, "은행명"] = fill_name
    return out

# =========================
# 5) 날짜 컬럼 확보 (없으면 시점NNN 생성)
# =========================
def ensure_date_columns(df: pd.DataFrame):
    date_cols = [c for c in df.columns if is_date_like_token(str(c))]
    if date_cols:
        return df, date_cols

    if not CREATE_TIMESLICES_IF_NO_DATE:
        raise ValueError("날짜(19/20..)로 시작하는 컬럼이 없습니다.")

    # 식별자 제외한 값 컬럼을 좌→우로 시점NNN으로 치환
    value_cols = [c for c in df.columns if (c not in ID_COLS and not is_date_like_token(str(c)))]
    mapping = {}
    for i, c in enumerate(value_cols, 1):
        mapping[c] = f"시점{i:03d}"
    new_df = df.rename(columns=mapping)
    return new_df, list(mapping.values())

# =========================
# 6) wide → long
# =========================
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    df2, date_cols = ensure_date_columns(df)
    id_vars = [c for c in df2.columns if c not in date_cols]

    long = df2.melt(id_vars=id_vars, value_vars=date_cols, var_name="날짜", value_name="값")

    # 숫자화 + 식별자 문자열 통일
    long["값"] = pd.to_numeric(clean_str_series(long["값"]).str.replace(",", "", regex=False), errors="coerce")
    for col in ["은행명", "구분", "코드", "날짜"]:
        if col in long.columns:
            long[col] = clean_str_series(long[col])
    return long

# =========================
# 7) long → pivot (열 라벨 = "은행명_구분_코드")
# =========================
def to_pivot(long: pd.DataFrame) -> pd.DataFrame:
    need = {"날짜", "은행명", "구분"}
    miss = need - set(long.columns)
    if miss:
        raise KeyError(f"피벗에 필요한 컬럼 없음: {sorted(miss)}")

    has_code = "코드" in long.columns
    key_cols = ["은행명", "구분"] + (["코드"] if has_code else [])

    dfp = long.copy()
    dfp["날짜"] = clean_str_series(dfp["날짜"]).replace({"": "미정"})
    for c in key_cols:
        dfp[c] = clean_str_series(dfp[c]).replace({"": "미정"})

    pt = dfp.pivot_table(
        index="날짜",
        columns=key_cols,
        values="값",
        aggfunc="sum",
        observed=True,
    )

    # 열 라벨 평탄화 + 유니크 처리
    if isinstance(pt.columns, pd.MultiIndex):
        labels = ["_".join(map(str, t)) for t in pt.columns.to_flat_index()]
    else:
        labels = [str(c) for c in pt.columns]

    seen = {}
    uniq = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            uniq.append(f"{lab}__{seen[lab]}")
        else:
            seen[lab] = 0
            uniq.append(lab)
    pt.columns = uniq

    return pt.reset_index()

# =========================
# 8) 폴더 처리
# =========================
if __name__ == "__main__":
    for f in INPUT_DIR.glob("*.xls*"):
        try:
            # 2줄 헤더(같은 날짜 반복) 감지 → 첫 서브만 유지
            try:
                df_raw = read_excel_header1_or_2sub_first(f)
                reader_used = "2row_first"
            except Exception:
                df_raw = read_excel_header1(f)
                reader_used = "1row"

            # 은행명 채우기
            df_filled = fill_bank_by_pattern(df_raw)

            # long / pivot
            df_long  = to_long(df_filled)
            df_pivot = to_pivot(df_long)

            # 저장
            out_long  = OUTPUT_DIR / f"{f.stem}_long.xlsx"
            out_pivot = OUTPUT_DIR / f"{f.stem}_pivot.xlsx"
            df_long.to_excel(out_long, index=False)
            df_pivot.to_excel(out_pivot, index=False)

            print(f"[완료] {f.name} ({reader_used}) → {out_long.name}, {out_pivot.name}")
        except Exception as e:
            print(f"[오류] {f.name}: {e}")
