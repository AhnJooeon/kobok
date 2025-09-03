import pandas as pd
import re
from pathlib import Path

# ===== 경로 설정 =====
INPUT_DIR  = Path("./bss_result/재무현황")     # 원본 폴더
OUTPUT_DIR = Path("output_folder")    # 결과 폴더
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_ROW = 1  # 0행 버리고 "1행을 헤더"(0-index 기준)

# ===== 공통 유틸 =====
def read_excel_header1(path: Path) -> pd.DataFrame:
    """엑셀을 헤더 없이 읽고 → 1행을 헤더로 강제 지정 (병합/빈칸 내성)"""
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
    else:
        raise ValueError(f"엑셀만 처리: {path.name}")
    if df.shape[0] <= HEADER_ROW:
        raise ValueError("데이터 행이 부족하여 1행을 헤더로 사용할 수 없습니다.")
    header = pd.Series(df.iloc[HEADER_ROW].values.flatten(), dtype="string")\
               .str.replace(r"^\s*nan\s*$", "", regex=True).str.strip()
    df = df.iloc[HEADER_ROW+1:].reset_index(drop=True)
    # 열 개수 보정
    if len(header) < df.shape[1]:
        header = list(header) + [f"col_extra_{i}" for i in range(df.shape[1]-len(header))]
    elif len(header) > df.shape[1]:
        header = list(header)[:df.shape[1]]
    df.columns = header
    # 전부 공백/결측인 열 제거
    stripped = df.apply(lambda s: s.astype(str).str.strip())
    all_blank_or_na = (stripped.eq("") | df.isna()).all(axis=0)
    return df.loc[:, ~all_blank_or_na]

def detect_first_cycle(codes: list[str]) -> list[str]:
    """구분 시퀀스에서 첫 반복 사이클 반환 (예: a,b,c,a,b,c -> a,b,c)"""
    if not codes:
        return []
    # 연속 중복 제거
    compact = [codes[0]] + [c for i, c in enumerate(codes[1:], 1) if c != codes[i-1]]
    pattern = []
    for c in compact:
        if not pattern:
            pattern.append(c)
            continue
        if c == pattern[0]:  # 시작 코드 재등장 → 한 사이클 끝
            break
        if c not in pattern:
            pattern.append(c)
    return pattern if pattern else [compact[0]]

def fill_bank_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """구분 패턴 단위 블록으로 묶고, 은행명 유니크 등장순으로 각 블록 빈칸 채움"""
    if "구분" not in df.columns or "은행명" not in df.columns:
        raise KeyError("필수 컬럼 누락: '구분', '은행명'")
    out = df.copy()
    out["구분"] = out["구분"].astype(str).str.strip()
    out["은행명"] = out["은행명"].astype(str).str.strip()
    out.loc[out["은행명"].eq("") | out["은행명"].isin(["nan","NaN","None"]), "은행명"] = pd.NA

    codes = out["구분"].tolist()
    pattern = detect_first_cycle(codes)
    start_code = pattern[0] if pattern else codes[0]

    # start_code 나올 때마다 블록 증가
    block_id = out["구분"].eq(start_code).cumsum()

    # 은행명 유니크 등장순
    seen = set()
    bank_order = []
    for x in out["은행명"]:
        if pd.isna(x) or x == "":
            continue
        if x not in seen:
            seen.add(x); bank_order.append(x)
    name_iter = iter(bank_order)

    # 블록별로 빈칸만 채움(이미 값 있으면 유지)
    for bid in sorted(block_id.unique()):
        fill_name = next(name_iter, None)
        if fill_name is None:
            break  # 더 채울 이름이 없으면 종료
        m_block = (block_id == bid)
        empty = out.loc[m_block, "은행명"].isna()
        if empty.any():
            out.loc[m_block & empty, "은행명"] = fill_name
    return out

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if re.match(r'^\s*(?:19|20)\d{2}', str(c))]
    if not date_cols:
        raise ValueError("날짜(기간) 컬럼을 찾지 못했습니다. 예: 2000년09월말, 2025-03 등")
    id_vars = [c for c in df.columns if c not in date_cols]
    long = df.melt(id_vars=id_vars, value_vars=date_cols, var_name="날짜", value_name="값")
    long["값"] = pd.to_numeric(
        long["값"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    return long

def to_pivot(long: pd.DataFrame) -> pd.DataFrame:
    """행=날짜, 열=(구분,코드) (코드 없으면 구분만)"""
    cols = ["구분"] + (["코드"] if "코드" in long.columns else [])
    pt = long.pivot_table(index="날짜", columns=cols, values="값", aggfunc="sum", observed=True)
    # 컬럼 평탄화
    if isinstance(pt.columns, pd.MultiIndex):
        pt.columns = ["_".join(map(str, t)) for t in pt.columns.to_flat_index()]
    else:
        pt.columns = [str(c) for c in pt.columns]
    return pt.reset_index()

# ===== 폴더 처리 =====
for f in INPUT_DIR.glob("*.xls*"):
    try:
        df_raw = read_excel_header1(f)
        df_filled = fill_bank_by_pattern(df_raw)
        df_long = to_long(df_filled)
        df_pivot = to_pivot(df_long)

        # 저장
        (OUTPUT_DIR / "per_file").mkdir(exist_ok=True)
        df_long.to_excel(OUTPUT_DIR / "per_file" / f"{f.stem}_long.xlsx", index=False)
        df_pivot.to_excel(OUTPUT_DIR / "per_file" / f"{f.stem}_pivot.xlsx", index=False)

        print(f"[완료] {f.name} → {f.stem}_long.xlsx, {f.stem}_pivot.xlsx")
    except Exception as e:
        print(f"[오류] {f.name}: {e}")
