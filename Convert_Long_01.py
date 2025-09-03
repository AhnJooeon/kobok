import pandas as pd
import re
from collections import OrderedDict

# 0행 버리고 1행을 헤더로 사용
path = "./bss_result/일반현황/영업점포 현황_20250820.xlsx"
df = pd.read_excel(path, header=1, dtype=str, engine="openpyxl")

if "구분" not in df.columns or "은행명" not in df.columns:
    raise KeyError("필수 컬럼 '구분' 또는 '은행명'이 없습니다.")

df["구분"] = df["구분"].astype(str).str.strip()
df["은행명"] = df["은행명"].astype(str).str.strip()
df.loc[df["은행명"].eq("") | df["은행명"].isin(["nan", "NaN", "None"]), "은행명"] = pd.NA

codes = df["구분"].tolist()
if not codes:
    raise ValueError("구분 데이터가 비어 있습니다.")

# ---- 2) 구분에서 반복 패턴 자동 탐지 (첫 사이클) ----
# 연속 중복 제거
codes_compact = [codes[0]] + [c for i, c in enumerate(codes[1:], 1) if c != codes[i-1]]

pattern = []
for c in codes_compact:
    if not pattern:
        pattern.append(c)
    else:
        if c == pattern[0]:   # 시작 코드 재등장 → 첫 사이클 끝
            break
        if c not in pattern:
            pattern.append(c)
if not pattern:
    pattern = [codes_compact[0]]
start_code = pattern[0]

# ---- 3) 패턴 한 사이클마다 블록 묶기 ----
block_id = (df["구분"] == start_code).cumsum()  # start_code 나올 때마다 블록 증가

# ---- 4) 은행명 유니크(등장 순서 유지) 목록 ----
bank_order = [x for x in df["은행명"].dropna().tolist() if x != ""]
# 순서 유지 중복 제거
bank_seen = set(); bank_order = [x for x in bank_order if not (x in bank_seen or bank_seen.add(x))]

name_iter = iter(bank_order)

# ---- 5) 블록별로 '은행명' 빈칸만 순서대로 채우기 ----
for bid in sorted(block_id.unique()):
    fill_name = next(name_iter, None)
    if fill_name is None:
        break  # 더 채울 이름이 없으면 종료
    mask_block = (block_id == bid)
    empty = df.loc[mask_block, "은행명"].isna()
    if empty.any():
        df.loc[mask_block & empty, "은행명"] = fill_name

# ---- 6) 저장 ----
df.to_excel("./output_folder/임직원현황_filled.xlsx", index=False)
print("✅ 은행명 채움 완료 → 임직원현황_filled.xlsx")

# 1) 날짜 컬럼 찾기(연도로 시작)
date_cols = [c for c in df.columns if re.match(r'^(?:19|20)\d{2}', str(c))]
if not date_cols:
    raise ValueError("날짜 컬럼이 없습니다. 예: 2000년09월말, 2025-03 등")

id_vars = [c for c in df.columns if c not in date_cols]

# 2) Wide → Long (날짜를 행으로 내림)
df_long = df.melt(id_vars=id_vars, value_vars=date_cols,
                  var_name="날짜", value_name="값")
df_long["값"] = pd.to_numeric(
    df_long["값"].astype(str).str.replace(",", "", regex=False), errors="coerce"
)

# 5) (선택) 저장
df_long.to_excel("./output_folder/임직원현황_long.xlsx", index=False)

def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

df_long = df_long.copy()
df_long["값"] = _to_num(df_long["값"])

has_code = "코드" in df_long.columns

pivots = {}

# 1) 날짜 × (은행명×구분)  → 가장 많이 쓰는 형태
pivots["P1_날짜_by_은행명x구분"] = (
    df_long.pivot_table(index="날짜", columns=["은행명","구분"], values="값", aggfunc="sum", observed=True)
)

# 2) 날짜 × (은행명×코드)  → 코드가 있을 때
if has_code:
    pivots["P2_날짜_by_은행명x코드"] = (
        df_long.pivot_table(index="날짜", columns=["은행명","코드"], values="값", aggfunc="sum", observed=True)
    )

# 3) (날짜×은행명) × 구분  → 은행명은 행으로 두고, 구분을 열로
pivots["P3_날짜x은행명_by_구분"] = (
    df_long.pivot_table(index=["날짜","은행명"], columns="구분", values="값", aggfunc="sum", observed=True)
)

# 4) (날짜×은행명) × 코드  → 코드가 있을 때
if has_code:
    pivots["P4_날짜x은행명_by_코드"] = (
        df_long.pivot_table(index=["날짜","은행명"], columns="코드", values="값", aggfunc="sum", observed=True)
    )

# 컬럼 평탄화 + 저장
def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(map(str, t)) for t in df.columns.to_flat_index()]
    return df.reset_index()

with pd.ExcelWriter("./output_folder/pivot_outputs.xlsx", engine="openpyxl") as w:
    for name, pt in pivots.items():
        out = flatten_cols(pt)
        out.to_excel(w, sheet_name=name[:31], index=False)  # 엑셀 시트명 31자 제한

print("✅ 저장: pivot_outputs.xlsx")