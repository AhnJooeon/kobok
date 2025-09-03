import pandas as pd
import re

# --- 0) 로드 ---
df = pd.read_csv("./Result/ecos_903Y001.csv")  # 방금 주신 컬럼 구조

# --- 1) TIME → datetime (주기 자동 인식: 연/월/분기/반기) ---
def parse_kosis_time(s: str) -> pd.Timestamp:
    s = str(s).strip()
    # YYYYQn 형태 (예: 2012Q1)
    m_q = re.fullmatch(r"(\d{4})[Qq]([1-4])", s)
    if m_q:
        y, q = int(m_q.group(1)), int(m_q.group(2))
        month = q * 3  # 분기 말월
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)
    # YYYY-Hn (반기) 예외 처리
    m_h = re.fullmatch(r"(\d{4})[Hh]([1-2])", s)
    if m_h:
        y, h = int(m_h.group(1)), int(m_h.group(2))
        month = 6 if h == 1 else 12
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)
    # YYYYMM
    m_m = re.fullmatch(r"(\d{4})(\d{2})", s)
    if m_m:
        y, m = int(m_m.group(1)), int(m_m.group(2))
        return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(1)
    # YYYY
    m_y = re.fullmatch(r"(\d{4})", s)
    if m_y:
        y = int(m_y.group(1))
        return pd.Timestamp(year=y, month=12, day=31)
    # 그 외는 pandas에게 맡김
    return pd.to_datetime(s, errors="coerce")

df["date"] = df["TIME"].map(parse_kosis_time)

# --- 2) value 숫자화 ---
df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")

# --- 3) 어떤 차원을 변수로 쓸지 선택 ---
# 보통 ITEM_NAME4가 가장 구체(예: '근로소득(전년도)(만원)')
# 필요에 따라 ITEM_NAME2=지역, ITEM_NAME3=범주 등을 포함 가능
dim_as_suffix = ["ITEM_NAME4"]             # 가장 세부 항목
dim_as_group  = ["ITEM_NAME2", "ITEM_NAME3"]  # 그룹(예: 지역/분류 등)
unit_col      = "UNIT_NAME"                # 단위 보존(선택)

# 변수키 만들기: ex) '근로소득(전년도)(만원)[만원]'
def make_var(row):
    base = " / ".join([str(row[c]) for c in dim_as_suffix if pd.notna(row[c])])
    unit = f"[{row[unit_col]}]" if unit_col in row and pd.notna(row[unit_col]) else ""
    return (base + unit).strip()

df["variable_id"] = df.apply(make_var, axis=1)

# --- 4A) 시계열(원주기) long-format (그룹별 멀티인덱스) ---
# 그룹 차원 + 시점으로 정렬
group_cols = dim_as_group  # 예: ['ITEM_NAME2','ITEM_NAME3'] = ['전국', '지표그룹'] 등
ts_long = (df
           .loc[:, group_cols + ["variable_id", "date", "value"]]
           .dropna(subset=["date"])
           .sort_values(["ITEM_NAME2","ITEM_NAME3","variable_id","date"])
          )

# 멀티인덱스(그룹+date)로 정리하면, "월 묶기 전"의 시계열이 그대로 유지됩니다.
ts_long = ts_long.set_index(group_cols + ["date", "variable_id"]).sort_index()

# --- 4B) 같은 걸 wide-format으로 (그룹×date 인덱스, 열=variable_id) ---
ts_wide = (ts_long
           .reset_index()
           .pivot_table(index=group_cols + ["date"],
                        columns="variable_id",
                        values="value",
                        aggfunc="last")
           .sort_index()
          )

ts_wide.to_csv('./test.csv')

# === 사용 팁 ===
# 1) '전국'만 보고 싶다:
#    ts_wide.xs('전국', level='ITEM_NAME2')  # 그룹 레벨 이름에 맞춰 변경

# 2) 멀티인덱스가 불편하면 평평하게:
#    flat = ts_wide.reset_index()  # date와 그룹이 컬럼으로 내려옴

# 3) (선택) 같은 항목이 중복로우라면 집계 규칙 지정:
#    pivot_table에서 aggfunc="last"/"mean"/"sum" 중 선택

# 4) (선택) 월/분기 변환은 '나중 단계'에서:
#    monthly = (ts_wide
#               .groupby(level=group_cols)
#               .resample('M', level='date')
#               .ffill())
