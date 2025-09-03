import pandas as pd
import glob
import re
import os

def parse_kosis_time(s: str) -> pd.Timestamp:
    s = str(s).strip()
    m_q = re.fullmatch(r"(\d{4})[Qq]([1-4])", s)
    if m_q:
        y, q = int(m_q.group(1)), int(m_q.group(2))
        month = q * 3
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)
    m_h = re.fullmatch(r"(\d{4})[Hh]([1-2])", s)
    if m_h:
        y, h = int(m_h.group(1)), int(m_h.group(2))
        month = 6 if h == 1 else 12
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)
    m_m = re.fullmatch(r"(\d{4})(\d{2})", s)
    if m_m:
        y, m = int(m_m.group(1)), int(m_m.group(2))
        return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(1)
    m_y = re.fullmatch(r"(\d{4})", s)
    if m_y:
        y = int(m_y.group(1))
        return pd.Timestamp(year=y, month=12, day=31)
    return pd.to_datetime(s, errors="coerce")

def transform_kosis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = df["TIME"].map(parse_kosis_time)
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")

    # 변수명 만들기 (예: ITEM_NAME4 + 단위)
    def make_var(row):
        base = row["ITEM_NAME4"] if pd.notna(row["ITEM_NAME4"]) else row["ITEM_NAME3"]
        unit = f"[{row['UNIT_NAME']}]" if pd.notna(row['UNIT_NAME']) else ""
        return (str(base) + unit).strip()

    df["variable_id"] = df.apply(make_var, axis=1)

    # 파일명 컬럼 추가 (어느 원본에서 왔는지 추적용)
    df["source_file"] = os.path.basename(path)
    return df

# --- 실행 ---
all_files = glob.glob("Result/*.csv")  # 폴더 경로
frames = []

for f in all_files:
    frames.append(transform_kosis(f))

df_all = pd.concat(frames, ignore_index=True)

# wide-format (원하면)
df_wide = (df_all
           .pivot_table(index=["source_file","date"],
                        columns="variable_id",
                        values="value",
                        aggfunc="last")
           .reset_index()
          )

# 결과 저장
df_all.to_csv("all_ecos_long.csv", index=False)
df_wide.to_csv("all_ecos_wide.csv", index=False)
