import pandas as pd
import glob
import re
import os

def parse_kosis_time(s: str) -> pd.Timestamp:
    s = str(s).strip()

    # 분기
    m_q = re.fullmatch(r"(\d{4})[Qq]([1-4])", s)
    if m_q:
        y, q = int(m_q.group(1)), int(m_q.group(2))
        month = q * 3
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)

    # 반기: H1/H2
    m_h = re.fullmatch(r"(\d{4})[Hh]([1-2])", s)
    if m_h:
        y, h = int(m_h.group(1)), int(m_h.group(2))
        month = 6 if h == 1 else 12
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)

    # 반기: S1/S2 (여기가 추가!)
    m_s = re.fullmatch(r"(\d{4})[Ss]([1-2])", s)
    if m_s:
        y, sidx = int(m_s.group(1)), int(m_s.group(2))
        month = 6 if sidx == 1 else 12
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(1)

    # 월(YYYYMM)
    m_m = re.fullmatch(r"(\d{4})(\d{2})", s)
    if m_m:
        y, m = int(m_m.group(1)), int(m_m.group(2))
        return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(1)

    # 연(YYYY)
    m_y = re.fullmatch(r"(\d{4})", s)
    if m_y:
        y = int(m_y.group(1))
        return pd.Timestamp(year=y, month=12, day=31)

    # 최후 보루
    return pd.to_datetime(s, errors="coerce")


def transform_kosis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = df["TIME"].map(parse_kosis_time)
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")

    def make_var(row):
        # ITEM_NAME1 ~ 4 중 NaN이 아닌 것만 모아서 _로 연결
        parts = [str(row[c]) for c in ["ITEM_NAME1", "ITEM_NAME2", "ITEM_NAME3", "ITEM_NAME4"]
                 if pd.notna(row.get(c)) and str(row[c]).strip() != ""]
        base = "_".join(parts)

        # 단위 붙이기
        unit = f"[{row['UNIT_NAME']}]" if pd.notna(row.get("UNIT_NAME")) else ""
        return (base + unit).strip()

    df["variable_id"] = df.apply(make_var, axis=1)
    df["source_file"] = os.path.basename(path)
    return df

# --- 실행 ---
all_files = glob.glob("Result/*.csv")
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

for f in all_files:
    df_long = transform_kosis(f)

    # wide-format 변환
    df_wide = (df_long
               .pivot_table(index=["date"],
                            columns="variable_id",
                            values="value",
                            aggfunc="last")
               .reset_index()
              )

    # 저장 파일명
    base = os.path.splitext(os.path.basename(f))[0]
    df_long.to_csv(os.path.join(output_dir, f"{base}_long.csv"), index=False)
    df_wide.to_csv(os.path.join(output_dir, f"{base}_wide.csv"), index=False)
