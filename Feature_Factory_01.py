# ==== Feature Factory: 시계열 파생 (누수 방지) ==================================
import numpy as np
import pandas as pd

def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    if not np.issubdtype(d[date_col].dtype, np.datetime64):
        s = d[date_col].astype(str).str.strip()
        if s.str.fullmatch(r"\d{8}").all():
            d[date_col] = pd.to_datetime(s, format="%Y%m%d")
        else:
            d[date_col] = pd.to_datetime(s, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)
    return d

def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """
    최근 window 구간의 선형추세 기울기(시간 인덱스 0..w-1). 현재값 포함 방지 위해 y는 미리 shift(1)된 상태로 들어옴.
    """
    if y.isna().all() or window <= 1:
        return pd.Series(index=y.index, dtype=float)
    x = np.arange(window)
    def _fit(a):
        if np.isnan(a).any(): return np.nan
        # slope = cov(x,y)/var(x)
        vx = x - x.mean()
        vy = a - a.mean()
        denom = (vx**2).sum()
        return float((vx*vy).sum()/denom) if denom > 0 else np.nan
    return y.rolling(window, min_periods=window).apply(_fit, raw=True)

def make_time_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    lags=(1,2,3,6,12),
    roll_windows=(3,6,12),
    add_calendar=True
) -> tuple[pd.DataFrame, list[str]]:
    """
    내부 데이터(날짜+타깃)만으로 누수 없는 파생변수 생성.
    - 모든 롤링/변화율/추세는 y_shift = y.shift(1) 기반으로 계산 → 현재값 정보 누수 차단
    - 반환: (피처가 추가된 DataFrame, 학습용 X_cols 리스트)
    """
    d = _ensure_datetime(df, date_col).copy()

    # 숫자형 타깃만 사용
    if not np.issubdtype(d[target_col].dtype, np.number):
        d[target_col] = pd.to_numeric(d[target_col], errors="coerce")

    y = d[target_col]
    y_shift = y.shift(1)  # ★ 누수 방지 핵심

    new_cols = []

    # 1) Lag 피처
    for L in lags:
        col = f"lag{L}"
        d[col] = y.shift(L)
        new_cols.append(col)

    # 2) 차이 / 수익률(변화율)
    d["diff_1"] = y_shift.diff(1)
    d["diff_3"] = y_shift.diff(3)
    d["pct_1"]  = y_shift.pct_change(1).replace([np.inf,-np.inf], np.nan)
    d["pct_3"]  = y_shift.pct_change(3).replace([np.inf,-np.inf], np.nan)
    new_cols += ["diff_1","diff_3","pct_1","pct_3"]

    # 3) 롤링 통계 (현재 제외: y_shift 기준)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = y_shift.rolling(w, min_periods=w).mean()
        d[f"roll_std_{w}"]  = y_shift.rolling(w, min_periods=w).std()
        d[f"roll_min_{w}"]  = y_shift.rolling(w, min_periods=w).min()
        d[f"roll_max_{w}"]  = y_shift.rolling(w, min_periods=w).max()
        d[f"vol_ratio_{w}"] = d[f"roll_std_{w}"] / (d[f"roll_mean_{w}"].abs() + 1e-9)
        new_cols += [f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}", f"roll_max_{w}", f"vol_ratio_{w}"]

    # 4) EWMA(지수이동평균) – 민감/완만 두 가지
    d["ewm_3"]  = y_shift.ewm(span=3,  adjust=False).mean()
    d["ewm_12"] = y_shift.ewm(span=12, adjust=False).mean()
    new_cols += ["ewm_3","ewm_12"]

    # 5) 추세 기울기(선형회귀 slope)
    for w in roll_windows:
        col = f"trend_slope_{w}"
        d[col] = _rolling_slope(y_shift, w)
        new_cols.append(col)

    # 6) 전환점 더미 (증감 부호 변화)
    d["turning_point"] = (
        np.sign(y_shift.diff(1)).fillna(0) != np.sign(y_shift.diff(1)).shift(1).fillna(0)
    ).astype(int)
    new_cols.append("turning_point")

    # 7) 달력 특성
    if add_calendar:
        dt = pd.to_datetime(d[date_col])
        d["month"] = dt.dt.month.astype("int16")
        d["quarter"] = dt.dt.quarter.astype("int16")
        d["day"] = dt.dt.day.astype("int16")
        d["weekday"] = dt.dt.weekday.astype("int16")  # 0=월
        d["is_month_end"] = dt.dt.is_month_end.astype("int8")
        d["is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")
        # 말일 ±3일 윈도우 더미 (월말 효과 캡쳐)
        day_in_month = dt.dt.day
        last_day = (dt + pd.offsets.MonthEnd(0)).dt.day
        d["eom_window_3d"] = (abs(last_day - day_in_month) <= 3).astype("int8")
        new_cols += ["month","quarter","day","weekday","is_month_end","is_quarter_end","eom_window_3d"]

    # 8) 결측/무한 처리
    d.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 타깃/날짜/현재값 누수로 생긴 NaN 제거 (학습 구간에서만)
    X_cols = new_cols.copy()
    return d, X_cols

# ---------------------- 사용 예 ----------------------
# df_feat, X_cols = make_time_features(df, date_col="date", target_col="target")
# X = df_feat[X_cols].to_numpy(float); y = df_feat["target"].to_numpy(float)
# (주의) 초기 몇 행은 파생으로 NaN → 학습 시 자동으로 dropna 하거나 split 시 마스크 적용
