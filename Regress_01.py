# impute_target_with_ml.py
# 목적: '저원가성' 타깃의 0을 결측으로 보고, ML로 예측해 과거 구간을 채워넣기 (GRU 아님)
# - 입력: '년월' + 65개 변수 (타깃 포함), 모두 월말 기준으로 간주
# - 특징: 타깃의 라그는 사용하지 않고, 외생변수(타깃 제외) 라그/달력피처로 예측 → 역사 구간도 일괄 예측 가능

import warnings, math, random
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# (선택) 설치되어 있으면 자동 포함
HAS_XGB = False
HAS_LGBM = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    pass
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    pass

warnings.filterwarnings("ignore")

# ====== 설정(여기만 수정) ======
CSV_PATH   = "./master.csv"    # 원본 파일 (년월 + 65개 변수)
DATE_COL   = "년월"
TARGET_COL = "저원가성"

LAGS       = [1, 3, 6, 12]     # 외생변수 라그
N_SPLITS   = 5                 # 시계열 CV 분할 수
SEED       = 42

SAVE_DIR   = Path("./artifacts_impute_ml")
OUT_CSV    = SAVE_DIR / "master_imputed.csv"
REPORT_CSV = SAVE_DIR / "model_report.csv"
FEATS_CSV  = SAVE_DIR / "feature_columns.txt"
# ==============================

def seed_all(s=42):
    random.seed(s); np.random.seed(s)

def to_month_end(s):
    s = pd.to_datetime(s)
    return s.dt.to_period("M").dt.to_timestamp("M")

def clean_numeric_cols(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = (out[c].astype(str)
                            .str.replace(",", "", regex=False)
                            .str.replace("%", "", regex=False)
                            .str.strip())
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def add_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=idx)
    cal["cal_year"] = idx.year
    cal["cal_month"] = idx.month
    cal["cal_quarter"] = idx.quarter
    # 원-핫 대신 연속형으로 둠(데이터 적을 수 있어서)
    return cal

def make_lagged_features(df_num: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    타깃 컬럼을 제외한 모든 수치형에 대해 라그 생성
    """
    base = df_num.copy()
    exog_cols = [c for c in base.columns if c != target_col]
    for col in exog_cols:
        for L in lags:
            base[f"{col}_lag{L}"] = base[col].shift(L)
    return base

def build_design(df_num: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    # 라그 생성
    lagged = make_lagged_features(df_num, target_col, lags)
    # 달력 피처
    cal = add_calendar_features(lagged.index)
    X = pd.concat([lagged.drop(columns=[target_col]), cal], axis=1)
    y = df_num[target_col]
    return X, y

def drop_all_nan_rows(X: pd.DataFrame, y: pd.Series):
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    return X.loc[mask], y.loc[mask]

def models_candidate():
    cands = []
    # RidgeCV (알파 자동 튜닝)
    cands.append(("RidgeCV", RidgeCV(alphas=np.logspace(-3, 3, 13))))
    # RandomForest
    cands.append(("RandomForest", RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=-1)))
    # GradientBoosting
    cands.append(("GradBoost", GradientBoostingRegressor(random_state=SEED)))
    # (옵션) XGBoost
    if HAS_XGB:
        cands.append(("XGB", xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1)))
    # (옵션) LightGBM
    if HAS_LGBM:
        cands.append(("LGBM", lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1)))
    return cands

def timeseries_cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits=5):
    """
    시계열 CV로 RMSE/MAE 평균 산출
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes = [], []
    # 간단 스케일: 연속형 전체 StandardScaler (트리 모델도 일관성 위해 파이프 사용)
    feat_cols = X.columns.tolist()
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model),
    ])
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xva)
        rmse = math.sqrt(((yva.values - p) ** 2).mean())
        mae = mean_absolute_error(yva.values, p)
        rmses.append(rmse); maes.append(mae)
    return np.mean(rmses), np.mean(maes)

def fit_best_model(X: pd.DataFrame, y: pd.Series, n_splits=5):
    cands = models_candidate()
    scores = []
    best_name, best_model, best_rmse = None, None, float("inf")
    for name, mdl in cands:
        rmse, mae = timeseries_cv_score(mdl, X, y, n_splits=n_splits)
        scores.append((name, rmse, mae))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = mdl
    # 최종 모델(훈련 전체로 적합)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", best_model),
    ])
    pipe.fit(X, y)
    rep = pd.DataFrame(scores, columns=["model", "cv_rmse", "cv_mae"]).sort_values("cv_rmse")
    return best_name, pipe, rep

if __name__ == "__main__":
    seed_all(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 로드 & 정리
    df = pd.read_csv(CSV_PATH)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise KeyError("DATE_COL/TARGET_COL 확인 필요")
    df[DATE_COL] = to_month_end(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)

    # 숫자 변환
    df = clean_numeric_cols(df, exclude=[])  # 날짜 제외 전부 숫자 시도
    if not np.issubdtype(df[TARGET_COL].dtype, np.number):
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 2) 타깃 0 → NaN (과거 0은 결측 표기로 가정)
    df[TARGET_COL] = df[TARGET_COL].replace(0, np.nan)

    # 3) 특징/라벨 구성 (타깃 라그는 사용하지 않음)
    X_full, y_full = build_design(df, TARGET_COL, LAGS)

    # 4) 학습 가능한 행만 사용하여 모델 선택/학습
    X_train, y_train = drop_all_nan_rows(X_full, y_full)
    if len(X_train) < 36:
        print("[경고] 학습 가능한 표본이 적습니다(36 미만). 결과가 불안정할 수 있습니다.")
    best_name, final_model, report = fit_best_model(X_train, y_train, n_splits=N_SPLITS)

    # 5) 전체 구간 예측
    #    - 라그 때문에 맨 앞부분은 자연스레 NaN → 그 구간은 예측 불가
    #    - 라그가 만들어진 이후 구간은 타깃이 비어 있어도 외생변수 라그만으로 예측 가능
    yhat_all = pd.Series(index=X_full.index, dtype=float)
    # 라그가 없는 앞 구간 제외
    valid_mask = ~X_full.isna().any(axis=1)
    yhat_all.loc[valid_mask] = final_model.predict(X_full.loc[valid_mask])

    # 6) 채워넣기: 타깃이 NaN인 곳만 예측값으로 채움(관측치는 보존)
    y_filled = y_full.copy()
    fill_mask = y_filled.isna() & valid_mask
    y_filled.loc[fill_mask] = yhat_all.loc[fill_mask]

    # 7) 결과 저장
    out = df.copy()
    out[TARGET_COL] = y_filled
    out = out.reset_index().rename(columns={DATE_COL: "년월"})
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    with open(FEATS_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(X_full.columns.tolist()))

    # 8) 요약 출력
    n_total = len(df)
    n_nan_before = int(y_full.isna().sum())
    n_imputed = int(fill_mask.sum())
    print(f"[모델 선택] best = {best_name}")
    print(f"[행 개수] 전체={n_total:,} | 학습가능={len(X_train):,}")
    print(f"[결측] 타깃 NaN(0 포함)={n_nan_before:,} | 예측으로 채운 개수={n_imputed:,}")
    print(f"[저장] 채운 데이터: {OUT_CSV}")
    print(f"[저장] 모델 CV 리포트: {REPORT_CSV}")
    print(f"[저장] 사용 피처 목록: {FEATS_CSV}")
