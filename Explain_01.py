# B_explain_elasticnet_shap.py
# 목적: 외부 변수(원본+변형)를 그대로 사용해 예측 + 변수기여도 해석(SHAP/계수)
import warnings, math, random
from pathlib import Path
from typing import List
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

HAS_LGBM=False
try:
    import lightgbm as lgb; HAS_LGBM=True
except Exception: pass

# SHAP (설치 필요: pip install shap)
HAS_SHAP=False
try:
    import shap; HAS_SHAP=True
except Exception: pass

warnings.filterwarnings("ignore")

# ===== 설정 =====
CSV_PATH   = "./master.csv"
DATE_COL   = "년월"
TARGET_COL = "저원가성"
SAVE_DIR   = Path("./artifacts_B")
N_SPLITS   = 5
SEED       = 42
# ==============

def seed_all(s=42):
    random.seed(s); np.random.seed(s)

def to_month_end_yyyymm(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s.astype(str), format="%Y%m", errors="coerce")
    return dt + pd.offsets.MonthEnd(0)

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for c in out.columns:
        if out[c].dtype==object:
            out[c]=(out[c].astype(str).str.replace(",","",regex=False).str.replace("%","",regex=False).str.strip())
        out[c]=pd.to_numeric(out[c], errors="coerce")
    return out

def winsorize(s: pd.Series, p=0.01): lo,hi=s.quantile(p),s.quantile(1-p); return s.clip(lo,hi)

def make_exog_transforms(df_num: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out=df_num.copy()
    exog=[c for c in out.columns if c!=target_col]
    for c in exog:
        s=winsorize(out[c].astype(float),p=0.01)
        out[c]=s
        out[f"{c}_d1"]=s.diff(1)
        out[f"{c}_mom%"]=(s.pct_change(1)*100)
        out[f"{c}_yoy%"]=(s.pct_change(12)*100)
        for k in (3,6,12):
            out[f"{c}_ma{k}"]=s.rolling(k,min_periods=1).mean()
            out[f"{c}_std{k}"]=s.rolling(k,min_periods=2).std()
        for L in (1,3,6,12):
            out[f"{c}_lag{L}"]=s.shift(L)
            out[f"{c}_d1_lag{L}"]=out[f"{c}_d1"].shift(L)
        out[f"{c}_level_x_d1"]=s*out[f"{c}_d1"]
    out=out.replace([np.inf,-np.inf],np.nan).fillna(method="ffill").fillna(method="bfill")
    return out

def mape(y,p,eps=1e-8):
    y=np.asarray(y); p=np.asarray(p); d=np.where(np.abs(y)<eps,eps,np.abs(y))
    return float(np.mean(np.abs((y-p)/d))*100)
def rmse(y,p): y=np.asarray(y); p=np.asarray(p); return float(math.sqrt(((y-p)**2).mean()))

def main():
    seed_all(SEED); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(CSV_PATH)
    df[DATE_COL]=to_month_end_yyyymm(df[DATE_COL])
    df=df.sort_values(DATE_COL).drop_duplicates([DATE_COL],keep="last").set_index(DATE_COL)
    df=clean_numeric(df)
    df[TARGET_COL]=pd.to_numeric(df[TARGET_COL],errors="coerce").replace(0,np.nan)

    # 변형 생성(원본 유지)
    num=df.select_dtypes(include=[np.number]).copy()
    fe =make_exog_transforms(num, TARGET_COL)
    y  =fe[TARGET_COL]
    X  =fe.drop(columns=[TARGET_COL],errors="ignore")

    # 피처 결측 간단 임퓨트 + 스케일
    imputer=SimpleImputer(strategy="median")
    scaler =StandardScaler()
    X_imp  =pd.DataFrame(imputer.fit_transform(X),index=X.index,columns=X.columns)
    X_s    =pd.DataFrame(scaler.fit_transform(X_imp), index=X.index, columns=X.columns)

    # 관측 구간만 학습
    mask_train=(~y.isna())
    Xtr=X_s.loc[mask_train]; ytr=y.loc[mask_train]
    n_splits=max(2, min(N_SPLITS, len(Xtr)-1)) if len(Xtr)>2 else 2
    tscv=TimeSeriesSplit(n_splits=n_splits)

    # ElasticNetCV (계수 해석)
    enet=ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                      alphas=np.logspace(-3,1,20), max_iter=5000, cv=tscv, n_jobs=-1, random_state=SEED)
    enet.fit(Xtr, ytr)
    p_en = pd.Series(enet.predict(X_s), index=X.index)

    rows=[("ElasticNet",
           rmse(ytr, enet.predict(Xtr.loc[Xtr.index.intersection(ytr.index)])),
           mean_absolute_error(ytr, enet.predict(Xtr.loc[Xtr.index.intersection(ytr.index)])),
           mape(ytr, enet.predict(Xtr.loc[Xtr.index.intersection(ytr.index)])))]

    # LightGBM + SHAP (있으면)
    if HAS_LGBM:
        lgbm=lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.05,num_leaves=31,
                               subsample=0.9,colsample_bytree=0.9,random_state=SEED,n_jobs=-1)
        # 간단 시계열 CV 점수
        rmses,maes,mapes=[],[],[]
        for tr,va in tscv.split(Xtr):
            X_tr,X_va=Xtr.iloc[tr],Xtr.iloc[va]; y_tr,y_va=ytr.iloc[tr],ytr.iloc[va]
            lgbm.fit(X_tr,y_tr)
            p=lgbm.predict(X_va)
            rmses.append(rmse(y_va,p)); maes.append(mean_absolute_error(y_va,p)); mapes.append(mape(y_va,p))
        rows.append(("LGBM", float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(mapes))))
        p_lgb = pd.Series(lgbm.fit(Xtr,ytr).predict(X_s), index=X.index)
    else:
        p_lgb = pd.Series(index=X.index, dtype=float)

    cv=pd.DataFrame(rows, columns=["model","cv_rmse","cv_mae","cv_mape"]).sort_values("cv_rmse")
    cv.to_csv(SAVE_DIR/"cv_report_explain.csv", index=False, encoding="utf-8-sig")
    (SAVE_DIR/"who_won.txt").write_text(f"winner={cv.iloc[0]['model']}\n", encoding="utf-8")
    print(cv)

    # == 해석 ==
    # 1) ElasticNet 계수 (표)
    coef_df = pd.DataFrame({"feature": X.columns, "coef": enet.coef_}).sort_values("coef", ascending=False)
    coef_df.to_csv(SAVE_DIR/"elasticnet_coefs.csv", index=False, encoding="utf-8-sig")

    # 2) LGBM SHAP (가능 시)
    if HAS_LGBM and HAS_SHAP:
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(Xtr)
        # 전체 중요도
        imp = pd.DataFrame({"feature": X.columns, "shap_abs_mean": np.mean(np.abs(shap_values), axis=0)}) \
                .sort_values("shap_abs_mean", ascending=False)
        imp.to_csv(SAVE_DIR/"shap_importance.csv", index=False, encoding="utf-8-sig")
        # 개별월 설명 예시(최근 1개)
        last_idx = Xtr.index[-1]
        shap_df = pd.DataFrame({"feature": X.columns, "shap_value": shap_values[-1], "value": Xtr.iloc[-1].values})
        shap_df.sort_values("shap_value", key=np.abs, ascending=False).head(30) \
               .to_csv(SAVE_DIR/"shap_last_point_top30.csv", index=False, encoding="utf-8-sig")
        print("[해석] SHAP 중요도/사례 저장 완료")

    # (선택) 두 모델 평균 예측도 저장
    p_avg = pd.concat([p_en.rename("ElasticNet"), p_lgb.rename("LGBM")], axis=1).mean(axis=1)
    pd.DataFrame({
        "y": y,
        "pred_elasticnet": p_en,
        "pred_lgbm": p_lgb,
        "pred_avg": p_avg
    }).to_csv(SAVE_DIR/"explain_predictions.csv", index=True, encoding="utf-8-sig")
    print("[저장] 설명 자료:", SAVE_DIR)

if __name__=="__main__":
    main()
