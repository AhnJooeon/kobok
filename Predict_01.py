# A_predict_pca.py
# 목적: 외부변수(원본+변형) → 스케일 → PCA → (GRU/GBM) CV 비교 → 우승 모델로 '저원가성' NaN 채움
import warnings, math, random
from pathlib import Path
from typing import List, Tuple
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

# (옵션) 설치 시 자동 사용
HAS_XGB = HAS_LGBM = False
try:
    import xgboost as xgb; HAS_XGB = True
except Exception: pass
try:
    import lightgbm as lgb; HAS_LGBM = True
except Exception: pass

# GRU
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ===== 설정 =====
CSV_PATH   = "./master.csv"     # '년월(YYYYMM)' + 65개 변수(저원가성 포함)
DATE_COL   = "년월"
TARGET_COL = "저원가성"
SAVE_DIR   = Path("./artifacts_A")
N_PCS      = 50                 # PCA 주성분 개수
N_SPLITS   = 5
SEED       = 42

# GRU HP
LOOKBACK   = 12; HIDDEN=64; LAYERS=1; DROPOUT=0.2
EPOCHS     = 300; BATCH=16; LR=1e-3; PATIENCE=30
# ==============

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_month_end_yyyymm(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s.astype(str), format="%Y%m", errors="coerce")
    return dt + pd.offsets.MonthEnd(0)

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = (out[c].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.replace("%", "", regex=False)
                          .str.strip())
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def winsorize(s: pd.Series, p=0.01): lo,hi=s.quantile(p),s.quantile(1-p); return s.clip(lo,hi)

def make_exog_transforms(df_num: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """외부 변수 변형(Δ, MoM%, YoY%, MA/STD, 라그)을 생성. 타깃은 변형 안 함."""
    out = df_num.copy()
    exog = [c for c in out.columns if c != target_col]
    for c in exog:
        s = winsorize(out[c].astype(float), p=0.01)
        out[c] = s
        out[f"{c}_d1"]    = s.diff(1)
        out[f"{c}_mom%"]  = (s.pct_change(1)*100)
        out[f"{c}_yoy%"]  = (s.pct_change(12)*100)
        for k in (3,6,12):
            out[f"{c}_ma{k}"]  = s.rolling(k, min_periods=1).mean()
            out[f"{c}_std{k}"] = s.rolling(k, min_periods=2).std()
        for L in (1,3,6,12):
            out[f"{c}_lag{L}"]   = s.shift(L)
            out[f"{c}_d1_lag{L}"]= out[f"{c}_d1"].shift(L)
        out[f"{c}_level_x_d1"] = s * out[f"{c}_d1"]
    out = out.replace([np.inf,-np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return out

def mape(y,p,eps=1e-8):
    y=np.asarray(y); p=np.asarray(p); d=np.where(np.abs(y)<eps,eps,np.abs(y))
    return float(np.mean(np.abs((y-p)/d))*100)
def rmse(y,p): y=np.asarray(y); p=np.asarray(p); return float(math.sqrt(((y-p)**2).mean()))

class SeqDS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32).view(-1,1)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

class GRUReg(nn.Module):
    def __init__(self,in_dim,hidden=64,layers=1,dropout=0.2):
        super().__init__()
        self.gru=nn.GRU(in_dim,hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0.0)
        self.head=nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden,1))
    def forward(self,x): out,_=self.gru(x); return self.head(out[:,-1,:])

def build_windows(X,y,lb):
    Xs,ys=[],[]
    for t in range(lb,len(y)):
        if np.isnan(y[t]) or np.isnan(X[t-lb:t,:]).any(): continue
        Xs.append(X[t-lb:t,:]); ys.append(y[t])
    return (np.stack(Xs), np.array(ys)) if Xs else (None,None)

def main():
    seed_all(SEED); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df[DATE_COL]=to_month_end_yyyymm(df[DATE_COL])
    df=df.sort_values(DATE_COL).drop_duplicates([DATE_COL],keep="last").set_index(DATE_COL)
    df=clean_numeric(df)
    # 타깃 0→NaN
    df[TARGET_COL]=pd.to_numeric(df[TARGET_COL],errors="coerce").replace(0,np.nan)

    # 변형 생성
    num = df.select_dtypes(include=[np.number]).copy()
    fe  = make_exog_transforms(num, TARGET_COL)

    # X_all (타깃 제외), y
    X_all = fe.drop(columns=[TARGET_COL], errors="ignore")
    y     = fe[TARGET_COL]

    # 결측 임시 대체(피처만; 타깃은 그대로)
    X_all = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_all), index=X_all.index)

    # PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all.values)
    pca = PCA(n_components=min(N_PCS, Xs.shape[1]), random_state=SEED)
    Xp = pca.fit_transform(Xs)
    Xp_df = pd.DataFrame(Xp, index=X_all.index, columns=[f"PC{i+1}" for i in range(Xp.shape[1])])

    # --- GBM 계열 CV ---
    rows=[]
    def gbm_cands():
        c=[("GradBoost", GradientBoostingRegressor(random_state=SEED))]
        if HAS_XGB: c.append(("XGB", xgb.XGBRegressor(n_estimators=800,learning_rate=0.05,max_depth=6,
                                                     subsample=0.9,colsample_bytree=0.9,random_state=SEED,n_jobs=-1)))
        if HAS_LGBM: c.append(("LGBM", lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.05,
                                                         num_leaves=31,subsample=0.9,colsample_bytree=0.9,
                                                         random_state=SEED,n_jobs=-1)))
        return c

    Xtr_full = Xp_df.copy(); y_full = y.copy()
    mask_train = (~y_full.isna())
    Xtr = Xtr_full.loc[mask_train]; ytr = y_full.loc[mask_train]
    n_splits = max(2, min(N_SPLITS, len(Xtr)-1)) if len(Xtr)>2 else 2

    for name,mdl in gbm_cands():
        tscv=TimeSeriesSplit(n_splits=n_splits)
        rmses,maes,mapes=[],[],[]
        for tr,va in tscv.split(Xtr):
            X_tr,X_va=Xtr.iloc[tr],Xtr.iloc[va]; y_tr,y_va=ytr.iloc[tr],ytr.iloc[va]
            mdl.fit(X_tr.values, y_tr.values)
            p = mdl.predict(X_va.values)
            rmses.append(rmse(y_va,p)); maes.append(mean_absolute_error(y_va,p)); mapes.append(mape(y_va,p))
        rows.append((name,float(np.mean(rmses)),float(np.mean(maes)),float(np.mean(mapes))))

    # --- GRU CV ---
    # 입력은 Xp_df, 타깃 y
    def gru_cv():
        scalerY=StandardScaler()
        yv=y_full.values.astype(float)
        y_s=scalerY.fit_transform(yv.reshape(-1,1)).ravel()
        Xv=Xp_df.values
        tscv=TimeSeriesSplit(n_splits=n_splits)
        rmses,maes,mapes=[],[],[]
        for tr,va in tscv.split(Xp_df):
            X_tr,X_va=Xv[tr],Xv[va]; y_tr,y_va=y_s[tr],y_s[va]
            Xtr_seq,ytr_seq=build_windows(X_tr,y_tr,LOOKBACK)
            Xva_seq,yva_seq=build_windows(np.vstack([X_tr[-LOOKBACK:],X_va]),
                                          np.concatenate([y_tr[-LOOKBACK:],y_va]),LOOKBACK)
            if Xtr_seq is None or Xva_seq is None: continue
            dev="cuda" if torch.cuda.is_available() else "cpu"
            model=GRUReg(in_dim=Xtr_seq.shape[2],hidden=HIDDEN,layers=LAYERS,dropout=DROPOUT).to(dev)
            opt=torch.optim.Adam(model.parameters(),lr=LR); crit=nn.MSELoss()
            Ltr=DataLoader(SeqDS(Xtr_seq,ytr_seq),batch_size=BATCH,shuffle=True)
            Lva=DataLoader(SeqDS(Xva_seq,yva_seq),batch_size=BATCH,shuffle=False)
            best=float("inf"); snap=None; stall=0
            for ep in range(1,EPOCHS+1):
                model.train()
                for xb,yb in Ltr:
                    xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); pred=model(xb); loss=crit(pred,yb); loss.backward(); opt.step()
                model.eval(); va_loss=0.0
                with torch.no_grad():
                    for xb,yb in Lva:
                        xb,yb=xb.to(dev),yb.to(dev); va_loss+=crit(model(xb),yb).item()*xb.size(0)
                va_loss/=max(1,len(Lva.dataset))
                if va_loss<best-1e-7: best=va_loss; snap={k:v.cpu().clone() for k,v in model.state_dict().items()}; stall=0
                else: stall+=1
                if stall>=PATIENCE: break
            if snap: model.load_state_dict(snap)
            # 예측(역스케일)
            preds=[]
            with torch.no_grad():
                for xb,yb in Lva:
                    xb=xb.to(dev); p=model(xb).cpu().numpy().ravel(); preds.append(p)
            p_s=np.concatenate(preds)
            y_true=scalerY.inverse_transform(yva_seq.reshape(-1,1)).ravel()
            y_pred=scalerY.inverse_transform(p_s.reshape(-1,1)).ravel()
            rmses.append(rmse(y_true,y_pred)); maes.append(mean_absolute_error(y_true,y_pred)); mapes.append(mape(y_true,y_pred))
        return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(mapes))

    if mask_train.sum()-LOOKBACK>0:
        g_rmse,g_mae,g_mape = gru_cv()
        rows.append(("GRU_PCA", g_rmse, g_mae, g_mape))
    else:
        print("[알림] GRU 스킵: 관측치 부족")

    cv = pd.DataFrame(rows, columns=["model","cv_rmse","cv_mae","cv_mape"]).sort_values("cv_rmse")
    cv.to_csv(SAVE_DIR/"cv_report.csv", index=False, encoding="utf-8-sig")
    winner=cv.iloc[0]["model"]
    (SAVE_DIR/"winner.txt").write_text(f"winner={winner}\n", encoding="utf-8")
    print(cv, "\n[우승]", winner)

    # --- 전 구간 예측 & 임퓨트 ---
    preds=pd.Series(index=df.index,dtype=float)

    if winner=="GRU_PCA":
        # 풀학습
        scalerY=StandardScaler()
        yv=y.values.astype(float); y_s=scalerY.fit_transform(yv.reshape(-1,1)).ravel()
        Xv=Xp_df.values
        Xseq,yseq=build_windows(Xv,y_s,LOOKBACK)
        if Xseq is not None:
            dev="cuda" if torch.cuda.is_available() else "cpu"
            model=GRUReg(Xseq.shape[2],HIDDEN,LAYERS,DROPOUT).to(dev)
            opt=torch.optim.Adam(model.parameters(),lr=LR); crit=nn.MSELoss()
            L=DataLoader(SeqDS(Xseq,yseq),batch_size=BATCH,shuffle=True)
            best=float("inf"); snap=None; stall=0
            for ep in range(1,EPOCHS+1):
                tr=0.0; model.train()
                for xb,yb in L:
                    xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); p=model(xb); loss=crit(p,yb); loss.backward(); opt.step(); tr+=loss.item()*xb.size(0)
                tr/=max(1,len(L.dataset))
                if tr<best-1e-6: best=tr; snap={k:v.cpu().clone() for k,v in model.state_dict().items()}; stall=0
                else: stall+=1
                if stall>=PATIENCE: break
            if snap: model.load_state_dict(snap)
            with torch.no_grad():
                for t in range(LOOKBACK,len(Xv)):
                    if np.isnan(Xv[t-LOOKBACK:t,:]).any(): continue
                    xb=torch.tensor(Xv[t-LOOKBACK:t,:],dtype=torch.float32).unsqueeze(0).to(dev)
                    p_s=model(xb).cpu().numpy().ravel()[0]
                    preds.iloc[t]=StandardScaler().fit(y.values.reshape(-1,1))  # dummy (안씀)
                    preds.iloc[t]=scalerY.inverse_transform([[p_s]])[0,0]
    else:
        # GBM 우승
        mdl=[m for m in gbm_cands() if m[0]==winner][0][1]
        mask_train = (~y.isna())
        mdl.fit(Xp_df.loc[mask_train].values, y.loc[mask_train].values)
        preds.loc[:] = np.nan
        preds.loc[:] = mdl.predict(Xp_df.values)

    # 임퓨트: 관측치 보존, NaN만 채움
    y_filled=y.copy()
    fill_mask=y_filled.isna() & preds.notna()
    y_filled.loc[fill_mask]=preds.loc[fill_mask]
    out=df.copy(); out[TARGET_COL]=y_filled
    out.reset_index().rename(columns={DATE_COL:"년월"}).to_csv(SAVE_DIR/"master_imputed.csv",index=False,encoding="utf-8-sig")
    print("[저장] 결과:", SAVE_DIR/"master_imputed.csv")

if __name__=="__main__":
    main()
