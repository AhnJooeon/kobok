# train_gru_with_pca.py
import math, random, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ===== 설정(여기만 수정) =====
CSV_PATH   = "./finance_data.csv"  # CSV 경로
DATE_COL   = "날짜"                 # 날짜 컬럼
TARGET_COL = "예금잔액"             # 타깃 컬럼
LOOKBACK   = 12                     # 입력 시퀀스 길이
TEST_TAIL  = 12                     # 테스트 길이
VAL_TAIL   = 6                      # 검증 길이
EPOCHS     = 200
BATCH_SIZE = 16
HIDDEN     = 64
LAYERS     = 1
DROPOUT    = 0.2
LR         = 1e-3
SCALER_X   = "standard"             # "standard" | "minmax"
SCALER_Y   = "standard"             # "standard" | "minmax"
USE_PCA    = True                   # PCA 사용 여부
PCA_N      = 32                     # PCA 차원 수 (입력 피처 수보다 클 수 없음)
SEED       = 42
# ===========================

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mape(y, yhat, eps=1e-8):
    y, yhat = np.array(y), np.array(yhat)
    denom = np.where(np.abs(y) < eps, eps, np.abs(y))
    return np.mean(np.abs((y - yhat) / denom)) * 100

def infer_index(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)

def cast_numeric(df, exclude=[]):
    df = df.copy()
    for c in df.columns:
        if c in exclude: continue
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def split_tail_idx(idx_len, test_tail=12, val_tail=6):
    tr = slice(0, idx_len - (test_tail + val_tail))
    va = slice(idx_len - (test_tail + val_tail), idx_len - test_tail)
    te = slice(idx_len - test_tail, idx_len)
    return tr, va, te

def build_windows_from_Xy(X_arr, y_arr, lookback):
    Xs, ys = [], []
    T = X_arr.shape[0]
    for t in range(lookback, T):
        Xs.append(X_arr[t - lookback:t, :])
        ys.append(y_arr[t])  # 1-step ahead
    return np.stack(Xs), np.array(ys)

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class GRUReg(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.gru(x)         # (B,T,H)
        last = out[:, -1, :]         # (B,H)
        return self.fc(last)         # (B,1)

def train_model(model, Ltr, Lva, epochs=200, lr=1e-3, patience=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best = float("inf"); snap = None; stall = 0
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in Ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb); loss = crit(pred, yb)
            loss.backward(); opt.step(); tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(Ltr.dataset)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in Lva:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb); loss = crit(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(Lva.dataset)

        if va_loss < best - 1e-7:
            best = va_loss; snap = {k: v.cpu().clone() for k, v in model.state_dict().items()}; stall = 0
        else:
            stall += 1

        if ep % 20 == 0 or ep == 1:
            print(f"[{ep:04d}] train MSE={tr_loss:.6f} | val MSE={va_loss:.6f}")
        if stall >= patience:
            print(f"Early stop @ {ep}, best val MSE={best:.6f}")
            break

    if snap: model.load_state_dict(snap)
    return model

def evaluate_and_report(model, Lte, scaler_y):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval(); Ys=[]; Ps=[]
    with torch.no_grad():
        for xb, yb in Lte:
            xb = xb.to(device)
            p = model(xb).cpu().numpy().ravel()
            Ys.append(yb.numpy().ravel())
            Ps.append(p)
    y_s = np.concatenate(Ys)   # scaled
    p_s = np.concatenate(Ps)   # scaled
    # 역스케일
    y = scaler_y.inverse_transform(y_s.reshape(-1,1)).ravel()
    p = scaler_y.inverse_transform(p_s.reshape(-1,1)).ravel()
    rmse = math.sqrt(((y - p) ** 2).mean())
    mae  = mean_absolute_error(y, p)
    mp   = mape(y, p)
    print(f"[TEST] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mp:.2f}%")
    return y, p

if __name__ == "__main__":
    seed_all(SEED)

    # 1) 데이터 로드 & 정렬
    df = pd.read_csv(CSV_PATH)
    df = cast_numeric(df, exclude=[DATE_COL])
    df = infer_index(df, DATE_COL)

    # 2) 숫자형만 사용 & 결측 보간
    num = df.select_dtypes(include=[np.number]).copy()
    if TARGET_COL not in num.columns:
        raise ValueError("타깃 컬럼이 숫자형이 아닙니다. 미리 숫자로 변환하세요.")
    num = num.fillna(method="ffill").fillna(method="bfill")

    # 3) X, y 분리
    y_full = num[[TARGET_COL]].copy()               # (T,1)
    X_full = num.drop(columns=[TARGET_COL]).copy()  # (T,F)

    # 4) 시계열 분할 인덱스
    T = len(num)
    tr_s, va_s, te_s = split_tail_idx(T, test_tail=TEST_TAIL, val_tail=VAL_TAIL)

    # 5) 스케일링 (훈련 구간만 적합)
    ScX = StandardScaler() if SCALER_X == "standard" else MinMaxScaler()
    ScY = StandardScaler() if SCALER_Y == "standard" else MinMaxScaler()

    X_tr = ScX.fit_transform(X_full.iloc[tr_s].values)
    X_va = ScX.transform(X_full.iloc[va_s].values)
    X_te = ScX.transform(X_full.iloc[te_s].values)

    y_tr = ScY.fit_transform(y_full.iloc[tr_s].values).ravel()
    y_va = ScY.transform(y_full.iloc[va_s].values).ravel()
    y_te = ScY.transform(y_full.iloc[te_s].values).ravel()

    # 6) (옵션) PCA — X에만 적용 (훈련 적합 → 분할별 변환)
    pca = None
    if USE_PCA:
        in_dim = X_tr.shape[1]
        k = min(PCA_N, in_dim)
        pca = PCA(n_components=k)
        pca.fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)

    # 7) Windows (검증/테스트는 경계 연결해 leak 없이 만들기)
    Xva_build = np.vstack([X_tr[-LOOKBACK:], X_va])
    yva_build = np.concatenate([y_tr[-LOOKBACK:], y_va])

    Xte_build = np.vstack([np.vstack([X_tr, X_va])[-LOOKBACK:], X_te])
    yte_build = np.concatenate([np.concatenate([y_tr, y_va])[-LOOKBACK:], y_te])

    Xtr_win, ytr_win = build_windows_from_Xy(X_tr, y_tr, LOOKBACK)
    Xva_win, yva_win = build_windows_from_Xy(Xva_build, yva_build, LOOKBACK)
    Xte_win, yte_win = build_windows_from_Xy(Xte_build, yte_build, LOOKBACK)

    # 8) DataLoader
    Ltr = DataLoader(SeqDS(Xtr_win, ytr_win), batch_size=BATCH_SIZE, shuffle=True)
    Lva = DataLoader(SeqDS(Xva_win, yva_win), batch_size=BATCH_SIZE, shuffle=False)
    Lte = DataLoader(SeqDS(Xte_win, yte_win), batch_size=BATCH_SIZE, shuffle=False)

    # 9) 모델 학습
    in_dim = Xtr_win.shape[2]  # PCA 적용 시 축소된 차원
    model = GRUReg(in_dim=in_dim, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT)
    model = train_model(model, Ltr, Lva, epochs=EPOCHS, lr=LR, patience=20)

    # 10) 평가 (역스케일 포함)
    y_true, y_pred = evaluate_and_report(model, Lte, ScY)

    # (선택) 결과를 CSV로 저장하고 싶으면 아래 주석 해제
    # out = pd.DataFrame({
    #     "date": num.iloc[te_s].index[LOOKBACK:],
    #     "y_true": y_true,
    #     "y_pred": y_pred
    # })
    # out.to_csv("./gru_pca_backtest.csv", index=False)
