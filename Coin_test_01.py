# pip install yfinance requests pandas pyarrow pytz python-dateutil

import os, time, warnings, math, json, requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timezone

# =========================
# 기본 설정
# =========================
TZ = "Asia/Seoul"
START = "2015-01-01"     # 필요에 맞게 조정
END   = None             # None이면 오늘까지
SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 주식/지수/환율 (야후 파이낸스 티커)
# - 코스피: ^KS11, 코스닥: ^KQ11, 달러/원: KRW=X
STOCK_TICKERS = [
    "^KS11", "^KQ11",                   # KOSPI, KOSDAQ
    "^GSPC", "^IXIC", "^NDX", "^DJI",   # S&P500, Nasdaq Composite, Nasdaq 100, Dow
    "^RUT", "^VIX",                     # Russell 2000, VIX
    "KRW=X"                             # USDKRW
]

# 코인: 업비트 KRW 마켓(다년치 수집용)
UPBIT_MARKETS = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP"]

# =========================
# 유틸: 타임존/캘린더/정렬
# =========================
def _coerce_to_tz(ts, tz):
    """naive/aware 상관없이 지정 tz로 통일"""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(tz)
    return t.tz_convert(tz)

def make_calendar(start, end=None, tz=TZ):
    end = pd.Timestamp.now(tz) if end is None else end
    start = _coerce_to_tz(start, tz)
    end   = _coerce_to_tz(end, tz)
    if end < start:
        raise ValueError("END가 START보다 빠릅니다.")
    rng = pd.date_range(start=start.normalize(),
                        end=end.normalize(),
                        freq="D")
    return pd.Index(rng)

def as_date_index(df, tz=TZ):
    """DatetimeIndex -> 지정 tz로 변환 후 자정(normalize) 기준 일자 인덱스"""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    # utc로 정규화한 뒤 지정 tz로
    df = df.tz_localize("UTC") if df.index.tz is None else df
    df = df.tz_convert(tz)
    df.index = df.index.normalize()
    return df

def safe_ffill_align(df, calendar):
    """캘린더로 리인덱스 후 전일값으로 채우기(ffill)"""
    df = df.copy()
    df = df.reindex(calendar)
    return df.ffill()

# =========================
# 수집: 야후(지수/환율)
# =========================
def fetch_yf_close(tickers, start=START, end=END):
    df = yf.download(
        tickers=tickers,
        start=start, end=end,
        interval="1d",
        auto_adjust=True,      # 배당/분할 반영 (지수/환율에는 영향 거의 없음)
        progress=False, threads=True
    )

    # 멀티인덱스 처리: Close 우선
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(0)):
            df = df["Close"]
        elif ("Adj Close" in df.columns.get_level_values(0)):
            df = df["Adj Close"]
        else:
            # 열 이름 예외 케이스 보호
            try:
                df = df.xs("Close", axis=1, level=0, drop_level=True)
            except Exception:
                # 마지막 수단: 각 티커의 Close만 수집
                cols = []
                for t in tickers:
                    try:
                        cols.append(df[("Close", t)])
                    except Exception:
                        pass
                if cols:
                    df = pd.concat(cols, axis=1)
                else:
                    raise RuntimeError("yfinance 응답 형식이 예상과 다릅니다.")
        # 컬럼명 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[1] for c in df.columns]
    else:
        # 단일 티커인 경우
        if "Close" in df.columns:
            df = df[["Close"]]
            df.columns = [tickers[0]]

    # 인덱스 타임존 정리
    df.index = pd.to_datetime(df.index, utc=True)
    df = as_date_index(df)
    return df

# =========================
# 수집: 업비트(다년치 일봉)
# - 한 번에 최대 200봉, to 파라미터로 뒤로 페이징
# =========================

# (신규) 업비트 마켓 리스트 확인 후 유효한 것만 사용
def get_upbit_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, params={"isDetails": "false"}, timeout=30, headers={"accept":"application/json"})
    r.raise_for_status()
    data = r.json()
    return {m["market"] for m in data}  # 예: {"KRW-BTC","KRW-ETH",...}

def _fmt_to_utc(dt):
    """Upbit가 가장 안전하게 받는 UTC 문자열 ('YYYY-MM-DD HH:MM:SS')"""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Seoul")  # 우리의 기준은 KST
    ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")  # 공백 구분, TZ 오프셋 없음
def fetch_upbit_days(markets=("KRW-BTC","KRW-ETH","KRW-XRP","KRW-SOL"),
                     start="2016-01-01", end=None, pause=0.12):
    tz = "Asia/Seoul"
    sdt = pd.Timestamp(start, tz=tz)
    edt = pd.Timestamp(end or pd.Timestamp.now(tz=tz))

    # 1) 마켓 유효성 필터링
    valid = get_upbit_markets()             # REST: /v1/market/all
    todo  = [m for m in markets if m in valid]
    if not todo:
        warnings.warn("업비트 유효 마켓이 없습니다.")
        return pd.DataFrame()

    frames = []
    for mkt in todo:
        to_dt = edt
        rows = []

        while True:
            # 2) UTC 문자열로 'to' 구성 (exclusive)
            params = {"market": mkt, "count": 200, "to": _fmt_to_utc(to_dt)}
            r = requests.get("https://api.upbit.com/v1/candles/days",
                             params=params, timeout=30,
                             headers={"accept":"application/json"})
            if r.status_code == 400:
                # 포맷 이슈가 남아있을 경우 대체 포맷 1회 재시도 (ISO8601 'Z')
                alt = to_dt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                r = requests.get("https://api.upbit.com/v1/candles/days",
                                 params={"market": mkt, "count": 200, "to": alt},
                                 timeout=30, headers={"accept":"application/json"})
            if r.status_code != 200:
                warnings.warn(f"Upbit {mkt} HTTP {r.status_code}: {r.text[:120]}")
                break

            data = r.json()
            if not data:
                break

            stop = False
            for item in data:  # 최신→과거
                # 응답은 UTC/KST 모두 제공되는데, 일봉은 KST 키가 가장 직관적
                ts = pd.Timestamp(item["candle_date_time_kst"]).tz_localize(tz).normalize()
                if ts < sdt:
                    stop = True
                    break
                if ts <= edt:
                    rows.append((ts, float(item["trade_price"])))

            oldest_kst = pd.Timestamp(data[-1]["candle_date_time_kst"]).tz_localize(tz)
            if stop or oldest_kst <= sdt or len(data) < 200:
                break

            # 다음 페이지: 가장 오래 캔들 바로 이전 시각으로 이동
            to_dt = oldest_kst - pd.Timedelta(seconds=1)
            time.sleep(pause)  # 레이트리밋 배려 (초당 10회 이하)

        if rows:
            s = pd.Series(dict(rows), name=mkt.split("-")[1]).sort_index()  # 'KRW-BTC' -> 'BTC'
            frames.append(s.to_frame())

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

# =========================
# 파생지표
# =========================
def make_features(price_df, prefix=""):
    df = price_df.copy()
    # 수익률
    ret1  = df.pct_change(1).add_prefix(f"{prefix}ret1_")
    ret5  = df.pct_change(5).add_prefix(f"{prefix}ret5_")
    ret20 = df.pct_change(20).add_prefix(f"{prefix}ret20_")
    # 변동성(20일 롤링, 일 변동성)
    vol20 = df.pct_change().rolling(20).std().add_prefix(f"{prefix}vol20_")
    # 모멘텀
    mom20 = (df / df.shift(20) - 1.0).add_prefix(f"{prefix}mom20_")
    mom60 = (df / df.shift(60) - 1.0).add_prefix(f"{prefix}mom60_")
    # 드로우다운
    dd = (df / df.cummax() - 1.0).add_prefix(f"{prefix}dd_")

    features = pd.concat(
        [df.add_prefix(f"{prefix}px_"), ret1, ret5, ret20, vol20, mom20, mom60, dd],
        axis=1
    )
    return features

# =========================
# 실행
# =========================
def main():
    # 1) 캘린더
    calendar = make_calendar(START, END, tz=TZ)

    # 2) 주가/지수/환율 (야후)
    yf_raw = fetch_yf_close(STOCK_TICKERS, START, END)  # ^KS11, ^KQ11, KRW=X
    yf_aligned = safe_ffill_align(yf_raw, calendar)

    # 3) 코인 (업비트 다년치 KRW 마켓)
    upbit_raw = fetch_upbit_days(UPBIT_MARKETS, START, END, pause=0.12)
    upbit_aligned = safe_ffill_align(upbit_raw, calendar)

    # 4) 피처 생성
    stock_feat = make_features(yf_aligned, prefix="stk_")
    coin_feat  = make_features(upbit_aligned, prefix="cc_")

    # 5) 합치기
    feat = pd.concat([stock_feat, coin_feat], axis=1)

    # 6) 저장
    out_parquet = os.path.join(SAVE_DIR, "market_features.parquet")
    out_csv     = os.path.join(SAVE_DIR, "market_features.csv")
    feat.to_parquet(out_parquet)
    feat.to_csv(out_csv, encoding="utf-8")

    # 7) 요약 출력
    print("DONE")
    print("rows:", len(feat), "cols:", len(feat.columns))
    print("date range:", feat.index.min().date(), "→", feat.index.max().date())
    print("saved to:", os.path.abspath(SAVE_DIR))

    # ---- (예금 데이터 병합 예시) ----
    # deposit_df = pd.read_csv("your_deposit_daily.csv")  # columns: date, deposit_amount
    # deposit_df["date"] = pd.to_datetime(deposit_df["date"]).dt.tz_localize(TZ).dt.normalize()
    # merged = deposit_df.set_index("date").join(feat, how="left")
    # # 누설 방지를 위해 시장 피처에 래그 추가 권장:
    # lagged = merged.copy()
    # for l in [1, 3, 5]:
    #     lagged[[c for c in feat.columns]] = merged[[c for c in feat.columns]].shift(l)
    # lagged.to_parquet(os.path.join(SAVE_DIR, "deposit_with_market_lag.parquet"))

if __name__ == "__main__":
    main()
