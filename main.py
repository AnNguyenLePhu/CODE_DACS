# =============================================================================
# main.py
# Pipeline dự đoán giá đóng cửa VN30 — chia thành 12 hàm step rõ ràng.
#
# STEP 01  step01_load_validate()       — Đọc CSV, kiểm tra cơ bản
# STEP 02  step02_handle_missing()      — Xử lý missing (chỉ ffill, không bfill)
# STEP 03  step03_feature_engineering() — Tính chỉ báo kỹ thuật (per-ticker)
# STEP 04  step04_clean_features()      — Xóa NaN sau FE, kiểm tra leakage
# STEP 05  step05_split()               — Chronological split 70/15/15
# STEP 06  step06_save_pretrain()       — Lưu data/artifacts trước khi train
# STEP 07  step07_scale()               — Fit scaler CHỈ trên train
# STEP 08  step08_sliding_windows()     — Tạo X/Y windows 3 kịch bản
# STEP 09  step09_train()               — Train 6 model (anti-overfit callbacks)
# STEP 10  step10_predict_evaluate()    — Dự báo, inverse transform, metrics
# STEP 11  step11_overfitting_audit()   — Kiểm tra overfit/lazy/copy giá
# STEP 12  step12_ensemble_and_save()   — Ensemble + lưu kết quả tổng hợp
# =============================================================================

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models, regularizers, callbacks as keras_callbacks

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import (
    DATA_FILE, RESULTS_DIR,
    TRAIN_RATIO, VAL_RATIO,
    FEATURE_COLS, N_FEATURES,
    SCENARIOS, MODEL_NAMES,
    EPOCHS, BATCH_SIZE, INITIAL_LR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MONITOR,
    REDUCE_LR_MONITOR, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_LR,
    H3_EARLY_STOPPING_PATIENCE, H3_REDUCE_LR_PATIENCE, H3_WARMUP_EPOCHS,
    H7_EARLY_STOPPING_PATIENCE, H7_REDUCE_LR_PATIENCE,
    DROPOUT_RATE, L2_LAMBDA,
    PARHYBRID_L2_OVERRIDE, PARHYBRID_SPATIAL_DROP, PARHYBRID_GRAD_CLIP,
    LABEL_SMOOTH_ALPHA, AUG_NOISE_STD_X, AUG_NOISE_STD_Y, AUG_PROB, MC_DROPOUT_SAMPLES,
    DIR_LOSS_WEIGHT_H1, DIR_LOSS_WEIGHT_H3, DIR_LOSS_WEIGHT_H7, USE_DIR_LOSS,
    OVERFIT_R2_GAP_THRESHOLD, DA_PASS_THRESHOLD, DA_WARN_THRESHOLD,
    VOLRATIO_TARGET_MIN, VOLRATIO_TARGET_MAX,
    LAZY_PASS_THRESHOLD, MAPE_MIN_THRESHOLD, MAPE_MAX_THRESHOLD,
    MIN_TICKER_ROWS, USE_TEST_MODE, TEST_TICKERS, SEED,
    USE_SAMPLE_WEIGHTS, SAMPLE_WEIGHT_MULTIPLIER, SAMPLE_WEIGHT_CLIP_MAX,
    HIGH_VOL_THRESHOLD, HIGH_VOL_WEIGHT,
    USE_HUBER_LOSS, HUBER_DELTA,
    PASS_REQUIRE_BEATS_NAIVE, PASS_MIN_DA, PASS_REQUIRE_MAPE_BEAT, PASS_MAX_COPY_RATIO,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# STEP 01 — LOAD & VALIDATE
# =============================================================================

def step01_load_validate(filepath: str) -> pd.DataFrame:
    """
    Đọc CSV gốc, kiểm tra cơ bản: shape, dtypes, duplicates, sort.
    KHÔNG xử lý missing ở bước này.
    """
    _banner("STEP 01 — LOAD & VALIDATE")

    df = pd.read_csv(filepath)
    _log(f"File         : '{filepath}'")
    _log(f"Shape (raw)  : {df.shape}")

    # Parse ngày
    df["TradingDate"] = pd.to_datetime(df["TradingDate"], errors="coerce")
    bad_dates = df["TradingDate"].isna().sum()
    if bad_dates:
        _warn(f"{bad_dates} rows có TradingDate không hợp lệ → drop")
        df = df.dropna(subset=["TradingDate"])

    # Sort — BẮT BUỘC trước mọi thao tác
    df = df.sort_values(["Ticker", "TradingDate"]).reset_index(drop=True)

    # Drop duplicates
    n_dup = df.duplicated(subset=["Ticker", "TradingDate"]).sum()
    if n_dup:
        _warn(f"{n_dup} dòng duplicate (Ticker, Date) → drop")
        df = df.drop_duplicates(subset=["Ticker", "TradingDate"]).reset_index(drop=True)

    # Drop Close <= 0 (lỗi dữ liệu: VIB 2018-07-23)
    bad_close = df["Close"] <= 0
    if bad_close.any():
        _warn(f"Drop {bad_close.sum()} dòng Close<=0: "
              f"{df.loc[bad_close, ['Ticker','TradingDate','Close']].values.tolist()}")
        df = df[~bad_close].reset_index(drop=True)

    tickers = sorted(df["Ticker"].unique().tolist())
    _log(f"Tickers      : {len(tickers)} — {tickers}")
    _log(f"Rows         : {len(df):,}")
    _log(f"Date range   : {df['TradingDate'].min().date()} → {df['TradingDate'].max().date()}")
    _log(f"Missing      : {df[['Open','High','Low','Close','Volume','VN30_Close','VN30_Volume']].isna().sum().to_dict()}")
    return df


# =============================================================================
# STEP 02 — HANDLE MISSING
# =============================================================================

def step02_handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý NaN cho các cột OHLCV và VN30.
    Chỉ ffill (forward fill) — TUYỆT ĐỐI KHÔNG bfill.
    bfill dùng dữ liệu tương lai để lấp quá khứ → data leakage.
    Drop nếu đầu chuỗi vẫn còn NaN (chưa có giá trị trước để ffill).
    """
    _banner("STEP 02 — HANDLE MISSING")

    fill_cols = ["Open", "High", "Low", "Close", "Volume", "VN30_Close", "VN30_Volume"]
    before = df[fill_cols].isna().sum()
    _log(f"Missing trước ffill: {before[before > 0].to_dict()}")

    # ffill theo từng ticker — KHÔNG bfill
    df[fill_cols] = df.groupby("Ticker")[fill_cols].transform(lambda g: g.ffill())

    still_na = df[fill_cols].isna().sum().sum()
    if still_na:
        _warn(f"Còn {still_na} NaN sau ffill (đầu chuỗi) → drop")
        df = df.dropna(subset=fill_cols).reset_index(drop=True)

    after = df[fill_cols].isna().sum().sum()
    _log(f"Missing sau xử lý: {after}  ✓")
    _log(f"Shape sau clean  : {df.shape}")

    assert after == 0, "Còn NaN sau step02!"
    return df


# =============================================================================
# STEP 03 — FEATURE ENGINEERING (per-ticker)
# =============================================================================

def _fe_one_ticker(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Tính chỉ báo kỹ thuật cho 1 ticker.
    Chỉ dùng shift/rolling/ewm theo chiều quá khứ — KHÔNG look-ahead.
    """
    t   = df_t.copy().reset_index(drop=True)
    c   = t["Close"]
    h   = t["High"]
    lo  = t["Low"]
    o   = t["Open"]
    vol = t["Volume"].clip(lower=0)
    eps = 1e-9

    # ── Returns ──────────────────────────────────────────────────────────────
    t["return_1d"]    = c.pct_change()
    t["log_return_1d"] = np.log((c / (c.shift(1) + eps)).clip(lower=eps))

    t["return_3d"] = c.pct_change(3)
    t["return_5d"] = c.pct_change(5)

    t["volatility_5d"] = t["log_return_1d"].rolling(5).std()
    t["volatility_10d"] = t["log_return_1d"].rolling(10).std()

    # ── Spread nội ngày ──────────────────────────────────────────────────────
    t["HL_Range"]  = h - lo
    t["OC_Change"] = c - o

    # ── Volume ───────────────────────────────────────────────────────────────
    t["Volume_Change"] = np.log((vol / (vol.shift(1) + eps)).clip(lower=eps))

    # ── Moving Averages ──────────────────────────────────────────────────────
    t["MA5"]  = c.rolling(5).mean()
    t["MA10"] = c.rolling(10).mean()
    t["MA20"] = c.rolling(20).mean()

    volume_ma20 = vol.rolling(20).mean()
    t["volume_ma_ratio"] = vol / (volume_ma20 + eps)

    t["price_ma5_gap"] = c / (t["MA5"] + eps) - 1
    t["price_ma20_gap"] = c / (t["MA20"] + eps) - 1

    # ── EMA (log-return để stationary) ───────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    t["EMA12"] = np.log((ema12 / (ema12.shift(1) + eps)).clip(lower=eps))
    t["EMA26"] = np.log((ema26 / (ema26.shift(1) + eps)).clip(lower=eps))

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta   = c.diff()
    avg_g   = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l   = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    t["RSI14"] = 100.0 - (100.0 / (1.0 + avg_g / (avg_l + eps)))

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line      = ema12 - ema26
    macd_signal    = macd_line.ewm(span=9, adjust=False).mean()
    t["MACD"]        = macd_line
    t["MACD_Signal"] = macd_signal

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────────────
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    t["BB_Upper"] = bb_mid + 2 * bb_std
    t["BB_Lower"] = bb_mid - 2 * bb_std

    # ── ATR(14) ───────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    t["ATR14"] = tr.rolling(14).mean()

    # ── Gap_Flag (khoảng trống giao dịch) ────────────────────────────────────
    days_gap      = t["TradingDate"].diff().dt.days.fillna(1)
    t["Gap_Flag"] = (days_gap > 5).astype(int)

    # ── [NEW] Breakout features ──────────────────────────────────────────
    # Rolling High/Low 20 phiên (shift(1): chỉ dùng dữ liệu quá khứ, không look-ahead)
    t["rolling_max_20"] = c.rolling(20).max()
    t["rolling_min_20"] = c.rolling(20).min()

    t["breakout_up"] = (
        c > t["rolling_max_20"].shift(1)
    ).astype(int)

    t["breakout_down"] = (
        c < t["rolling_min_20"].shift(1)
    ).astype(int)

    volume_ma20 = vol.shift(1).rolling(20).mean()
    t["volume_spike"]  = (vol > 2.0 * (volume_ma20 + eps)).astype(np.float32)
    return t


def step03_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Chạy FE cho tất cả tickers. Trả về dict {ticker: DataFrame}.
    """
    _banner("STEP 03 — FEATURE ENGINEERING")
    result = {}
    for ticker in sorted(df["Ticker"].unique()):
        df_t  = df[df["Ticker"] == ticker].copy()
        df_fe = _fe_one_ticker(df_t)
        result[ticker] = df_fe
        _log(f"[{ticker}] {len(df_fe):,} rows | indicators computed")
    _log(f"Tổng: {len(result)} tickers")
    return result


# =============================================================================
# STEP 04 — CLEAN FEATURES (dropna, validate, no bfill)
# =============================================================================

def step04_clean_features(feat_dict: dict) -> dict:
    """
    Xóa NaN sau FE bằng dropna() — KHÔNG ffill/bfill trên FEATURE_COLS.
    Kiểm tra: không NaN, không Inf, đủ MIN_TICKER_ROWS.
    Trả về dict {ticker: DataFrame} (loại ticker không đủ dữ liệu).
    """
    _banner("STEP 04 — CLEAN FEATURES")
    clean = {}
    for ticker, df_fe in feat_dict.items():
        # dropna CHỈ trên FEATURE_COLS — không fill
        df_c = df_fe.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # Kiểm tra Inf
        inf_count = np.isinf(df_c[FEATURE_COLS].values.astype(float)).sum()
        if inf_count:
            _warn(f"[{ticker}] {inf_count} Inf values → drop")
            df_c = df_c.replace([np.inf, -np.inf], np.nan)
            df_c = df_c.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        if len(df_c) < MIN_TICKER_ROWS:
            _warn(f"[{ticker}] Bỏ qua — chỉ {len(df_c)} rows sau dropna (cần ≥ {MIN_TICKER_ROWS})")
            continue

        # Xác nhận không còn NaN/Inf
        assert df_c[FEATURE_COLS].isna().sum().sum() == 0, f"[{ticker}] còn NaN!"
        assert not np.isinf(df_c[FEATURE_COLS].values.astype(float)).any(), f"[{ticker}] còn Inf!"

        clean[ticker] = df_c
        _log(f"[{ticker}] {len(df_c):,} rows | {N_FEATURES} features | OK")

    _log(f"Tổng hợp lệ: {len(clean)}/{len(feat_dict)} tickers")
    return clean


# =============================================================================
# STEP 05 — CHRONOLOGICAL SPLIT 70/15/15
# =============================================================================

def step05_split(df_ticker: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Chia theo thứ tự thời gian — KHÔNG shuffle, KHÔNG random_state.
    Returns (df_train, df_val, df_test, split_info).
    """
    n      = len(df_ticker)
    n_tr   = int(n * TRAIN_RATIO)
    n_va   = int(n * VAL_RATIO)

    df_tr  = df_ticker.iloc[:n_tr].copy()
    df_va  = df_ticker.iloc[n_tr : n_tr + n_va].copy()
    df_te  = df_ticker.iloc[n_tr + n_va:].copy()

    info = {
        "n_total":    n,
        "n_train":    len(df_tr),
        "n_val":      len(df_va),
        "n_test":     len(df_te),
        "train_start": str(df_tr["TradingDate"].iloc[0].date()),
        "train_end":   str(df_tr["TradingDate"].iloc[-1].date()),
        "val_start":   str(df_va["TradingDate"].iloc[0].date()),
        "val_end":     str(df_va["TradingDate"].iloc[-1].date()),
        "test_start":  str(df_te["TradingDate"].iloc[0].date()),
        "test_end":    str(df_te["TradingDate"].iloc[-1].date()),
    }
    _log(f"  Train {len(df_tr):,}  ({info['train_start']} → {info['train_end']})")
    _log(f"  Val   {len(df_va):,}  ({info['val_start']}   → {info['val_end']})")
    _log(f"  Test  {len(df_te):,}  ({info['test_start']}  → {info['test_end']})")
    return df_tr, df_va, df_te, info


# =============================================================================
# STEP 06 — SAVE PRE-TRAIN ARTIFACTS
# =============================================================================

def step06_save_pretrain(
    ticker: str,
    scenario: dict,
    df_full: pd.DataFrame,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    split_info: dict,
    out_root: str,
) -> str:
    """
    Lưu data trước khi train:
      {out_root}/{ticker}/{scenario}/
        ├── processed_full.csv
        ├── train.csv / val.csv / test.csv
        ├── split_info.json
        └── feature_list.json
    Scaler (pkl) được lưu sau khi fit ở step07.
    Returns thư mục đã lưu.
    """
    folder = os.path.join(out_root, ticker, scenario["name"])
    os.makedirs(folder, exist_ok=True)

    df_full.to_csv(os.path.join(folder, "processed_full.csv"), index=False)
    df_tr.to_csv(  os.path.join(folder, "train.csv"),          index=False)
    df_va.to_csv(  os.path.join(folder, "val.csv"),            index=False)
    df_te.to_csv(  os.path.join(folder, "test.csv"),           index=False)

    with open(os.path.join(folder, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    with open(os.path.join(folder, "feature_list.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "n_features": N_FEATURES}, f, indent=2)
    
    metadata = {
    "ticker": ticker,
    "scenario": scenario["name"],
    "lookback": scenario["lookback"],
    "horizon": scenario["horizon"],
    "n_features": N_FEATURES,
    "feature_cols": FEATURE_COLS,
    "full_rows": len(df_full),
    "train_rows": len(df_tr),
    "val_rows": len(df_va),
    "test_rows": len(df_te),
    "target": f"Future_Log_Return_{scenario['horizon']}d",
    "target_formula": "log(Future_Close_h / Current_Close)",
    "prediction_formula": "Pred_Close = Base_Close * exp(Pred_Log_Return)"
    }

    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    target_info = {
        "target_type": "future_log_return",
        "horizon": scenario["horizon"],
        "target_column": f"Future_Log_Return_{scenario['horizon']}d",
        "future_close_column": f"Future_Close_{scenario['horizon']}d",
        "base_close_column": "Close",
        "formula": "Future_Log_Return_h = log(Future_Close_h / Close)",
        "inverse_formula": "Pred_Close = Base_Close * exp(Pred_Log_Return)"
}

    with open(os.path.join(folder, "target_info.json"), "w", encoding="utf-8") as f:
        json.dump(target_info, f, indent=2, ensure_ascii=False)
    _log(f"[{ticker}|{scenario['name']}] Pre-train artifacts → {folder}")
    return folder


# =============================================================================
# STEP 07 — SCALE (fit CHỈ trên train)
# =============================================================================

def step07_scale(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    folder: str,
) -> tuple:
    """
    MinMaxScaler(X): fit CHỈ trên train, transform val+test.
    Lưu scaler_x.pkl và scaler_y.pkl vào folder.
    target_scaler (StandardScaler cho Y) được fit ở step08 sau khi tạo windows.

    Returns (scaler_x, tr_scaled, va_scaled, te_scaled,
             tr_dates, va_dates, te_dates,
             raw_close_tr, raw_close_va, raw_close_te)
    """
    scaler_x   = MinMaxScaler(feature_range=(0, 1))
    tr_scaled  = scaler_x.fit_transform(df_tr[FEATURE_COLS].values)   # fit CHỈ train
    va_scaled  = scaler_x.transform(df_va[FEATURE_COLS].values)       # transform bằng scaler train
    te_scaled  = scaler_x.transform(df_te[FEATURE_COLS].values)       # transform bằng scaler train

    joblib.dump(scaler_x, os.path.join(folder, "scaler_x.pkl"))
    _log(f"  scaler_x fit on train ({len(df_tr)} rows) — saved scaler_x.pkl")

    return (
        scaler_x,
        tr_scaled, va_scaled, te_scaled,
        df_tr["TradingDate"].values, df_va["TradingDate"].values, df_te["TradingDate"].values,
        df_tr["Close"].values.astype(np.float64),
        df_va["Close"].values.astype(np.float64),
        df_te["Close"].values.astype(np.float64),
    )


# =============================================================================
# STEP 08 — SLIDING WINDOWS
# =============================================================================

def _make_windows(
    scaled: np.ndarray,
    raw_close: np.ndarray,
    dates: np.ndarray,
    lookback: int,
    horizon: int,
    context_scaled: np.ndarray = None,
    context_close:  np.ndarray = None,
    context_dates:  np.ndarray = None,
) -> tuple:
    """
    X[i]      = scaled[i-lookback : i]   shape (lookback, N_FEATURES)
    Y[i]      = log_return tại [i..i+horizon-1]  shape (horizon,)
    Y_dates   = dates tương ứng với Y
    last_close= Close tại i-1 (để inverse về giá)
    """
    if context_scaled is not None:
        scaled    = np.concatenate([context_scaled, scaled], axis=0)
        raw_close = np.concatenate([context_close,  raw_close], axis=0)
        dates     = np.concatenate([context_dates,  dates], axis=0)

    X, Y, Y_dates, last_close = [], [], [], []
    total = len(scaled)
    for i in range(lookback, total - horizon + 1):
        close_w = raw_close[i - 1 : i + horizon]          # Close[t-1], Close[t], ..., Close[t+h-1]
        log_ret = np.log(close_w[1:] / (close_w[:-1] + 1e-9))
        X.append(scaled[i - lookback : i])
        Y.append(log_ret)
        Y_dates.append(dates[i : i + horizon])
        last_close.append(close_w[0])                      # Close tại t-1

    if not X:
        return None, None, None, None
    return (
        np.array(X,          dtype=np.float32),
        np.array(Y,          dtype=np.float32),
        np.array(Y_dates,    dtype=object),
        np.array(last_close, dtype=np.float64),
    )


def step08_sliding_windows(
    tr_scaled: np.ndarray,  va_scaled: np.ndarray,  te_scaled: np.ndarray,
    raw_close_tr: np.ndarray, raw_close_va: np.ndarray, raw_close_te: np.ndarray,
    tr_dates: np.ndarray,   va_dates: np.ndarray,   te_dates: np.ndarray,
    scenario: dict,
    folder: str,
) -> tuple | None:
    """
    Tạo sliding windows cho 3 tập. Fit StandardScaler trên Y_train.
    Lưu scaler_y.pkl. Returns None nếu không đủ windows.
    """
    lb = scenario["lookback"]
    h  = scenario["horizon"]

    X_tr, Y_tr, Yd_tr, Lc_tr = _make_windows(tr_scaled, raw_close_tr, tr_dates, lb, h)
    X_va, Y_va, Yd_va, Lc_va = _make_windows(
        va_scaled, raw_close_va, va_dates, lb, h,
        context_scaled=tr_scaled[-lb:],
        context_close=raw_close_tr[-lb:],
        context_dates=tr_dates[-lb:],
    )
    X_te, Y_te, Yd_te, Lc_te = _make_windows(
        te_scaled, raw_close_te, te_dates, lb, h,
        context_scaled=va_scaled[-lb:],
        context_close=raw_close_va[-lb:],
        context_dates=va_dates[-lb:],
    )

    if X_tr is None or len(X_tr) < 10 or X_va is None or len(X_va) < 1 or X_te is None or len(X_te) < 1:
        _warn(f"  Không đủ windows (tr={0 if X_tr is None else len(X_tr)}, "
              f"va={0 if X_va is None else len(X_va)}, "
              f"te={0 if X_te is None else len(X_te)}) → bỏ qua")
        return None

    # StandardScaler cho Y — fit CHỈ trên Y_train
    scaler_y  = StandardScaler()
    Y_tr_s    = scaler_y.fit_transform(Y_tr.reshape(-1, 1)).reshape(Y_tr.shape)
    Y_va_s    = scaler_y.transform(Y_va.reshape(-1, 1)).reshape(Y_va.shape)
    # Y_te không scale (chỉ dùng cho inverse)

    joblib.dump(scaler_y, os.path.join(folder, "scaler_y.pkl"))

    _log(f"  Windows: train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,} | "
         f"features={X_tr.shape[2]} | horizon={h}")

    return {
        "X_tr": X_tr,  "Y_tr_s": Y_tr_s,  "Y_tr": Y_tr, "Yd_tr": Yd_tr, "Lc_tr": Lc_tr,
        "X_va": X_va,  "Y_va_s": Y_va_s,  "Y_va": Y_va, "Yd_va": Yd_va, "Lc_va": Lc_va,
        "X_te": X_te,                      "Y_te": Y_te, "Yd_te": Yd_te, "Lc_te": Lc_te,
        "scaler_y": scaler_y,
    }


# =============================================================================
# STEP 09 — BUILD & TRAIN MODELS
# =============================================================================

# ── Custom loss: Huber + Directional penalty ─────────────────────────────────────────────
# Phạt mạnh khi dự đoán sai chiều (actual tăng nhưng pred giảm, hoặc ngược lại).
# Hệ số 0.20 là điểm khởi đầu — tăng lên 0.30/0.40 nếu DA vẫn < 53%.
# Không nên vượt 0.50 (model có thể chỉ đoán chiều, MAPE tăng mạnh).
DIR_PENALTY_WEIGHT = 0.30

def directional_huber_loss(y_true, y_pred):
    """
    Loss = Huber(delta=1.0) + DIR_PENALTY_WEIGHT * directional_error

    directional_error = tỷ lệ các mẫu dự đoán SAI chiều (sign khác nhau).
    """
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

    true_sign = tf.sign(y_true)
    pred_sign = tf.sign(y_pred)
    direction_error    = tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
    direction_penalty  = tf.reduce_mean(direction_error)

    return huber + DIR_PENALTY_WEIGHT * direction_penalty


def _build_model(name: str, lookback: int, horizon: int) -> tf.keras.Model:
    """Tạo model theo tên. Tất cả dùng padding='causal' cho Conv1D."""
    nf = N_FEATURES
    dr = DROPOUT_RATE
    l2 = L2_LAMBDA
    inp = layers.Input(shape=(lookback, nf))

    if name == "LSTM":
        x = layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.LSTM(32, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    elif name == "GRU":
        x = layers.GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.GRU(32, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    elif name == "RNN":
        x = layers.SimpleRNN(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.SimpleRNN(32, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    else:
        raise ValueError(f"Unknown model: {name}")


    out = layers.Dense(horizon)(x)
    m   = models.Model(inp, out, name=name)

    opt = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR,
                                   clipnorm=PARHYBRID_GRAD_CLIP if name == "ParHybrid" else 1.0)
    m.compile(optimizer=opt, loss=directional_huber_loss, metrics=["mae"])
    return m


def _build_callbacks(horizon: int, ckpt_path: str) -> list:
    """EarlyStopping + ReduceLROnPlateau + ModelCheckpoint."""
    if horizon == 3:
        es_p, lr_p = H3_EARLY_STOPPING_PATIENCE, H3_REDUCE_LR_PATIENCE
    elif horizon == 7:
        es_p, lr_p = H7_EARLY_STOPPING_PATIENCE, H7_REDUCE_LR_PATIENCE
    else:
        es_p, lr_p = EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE

    return [
        keras_callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=es_p,
            restore_best_weights=True,    # ← giữ weights tốt nhất
            verbose=0,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor=REDUCE_LR_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=lr_p,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=0,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def _label_smooth(Y: np.ndarray, alpha: float = LABEL_SMOOTH_ALPHA) -> np.ndarray:
    return (1.0 - alpha) * Y + alpha * float(np.mean(Y))


def _augment(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Gaussian noise injection để chống memorization."""
    mask = np.random.rand(len(X)) < AUG_PROB
    X_aug = X.copy()
    Y_aug = Y.copy()
    X_aug[mask] += np.random.normal(0, AUG_NOISE_STD_X, X_aug[mask].shape).astype(np.float32)
    Y_aug[mask] += np.random.normal(0, AUG_NOISE_STD_Y, Y_aug[mask].shape).astype(np.float32)
    return X_aug, Y_aug


def _mc_predict(model: tf.keras.Model, X: np.ndarray, n: int = MC_DROPOUT_SAMPLES) -> np.ndarray:
    """Monte Carlo dropout inference."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n)], axis=0)
    return preds.mean(axis=0)


def step09_train(
    w: dict,
    model_name: str,
    scenario: dict,
    folder: str,
) -> dict | None:
    """
    Build, augment, label-smooth, train model với callbacks chống overfit.
    shuffle=False — time-series.
    Returns dict kết quả hoặc None nếu lỗi.
    """
    lb = scenario["lookback"]
    h  = scenario["horizon"]
    sc = scenario["name"]

    tf.keras.backend.clear_session()
    model    = _build_model(model_name, lb, h)
    ckpt_path = os.path.join(folder, f"best_{model_name}_{sc}.keras")
    cbs       = _build_callbacks(h, ckpt_path)

    X_tr_aug, Y_tr_aug = _augment(w["X_tr"], w["Y_tr_s"])
    Y_smooth = _label_smooth(Y_tr_aug)

    # ── Sample weights: nhấn mạnh ngày biến động lớn ────────────────────────
    # Ngày |log_return| > HIGH_VOL_THRESHOLD → weight = HIGH_VOL_WEIGHT
    # Ngày thường → weight = 1.0
    # HIGH_VOL_THRESHOLD=0.015, HIGH_VOL_WEIGHT=5.0 (config)
    # Giảm từ 10 → 5 so với trước để gradient ổn định hơn.
    sw = None
    if USE_SAMPLE_WEIGHTS:
        abs_y = np.abs(Y_tr_aug[:, 0]) if Y_tr_aug.ndim == 2 else np.abs(Y_tr_aug)
        sw    = np.where(abs_y > HIGH_VOL_THRESHOLD,
                         HIGH_VOL_WEIGHT, 1.0).astype(np.float32)


    history = model.fit(
        X_tr_aug, Y_smooth,
        sample_weight=sw,
        validation_data=(w["X_va"], w["Y_va_s"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        shuffle=False,              # time-series: KHÔNG shuffle
        verbose=0,
    )

    # Load best checkpoint
    if os.path.exists(ckpt_path):
        try:
            model = tf.keras.models.load_model(ckpt_path, compile=False)
        except Exception as e:
            _warn(f"Không load được checkpoint: {e}")

    ep_run    = len(history.history["loss"])
    best_val  = float(min(history.history.get("val_loss", [float("inf")])))
    _log(f"  [{model_name}] epochs={ep_run} | best_val_loss={best_val:.6f}")

    # Lưu full model + history
    model.save(os.path.join(folder, f"model_{model_name}_{sc}.keras"))
    with open(os.path.join(folder, f"history_{model_name}_{sc}.json"), "w") as f:
        json.dump({
            "loss":     [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        }, f)

    return {
        "model":     model,
        "best_val":  best_val,
        "ep_run":    ep_run,
    }



# =============================================================================
# STEP 10 — PREDICT & EVALUATE
# =============================================================================

def _inverse_to_price(
    pred_scaled: np.ndarray,
    last_close: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """
    1. Inverse StandardScaler → log-return thực
    2. Close_pred = last_close × exp(cumsum(log_ret))
    """
    bs, h     = pred_scaled.shape
    log_ret   = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(bs, h)
    prices    = np.zeros((bs, h), dtype=np.float64)
    for i in range(bs):
        prices[i] = float(last_close[i]) * np.exp(np.cumsum(log_ret[i]))
    return prices


def _actual_price(Y_raw: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    bs, h  = Y_raw.shape
    prices = np.zeros((bs, h), dtype=np.float64)
    for i in range(bs):
        prices[i] = float(last_close[i]) * np.exp(np.cumsum(Y_raw[i]))
    return prices

def _directional_accuracy(actual, pred, last_close):
    a = actual.flatten()
    p = pred.flatten()

    lc = np.repeat(last_close, actual.shape[1])

    true_dir = np.sign(a - lc)
    pred_dir = np.sign(p - lc)

    return float(np.mean(true_dir == pred_dir) * 100)

def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    a = actual.flatten().astype(np.float64)
    p = pred.flatten().astype(np.float64)
    mask = a != 0
    a_m, p_m = a[mask], p[mask]
    rmse = float(np.sqrt(np.mean((a_m - p_m) ** 2)))
    mae  = float(np.mean(np.abs(a_m - p_m)))
    mape = float(np.mean(np.abs((a_m - p_m) / (np.abs(a_m) + 1e-9))) * 100)
    ss_res = np.sum((a_m - p_m) ** 2)
    ss_tot = np.sum((a_m - np.mean(a_m)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    da   = float(np.mean(np.sign(np.diff(a_m)) == np.sign(np.diff(p_m))) * 100) if len(a_m) > 1 else 50.0
    vr   = float(np.std(np.diff(p_m)) / (np.std(np.diff(a_m)) + 1e-9))
    return {"RMSE": round(rmse,6), "MAE": round(mae,6), "MAPE": round(mape,4),
            "R2": round(r2,6), "DA": round(da,2), "VolRatio": round(vr,4)}


def _naive_metrics(actual: np.ndarray, last_close: np.ndarray, h: int) -> dict:
    naive = np.tile(last_close.reshape(-1, 1), (1, h))
    a = actual.flatten(); n = naive.flatten()
    mask = a != 0
    return {
        "Naive_RMSE": round(float(np.sqrt(np.mean((a[mask]-n[mask])**2))), 6),
        "Naive_MAE":  round(float(np.mean(np.abs(a[mask]-n[mask]))), 6),
        "Naive_MAPE": round(float(np.mean(np.abs((a[mask]-n[mask])/(np.abs(a[mask])+1e-9)))*100), 4),
    }


def _build_pred_df(Y_dates, actual, pred, h) -> pd.DataFrame:
    rows = []
    if h == 1:
        for d, a, p in zip(Y_dates.flatten(), actual.flatten(), pred.flatten()):
            rows.append({"Date": pd.Timestamp(d).date(), "Actual": a, "Predicted": p})
    else:
        for i in range(len(Y_dates)):
            for j in range(h):
                rows.append({"Date": pd.Timestamp(Y_dates[i][j]).date(),
                             "Actual": actual[i][j], "Predicted": pred[i][j]})
    return (pd.DataFrame(rows)
            .groupby("Date", as_index=False).mean()
            .sort_values("Date").reset_index(drop=True)
            .round(4))


def step10_predict_evaluate(
    model_result: dict,
    w: dict,
    scenario: dict,
    model_name: str,
    ticker: str,
    folder: str,
) -> dict:
    """
    Dự báo train/val/test, inverse transform → giá Close,
    tính metrics 3 tập.
    """
    model    = model_result["model"]
    scaler_y = w["scaler_y"]
    h        = scenario["horizon"]
    sc       = scenario["name"]

    pred_tr_s = _mc_predict(model, w["X_tr"])
    pred_va_s = _mc_predict(model, w["X_va"])
    pred_te_s = _mc_predict(model, w["X_te"])

    pred_tr = _inverse_to_price(pred_tr_s, w["Lc_tr"], scaler_y)
    pred_va = _inverse_to_price(pred_va_s, w["Lc_va"], scaler_y)
    pred_te = _inverse_to_price(pred_te_s, w["Lc_te"], scaler_y)

    act_tr  = _actual_price(w["Y_tr"], w["Lc_tr"])
    act_va  = _actual_price(w["Y_va"], w["Lc_va"])
    act_te  = _actual_price(w["Y_te"], w["Lc_te"])

    m_tr    = _metrics(act_tr, pred_tr)
    m_va    = _metrics(act_va, pred_va)
    m_te    = _metrics(act_te, pred_te)
    naive   = _naive_metrics(act_te, w["Lc_te"], h)
    
    m_tr["DA"] = _directional_accuracy(
    act_tr, pred_tr, w["Lc_tr"]
    )

    m_va["DA"] = _directional_accuracy(
    act_va, pred_va, w["Lc_va"]
    )

    m_te["DA"] = _directional_accuracy(
    act_te, pred_te, w["Lc_te"]
    )
    
    # Lưu prediction CSV (test)
    df_pred = _build_pred_df(w["Yd_te"], act_te, pred_te, h)
    df_pred.to_csv(os.path.join(folder, f"pred_{model_name}_{sc}.csv"), index=False)

    _log(f"  [{model_name}|{sc}] "
         f"Tr MAPE={m_tr['MAPE']:.2f}% | Va MAPE={m_va['MAPE']:.2f}% | "
         f"Te MAPE={m_te['MAPE']:.2f}% | Naive={naive['Naive_MAPE']:.2f}%")

    return {
        "ticker": ticker, 
        "model": model_name, 
        "scenario": sc,
        "m_tr": m_tr, 
        "m_va": m_va, 
        "m_te": m_te, 
        "naive": naive,
        "pred_te": pred_te, 
        "act_te": act_te,
        "pred_tr": pred_tr, 
        "act_tr": act_tr,
        "Yd_te": w["Yd_te"], 
        "Lc_te": w["Lc_te"],
        "pred_va": pred_va,
        "act_va": act_va,
        "Yd_va": w["Yd_va"],
        "Lc_va": w["Lc_va"],
    }


# =============================================================================
# STEP 11 — OVERFITTING / LAZY / COPY PRICE AUDIT
# =============================================================================

def step11_overfitting_audit(ev: dict) -> dict:
    """
    Kiểm tra:
      R2_gap     : R2_train - R2_test   (> 0.02 → WARN_OVERFIT)
      LazyRatio  : % dự báo cứng nhắc gần Close[t-1]
      CopyRatio  : MAPE_test / Naive_MAPE (gần 1 → copy giá)
      Beats_Naive: model tốt hơn naive về RMSE
      DA         : directional accuracy
      Status     : PASS | WARN_OVERFIT | WARN_LAZY | WARN_MAPE | FAIL
    """
    m_tr = ev["m_tr"]; m_va = ev["m_va"]; m_te = ev["m_te"]
    naive = ev["naive"]

    r2_gap    = m_tr["R2"] - m_te["R2"]
    val_gap   = m_tr["R2"] - m_va["R2"]
    lazy_r    = float(m_te["RMSE"]) / (float(naive["Naive_RMSE"]) + 1e-9)
    copy_r    = float(m_te["MAPE"]) / (float(naive["Naive_MAPE"]) + 1e-9)
    beats     = m_te["RMSE"] < naive["Naive_RMSE"]

    # Detect lazy (copy giá): |pred - last_close| rất nhỏ
    pred_flat = ev["pred_te"].flatten()
    lc_rep    = np.repeat(ev["Lc_te"], ev["pred_te"].shape[1])
    diff_r    = np.abs(pred_flat - lc_rep) / (np.abs(lc_rep) + 1e-9)
    lazy_cnt  = int((diff_r < 1e-4).sum())
    lazy_pct  = lazy_cnt / max(len(pred_flat), 1)

    issues = []
    if r2_gap > OVERFIT_R2_GAP_THRESHOLD or val_gap > OVERFIT_R2_GAP_THRESHOLD:
        issues.append("OVERFIT")

    # Điều kiện chống lazy/copy giá (nghiêm hơn trước)
    lazy_fail = (
        (PASS_REQUIRE_BEATS_NAIVE and not beats) or        # phải vượt naive RMSE
        lazy_pct > LAZY_PASS_THRESHOLD or                  # predict không được quá cứng
        m_te["DA"] < DA_WARN_THRESHOLD or                  # DA phải >= 52%
        (PASS_REQUIRE_MAPE_BEAT and copy_r >= PASS_MAX_COPY_RATIO)  # MAPE < 95% naive
    )
    if lazy_fail:
        issues.append("LAZY")

    if m_te["MAPE"] < MAPE_MIN_THRESHOLD or m_te["MAPE"] > MAPE_MAX_THRESHOLD:
        issues.append("MAPE")
    if m_te["VolRatio"] < VOLRATIO_TARGET_MIN or m_te["VolRatio"] > VOLRATIO_TARGET_MAX:
        issues.append("VOLRATIO")

    if not issues:
        status = "PASS"
    elif len(issues) >= 2:
        status = "FAIL"
    else:
        status = f"WARN_{issues[0]}"

    result = {
        "R2_gap":     round(r2_gap,  4),
        "ValR2_gap":  round(val_gap, 4),
        "LazyRatio":  round(lazy_pct, 4),
        "CopyRatio":  round(copy_r,  4),
        "Beats_Naive": beats,
        "DA_test":    m_te["DA"],
        "VolRatio":   m_te["VolRatio"],
        "Status":     status,
    }
    _log(f"  Audit [{ev['model']}|{ev['scenario']}]: "
         f"R2_gap={r2_gap:.4f} LazyPct={lazy_pct*100:.2f}% "
         f"CopyRatio={copy_r:.3f} Beats={beats} DA={m_te['DA']:.1f}% → {status}")
    return result


# =============================================================================
# STEP 12 — ENSEMBLE + SAVE ALL RESULTS
# =============================================================================

def step12_ensemble_and_save(
    ticker: str,
    scenario: dict,
    ev_dict: dict,          # {model_name: eval_result}
    audit_dict: dict,       # {model_name: audit_result}
    folder: str,
    all_rows: list,         # danh sách metrics rows (append vào đây)
) -> None:
    """
    - ensemble is disabled
    - only save per-model metrics
    - Append all metrics rows into all_rows
    """
    sc = scenario["name"]
    h  = scenario["horizon"]

    # ── Per-model rows ───────────────────────────────────────────────────────
    for mn, ev in ev_dict.items():
        aud = audit_dict.get(mn, {})
        row = {
            "Ticker": ticker, "Scenario": sc, "Model": mn,
            **{f"Tr_{k}": v for k, v in ev["m_tr"].items()},
            **{f"Va_{k}": v for k, v in ev["m_va"].items()},
            **{f"Te_{k}": v for k, v in ev["m_te"].items()},
            **ev["naive"],
            **aud,
        }
        all_rows.append(row)

    # ── Ensemble đã bị tắt — chỉ lưu per-model metrics ─────────────────────



# =============================================================================
# HELPERS
# =============================================================================

def _banner(msg: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")

def _log(msg: str)  -> None: print(f"  {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}")


def _save_all_metrics(all_rows: list, results_dir: str) -> pd.DataFrame:
    if not all_rows:
        _warn("Không có metrics để lưu!")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    path = os.path.join(results_dir, "merged_metrics_ALL.csv")
    df.to_csv(path, index=False)
    _log(f"Lưu tổng hợp → {path}  ({len(df)} rows)")

    # Lưu riêng từng ticker
    for ticker, grp in df.groupby("Ticker"):
        p = os.path.join(results_dir, ticker, f"metrics_{ticker}.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        grp.to_csv(p, index=False)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["Ticker","Scenario","Model","Te_MAPE","Te_R2","LazyRatio","Beats_Naive","Status"]
    cols = [c for c in cols if c in df.columns]
    ok   = df[df["Status"] == "PASS"].sort_values("Te_MAPE") if "Te_MAPE" in df.columns else df
    _banner("TOP KẾT QUẢ — Status=PASS (sort by Te_MAPE)")
    if not ok.empty:
        print(ok[cols].head(20).to_string(index=False))
    else:
        _warn("Không có Status=PASS — xem merged_metrics_ALL.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    _banner("VN30 STOCK PRICE PREDICTION — 12-STEP PIPELINE")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    # ── STEP 01 ───────────────────────────────────────────────────────────────
    df_raw = step01_load_validate(DATA_FILE)

    # ── STEP 02 ───────────────────────────────────────────────────────────────
    df_clean = step02_handle_missing(df_raw)

    # ── STEP 03 ───────────────────────────────────────────────────────────────
    feat_dict = step03_feature_engineering(df_clean)

    # ── STEP 04 ───────────────────────────────────────────────────────────────
    clean_dict = step04_clean_features(feat_dict)

    all_tickers = sorted(clean_dict.keys())

    if USE_TEST_MODE:
        tickers = [t for t in TEST_TICKERS if t in all_tickers]
        missing = [t for t in TEST_TICKERS if t not in all_tickers]

        if missing:
            _warn(f"[TEST MODE] Không tìm thấy ticker: {missing}")

        _log(f"[TEST MODE] Chỉ chạy thử: {tickers}")
    else:
        tickers = all_tickers

    total = len(tickers)
    _log(f"Bắt đầu train: {total} tickers")

    for idx, ticker in enumerate(tickers, 1):
        _banner(f"[{idx:02d}/{total}] TICKER: {ticker}")
        df_tick = clean_dict[ticker]

        for scenario in SCENARIOS:
            sc   = scenario["name"]
            _log(f"\n  ── {scenario['label']} ──")

            folder = os.path.join(RESULTS_DIR, ticker, sc)
            os.makedirs(folder, exist_ok=True)

            # ── STEP 05 ──────────────────────────────────────────────────────
            _banner(f"STEP 05 — SPLIT [{ticker}|{sc}]")
            df_tr, df_va, df_te, split_info = step05_split(df_tick)

            # ── STEP 06 ──────────────────────────────────────────────────────
            _banner(f"STEP 06 — SAVE PRE-TRAIN [{ticker}|{sc}]")
            step06_save_pretrain(ticker, scenario, df_tick, df_tr, df_va, df_te, split_info, RESULTS_DIR)

            # ── STEP 07 ──────────────────────────────────────────────────────
            _banner(f"STEP 07 — SCALE [{ticker}|{sc}]")
            (scaler_x,
             tr_sc, va_sc, te_sc,
             tr_dt, va_dt, te_dt,
             raw_tr, raw_va, raw_te) = step07_scale(df_tr, df_va, df_te, folder)

            # ── STEP 08 ──────────────────────────────────────────────────────
            _banner(f"STEP 08 — SLIDING WINDOWS [{ticker}|{sc}]")
            w = step08_sliding_windows(
                tr_sc, va_sc, te_sc,
                raw_tr, raw_va, raw_te,
                tr_dt, va_dt, te_dt,
                scenario, folder,
            )
            if w is None:
                continue

            ev_dict    = {}
            audit_dict = {}

            for model_name in MODEL_NAMES:
                # ── STEP 09 ──────────────────────────────────────────────────
                _banner(f"STEP 09 — TRAIN {model_name} [{ticker}|{sc}]")
                model_result = step09_train(w, model_name, scenario, folder)


                if model_result is None:
                    continue

                # ── STEP 10 ──────────────────────────────────────────────────
                _banner(f"STEP 10 — EVALUATE {model_name} [{ticker}|{sc}]")
                ev = step10_predict_evaluate(model_result, w, scenario, model_name, ticker, folder)
                ev_dict[model_name] = ev

                # ── STEP 11 ──────────────────────────────────────────────────
                _banner(f"STEP 11 — AUDIT {model_name} [{ticker}|{sc}]")
                aud = step11_overfitting_audit(ev)
                audit_dict[model_name] = aud

                del model_result["model"]
                tf.keras.backend.clear_session()

            # ── STEP 12 ──────────────────────────────────────────────────────
            _banner(f"STEP 12 — ENSEMBLE & SAVE [{ticker}|{sc}]")
            step12_ensemble_and_save(ticker, scenario, ev_dict, audit_dict, folder, all_rows)

        _log(f"Hoàn thành {ticker}")

    # ── Lưu tổng hợp ─────────────────────────────────────────────────────────
    df_all = _save_all_metrics(all_rows, RESULTS_DIR)
    _print_summary(df_all)
    _banner("PIPELINE HOÀN TẤT")
    _log(f"Kết quả tại: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
