# =============================================================================
# config.py
# Toàn bộ hằng số tập trung tại đây — các file khác chỉ import từ đây.
# =============================================================================

import os

# ── Đường dẫn ────────────────────────────────────────────────────────────────
DATA_FILE   = "D:/DACS/3.0/Dataset_VN30_2017_2026.csv"
RESULTS_DIR = "D:/DACS/3.0/results"

# ── Dữ liệu ──────────────────────────────────────────────────────────────────
MIN_TICKER_ROWS = 300   # Bỏ qua ticker có < 300 rows sau feature engineering

# ── Test mode ────────────────────────────────────────────────────────────────
USE_TEST_MODE = True # True = test, False = full run
TEST_TICKERS = ["VNM","VPB","VRE",]

# ── Chronological split ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.70      # 70% train
VAL_RATIO   = 0.15      # 15% validation
# TEST_RATIO = 0.15     # 15% test (phần còn lại)

# ── 3 kịch bản sliding window ────────────────────────────────────────────────
SCENARIOS = [
    {"name": "scenario_1", "label": "Scenario 1 (Lookback=20,  Horizon=1)", "lookback": 20,  "horizon": 1},
    {"name": "scenario_2", "label": "Scenario 2 (Lookback=40,  Horizon=3)", "lookback": 40,  "horizon": 3},
    {"name": "scenario_3", "label": "Scenario 3 (Lookback=100, Horizon=7)", "lookback": 100, "horizon": 7},
]

# ── Features dùng để train ────────────────────────────────────────────────────
# Bao gồm Open, High, Low, Close (4 cột bắt buộc theo yêu cầu)
# Loại bỏ DELTA_* (không trong yêu cầu)
FEATURE_COLS = [
    # 4 cột OHLC bắt buộc
    "Open",
    "High",
    "Low",
    "Close",
    # Returns
    "return_1d",
    "log_return_1d",
    # Spread nội ngày
    "HL_Range",
    "OC_Change",
    # Volume
    "Volume",
    "Volume_Change",
    # Moving Averages
    "MA5",
    "MA10",
    "MA20",
    # EMA
    "EMA12",
    "EMA26",
    # Momentum oscillators
    "RSI14",
    "MACD",
    "MACD_Signal",
    # Bollinger Bands
    "BB_Upper",
    "BB_Lower",
    # Volatility
    "ATR14",
    # Gap
    "Gap_Flag",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volatility_10d",
    "volume_ma_ratio",
    "price_ma5_gap",
    "price_ma20_gap",
    # Breakout features
    "rolling_max_20",
    "rolling_min_20",
    "breakout_up",
    "breakout_down",
    "volume_spike",
]

N_FEATURES = len(FEATURE_COLS)   # = 34

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_NAMES   = ["RNN", "LSTM", "GRU" ]

# ── Huấn luyện ───────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 64
INITIAL_LR = 3e-4

# EarlyStopping
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MONITOR  = "val_loss"

# ReduceLROnPlateau
REDUCE_LR_MONITOR  = "val_loss"
REDUCE_LR_FACTOR   = 0.5
REDUCE_LR_PATIENCE = 7
REDUCE_LR_MIN_LR   = 1e-7

# Patience riêng cho horizon=3
H3_EARLY_STOPPING_PATIENCE = 25
H3_REDUCE_LR_PATIENCE      = 8
H3_WARMUP_EPOCHS           = 5

# Patience riêng cho horizon=7
H7_EARLY_STOPPING_PATIENCE = 25
H7_REDUCE_LR_PATIENCE      = 8

# ── Regularization ───────────────────────────────────────────────────────────
DROPOUT_RATE = 0.30
L2_LAMBDA    = 1e-4

# ParHybrid dùng regularization mạnh hơn
PARHYBRID_L2_OVERRIDE  = 1e-3
PARHYBRID_SPATIAL_DROP = 0.50
PARHYBRID_GRAD_CLIP    = 0.8

# ── Chống overfitting ────────────────────────────────────────────────────────
LABEL_SMOOTH_ALPHA = 0.02   # Giảm từ 0.10 → 0.02: train signal rõ hơn, loss không bị mờ
AUG_NOISE_STD_X    = 0.008  # Giảm từ 0.012 → 0.008: không át signal khi label smooth đã thấp
AUG_NOISE_STD_Y    = 0.002  # Giảm từ 0.004 → 0.002
AUG_PROB           = 0.40   # Giảm từ 0.60 → 0.40: augment ít hơn, model học signal rõ hơn
MC_DROPOUT_SAMPLES = 20     # Monte Carlo dropout inference

# Directional loss weights
DIR_LOSS_WEIGHT_H1 = 0.40
DIR_LOSS_WEIGHT_H3 = 0.30
DIR_LOSS_WEIGHT_H7 = 0.20

# Directional loss: phạt khi dự đoán sai chiều (actual tăng nhưng pred giảm)
# directional_loss = mean( max(0, -sign(y_true) * y_pred) )
USE_DIR_LOSS       = True   # Bật directional penalty

# Magnitude constraint
MAG_CONSTRAINT_RATIO  = 0.30
MAG_CONSTRAINT_WEIGHT = 0.15

# ── Ngưỡng đánh giá ──────────────────────────────────────────────────────────
OVERFIT_R2_GAP_THRESHOLD = 0.020   # R²_gap > 0.02 → WARN_OVERFIT
DA_PASS_THRESHOLD        = 55.0    # DA% ≥ 55% → PASS
DA_WARN_THRESHOLD        = 52.0    # DA% < 52% → WARN_LAZY
VOLRATIO_TARGET_MIN      = 0.85    # VolRatio < 0.85 → WARN
VOLRATIO_TARGET_MAX      = 1.15    # VolRatio > 1.15 → WARN
LAZY_PASS_THRESHOLD      = 0.05    # LazyRatio > 5% → WARN_LAZY
MAPE_MIN_THRESHOLD       = 0.80    # MAPE < 0.8% → suspect lazy
MAPE_MAX_THRESHOLD       = 2.00    # MAPE > 2.0% → too noisy

# ── [CẢI TIẾN 2] Sample weight cho ngày biến động lớn ────────────────────────
USE_SAMPLE_WEIGHTS       = True    # Bật/tắt sample weighting
SAMPLE_WEIGHT_GAMMA      = 2.0     # Mũ khuếch đại: weight = (|return| / median) ^ gamma
SAMPLE_WEIGHT_CLIP_MAX   = 5.0    # Clamp weight tối đa — tránh 1 sample chiếm quá nhiều gradient
SAMPLE_WEIGHT_MULTIPLIER = 5.0    # Hệ số nhân (giảm từ 10 → 5 để ổn định hơn)
# Ngưỡng phân loại "ngày biến động lớn": |log_return| > HIGH_VOL_THRESHOLD
# Ngày thường → weight=1.0, ngày biến động lớn → weight=HIGH_VOL_WEIGHT
HIGH_VOL_THRESHOLD       = 0.015  # 1.5% log-return
HIGH_VOL_WEIGHT          = 5.0    # Giảm từ 10 → 5: gradient ổn định hơn, không bị spike

# ── [CẢI TIẾN 3] Huber loss ───────────────────────────────────────────────────
USE_HUBER_LOSS           = True    # True = Huber, False = MSE
HUBER_DELTA              = 1.0     # delta=1.0: gần MAE khi sai số lớn, MSE khi nhỏ

# ── [CẢI TIẾN 4] Điều kiện PASS nghiêm hơn ───────────────────────────────────
PASS_REQUIRE_BEATS_NAIVE = True    # PASS phải có Beats_Naive=True (Te_RMSE < Naive_RMSE)
PASS_MIN_DA              = 52.0    # PASS phải có DA_test >= 52%
PASS_REQUIRE_MAPE_BEAT   = True    # PASS phải có Te_MAPE < Naive_MAPE (không copy giá)
PASS_MAX_COPY_RATIO      = 0.95    # PASS phải có CopyRatio < 0.95 (tức Te_MAPE < 95% Naive_MAPE)

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
