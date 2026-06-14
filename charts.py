# -*- coding: utf-8 -*-
"""
charts.py
=================
Vẽ nhiều biểu đồ so sánh các mô hình và kịch bản từ merged_metrics_ALL.csv,
và vẽ biểu đồ chuỗi thời gian (Actual vs Predicted) cho từng mã cổ phiếu.

Biểu đồ được tạo:
  A. So sánh Mô hình (trung bình qua tất cả kịch bản)
  B. So sánh Kịch bản (trung bình qua tất cả mô hình)
  C. Mô hình x Kịch bản (chi tiết nhất)
  D. Phân tích ĐẠT/TRƯỢT (Pass/Fail)
  E. Huấn luyện vs Kiểm định vs Kiểm tra (Kiểm tra Overfitting)
  F. Biểu đồ đường chuỗi thời gian (Thực tế vs Dự đoán) cho từng mã
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
METRICS_PATH = "D:/DACS/3.0/results/merged_metrics_ALL.csv"
OUT_DIR      = "D:/DACS/3.0/charts"

HORIZON_MAP = {"scenario_1": 1, "scenario_2": 3, "scenario_3": 7}

# Đổi tên cột cho đẹp (Tập Test)
METRIC_ALIAS = {
    "Te_MAPE":    "MAPE(%)",
    "Te_MAE":     "MAE",
    "Te_RMSE":    "RMSE",
    "Te_R2":      "R2",
    "DA_test":    "DA(%)",
    "LazyRatio":  "Tỷ lệ Lười (LazyRatio)",
    "CopyRatio":  "Tỷ lệ Copy (CopyRatio)",
}

# Metrics dùng cho radar
RADAR_METRICS = ["MAPE(%)", "RMSE", "R2", "DA(%)"]
LOWER_IS_BETTER = {"MAPE(%)", "MAE", "RMSE", "Tỷ lệ Lười (LazyRatio)", "Tỷ lệ Copy (CopyRatio)"}

DPI = 150
PALETTE = "tab10"

# ─────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(METRICS_PATH):
    sys.exit(f"[LỖI] File không tồn tại: {METRICS_PATH}\nHãy chạy 1.py trước.")

df = pd.read_csv(METRICS_PATH)
print("Số dòng dữ liệu:", df.shape[0])

# Tạo cột Horizon (Ngày)
df["Horizon"] = df["Scenario"].map(HORIZON_MAP)
if df["Horizon"].isna().any():
    df["Horizon"] = df["Horizon"].fillna(df["Scenario"].str.extract(r"(\d+)$")[0].astype(float))
df["Horizon"] = df["Horizon"].astype(int)

# Đổi tên cột
for src, alias in METRIC_ALIAS.items():
    if src in df.columns and alias not in df.columns:
        df[alias] = pd.to_numeric(df[src], errors="coerce")

num_metrics  = [a for a in METRIC_ALIAS.values() if a in df.columns and df[a].dtype != object]
radar_avail  = [m for m in RADAR_METRICS if m in df.columns]

if "Status" in df.columns:
    df["PassFail"] = df["Status"].apply(lambda s: "PASS" if "PASS" in str(s).upper() else "FAIL")

models    = sorted(df["Model"].dropna().unique())
scenarios = sorted(df["Scenario"].dropna().unique())
horizons  = sorted(df["Horizon"].dropna().unique())
tickers   = sorted(df["Ticker"].dropna().unique())

sns.set_theme(style="whitegrid", font_scale=1.05)
COLORS = sns.color_palette(PALETTE, n_colors=max(len(models), len(scenarios)))

def _save(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Đã lưu:", path)

def _normalize(series, lower_is_better=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if lower_is_better else norm

# ─────────────────────────────────────────────
# A. So sánh Mô hình (Bar & Radar)
# ─────────────────────────────────────────────
print("\n[A1] Biểu đồ cột: So sánh các Mô hình")
agg_model = df.groupby("Model")[num_metrics].mean().reset_index()
for metric in num_metrics:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg_model["Model"], agg_model[metric], color=COLORS[:len(models)], edgecolor="white")
    for bar, val in zip(bars, agg_model[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"So sánh {metric} giữa các Mô hình\n(Trung bình qua các Kịch bản & Mã Cổ phiếu)", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xlabel("Mô hình")
    _save(f"A1_model_compare_{metric.replace('%', '')}.png")

print("\n[A2] Biểu đồ Radar: Hiệu suất Tổng hợp")
if len(radar_avail) >= 3:
    summary = df.groupby("Model")[radar_avail].mean().reset_index()
    norm_df = summary[["Model"]].copy()
    for m in radar_avail:
        norm_df[m] = _normalize(summary[m], lower_is_better=(m in LOWER_IS_BETTER)).values

    angles = np.linspace(0, 2 * np.pi, len(radar_avail), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(norm_df.iterrows()):
        vals = row[radar_avail].tolist() + [row[radar_avail[0]]]
        ax.plot(angles, vals, linewidth=2, label=row["Model"], color=COLORS[i])
        ax.fill(angles, vals, alpha=0.1, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_avail, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Radar: Hiệu suất Tổng hợp của các Mô hình\n(Chuẩn hóa 0-1, diện tích càng lớn càng tốt)", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save("A2_radar_models.png")

# ─────────────────────────────────────────────
# B. So sánh Kịch bản (Heatmap)
# ─────────────────────────────────────────────
print("\n[B1] Bản đồ nhiệt (Heatmap): Kịch bản vs Mô hình")
for metric in num_metrics:
    pivot = df.pivot_table(index="Scenario", columns="Model", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn", ax=ax)
    ax.set_title(f"Bản đồ nhiệt {metric}: Kịch bản vs Mô hình", fontweight="bold")
    ax.set_xlabel("Mô hình")
    ax.set_ylabel("Kịch bản")
    _save(f"B2_heatmap_{metric.replace('%', '')}.png")

# ─────────────────────────────────────────────
# C. Chi tiết Mô hình x Kịch bản (Line chart)
# ─────────────────────────────────────────────
print("\n[C1] Biểu đồ đường: Theo chân trời dự báo (Horizon)")
for metric in num_metrics:
    agg_line = df.groupby(["Horizon", "Model"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=agg_line, x="Horizon", y=metric, hue="Model", marker="o", linewidth=2, ax=ax, palette=COLORS[:len(models)])
    ax.set_title(f"Sự thay đổi của {metric} theo Chân trời dự báo (Ngày)", fontweight="bold")
    ax.set_xticks(sorted(horizons))
    ax.set_xlabel("Chân trời dự báo (Ngày)")
    ax.set_ylabel(metric)
    _save(f"C2_line_{metric.replace('%', '')}_horizon.png")

# ─────────────────────────────────────────────
# D. Phân tích ĐẠT/TRƯỢT (Pass/Fail)
# ─────────────────────────────────────────────
if "PassFail" in df.columns:
    print("\n[D1] Tỷ lệ Đạt/Trượt theo Mô hình")
    pf_model = df.groupby(["Model", "PassFail"]).size().unstack(fill_value=0).reindex(columns=["PASS", "FAIL"], fill_value=0)
    pf_pct = pf_model.div(pf_model.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    pf_pct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Tỷ lệ mô hình ĐẠT/TRƯỢT kiểm định (%)", fontweight="bold")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xlabel("Mô hình")
    for bar in ax.patches:
        if bar.get_height() > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, f"{bar.get_height():.0f}%", 
                    ha="center", va="center", color="white", fontweight="bold")
    plt.xticks(rotation=0)
    _save("D1_passfail_model.png")

# ─────────────────────────────────────────────
# E. Kiểm tra Overfitting (Train vs Val vs Test)
# ─────────────────────────────────────────────
if all(c in df.columns for c in ["Tr_MAPE", "Va_MAPE", "Te_MAPE"]):
    print("\n[E1] Huấn luyện vs Kiểm định vs Kiểm tra (MAPE)")
    tvt = df.groupby("Model")[["Tr_MAPE", "Va_MAPE", "Te_MAPE"]].mean().reset_index()
    tvt.rename(columns={"Tr_MAPE": "Huấn luyện (Train)", "Va_MAPE": "Kiểm định (Val)", "Te_MAPE": "Kiểm tra (Test)"}, inplace=True)
    tvt_melt = tvt.melt(id_vars="Model", var_name="Tập dữ liệu", value_name="MAPE")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=tvt_melt, x="Model", y="MAPE", hue="Tập dữ liệu", ax=ax, palette="Blues")
    ax.set_title("MAPE: Huấn luyện vs Kiểm định vs Kiểm tra\n(Dùng để kiểm tra Overfitting)", fontweight="bold")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Mô hình")
    _save("E1_train_val_test_MAPE.png")

if all(c in df.columns for c in ["Tr_RMSE", "Va_RMSE", "Te_RMSE"]):
    print("\n[E2] Huấn luyện vs Kiểm định vs Kiểm tra (RMSE)")
    tvt = df.groupby("Model")[["Tr_RMSE", "Va_RMSE", "Te_RMSE"]].mean().reset_index()
    tvt.rename(columns={"Tr_RMSE": "Huấn luyện (Train)", "Va_RMSE": "Kiểm định (Val)", "Te_RMSE": "Kiểm tra (Test)"}, inplace=True)
    tvt_melt = tvt.melt(id_vars="Model", var_name="Tập dữ liệu", value_name="RMSE")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=tvt_melt, x="Model", y="RMSE", hue="Tập dữ liệu", ax=ax, palette="Oranges")
    ax.set_title("RMSE: Huấn luyện vs Kiểm định vs Kiểm tra\n(Dùng để kiểm tra Overfitting)", fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Mô hình")
    _save("E2_train_val_test_RMSE.png")


# ─────────────────────────────────────────────
# F. Biểu đồ chuỗi thời gian (Thực tế vs Dự đoán)
# ─────────────────────────────────────────────
print("\n[F] Vẽ biểu đồ chuỗi thời gian Thực tế vs Dự đoán...")
PRED_OUT_DIR = os.path.join(OUT_DIR, "predictions")
os.makedirs(PRED_OUT_DIR, exist_ok=True)

# Lọc những mã có dữ liệu
count_plots = 0
for ticker in tickers:
    tick_dir = os.path.join(PRED_OUT_DIR, ticker)
    
    for sc in scenarios:
        for mdl in models:
            pred_file = f"D:/DACS/3.0/results/{ticker}/{sc}/pred_{mdl}_{sc}.csv"
            if os.path.exists(pred_file):
                df_pred = pd.read_csv(pred_file)
                if df_pred.empty: continue
                
                os.makedirs(tick_dir, exist_ok=True)
                
                # Lấy metrics để đưa lên tiêu đề
                row = df[(df["Ticker"]==ticker) & (df["Scenario"]==sc) & (df["Model"]==mdl)]
                if not row.empty:
                    mape = row.iloc[0].get("MAPE(%)", np.nan)
                    r2 = row.iloc[0].get("R2", np.nan)
                    title_metrics = f"MAPE = {mape:.2f}% | R2 = {r2:.4f}"
                else:
                    title_metrics = ""
                
                horizon_val = df[(df["Scenario"]==sc)]["Horizon"].iloc[0] if not df[(df["Scenario"]==sc)].empty else "?"
                
                fig, ax = plt.subplots(figsize=(15, 5))
                
                # Biểu đồ Thực tế vs Dự đoán
                ax.plot(df_pred.index, df_pred["Actual"], label="Thực tế (giá)", color="#1f77b4", linewidth=1.8)
                ax.plot(df_pred.index, df_pred["Predicted"], label="Dự đoán (giá)", color="#ff7f0e", linestyle="--", linewidth=1.5)
                
                # Tiêu đề & Nhãn (Tiếng Việt 100%)
                ax.set_title(f"Mã CP: {ticker} | {mdl} - Chân trời {horizon_val} ngày | {title_metrics} [Mục tiêu=Giá đóng cửa]", fontsize=13, fontweight="bold")
                ax.set_xlabel("Chỉ số mẫu (Tập Test)", fontsize=11)
                ax.set_ylabel("Giá đóng cửa (Close Price)", fontsize=11)
                
                ax.legend(loc="upper left", fontsize=10)
                ax.grid(True, alpha=0.3)
                
                save_path = os.path.join(tick_dir, f"plot_{mdl}_{sc}.png")
                plt.tight_layout()
                plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
                plt.close()
                count_plots += 1

print(f"  Đã tạo {count_plots} biểu đồ chuỗi thời gian vào {PRED_OUT_DIR}")

print(f"\n[HOÀN TẤT] Tất cả biểu đồ đã lưu vào: {OUT_DIR}")
