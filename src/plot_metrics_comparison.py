"""Biểu đồ so sánh MAE / RMSE / WAPE của LSTM với 2 baseline (MA7, seasonal naive).

Đầu vào : output/lstm/metrics_valid_overall.csv, metrics_test_overall.csv, metrics_test_by_category.csv,
          metrics_test_by_group4.csv (do src/train_lstm.py tạo)
Đầu ra  : output/lstm/figures/fig_metrics_overall.png   (valid + test, toàn bộ)
          output/lstm/figures/fig_metrics_by_category.png (test, Thực phẩm vs Đồ tiêu dùng)
          output/lstm/figures/fig_metrics_by_group4.png   (test, 4 nhóm F1, F2, H1, H2)

Muốn thêm mô hình khác (ví dụ LightGBM): thêm cột <tên>_WAPE / _MAE / _RMSE vào các file metrics
và thêm 1 dòng vào MODELS (theo thứ tự màu cố định).

Chạy: python src/plot_metrics_comparison.py            (nhãn tiếng Việt)
      python src/plot_metrics_comparison.py --lang en  (nhãn tiếng Anh, file *_en.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "output" / "lstm"
OUT = IN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# (cột trong file metrics, nhãn hiển thị, màu) — thứ tự màu cố định, không xoay vòng
MODELS = [("lstm", "LSTM", "#2a78d6"), ("ma7", "MA7", "#eb6834"), ("seasonal_naive", "Seasonal naive", "#1baf7a")]
EN = "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en"
SUFFIX = "_en" if EN else ""
DEC = "." if EN else ","                       # dấu thập phân theo ngôn ngữ
if EN:
    METRICS = [("WAPE", "WAPE (lower is better)", "{:.3f}"),
               ("MAE", "MAE (units / day)", "{:.2f}"),
               ("RMSE", "RMSE (units / day)", "{:.2f}")]
    SPLIT_LABELS = ["Validation\n(Apr–Jun 2024)", "Test\n(Jul–Sep 2024)"]
    CAT_LABELS = ["Food", "Household"]
    GROUP_LABELS = ["F1 Produce", "F2 Meat", "H1 Personal\ncare", "H2 Laundry,\ncleaning"]
else:
    METRICS = [("WAPE", "WAPE (càng thấp càng tốt)", "{:.3f}"),
               ("MAE", "MAE (đơn vị / ngày)", "{:.2f}"),
               ("RMSE", "RMSE (đơn vị / ngày)", "{:.2f}")]
    SPLIT_LABELS = ["Valid\n(04–06/2024)", "Test\n(07–09/2024)"]
    CAT_LABELS = ["Thực phẩm", "Đồ tiêu dùng"]
    GROUP_LABELS = ["F1 Rau quả", "F2 Thịt", "H1 Chăm sóc\ncá nhân", "H2 Giặt giũ,\ntẩy rửa"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def grouped_panels(table, groups, group_labels, title, path, figsize=(11, 3.3), label_size=8):
    """table: DataFrame chỉ số theo dòng = nhóm; mỗi chỉ số 1 panel riêng (thang riêng, không dùng 2 trục)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    n, width, gap = len(MODELS), 0.26, 0.02
    x = np.arange(len(groups))
    for ax, (met, label, fmt) in zip(axes, METRICS):
        top = max(table.loc[g, f"{m}_{met}"] for g in groups for m, _, _ in MODELS)
        for k, (m, name, color) in enumerate(MODELS):
            vals = [table.loc[g, f"{m}_{met}"] for g in groups]
            pos = x + (k - (n - 1) / 2) * (width + gap)
            ax.bar(pos, vals, width, color=color, label=name, zorder=2)
            for p, v in zip(pos, vals):        # nhãn giá trị trực tiếp (màu chữ trung tính)
                ax.text(p, v + top * 0.015, fmt.format(v).replace(".", DEC), ha="center", va="bottom",
                        fontsize=label_size, color=INK)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xticks(x, group_labels)
        ax.set_ylim(0, top * 1.15)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", DEC)))
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    # tiêu đề hình đặt ở chú thích trong báo cáo (tham số title chỉ để ghi chú)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Đã lưu", path)


valid = pd.read_csv(IN / "metrics_valid_overall.csv", index_col=0).rename(index={"all": "valid"})
test = pd.read_csv(IN / "metrics_test_overall.csv", index_col=0).rename(index={"all": "test"})
overall = pd.concat([valid, test])
grouped_panels(overall, ["valid", "test"], SPLIT_LABELS,
               "Hình 1. So sánh WAPE, MAE, RMSE: LSTM và baseline", OUT / f"fig_metrics_overall{SUFFIX}.png")

cat = pd.read_csv(IN / "metrics_test_by_category.csv", index_col=0)
grouped_panels(cat, ["food", "household"], CAT_LABELS,
               "Hình 2. So sánh WAPE, MAE, RMSE trên tập test theo nhóm hàng", OUT / f"fig_metrics_by_category{SUFFIX}.png")

grp = pd.read_csv(IN / "metrics_test_by_group4.csv", index_col=0)
grouped_panels(grp, ["F1", "F2", "H1", "H2"], GROUP_LABELS,
               "Hình 3. So sánh WAPE, MAE, RMSE trên tập test theo 4 nhóm hàng", OUT / f"fig_metrics_by_group4{SUFFIX}.png",
               figsize=(14, 3.8), label_size=6.5)
