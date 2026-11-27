"""Figures for the NLinear-X forecast run (reads outputs/ written by nlinear_x_forecast.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MET = OUT / "metrics"
FIG = OUT / "figures"

# palette (reference instance, light mode)
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e4e4e4", "#ffffff"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]          # fixed categorical order
MODEL_COLOR = {"nlinear_x": SERIES[0], "seasonal_naive": SERIES[1]}
MODEL_LABEL = {"nlinear_x": "NLinear-X", "seasonal_naive": "Seasonal naive"}
MODEL_ORDER = ["seasonal_naive", "nlinear_x"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1757ac", "#84b9f3", "#efefef",
                                                      "#f19a9a", "#c42b2c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("saved", FIG / name)


def wape_bars(df, key, order, labels, title, fname, xlabel):
    piv = df.pivot(index=key, columns="model", values="WAPE").reindex(order)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(order) + 2), 4.2))
    for i, m in enumerate(MODEL_ORDER):
        vals = piv[m].to_numpy()
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=MODEL_COLOR[m],
                      label=MODEL_LABEL[m], edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8, color=TEXT2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, piv.to_numpy().max() * 1.15)
    ax.legend(loc="upper left", ncols=2)
    save(fig, fname)


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. WAPE by product group
    g4 = pd.read_csv(MET / "by_group4.csv")
    wape_bars(g4, "group4", ["F1", "F2", "H1", "H2"],
              ["F1\nFresh\nproduce", "F2\nFresh &\nchilled meat", "H1\nPersonal care\n& paper",
               "H2\nLaundry &\ncleaning"],
              "Test WAPE by product group", "01_wape_by_group4.png", "Product group")

    # 2. WAPE by promo phase
    ph = pd.read_csv(MET / "by_phase.csv")
    wape_bars(ph, "phase", ["pre_1_3", "during", "post_1_3", "post_4_7", "other"],
              ["1-3 days\nbefore promo", "During\npromo", "1-3 days\nafter promo",
               "4-7 days\nafter promo", "Other\ndays"],
              "Test WAPE by promotion phase", "02_wape_by_phase.png", "Promotion phase")

    # 3. WAPE by horizon step
    hz = pd.read_csv(MET / "by_horizon.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in MODEL_ORDER:
        d = hz[hz["model"] == m].sort_values("horizon_step")
        ax.plot(d["horizon_step"], d["WAPE"], color=MODEL_COLOR[m], lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABEL[m])
        last = d.iloc[-1]
        ax.annotate(f"{last['WAPE']:.3f}", (last["horizon_step"], last["WAPE"]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT2)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Horizon step h (days ahead)")
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Test WAPE by forecast horizon")
    ax.set_xlim(0.7, 7.6)
    ax.legend(loc="upper left", ncols=2)
    save(fig, "03_wape_by_horizon.png")

    # 4. Total daily sales: actual vs NLinear-X
    res = pd.read_csv(OUT / "final_result.csv", usecols=["date", "category", "actual_quantity",
                                                         "pred_nlinear_x"], parse_dates=["date"])
    daily = res.groupby(["category", "date"])[["actual_quantity", "pred_nlinear_x"]].sum().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, (cat, title) in zip(axes, [("food", "Food"), ("household", "Household")]):
        d = daily[daily["category"] == cat]
        ax.plot(d["date"], d["actual_quantity"], color=TEXT, lw=2, label="Actual")
        ax.plot(d["date"], d["pred_nlinear_x"], color=MODEL_COLOR["nlinear_x"], lw=2,
                label="NLinear-X")
        ax.set_title(f"{title}: total daily quantity, test period")
        ax.set_ylabel("Units per day")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", ncols=2)
    fig.autofmt_xdate()
    save(fig, "04_daily_total_actual_vs_nlinear_x.png")

    # 5. Effective weight heatmap
    wdf = pd.read_csv(MET / "nlinear_x_weights.csv")
    lag_cols = [c for c in wdf.columns if c.startswith("lag_")]
    W = wdf[lag_cols].to_numpy()
    L = W.shape[1]
    Wt = W.copy()
    Wt[:, -1] = W[:, -1] + 1.0 - W.sum(axis=1)
    lim = np.abs(Wt).max()
    fig, ax = plt.subplots(figsize=(max(8, L * 0.32), 3.8))
    im = ax.imshow(Wt, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    ax.grid(False)
    ax.set_yticks(range(7), [f"h={i}" for i in range(1, 8)])
    ax.set_xticks(range(L), [c.replace("lag_", "t-") for c in lag_cols], rotation=90, fontsize=7)
    ax.set_xlabel("Input day (t-k)")
    ax.set_title(f"NLinear-X effective weights  W~[h,j]  (L = {L}; last column includes 1 - sum W)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Weight")
    save(fig, "05_effective_weight_heatmap.png")

    # 6. Promotion effect by group (multiplicative on scale s = mean(x) + 1)
    V = pd.read_csv(MET / "nlinear_x_exo_coefficients.csv").set_index("feature")
    groups = ["F1", "F2", "H1", "H2"]
    effects = {
        "Promo day,\n20% discount": V.loc["promo_on"] + 0.2 * V.loc["depth"] + 0.04 * V.loc["depth_sq"],
        "1-3 days\nbefore promo": V.loc["pre_promo_1_3"],
        "1-3 days\nafter promo": V.loc["post_promo_1_3"],
        "4-7 days\nafter promo": V.loc["post_promo_4_7"],
    }
    names = list(effects)
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, g in enumerate(groups):
        vals = np.array([effects[n][g] for n in names])
        ax.bar(x + (i - 1.5) * (w + 0.01), vals, width=w, color=SERIES[i], label=g,
               edgecolor=SURFACE, linewidth=1)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Effect on forecast, in units of s = mean(x)+1")
    ax.set_title("NLinear-X promotion effects by product group")
    ax.legend(title="Group", loc="upper right", ncols=4)
    save(fig, "06_promo_effects_by_group.png")


if __name__ == "__main__":
    main()
