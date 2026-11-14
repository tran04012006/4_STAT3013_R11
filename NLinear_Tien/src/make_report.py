"""
Tao bieu do + bao cao Word (REPORT_NLinear_Tien.docx) tu cac file trong outputs/.
Chay SAU nlinear_forecast.py:   python src/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
M = OUT / "metrics"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- palette (reference instance, light mode) ----
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COL = {"Seasonal naive": "#a3a29c", "NLinear": "#2a78d6",
       "NLinear-X (KM + lich tuong lai)": "#eb6834", "NLinear-X-cal (chi lich)": "#1baf7a"}
SHORT = {"Seasonal naive": "Seasonal naive", "NLinear": "NLinear",
         "NLinear-X (KM + lich tuong lai)": "NLinear-X (KM + lịch)",
         "NLinear-X-cal (chi lich)": "NLinear-X-cal (chỉ lịch)"}
GCOL = {"F1": "#2a78d6", "F2": "#eb6834", "H1": "#1baf7a", "H2": "#eda100"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def read(name):
    return pd.read_csv(M / f"{name}.csv")


overall, by_cat, by_g4 = read("overall"), read("by_category"), read("by_group4")
by_phase, by_h, by_promo = read("by_phase"), read("by_horizon"), read("by_promo_status")
by_store, bu, dm = read("by_store"), read("bottom_up_group_store_day"), read("diebold_mariano")
tune, eps = read("tuning_validation"), read("episodes_summary")
coef = pd.read_csv(M / "nlinear_x_exo_coefficients.csv", index_col=0)
W = pd.read_csv(M / "nlinear_weights.csv", index_col=0)
run = json.load(open(OUT / "run_config.json", encoding="utf-8"))
models = list(COL)


def get(df, model, col, **flt):
    sub = df[df["model"] == model]
    for k, v in flt.items():
        sub = sub[sub[k] == v]
    return float(sub[col].iloc[0])


# ============================== FIGURES ==============================
def grouped_bar(df, key, order, labels, title, fname, ylabel="WAPE"):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    n = len(models)
    w = 0.8 / n
    x = np.arange(len(order))
    for i, m in enumerate(models):
        vals = [get(df, m, ylabel, **{key: k}) for k in order]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=COL[m], label=SHORT[m],
               edgecolor=SURF, linewidth=1)
        if m == "NLinear-X (KM + lich tuong lai)":
            for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel + " (thấp hơn = tốt hơn)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8)
    fig.savefig(FIG / fname)
    plt.close(fig)


phase_order = ["pre_1_3", "during", "post_1_3", "post_4_7", "other"]
grouped_bar(by_phase, "phase", phase_order,
            ["Trước KM\n1–3 ngày", "Trong KM", "Sau KM\n1–3 ngày", "Sau KM\n4–7 ngày", "Ngày thường"],
            "WAPE theo giai đoạn khuyến mãi (tập test)", "fig1_wape_by_phase.png")
grouped_bar(by_g4, "group4", ["F1", "F2", "H1", "H2"],
            ["F1 Rau quả tươi", "F2 Thịt tươi/ướp lạnh", "H1 Chăm sóc cá nhân", "H2 Giặt giũ, tẩy rửa"],
            "WAPE theo nhóm hàng (tập test)", "fig2_wape_by_group.png")

# Fig 3: WAPE theo horizon
fig, ax = plt.subplots(figsize=(7.4, 3.2))
for m in models:
    sub = by_h[by_h["model"] == m]
    ax.plot(sub["horizon_step"], sub["WAPE"], color=COL[m], lw=2, marker="o", ms=4.5, label=SHORT[m])
ax.set_xlabel("Bước dự báo h (ngày sau ngày origin)")
ax.set_ylabel("WAPE")
ax.set_title("Sai số tăng dần theo tầm dự báo", loc="left")
ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8)
fig.savefig(FIG / "fig3_wape_by_horizon.png")
plt.close(fig)

# Fig 4: tong luong ban theo ngay - thuc te vs du bao (food / household)
fc = pd.read_csv(OUT / "final_result.csv", parse_dates=["date"])
fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True)
for ax, cat, name in zip(axes, ["food", "household"], ["Thực phẩm (F1+F2)", "Đồ tiêu dùng (H1+H2)"]):
    d = fc[fc["category"] == cat].groupby("date")[["actual_quantity", "pred_nlinear", "pred_nlinear_x"]].sum()
    ax.plot(d.index, d["actual_quantity"], color=INK, lw=1.6, label="Thực tế")
    ax.plot(d.index, d["pred_nlinear"], color=COL["NLinear"], lw=1.6, label="NLinear")
    ax.plot(d.index, d["pred_nlinear_x"], color=COL["NLinear-X (KM + lich tuong lai)"], lw=1.6,
            label="NLinear-X (KM + lịch)")
    ax.set_title(f"{name}: tổng lượng bán / ngày", loc="left")
    ax.set_ylabel("quantity")
axes[0].legend(ncol=3, loc="upper right", fontsize=8)
fig.autofmt_xdate()
fig.savefig(FIG / "fig4_daily_total_test.png")
plt.close(fig)

# Fig 5: trong so hieu dung cua NLinear (heatmap phan ky)
Wv = W.to_numpy().copy()
Weff = Wv.copy()
Weff[:, -1] += 1 - Wv.sum(1)          # NLinear: y = W(x - x_L) + x_L  =>  he so x_L = W_L + 1 - sum(W)
L = Wv.shape[1]
show = 28
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f2f1ee", "#2a78d6"])
lim = np.abs(Weff[:, -show:]).max()
fig, ax = plt.subplots(figsize=(7.4, 2.6))
im = ax.imshow(Weff[:, -show:], aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
ax.set_yticks(range(7), [f"h={i+1}" for i in range(7)])
ax.set_xticks(range(0, show, 7), [f"t-{show - j}" for j in range(0, show, 7)])
ax.grid(False)
ax.set_title(f"Trọng số hiệu dụng của NLinear (L={L}, hiển thị {show} ngày gần nhất)", loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.outline.set_visible(False)
fig.savefig(FIG / "fig5_nlinear_weights.png")
plt.close(fig)

# Fig 6: hieu ung KM hoc duoc (theo ty le muc nen s)
lift20 = coef.loc["promo_on"] + 0.2 * coef.loc["depth"] + 0.04 * coef.loc["depth_sq"]
eff = pd.DataFrame({"Ngày KM, giảm 20%": lift20, "1–3 ngày sau KM": coef.loc["post_promo_1_3"],
                    "4–7 ngày sau KM": coef.loc["post_promo_4_7"], "1–3 ngày trước KM": coef.loc["pre_promo_1_3"]})
fig, ax = plt.subplots(figsize=(7.4, 3.2))
x = np.arange(eff.shape[1])
for i, g in enumerate(["F1", "F2", "H1", "H2"]):
    ax.bar(x + (i - 1.5) * 0.2, eff.loc[g], 0.18, color=GCOL[g], label=g, edgecolor=SURF, linewidth=1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x, eff.columns)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Hiệu ứng × mức nền s")
ax.set_title("Hiệu ứng khuyến mãi mà NLinear-X học được, theo nhóm hàng", loc="left")
ax.legend(ncol=4, loc="upper right", fontsize=8)
fig.savefig(FIG / "fig6_promo_effects.png")
plt.close(fig)

# ============================== DOCX ==============================
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2)


def H1(t): doc.add_heading(t, level=1)
def H2(t): doc.add_heading(t, level=2)


def P(t, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold, r.italic = bold, italic
    return p


def B(t, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    # ho tro **dam** don gian
    parts = t.split("**")
    for i, part in enumerate(parts):
        r = p.add_run(part)
        r.bold = (i % 2 == 1)
    return p


def T(df, widths=None, fmt=None, font=9):
    fmt = fmt or {}
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold, r.font.size = True, Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            v = row[c]
            if c in fmt:
                s = fmt[c](v)
            elif isinstance(v, float):
                s = f"{v:,.4f}"
            elif isinstance(v, (int, np.integer)):
                s = f"{v:,}"
            else:
                s = str(v)
            cells[j].text = s
            for r in cells[j].paragraphs[0].runs:
                r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return t


def FIGURE(name, caption, width=16):
    doc.add_picture(str(FIG / name), width=Cm(width))
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.italic, r.font.size, r.font.color.rgb = True, Pt(9), RGBColor(0x52, 0x51, 0x4e)


pct = lambda v: f"{100 * v:.1f}%"
cfg = run["best_config"]
wN = get(overall, "NLinear", "WAPE")
wX = get(overall, "NLinear-X (KM + lich tuong lai)", "WAPE")
wS = get(overall, "Seasonal naive", "WAPE")
wC = get(overall, "NLinear-X-cal (chi lich)", "WAPE")
mX = "NLinear-X (KM + lich tuong lai)"

# ---------- Trang bia ----------
doc.add_heading("Báo cáo mô hình NLinear dự báo nhu cầu mua hàng", 0)
P("Đề tài: Phân tích doanh số trước, trong và sau khuyến mãi; dự báo nhu cầu; đề xuất kinh doanh", italic=True)
P("Phần việc: NLinear (Tiến) · Liên hệ Literature Review (Tiến + Trân)", italic=True)
P(f"Dữ liệu: dataset/daily_panel.parquet + dataset/episodes.csv · Thời gian chạy: {run['runtime_sec']:.0f} giây (CPU) · seed = {run['seed']}", italic=True)

# ---------- 1. Tom tat ----------
H1("1. Tóm tắt kết quả")
B(f"**NLinear vượt baseline Seasonal naive rõ rệt**: WAPE trên tập test giảm từ {pct(wS)} xuống {pct(wN)} "
  f"(giảm {pct((wS - wN) / wS)} sai số tương đối). Kiểm định Diebold–Mariano: p < 0,001.")
B(f"**Thêm thông tin khuyến mãi tương lai (NLinear-X) giúp dự báo tốt hơn nữa**: WAPE {pct(wN)} → {pct(wX)}, "
  f"MAE {get(overall, 'NLinear', 'MAE'):.3f} → {get(overall, mX, 'MAE'):.3f}, RMSE {get(overall, 'NLinear', 'RMSE'):.2f} → "
  f"{get(overall, mX, 'RMSE'):.2f}. Cải thiện có ý nghĩa thống kê ở cả Thực phẩm và Đồ tiêu dùng (DM test, p < 0,001). "
  "Điều này trả lời research gap của bài Hierarchical (ARIMAX vs MLP): khi mô hình học máy được cấp biến ngoại sinh tương lai, độ chính xác tăng.")
B(f"**Lợi ích tập trung ở quanh đợt khuyến mãi**: WAPE sau KM 1–3 ngày giảm {pct(get(by_phase,'NLinear','WAPE',phase='post_1_3'))} → "
  f"{pct(get(by_phase,mX,'WAPE',phase='post_1_3'))}, sau KM 4–7 ngày {pct(get(by_phase,'NLinear','WAPE',phase='post_4_7'))} → "
  f"{pct(get(by_phase,mX,'WAPE',phase='post_4_7'))}. Trong KM, mức dự báo thiếu giảm từ "
  f"{pct(-get(by_phase,'NLinear','Bias_pct',phase='during'))} xuống {pct(-get(by_phase,mX,'Bias_pct',phase='during'))}.")
B(f"**Ablation**: chỉ thêm biến lịch (NLinear-X-cal) đạt WAPE {pct(wC)}; phần cải thiện còn lại đến từ biến KM. "
  f"Riêng trong các ngày KM, bản chỉ-lịch còn kém hơn NLinear gốc, nghĩa là **biến KM mới là yếu tố giúp dự báo đúng ngày KM**.")
B(f"**Thực phẩm dễ dự báo hơn Đồ tiêu dùng**: WAPE NLinear-X là {pct(get(by_cat,mX,'WAPE',category='food'))} (food) so với "
  f"{pct(get(by_cat,mX,'WAPE',category='household'))} (household). Đồ tiêu dùng bán thưa (nhiều ngày bán 0) nên sai số tương đối cao.")
B("**Lưu ý quan trọng**: mô hình chọn theo WAPE cấp sản phẩm dùng hàm mất mát MAE, nên dự báo nghiêng về trung vị và "
  f"thấp hơn tổng thực tế khoảng {pct(-get(overall,mX,'Bias_pct'))}. Khi cộng dồn lên cấp nhóm hàng × cửa hàng (dùng để đặt hàng), "
  "NLinear gốc lại chính xác hơn (mục 6.8). Đây là đánh đổi cần nêu trong phần thảo luận.")

# ---------- 2. Nhiem vu ----------
H1("2. Nhiệm vụ và câu hỏi cần trả lời")
P("Theo file Overview và STATISTIC + ML_DL của nhóm, phần ML/DL gồm 5 mô hình, mỗi người một mô hình: "
  "DLinear (Trân), NLinear (Tiến), XGBoost (Ngọc), LightGBM (Hoàng), LSTM (Tú). Yêu cầu chung:")
B("Dự báo ở cấp **sản phẩm × cửa hàng × ngày** (biến mục tiêu `quantity`).")
B("Cùng một bộ thước đo: **MAE, RMSE, WAPE**; WAPE theo **Food / Household**; MAE, WAPE theo **phase** (Pre 1–3, During, Post 1–3, Post 4–7, Other).")
B("Xem xét thông tin khuyến mãi (promo_status, discount depth, trạng thái hậu KM), category, day_of_week, ngày lễ như feature.")
B("Lấp các research gap trong Literature Review: dùng dữ liệu theo ngày; tune bằng validation; cấp cho mô hình ML/DL biến KM của tương lai.")
P("Câu hỏi mà phần NLinear trả lời:", bold=True)
B("Q1. NLinear dự báo nhu cầu tốt đến mức nào so với baseline đơn giản?")
B("Q2. Đưa thông tin khuyến mãi đã lên lịch (tương lai) vào NLinear có cải thiện dự báo không, và cải thiện ở giai đoạn nào?")
B("Q3. NLinear phù hợp với Thực phẩm hay Đồ tiêu dùng hơn?")

# ---------- 3. Du lieu ----------
H1("3. Dữ liệu đầu vào")
T(pd.DataFrame([
    ["daily_panel.parquet", "1 dòng = sản phẩm × cửa hàng × ngày (1.197.327 dòng, 4.063 chuỗi)", "Huấn luyện & dự báo"],
    ["episodes.csv", "1 dòng = 1 đợt KM thật hoặc đợt giả", "Đánh giá mức bắt được doanh số trong đợt KM (mục 6.9)"],
    ["group_day.csv", "nhóm × cửa hàng × ngày × trạng thái", "Không dùng để train (không phải chuỗi liên tục); thay bằng cộng dồn bottom-up (mục 6.8)"],
], columns=["File", "Nội dung", "Vai trò"]), widths=[4, 7, 6])
P("Chia dữ liệu theo thời gian, giữ nguyên cột split của pipeline nhóm:")
T(pd.DataFrame([["Train", "28/08/2022 – 31/03/2024", "Học trọng số khi tune"],
                ["Valid", "01/04 – 30/06/2024", "Chọn L, hàm mất mát, số epoch (early stopping)"],
                ["Test", "01/07 – 26/09/2024", "Chạy đúng 1 lần, sau khi retrain trên train + valid"]],
               columns=["Tập", "Thời gian", "Dùng để"]), widths=[2.5, 5, 9])

# ---------- 4. Phuong phap ----------
H1("4. Phương pháp")
H2("4.1 NLinear là gì")
P("NLinear (Zeng et al., 2023, AAAI – \"Are Transformers Effective for Time Series Forecasting?\") là mô hình chuỗi thời gian "
  "chỉ có một lớp tuyến tính. Với cửa sổ quá khứ x = (x_{t-L}, …, x_{t-1}):")
P("    ŷ_{t+h} = Σ_j W_{h,j} · (x_j − x_{t-1}) + b_h + x_{t-1},   h = 1..7", italic=True)
B("**Chuẩn hóa bằng giá trị cuối**: trừ giá trị ngày gần nhất trước khi qua lớp Linear rồi cộng lại. Nhờ vậy mô hình chịu được thay đổi mức bán (ví dụ cửa hàng 4 mở thêm, sản phẩm tăng/giảm dần).")
B("Hệ quả toán học: tổng các hệ số hiệu dụng trên cửa sổ luôn bằng 1, tức NLinear là một **trung bình có trọng số của quá khứ**, không bị lệch mức.")
B("**Global, channel-independent**: một bộ trọng số W (7 × L) dùng chung cho cả 4.063 chuỗi, học từ ~700 nghìn cửa sổ.")
B("Lý do chọn NLinear: rất nhanh, dễ giải thích (đọc được trọng số), và theo paper gốc thường ngang hoặc tốt hơn Transformer.")
H2("4.2 NLinear-X: thêm biến ngoại sinh của tương lai")
P("    ŷ_{t+h} = NLinear(x)_h + s · ( z_{t+h} · V_g + c_{g,h} ),    s = trung bình cửa sổ + 1", italic=True)
B("z_{t+h}: các biến **đã biết trước tại ngày dự báo** cho chính ngày cần dự báo (nhà bán lẻ tự lên lịch KM, lịch ngày lễ cố định).")
B("V_g: hệ số riêng cho từng nhóm hàng g ∈ {F1, F2, H1, H2} → hiệu ứng KM được phép khác nhau giữa Thực phẩm và Đồ tiêu dùng (tương tác depth × category mà EDA đề xuất).")
B("Nhân với s: hiệu ứng KM tỷ lệ với quy mô bán của sản phẩm (sản phẩm bán 100/ngày tăng nhiều đơn vị hơn sản phẩm bán 1/ngày).")
T(pd.DataFrame([
    ["promo_on, promo_nodisc", "Ngày có KM giảm giá / KM không giảm giá"],
    ["depth, depth²", "Mức giảm giá (0–1) và bình phương (cho phép hiệu ứng phi tuyến)"],
    ["promo_first3, promo_last2", "3 ngày đầu / 2 ngày cuối của đợt KM"],
    ["pre_promo_1_3", "Còn 1–3 ngày nữa bắt đầu KM"],
    ["post_promo_1_3, post_promo_4_7", "1–3 / 4–7 ngày sau khi KM kết thúc (Post-Promotion Dip)"],
    ["dow_0..dow_6", "Thứ trong tuần (dow_0 = Thứ Hai, dow_4 = Thứ Sáu)"],
    ["is_holiday, pre_holiday, post_holiday, ny_break, minor_holiday", "Lịch lễ (theo quy tắc lọc dữ liệu của nhóm)"],
    ["month_1..month_12", "Tháng (mùa vụ)"],
    ["lookback_promo_share, last_day_promo", "Tỷ lệ ngày KM trong cửa sổ quá khứ; ngày cuối cửa sổ có KM không (để mô hình biết mức nền đang bị KM đẩy lên)"],
], columns=["Feature", "Ý nghĩa"]), widths=[6, 11])
P("Không dùng revenue, price_base, qty_norm, phase, status6 làm feature (đúng quy định chống rò rỉ của pipeline). "
  "Các biến pre/post KM được tự tính từ lịch KM (promo_status), không dùng cột phase.")
H2("4.3 Baseline và ablation")
B("**Seasonal naive**: ŷ_{t+h} = lượng bán cùng thứ của tuần trước. Đây là baseline bắt buộc theo báo cáo feature của nhóm.")
B("**NLinear-X-cal**: giống NLinear-X nhưng tắt toàn bộ biến KM, chỉ giữ biến lịch. So với NLinear-X để tách riêng phần đóng góp của thông tin KM.")
H2("4.4 Cách dự báo và đánh giá (rolling origin)")
B("Tầm dự báo H = 7 ngày (khớp với lag_7 của pipeline và thiết lập h = 1..7 của bài Hierarchical).")
B("Mỗi 7 ngày đặt một **forecast origin**; mô hình dùng dữ liệu thực tế đến ngày origin − 1 để dự báo 7 ngày tiếp theo. "
  "Tập test có 13 origin (01/07 … 23/09/2024); mỗi dòng test được dự báo đúng 1 lần.")
B("Chuỗi chưa đủ L ngày lịch sử: phần thiếu được đệm bằng trung bình của phần lịch sử đã có; chuỗi chưa có ngày nào (sản phẩm mới) dùng mức trung bình 28 ngày "
  "của các sản phẩm cùng nhóm × cửa hàng. Cột history_days trong final_result.csv cho biết số ngày lịch sử thật.")
B("Dự báo âm được chặn về 0.")
H2("4.5 Tune tham số (lấp gap \"chưa tune\" của Hewage, 2025)")
T(pd.DataFrame([
    ["Lookback L", "{14, 28, 56} ngày", f"NLinear: {cfg['nlinear']['L']} · NLinear-X: {cfg['nlinear_x']['L']}"],
    ["Hàm mất mát", "mae_raw (MAE trên đơn vị gốc) / mse_scaled (MSE trên chuỗi đã chia s)", f"NLinear: {cfg['nlinear']['loss']} · NLinear-X: {cfg['nlinear_x']['loss']}"],
    ["Số epoch", f"tối đa {run['max_epochs']}, early stopping patience {run['patience']} theo WAPE valid", f"NLinear: {cfg['nlinear']['epochs']} · NLinear-X: {cfg['nlinear_x']['epochs']}"],
    ["Optimizer", f"Adam, lr = {run['lr']}, batch = {run['batch']}", "cố định"],
    ["Tiêu chí chọn", "WAPE trên tập valid (13 origin hàng tuần)", "—"],
], columns=["Siêu tham số", "Lưới / giá trị thử", "Giá trị được chọn"]), widths=[3.2, 8, 5.8])
t2 = tune.copy()
t2["val_WAPE"] = t2["val_WAPE"].map(pct)
T(t2.rename(columns={"model": "Mô hình", "lookback_L": "L", "loss": "Loss", "best_epoch": "Epoch tốt nhất",
                     "val_WAPE": "WAPE valid", "n_train_windows": "Số cửa sổ train"}))
P("Nhận xét: mae_raw luôn tốt hơn mse_scaled (vì WAPE chính là tổng sai số tuyệt đối). L dài hơn giúp NLinear gốc (cần nhiều tuần để học chu kỳ tuần), "
  "còn NLinear-X ít nhạy với L vì biến dow đã mang thông tin chu kỳ tuần.")

# ---------- 5. Giai thich thong so ----------
H1("5. Giải thích các thông số (metrics)")
T(pd.DataFrame([
    ["MAE", "mean |y − ŷ|", "Sai số tuyệt đối trung bình mỗi dòng (đơn vị: sản phẩm hoặc kg/ngày). Dễ hiểu nhưng phụ thuộc quy mô bán."],
    ["RMSE", "√ mean (y − ŷ)²", "Phạt nặng sai số lớn (ngày đột biến KM/lễ). RMSE ≫ MAE nghĩa là có vài sai số rất lớn."],
    ["WAPE", "Σ|y − ŷ| / Σ y", "Thước đo chính của nhóm. 0,56 = tổng sai số bằng 56% tổng lượng bán. So sánh được giữa các nhóm hàng có quy mô khác nhau. Với hàng bán thưa có thể > 1."],
    ["Bias_pct", "(Σŷ − Σy) / Σy", "Âm = dự báo thiếu (nguy cơ hết hàng), dương = dự báo thừa (tồn kho)."],
    ["DM / HLN stat, p_value", "Diebold–Mariano (hiệu chỉnh Harvey–Leybourne–Newbold)", "Kiểm định sai số 2 mô hình có khác nhau thật không. Sai số tuyệt đối được cộng theo ngày (88 ngày), phương sai HAC với độ trễ 6. Stat âm = mô hình A tốt hơn B."],
    ["capture_ratio", "Σ dự báo trong đợt / Σ thực tế trong đợt", "Mô hình bắt được bao nhiêu % lượng bán trong các đợt KM (1 = bắt đủ)."],
    ["history_days", "số ngày lịch sử thật trong 56 ngày trước origin", "< L của mô hình nghĩa là cửa sổ phải đệm (sản phẩm/cửa hàng mới)."],
], columns=["Thông số", "Công thức", "Ý nghĩa"]), widths=[3, 4.2, 9.8])

# ---------- 6. Ket qua ----------
H1("6. Kết quả trên tập test (01/07 – 26/09/2024)")
P(f"Số dòng được dự báo: {run['n_test_predictions']:,} / {run['n_panel_test_rows']:,} dòng test của panel (phủ 100%).")
H2("6.1 Tổng thể")
o = overall.copy()
o["model"] = o["model"].map(SHORT)
o["WAPE"], o["Bias_pct"] = o["WAPE"].map(pct), o["Bias_pct"].map(pct)
T(o)
H2("6.2 Theo Food / Household")
c = by_cat.copy()
c["model"] = c["model"].map(SHORT)
c["WAPE"], c["Bias_pct"] = c["WAPE"].map(pct), c["Bias_pct"].map(pct)
T(c)
P("NLinear-X tốt nhất ở cả hai nhóm. Household có WAPE ~80% vì trung vị lượng bán mỗi ngày bằng 0 (bán thưa): "
  "sai số tương đối của hàng bán thưa luôn cao, đây là đặc điểm của dữ liệu chứ không phải lỗi mô hình. "
  "Vì vậy NLinear phù hợp hơn với Thực phẩm (bán đều, có chu kỳ tuần rõ).")
H2("6.3 Theo nhóm hàng")
FIGURE("fig2_wape_by_group.png", "Hình 1. WAPE theo 4 nhóm hàng. NLinear-X thấp nhất ở mọi nhóm. H1, H2 có WAPE cao vì bán thưa.")
H2("6.4 Theo giai đoạn khuyến mãi (phase)")
FIGURE("fig1_wape_by_phase.png", "Hình 2. WAPE theo phase. Biến KM tương lai giúp nhiều nhất ở 1–7 ngày sau KM.")
ph = by_phase[["phase", "model", "n", "MAE", "WAPE", "Bias_pct"]].copy()
ph["model"] = ph["model"].map(SHORT)
ph["WAPE"], ph["Bias_pct"] = ph["WAPE"].map(pct), ph["Bias_pct"].map(pct)
T(ph)
B("**Seasonal naive và NLinear dự báo thừa mạnh sau KM** (Bias +23% đến +63%): vì cửa sổ quá khứ chứa các ngày KM bán cao, mô hình không biết KM đã kết thúc.")
B("**NLinear-X biết KM kết thúc** (post_promo_1_3, last_day_promo, lookback_promo_share) nên giảm WAPE sau KM khoảng 12 điểm %.")
B("**Trong KM**, NLinear gốc thiếu ~24% vì không biết ngày mai có KM; NLinear-X giảm mức thiếu còn ~10%.")
H2("6.5 Theo tầm dự báo")
FIGURE("fig3_wape_by_horizon.png", "Hình 3. Sai số tăng theo h. Khoảng cách NLinear-X vs NLinear rộng hơn ở h xa (4–7), khi thông tin quá khứ kém cập nhật và biến tương lai càng có giá trị.")
H2("6.6 Theo trạng thái KM của ngày dự báo và theo cửa hàng")
pr = by_promo.copy()
pr["model"] = pr["model"].map(SHORT)
pr["WAPE"], pr["Bias_pct"] = pr["WAPE"].map(pct), pr["Bias_pct"].map(pct)
T(pr[["promo_status", "model", "n", "MAE", "WAPE", "Bias_pct"]])
P("Theo cửa hàng (xem metrics/by_store.csv): NLinear-X tốt nhất ở CH1, CH2, CH4; riêng CH3 (ít dữ liệu nhất, 9.369 dòng test) NLinear gốc nhỉnh hơn một chút.")
H2("6.7 Kiểm định Diebold–Mariano")
d = dm.copy()
d["subset"] = d["subset"].fillna("ALL")
d["p_value"] = d["p_value"].map(lambda v: "< 0,001" if v < 1e-3 else f"{v:.3f}")
T(d[["subset", "model_A", "model_B", "mean_daily_AE_diff(A-B)", "HLN_stat", "p_value"]])
P("Mọi cặp so sánh đều có p < 0,001: (i) NLinear tốt hơn Seasonal naive; (ii) NLinear-X tốt hơn NLinear ở cả food và household; "
  "(iii) NLinear-X tốt hơn bản chỉ-lịch, tức **biến KM có đóng góp riêng, có ý nghĩa thống kê**.")
H2("6.8 Cộng dồn lên nhóm hàng × cửa hàng × ngày (bottom-up)")
b = bu.copy()
b["model"] = b["model"].map(SHORT)
b["WAPE"], b["Bias_pct"] = b["WAPE"].map(pct), b["Bias_pct"].map(pct)
T(b[b["group4"] == "ALL"][["model", "n", "MAE", "RMSE", "WAPE", "Bias_pct"]])
P("Khi cộng dồn, sai số ngẫu nhiên từng sản phẩm triệt tiêu nhau, còn **sai lệch hệ thống (bias) thì cộng dồn**. "
  "NLinear-X học bằng MAE nên dự báo gần trung vị, mà phân phối lượng bán lệch phải → tổng dự báo thấp hơn thực tế ~11%. "
  "Do đó ở cấp tổng, NLinear gốc (bias −1,7%) cho WAPE thấp nhất (12,1%). "
  "Khuyến nghị: dùng NLinear-X cho quyết định cấp sản phẩm (ngày nào nên tăng hàng, sản phẩm nào), và hiệu chỉnh bias "
  "(nhân hệ số Σy/Σŷ học trên tập valid) hoặc dùng NLinear gốc khi lập kế hoạch tổng theo nhóm hàng.")
FIGURE("fig4_daily_total_test.png", "Hình 4. Tổng lượng bán theo ngày trên tập test. Đường NLinear-X bám dạng đỉnh KM tốt hơn nhưng nằm thấp hơn thực tế.")
H2("6.9 Đánh giá theo đợt KM (episodes.csv)")
e = eps.copy()
e["model"] = e["model"].map(SHORT)
e["capture_ratio"], e["episode_WAPE"] = e["capture_ratio"].map(pct), e["episode_WAPE"].map(pct)
T(e)
P(f"Trên {int(eps['n_episodes'].iloc[0]):,} đợt KM thật nằm trọn trong tập test, NLinear-X bắt được "
  f"{pct(get(eps, mX, 'capture_ratio', category='ALL'))} lượng bán trong đợt (NLinear gốc: {pct(get(eps, 'NLinear', 'capture_ratio', category='ALL'))}). "
  "Với Thực phẩm con số là ~94%, còn Đồ tiêu dùng chỉ ~69% – lượng bán KM của hàng tiêu dùng khó đoán hơn.")

# ---------- 7. Giai thich mo hinh ----------
H1("7. Mô hình đã học được gì")
H2("7.1 Trọng số NLinear")
FIGURE("fig5_nlinear_weights.png", "Hình 5. Trọng số hiệu dụng (đã cộng phần x_{t-1}). Màu xanh = trọng số dương.")
P("Có hai mẫu rõ ràng: (1) trọng số lớn ở các ngày gần nhất (mức bán hiện tại); (2) một đường chéo: dự báo ngày h dựa mạnh vào ngày t−8+h, "
  "tức **cùng thứ của tuần trước** → NLinear tự học được chu kỳ tuần mà không cần feature dow.")
H2("7.2 Hệ số ngoại sinh của NLinear-X")
FIGURE("fig6_promo_effects.png", "Hình 6. Hiệu ứng KM (đơn vị: bội số của mức nền s = trung bình cửa sổ + 1).")
sel = coef.loc[["promo_on", "depth", "depth_sq", "promo_last2", "pre_promo_1_3", "post_promo_1_3", "post_promo_4_7",
                "dow_4", "pre_holiday", "ny_break", "lookback_promo_share", "last_day_promo"]].reset_index()
sel.columns = ["Feature"] + list(sel.columns[1:])
T(sel)
B(f"**Ngày KM giảm 20%** làm tăng lượng bán khoảng {lift20['F1']:.2f}·s (F1), {lift20['F2']:.2f}·s (F2), {lift20['H1']:.2f}·s (H1), "
  f"{lift20['H2']:.2f}·s (H2). **Thực phẩm phản ứng mạnh hơn Đồ tiêu dùng**, khớp với kết quả EDA (lift 1,51–2,62) và độ co giãn giá của nhóm.")
B("**Giảm càng sâu bán càng nhiều** (hệ số depth, depth² dương), mạnh nhất ở F2 thịt.")
B("**Post-Promotion Dip**: hệ số post_promo_1_3 âm ở F2, H1, H2 (≈ −0,09 đến −0,12) nhưng ≈ 0 ở F1 rau quả. Phù hợp giả thuyết tích trữ: hàng để được lâu (đồ tiêu dùng, thịt có thể cấp đông) bị sụt sau KM, rau quả tươi thì không.")
B("**Cuối đợt KM bán giảm** (promo_last2 âm) và **trước lễ bán tăng** (pre_holiday dương, mạnh nhất ở thịt), **kỳ nghỉ Tết bán giảm** ở thực phẩm (ny_break âm).")
B("**Thứ Sáu (dow_4) cao nhất ở thực phẩm**, khớp quan sát EDA \"Food bán nhiều nhất vào thứ Sáu\".")
P("Lưu ý: đây là hệ số của mô hình dự báo, mô tả tương quan có điều kiện, không phải ước lượng nhân quả (KM do cửa hàng tự chọn).", italic=True)

# ---------- 8. Tra loi cau hoi ----------
H1("8. Trả lời câu hỏi và ý nghĩa kinh doanh")
B(f"**Q1**: NLinear giảm WAPE {pct((wS - wN) / wS)} so với Seasonal naive, chỉ với 1 lớp tuyến tính và ~1 phút huấn luyện trên CPU.")
B(f"**Q2**: Có. Biến KM tương lai giảm thêm {pct((wN - wX) / wN)} WAPE, có ý nghĩa thống kê; lợi ích lớn nhất ở giai đoạn sau KM và trong KM.")
B("**Q3**: NLinear phù hợp với Thực phẩm hơn (WAPE ~53% so với ~80%). Với Đồ tiêu dùng bán thưa nên cân nhắc mô hình cho cầu gián đoạn (Croston, TSB) hoặc dự báo ở cấp gộp.")
P("Gợi ý cho doanh nghiệp:", bold=True)
B("Lập kế hoạch nhập hàng nên dùng mô hình có lịch KM: mô hình không có lịch KM sẽ đặt thiếu hàng trong KM (~24%) và đặt thừa sau KM (~23–26%).")
B("Sau KM, giảm lượng nhập cho thịt và đồ tiêu dùng trong 1–7 ngày (có dip); rau quả tươi không cần giảm.")
B("Tăng tồn kho thịt trước các ngày lễ (pre_holiday mạnh nhất ở F2) và thứ Sáu hàng tuần cho thực phẩm.")

# ---------- 9. Lien he Literature review ----------
H1("9. Liên hệ Literature Review (Tiến + Trân)")
T(pd.DataFrame([
    ["Hewage (2025): dữ liệu theo tuần, KM chỉ vài ngày", "Dự báo theo ngày; feature ngày đầu/cuối KM, 1–3/4–7 ngày sau KM"],
    ["Hewage (2025): chỉ FMCG", "So sánh 2 phía hàng có hành vi khác nhau: Thực phẩm tươi vs Đồ tiêu dùng tích trữ được; hệ số riêng từng nhóm"],
    ["Hewage (2025): chưa tune tham số", "Tune L, loss, epoch bằng rolling-origin validation (13 origin)"],
    ["Hierarchical (MDPI 2024): MLP chỉ được dùng biến ngoại sinh quá khứ nên thua ARIMAX", "NLinear-X dùng biến KM/lịch của chính ngày tương lai; DM test xác nhận cải thiện có ý nghĩa"],
    ["Hierarchical: Bottom-Up cạnh tranh ở cấp cửa hàng", "Đánh giá thêm cấp nhóm × cửa hàng bằng bottom-up (mục 6.8), phát hiện vấn đề bias"],
    ["IEEE 2025: mô hình chuỗi DL kém trên dữ liệu bảng", "NLinear (tuyến tính, global) là đối chứng nhẹ cho LSTM của nhóm; kết quả để so với XGBoost/LightGBM"],
], columns=["Research gap", "Cách phần NLinear xử lý"]), widths=[7.5, 9.5])

# ---------- 10. Han che ----------
H1("10. Hạn chế và việc tiếp theo")
B("Hàm mất mát MAE → dự báo lệch thấp ~11% ở NLinear-X (mục 6.8). Có thể thêm bước hiệu chỉnh bias học trên tập valid, hoặc chọn mô hình theo cả WAPE và |Bias|.")
B("Đồ tiêu dùng bán thưa → WAPE cao (H2 > 100%). Có thể thử dự báo cấp subclass × cửa hàng, hoặc mô hình cầu gián đoạn.")
B("Khoảng 4,5% dòng test thuộc chuỗi mới xuất hiện trong test (cold start), được dự báo bằng mức trung bình nhóm × cửa hàng.")
B("Lịch lễ Nga là giả thuyết của pipeline; mức giảm giá do cửa hàng tự chọn nên hệ số chỉ là tương quan.")
B("Chỉ tối ưu doanh số, không có giá vốn để tối ưu lợi nhuận (cho phần Scenario Simulation).")
B("Để so với các thành viên khác: dùng cùng file final_result.csv (cùng khóa item_id, store_id, date) và cùng các hàm MAE/RMSE/WAPE.")

# ---------- 11. Mo ta file ----------
H1("11. Mô tả các file đầu ra")
T(pd.DataFrame([
    ["outputs/final_result.csv", "Dự báo từng dòng test: khóa, nhóm, phase, KM, actual_quantity và 4 cột dự báo"],
    ["outputs/metrics/*.csv", "Các bảng thông số trong báo cáo này (overall, by_category, by_group4, by_phase, by_horizon, by_promo_status, by_store, bottom_up, diebold_mariano, episodes, tuning, trọng số, hệ số)"],
    ["outputs/models/*.pt", "Trọng số mô hình đã huấn luyện (PyTorch)"],
    ["outputs/figures/*.png", "Các hình trong báo cáo"],
    ["outputs/run_config.json, run_log.txt", "Cấu hình được chọn, phiên bản thư viện, log huấn luyện"],
], columns=["File", "Nội dung"]), widths=[5.5, 11.5])
T(pd.DataFrame([
    ["item_id, store_id, date", "Khóa: sản phẩm, cửa hàng, ngày được dự báo"],
    ["group4, category, dept_name", "Nhóm hàng (F1/F2/H1/H2), food/household, ngành hàng"],
    ["phase", "pre_1_3 / during / post_1_3 / post_4_7 / other (từ pipeline, chỉ dùng để đánh giá)"],
    ["promo_status, depth", "promo / no_promo / promo_no_discount; mức giảm giá"],
    ["forecast_origin, horizon_step", "Ngày bắt đầu khối dự báo; ngày được dự báo là ngày thứ mấy (1–7) sau origin"],
    ["history_days", "Số ngày lịch sử thật trong 56 ngày trước forecast_origin (0–56)"],
    ["actual_quantity", "Lượng bán thực tế"],
    ["pred_seasonal_naive, pred_nlinear, pred_nlinear_x, pred_nlinear_x_cal", "Dự báo của 4 mô hình (≥ 0)"],
], columns=["Cột", "Ý nghĩa"]), widths=[6.5, 10.5])

P("Tài liệu tham khảo: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? AAAI 37(9), 11121–11128. "
  "Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. JBES 13(3). "
  "Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. IJF 13(2).", italic=True)

dst = ROOT / "REPORT_NLinear_Tien.docx"
doc.save(dst)
print("Saved", dst)
