"""Regression: yếu tố nào liên quan đến lượng bán và mức độ liên hệ là bao nhiêu?

Phân tích 4 nhóm yếu tố: Discount Depth (khuyến mãi), Category, Store, Seasonality.

Biến phụ thuộc: y = ln(1 + quantity) → hệ số b được diễn giải là % thay đổi lượng bán ≈ 100·(e^b − 1).
Dùng toàn bộ panel (train + valid + test) vì đây là phân tích giải thích, không phải dự báo.

Ba mô hình:
  M1 – Pooled OLS: có Category và Store (khác biệt giữa các nhóm) + khuyến mãi + mùa vụ + năm.
  M2 – Fixed effects item×store (within): loại bỏ khác biệt mức bán cố định của từng cặp item×store
       → ước lượng sạch hơn cho Discount và Seasonality (Category/Store bị hấp thụ vào fixed effect).
  M3 – Như M2 nhưng thay depth tuyến tính bằng các khoảng depth × category → hình dạng phi tuyến của hiệu ứng.

Sai số chuẩn: cluster-robust theo item_id (các quan sát của cùng một sản phẩm tương quan theo thời gian).
Mức độ liên hệ (effect size): ΔR² khi bỏ cả khối biến và partial η² = ΔR² / (1 − R²_full + ΔR²).

OLS được tính bằng phương trình chuẩn X'X (k nhỏ, N = 1.2 triệu) để tiết kiệm RAM thay vì statsmodels.

Kết quả: outputs/regression_*.csv, outputs/fig_reg_*.png, outputs/regression_log.txt
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / 'data' / 'daily_panel.parquet'
OUT = HERE / 'outputs'; OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)

# Màu (palette tham chiếu của skill dataviz, slot 1–2): Food = xanh, Household = cam
C_FOOD, C_HH, INK, INK2, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})

# ======================================================================
# 1. Dữ liệu
COLS = ['date', 'item_id', 'store_id', 'category', 'quantity', 'promo_status', 'depth', 'phase',
        'dow', 'month', 'year', 'is_holiday', 'holiday_status', 'ny_break']
df = pd.read_parquet(DATA_PATH, columns=COLS)
df['y'] = np.log1p(df['quantity'].astype('float64'))
df['household'] = (df['category'] == 'household').astype(float)
df['pair'] = df['item_id'].astype(str) + '_' + df['store_id'].astype(str)
print(df.shape, '| số item:', df.item_id.nunique(), '| số cặp item×store:', df.pair.nunique())

# Depth chỉ có ý nghĩa trong khuyến mãi (ngoài KM = 0). Tách thành:
#   during (dummy)                   → hiệu ứng KM ở mức depth trung bình
#   depth10_c = (depth − mean)/0.10  → tăng thêm cho mỗi +10 điểm % chiết khấu (chỉ trong KM, ngoài KM = 0)
during = (df['phase'] == 'during')
DEPTH_MEAN = float(df.loc[during, 'depth'].mean())
df['depth10_c'] = np.where(during, (df['depth'] - DEPTH_MEAN) / 0.10, 0.0)
print(f'Depth trung bình trong KM: {DEPTH_MEAN:.4f}')

# Lễ: holiday_status −1 = trước lễ, NaN = ngày lễ, 1 = 1–7 ngày sau lễ
df['hol_pre'] = (df['holiday_status'] == -1).astype(float)
df['hol_day'] = (df['is_holiday'] == 1).astype(float)
df['hol_post'] = (df['holiday_status'] == 1).astype(float)

# ======================================================================
# 2. Ma trận thiết kế theo khối biến
def dummies(s, prefix, ref):
    d = pd.get_dummies(s.astype(str), prefix=prefix, dtype=float)
    return d.drop(columns=f'{prefix}_{ref}')

DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ph = dummies(df['phase'], 'phase', 'other')[['phase_pre_1_3', 'phase_during', 'phase_post_1_3', 'phase_post_4_7']]
BLOCKS = {
    # Khuyến mãi: phase + depth + tương tác với household (mức độ khác nhau giữa 2 ngành hàng)
    'Discount': pd.concat([ph, df[['depth10_c']],
                           (ph['phase_during'] * df['household']).rename('during_x_household'),
                           (df['depth10_c'] * df['household']).rename('depth10_x_household')], axis=1),
    'Category': df[['household']],
    'Store': dummies(df['store_id'], 'store', '1'),
    'DOW': dummies(df['dow'].map(dict(enumerate(DOW_NAMES))), 'dow', 'Mon'),
    'Month': dummies(df['month'].map(lambda m: f'{m:02d}'), 'month', '01'),
    'Holiday': df[['hol_pre', 'hol_day', 'hol_post', 'ny_break']],
    'Year': dummies(df['year'], 'year', '2022'),
}
SEASONALITY = ['DOW', 'Month', 'Holiday']
y_all = df['y'].to_numpy()
clusters = pd.factorize(df['item_id'])[0]
pair_codes = pd.factorize(df['pair'])[0]

# ======================================================================
# 3. OLS + cluster-robust SE (CR1)
def demean(a, groups):
    """Trừ trung bình theo nhóm (within transformation cho fixed effects)."""
    a = np.asarray(a, dtype=np.float64)
    cnt = np.bincount(groups)
    if a.ndim == 1:
        return a - (np.bincount(groups, a) / cnt)[groups]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(groups, a[:, j]) / cnt)[groups]
    return out

def ols(y, X, names, clusters=None, n_absorbed=0):
    """Trả về (bảng hệ số, R²). R² của mô hình FE là within-R² (y đã demean)."""
    X = np.asarray(X, dtype=np.float64)
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    if clusters is None:
        return None, r2
    n, k = X.shape
    bread = np.linalg.inv(XtX)
    G = clusters.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, clusters, X * e[:, None])
    k_eff = k + n_absorbed
    V = bread @ (S.T @ S) @ bread * (G / (G - 1)) * ((n - 1) / (n - k_eff))
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=G - 1)
    q = stats.t.ppf(0.975, df=G - 1)
    tab = pd.DataFrame({'term': names, 'coef': beta, 'se': se, 't': t, 'p': p,
                        'ci_low': beta - q * se, 'ci_high': beta + q * se})
    for c in ['coef', 'ci_low', 'ci_high']:
        tab[f'pct_{c}'] = 100 * np.expm1(tab[c])          # % thay đổi lượng bán
    return tab, r2

def build(blocks, intercept):
    X = pd.concat([BLOCKS[b] for b in blocks], axis=1)
    if intercept:
        X.insert(0, 'const', 1.0)
    return X

def effect_sizes(blocks, intercept, y, transform=lambda a: a, label=''):
    """ΔR² và partial η² khi bỏ từng khối (và cả khối Seasonality gộp)."""
    X = build(blocks, intercept)
    _, r2_full = ols(y, transform(X.to_numpy()), X.columns)
    groups = {b: [b] for b in blocks}
    if all(s in blocks for s in SEASONALITY):
        groups['Seasonality (DOW+Month+Holiday)'] = SEASONALITY
    rows = []
    for gname, drop in groups.items():
        Xr = build([b for b in blocks if b not in drop], intercept)
        _, r2_red = ols(y, transform(Xr.to_numpy()), Xr.columns)
        d = r2_full - r2_red
        rows.append({'model': label, 'block': gname, 'n_terms': X.shape[1] - Xr.shape[1],
                     'R2_full': r2_full, 'delta_R2': d, 'partial_eta2': d / (1 - r2_full + d)})
    return pd.DataFrame(rows)

# ---------- M1: Pooled OLS ----------
M1_BLOCKS = ['Discount', 'Category', 'Store', 'DOW', 'Month', 'Holiday', 'Year']
X1 = build(M1_BLOCKS, intercept=True)
m1, r2_1 = ols(y_all, X1.to_numpy(), X1.columns, clusters)
m1.insert(0, 'model', 'M1_pooled')
print(f'\nM1 pooled OLS: R² = {r2_1:.4f}, N = {len(y_all):,}, clusters = {clusters.max() + 1}')
print(m1[['term', 'coef', 'se', 'p', 'pct_coef']].round(4).to_string(index=False))
es1 = effect_sizes(M1_BLOCKS, True, y_all, label='M1_pooled')

# ---------- M2: Fixed effects item×store ----------
M2_BLOCKS = ['Discount', 'DOW', 'Month', 'Holiday', 'Year']      # Category, Store bị hấp thụ
X2 = build(M2_BLOCKS, intercept=False)
y_w = demean(y_all, pair_codes)
fe = lambda a: demean(a, pair_codes)
n_pairs = pair_codes.max() + 1
m2, r2_2 = ols(y_w, fe(X2.to_numpy()), X2.columns, clusters, n_absorbed=n_pairs)
m2.insert(0, 'model', 'M2_FE_item_store')
# R² tổng thể của M2 (tính cả fixed effects) để so với M1
r2_2_overall = 1 - (1 - r2_2) * np.sum(y_w ** 2) / np.sum((y_all - y_all.mean()) ** 2)
print(f'\nM2 FE item×store: within-R² = {r2_2:.4f}, overall R² (kèm FE) = {r2_2_overall:.4f}, số FE = {n_pairs}')
print(m2[['term', 'coef', 'se', 'p', 'pct_coef']].round(4).to_string(index=False))
es2 = effect_sizes(M2_BLOCKS, False, y_w, transform=fe, label='M2_FE_item_store (within)')

# Phần giải thích bởi đặc tính cố định item×store (fixed effects) so với các yếu tố còn lại
r2_pairs = 1 - np.sum(y_w ** 2) / np.sum((y_all - y_all.mean()) ** 2)
print(f'R² chỉ với fixed effects item×store: {r2_pairs:.4f}')

# ---------- M3: Depth bins × category (FE) ----------
BINS = [0, 0.10, 0.20, 0.30, 0.40, 1.01]
BIN_LABELS = ['<10%', '10–20%', '20–30%', '30–40%', '≥40%']
dbin = pd.cut(df['depth'], BINS, right=False, labels=BIN_LABELS)
cols = {}
for cat, is_hh in [('food', 0.0), ('household', 1.0)]:
    cm = (df['household'] == is_hh).to_numpy()
    for lab in BIN_LABELS:
        cols[f'during[{lab}]|{cat}'] = (during.to_numpy() & cm & (dbin == lab).to_numpy()).astype(float)
    for p in ['pre_1_3', 'post_1_3', 'post_4_7']:
        cols[f'phase_{p}|{cat}'] = ((df['phase'] == p).to_numpy() & cm).astype(float)
X3 = pd.concat([pd.DataFrame(cols, index=df.index)] + [BLOCKS[b] for b in ['DOW', 'Month', 'Holiday', 'Year']], axis=1)
m3, r2_3 = ols(y_w, fe(X3.to_numpy()), X3.columns, clusters, n_absorbed=n_pairs)
m3.insert(0, 'model', 'M3_FE_depthbins')
bin_n = pd.crosstab(dbin[during], df.loc[during, 'category'])
print(f'\nM3 depth bins × category (FE): within-R² = {r2_3:.4f}')
print(m3[m3.term.str.contains(r'\|')][['term', 'coef', 'p', 'pct_coef', 'pct_ci_low', 'pct_ci_high']].round(3).to_string(index=False))
print('Số dòng theo khoảng depth:'); print(bin_n)

# ======================================================================
# 4. Lưu bảng
coefs = pd.concat([m1, m2, m3], ignore_index=True)
coefs.to_csv(OUT / 'regression_coefficients.csv', index=False)
es = pd.concat([es1, es2], ignore_index=True)
es.to_csv(OUT / 'regression_effect_sizes.csv', index=False)
fit = pd.DataFrame([
    {'model': 'M1_pooled', 'N': len(y_all), 'k': X1.shape[1], 'R2': r2_1, 'within_R2': np.nan},
    {'model': 'M2_FE_item_store', 'N': len(y_all), 'k': X2.shape[1], 'R2': r2_2_overall, 'within_R2': r2_2},
    {'model': 'M3_FE_depthbins', 'N': len(y_all), 'k': X3.shape[1], 'R2': np.nan, 'within_R2': r2_3},
    {'model': 'FE_only (item×store)', 'N': len(y_all), 'k': n_pairs, 'R2': r2_pairs, 'within_R2': np.nan},
])
fit.to_csv(OUT / 'regression_fit.csv', index=False)
print('\nEffect sizes:'); print(es.round(4).to_string(index=False))
print('\nFit:'); print(fit.round(4).to_string(index=False))

# Lượng bán trung bình theo nhóm (thống kê mô tả đi kèm để đọc hệ số)
desc = (df.groupby(['category', 'phase'], observed=True)['quantity'].agg(['mean', 'median', 'count'])
          .reset_index())
desc.to_csv(OUT / 'regression_descriptive_by_phase.csv', index=False)

# ======================================================================
# 5. Biểu đồ
def coef_row(tab, term):
    r = tab[tab.term == term].iloc[0]
    return r['pct_coef'], r['pct_ci_low'], r['pct_ci_high']

def hbar_ci(ax, labels, vals, lo, hi, colors, title):
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, color=colors, height=0.62, edgecolor='white', linewidth=2)
    ax.errorbar(vals, yy, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                fmt='none', ecolor=INK, elinewidth=1, capsize=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy, labels)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, fontweight='bold')
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    span = max(max(hi), 0) - min(min(lo), 0)
    ax.set_xlim(min(min(lo), 0) - 0.04 * span, max(max(hi), 0) + 0.22 * span)
    for v, yv, l, h in zip(vals, yy, lo, hi):
        x_txt = (h if v >= 0 else max(h, 0)) + 0.03 * span      # đặt nhãn sau đầu error bar
        ax.text(x_txt, yv, f'{v:+.1f}%', va='center', fontsize=8.5, color=INK)

# Hình 1: tổng quan 4 nhóm yếu tố (Discount & Seasonality từ M2; Category & Store từ M1)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
NEU = '#8a8986'
items = [('phase_during', 'Đang KM (Food, depth TB)'), ('during_x_household', '+ chênh lệch Household'),
         ('depth10_c', '+10 điểm % chiết khấu (Food)'), ('depth10_x_household', '+ chênh lệch Household / +10đ%'),
         ('phase_pre_1_3', '1–3 ngày trước KM'), ('phase_post_1_3', '1–3 ngày sau KM'), ('phase_post_4_7', '4–7 ngày sau KM')]
v = [coef_row(m2, t) for t, _ in items]
hbar_ci(axes[0, 0], [l for _, l in items], [a for a, _, _ in v], [b for _, b, _ in v], [c for _, _, c in v],
        [C_FOOD, C_HH, C_FOOD, C_HH, NEU, NEU, NEU], 'Discount / Phase KM (M2, FE item×store)')
items = [('household', 'Household vs Food')]
items_s = [('store_2', 'Store 2 vs Store 1'), ('store_3', 'Store 3 vs Store 1'), ('store_4', 'Store 4 vs Store 1')]
v = [coef_row(m1, t) for t, _ in items + items_s]
hbar_ci(axes[0, 1], [l for _, l in items + items_s], [a for a, _, _ in v], [b for _, b, _ in v], [c for _, _, c in v],
        [C_HH, NEU, NEU, NEU], 'Category & Store (M1, pooled OLS)')
dows = [f'dow_{d}' for d in DOW_NAMES[1:]]
v = [coef_row(m2, t) for t in dows]
hbar_ci(axes[1, 0], [f'{d} vs Mon' for d in DOW_NAMES[1:]], [a for a, _, _ in v], [b for _, b, _ in v],
        [c for _, _, c in v], [NEU] * 6, 'Seasonality – ngày trong tuần (M2)')
mon_terms = [f'month_{m:02d}' for m in range(2, 13)]
v = [coef_row(m2, t) for t in mon_terms]
ax = axes[1, 1]
mm = np.arange(1, 13)
pct = [0] + [a for a, _, _ in v]; lo = [0] + [b for _, b, _ in v]; hi = [0] + [c for _, _, c in v]
ax.fill_between(mm, lo, hi, color=NEU, alpha=0.18, lw=0)
ax.plot(mm, pct, color=INK2, lw=2, marker='o', ms=5, markeredgecolor='white', markeredgewidth=1.5)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(mm, ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])
ax.set_ylabel('% so với tháng 1'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('Seasonality – tháng (M2, KTC 95%)', loc='left', fontsize=10.5, color=INK, fontweight='bold')
for a in axes.flat[:3]:
    a.set_xlabel('% thay đổi lượng bán (KTC 95%)')
fig.suptitle('Mức độ liên hệ của các yếu tố với lượng bán, y = ln(1 + quantity)', x=0.01, ha='left',
             fontsize=12, color=INK, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'fig_reg_effects.png'); plt.close(fig)

# Hình 2: uplift theo khoảng depth × category (M3)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = np.arange(len(BIN_LABELS)); w = 0.38
for i, (cat, col, name) in enumerate([('food', C_FOOD, 'Food'), ('household', C_HH, 'Household')]):
    v = [coef_row(m3, f'during[{lab}]|{cat}') for lab in BIN_LABELS]
    vals = np.array([a for a, _, _ in v]); his = np.array([c for _, _, c in v])
    xs = x + (i - 0.5) * w
    ax.bar(xs, vals, w - 0.04, color=col, label=name, edgecolor='white', linewidth=2)
    ax.errorbar(xs, vals, yerr=[vals - [b for _, b, _ in v], his - vals],
                fmt='none', ecolor=INK, elinewidth=1, capsize=3)
    for xv, val, h in zip(xs, vals, his):
        ax.text(xv, h + 6, f'{val:.0f}%', ha='center', fontsize=8, color=INK)   # nhãn trên đầu error bar
ax.set_xticks(x, BIN_LABELS); ax.set_xlabel('Mức chiết khấu (discount depth)')
ax.set_ylabel('% tăng lượng bán so với ngày thường')
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper left')
ax.set_title('Hiệu ứng khuyến mãi theo mức chiết khấu và ngành hàng (M3, FE item×store)',
             loc='left', fontsize=10.5, color=INK, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT / 'fig_reg_depth_bins.png'); plt.close(fig)

# Hình 3: effect size (partial η²) theo khối
fig, ax = plt.subplots(figsize=(7.6, 3.8))
e1 = es1.set_index('block')['partial_eta2']; e2 = es2.set_index('block')['partial_eta2']
blocks_show = ['Discount', 'Category', 'Store', 'Seasonality (DOW+Month+Holiday)', 'DOW', 'Month', 'Holiday', 'Year']
yy = np.arange(len(blocks_show))[::-1]; h = 0.38
# Hai mô hình không phải Food/Household → dùng 2 sắc độ trung tính, tránh trùng nghĩa màu xanh/cam
ax.barh(yy + h / 2, [e1.get(b, np.nan) for b in blocks_show], h - 0.04, color='#3d3c39', label='M1 pooled OLS')
ax.barh(yy - h / 2, [e2.get(b, np.nan) for b in blocks_show], h - 0.04, color='#a9a8a3', label='M2 FE item×store (within)')
for b, yv in zip(blocks_show, yy):
    for val, off in [(e1.get(b, np.nan), h / 2), (e2.get(b, np.nan), -h / 2)]:
        if np.isfinite(val):
            ax.text(val + 0.001, yv + off, f'{val:.3f}', va='center', fontsize=8, color=INK)
ax.set_yticks(yy, blocks_show); ax.set_xlabel('partial η²')
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right')
ax.set_title('Effect size theo nhóm yếu tố', loc='left', fontsize=10.5, color=INK, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT / 'fig_reg_effect_sizes.png'); plt.close(fig)
print('\nĐã lưu bảng và hình vào', OUT)
