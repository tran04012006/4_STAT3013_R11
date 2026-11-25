"""Regression cấp ĐỢT KM – làm theo EDA (mục 4 "Discount Depth vs Sales Lift") và pipeline chung của nhóm.

EDA: "dùng hồi quy để kiểm tra quan hệ giữa depth và log_lift, đồng thời kiểm soát store_id, category, promo_type và
seasonality" + "kiểm tra quan hệ discount_depth × category" + "so sánh với giai đoạn đối chứng (đợt giả)".

Dữ liệu: data/episodes.csv (pipeline chung, src/prepare_data.py): 10,044 đợt KM thật + 9,700 đợt giả (placebo).
  log_lift = ln((during_qty + 1/3) / (pre_qty + 1/3)), pre = 1–3 ngày trước đợt.
Sai số chuẩn HC3 (quy tắc E4 của pipeline: Welch / HC3, luôn báo cáo cỡ hiệu ứng + KTC 95%).

Mô hình:
  E1 (chỉ đợt KM thật):  log_lift ~ depth·category + store + promo_type + tháng + lễ + năm + ln(duration)
  E2 (KM thật + đợt giả): log_lift ~ promo × category + store + tháng + lễ + năm
        → mức tăng của KM so với nhóm đối chứng (đợt giả cũng có lift ≠ 0 do hồi quy về trung bình)
  E3 (KM thật + đợt giả): log_lift ~ (khoảng depth × category) + category + kiểm soát như E2
        → lift theo mức chiết khấu, so với đợt giả cùng ngành hàng

Kết quả: outputs/reg_episode_*.csv, outputs/fig_reg_ep_*.png, outputs/regression_episodes_log.txt
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EP_PATH = HERE.parent / 'data' / 'episodes.csv'
OUT = HERE / 'outputs'; OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 200)

C_FOOD, C_HH, INK, INK2, GRID, NEU = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#8a8986'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})

# ======================================================================
# 1. Dữ liệu
ep = pd.read_csv(EP_PATH, parse_dates=['start', 'end'])
ep['household'] = (ep['category'] == 'household').astype(int)
ep['promo'] = (ep['source'] == 'promo').astype(int)
ep['depth10'] = ep['depth'] * 10                                  # đơn vị: +10 điểm % chiết khấu
ep['log_dur'] = np.log(ep['duration'])
# promo_type: giữ 2 loại phổ biến (5: 63%, 6: 31% số đợt), gộp phần còn lại
ep['ptype'] = np.where(ep['promo_type'].isin([5, 6]), ep['promo_type'].fillna(0).astype(int).astype(str), 'other')
ep['store'] = ep['store_id'].astype(str)
ep['hol'] = ep['hol_status']
pr = ep[ep.promo == 1].copy()
print(f'Đợt KM thật: {len(pr):,} (food {int((pr.household == 0).sum()):,}, household {int(pr.household.sum()):,}) | '
      f'đợt giả: {int((ep.promo == 0).sum()):,}')
print('Trung vị log_lift / lift:'); print(ep.groupby(['source', 'category'])[['log_lift', 'lift']].median().round(3))

# Mùa vụ cấp đợt: tháng bắt đầu + trạng thái lễ. KHÔNG dùng thứ bắt đầu đợt: KM gần như luôn bắt đầu Thứ 2–5
# (69% vào Thứ 5, chỉ 2 đợt vào Chủ nhật) nên đó là thuộc tính lịch KM, không phải mùa vụ nhu cầu.
# Mùa vụ theo ngày trong tuần được phân tích ở regression_analysis.py (cấp ngày).
SEAS = "C(start_month) + C(hol, Treatment('normal'))"
CTRL = f"C(store, Treatment('1')) + {SEAS} + C(year)"

def tidy(res, model):
    ci = res.conf_int()
    t = pd.DataFrame({'model': model, 'term': res.params.index, 'coef': res.params.values, 'se': res.bse.values,
                      'p': res.pvalues.values, 'ci_low': ci[0].values, 'ci_high': ci[1].values})
    for c in ['coef', 'ci_low', 'ci_high']:
        t[f'pct_{c}'] = 100 * np.expm1(t[c])           # % thay đổi của lift (tỉ lệ during / pre)
    return t

def partial_eta2(data, full, blocks, model):
    """ΔR² và partial η² khi bỏ từng khối biến khỏi công thức đầy đủ."""
    r2 = smf.ols(full, data).fit().rsquared
    rows = []
    for name, terms in blocks.items():
        f = full
        for term in terms:
            f = f.replace(f' + {term}', '').replace(f'{term} + ', '')
        r2r = smf.ols(f, data).fit().rsquared
        d = r2 - r2r
        rows.append({'model': model, 'block': name, 'R2_full': r2, 'delta_R2': d, 'partial_eta2': d / (1 - r2 + d)})
    return pd.DataFrame(rows)

# ======================================================================
# 2. E1 – chỉ đợt KM thật: depth × category + store + promo_type + seasonality
F1 = (f"log_lift ~ depth10 + depth10:household + household + C(ptype, Treatment('5')) + log_dur + {CTRL}")
e1 = smf.ols(F1, pr).fit(cov_type='HC3')
t1 = tidy(e1, 'E1_promo_only')
print(f'\nE1 (N = {int(e1.nobs):,}, R² = {e1.rsquared:.4f}, adj R² = {e1.rsquared_adj:.4f})')
print(t1[['term', 'coef', 'se', 'p', 'pct_coef']].round(4).to_string(index=False))
es1 = partial_eta2(pr, F1, {
    'Discount depth (depth + depth×category)': ['depth10', 'depth10:household'],
    'Category': ['household'],
    'Store': ["C(store, Treatment('1'))"],
    'Promo type': ["C(ptype, Treatment('5'))"],
    'Seasonality (tháng + lễ)': ['C(start_month)', "C(hol, Treatment('normal'))"],
    'Năm': ['C(year)'],
    'Độ dài đợt': ['log_dur']}, 'E1_promo_only')

# Kiểm định Wald: hiệu ứng depth trong Household (depth10 + depth10:household = 0)
w = e1.t_test('depth10 + depth10:household = 0')
hh_slope = {'term': 'depth10 (household = depth10 + depth10:household)', 'coef': float(w.effect[0]),
            'se': float(w.sd[0][0]), 'p': float(w.pvalue)}
ci = w.conf_int()[0]; hh_slope.update({'ci_low': ci[0], 'ci_high': ci[1]})
print(f"\nĐộ dốc depth của Household: {hh_slope['coef']:.4f} (SE {hh_slope['se']:.4f}, p = {hh_slope['p']:.3g})")
t1 = pd.concat([t1, pd.DataFrame([{**hh_slope, 'model': 'E1_promo_only',
                                   **{f'pct_{k}': 100 * np.expm1(hh_slope[k]) for k in ['coef', 'ci_low', 'ci_high']}}])],
               ignore_index=True)

# ======================================================================
# 3. E2 – KM thật vs đợt giả (nhóm đối chứng)
F2 = f"log_lift ~ promo * household + {CTRL}"
e2 = smf.ols(F2, ep).fit(cov_type='HC3')
t2 = tidy(e2, 'E2_promo_vs_placebo')
w = e2.t_test('promo + promo:household = 0')
hh = {'model': 'E2_promo_vs_placebo', 'term': 'promo (household = promo + promo:household)',
      'coef': float(w.effect[0]), 'se': float(w.sd[0][0]), 'p': float(w.pvalue),
      'ci_low': w.conf_int()[0][0], 'ci_high': w.conf_int()[0][1]}
hh.update({f'pct_{k}': 100 * np.expm1(hh[k]) for k in ['coef', 'ci_low', 'ci_high']})
t2 = pd.concat([t2, pd.DataFrame([hh])], ignore_index=True)
print(f'\nE2 (N = {int(e2.nobs):,}, R² = {e2.rsquared:.4f})')
print(t2[t2.term.str.contains('promo|household')][['term', 'coef', 'p', 'pct_coef', 'pct_ci_low', 'pct_ci_high']].round(4).to_string(index=False))
es2 = partial_eta2(ep, F2, {'KM thật vs đợt giả (promo + promo×category)': ['promo', 'promo:household'],
                            'Category': ['household'], 'Store': ["C(store, Treatment('1'))"],
                            'Seasonality (tháng + lễ)': ['C(start_month)', "C(hol, Treatment('normal'))"],
                            'Năm': ['C(year)']}, 'E2_promo_vs_placebo')
# Lưu ý: bỏ promo nhưng còn promo:household → khối "promo" được bỏ trọn bằng cách thay công thức
F2r = f"log_lift ~ household + {CTRL}"
r2r = smf.ols(F2r, ep).fit().rsquared; d = e2.rsquared - r2r
es2.loc[es2.block.str.startswith('KM thật'), ['delta_R2', 'partial_eta2']] = [d, d / (1 - e2.rsquared + d)]
F2c = f"log_lift ~ promo + {CTRL}"
r2c = smf.ols(F2c, ep).fit().rsquared; d = e2.rsquared - r2c
es2.loc[es2.block == 'Category', ['delta_R2', 'partial_eta2']] = [d, d / (1 - e2.rsquared + d)]

# ======================================================================
# 4. E3 – lift theo khoảng depth × category, so với đợt giả cùng ngành hàng
BINS = [0, 0.10, 0.20, 0.30, 0.40, 1.01]
BIN_LABELS = ['<10%', '10–20%', '20–30%', '30–40%', '≥40%']
ep['dbin'] = pd.cut(ep['depth'], BINS, right=False, labels=['b1', 'b2', 'b3', 'b4', 'b5']).astype(str)
cell_cols = []
for cat, hhv in [('food', 0), ('household', 1)]:
    for b in ['b1', 'b2', 'b3', 'b4', 'b5']:
        c = f'p_{cat}_{b}'; cell_cols.append(c)
        ep[c] = ((ep.promo == 1) & (ep.household == hhv) & (ep.dbin == b)).astype(int)
F3 = 'log_lift ~ ' + ' + '.join(cell_cols) + f' + household + {CTRL}'
e3 = smf.ols(F3, ep).fit(cov_type='HC3')
t3 = tidy(e3, 'E3_depthbins_vs_placebo')
n3 = ep[ep.promo == 1].groupby(['category', 'dbin']).size().unstack(0)
n3.index = BIN_LABELS
print(f'\nE3 (N = {int(e3.nobs):,}, R² = {e3.rsquared:.4f}) – % lift so với đợt giả cùng ngành hàng:')
print(t3[t3.term.str.startswith('p_')][['term', 'pct_coef', 'pct_ci_low', 'pct_ci_high', 'p']].round(2).to_string(index=False))
print('Số đợt KM theo khoảng depth:'); print(n3)

# ======================================================================
# 5. Lưu
coefs = pd.concat([t1, t2, t3], ignore_index=True)
coefs.to_csv(OUT / 'reg_episode_coefficients.csv', index=False)
es = pd.concat([es1, es2], ignore_index=True); es.to_csv(OUT / 'reg_episode_effect_sizes.csv', index=False)
fit = pd.DataFrame([{'model': m, 'N': int(r.nobs), 'R2': r.rsquared, 'adj_R2': r.rsquared_adj, 'cov_type': 'HC3'}
                    for m, r in [('E1_promo_only', e1), ('E2_promo_vs_placebo', e2), ('E3_depthbins_vs_placebo', e3)]])
fit.to_csv(OUT / 'reg_episode_fit.csv', index=False)
n3.to_csv(OUT / 'reg_episode_depthbin_counts.csv')
print('\nEffect sizes:'); print(es.round(4).to_string(index=False))
print('\nFit:'); print(fit.round(4).to_string(index=False))

# ======================================================================
# 6. Biểu đồ
def row(t, term):
    r = t[t.term == term].iloc[0]
    return r['pct_coef'], r['pct_ci_low'], r['pct_ci_high']

# Hình 1: lift theo khoảng depth × category (E3) – phiên bản kiểm định của biểu đồ EDA "Discount Depth vs Sales Lift"
fig, ax = plt.subplots(figsize=(7.8, 4.5))
x = np.arange(len(BIN_LABELS)); w_ = 0.38
for i, (cat, col, name) in enumerate([('food', C_FOOD, 'Food'), ('household', C_HH, 'Household')]):
    v = [row(t3, f'p_{cat}_{b}') for b in ['b1', 'b2', 'b3', 'b4', 'b5']]
    vals = np.array([a for a, _, _ in v]); lo = np.array([b for _, b, _ in v]); hi = np.array([c for _, _, c in v])
    xs = x + (i - 0.5) * w_
    ax.bar(xs, vals, w_ - 0.04, color=col, label=name, edgecolor='white', linewidth=2)
    ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt='none', ecolor=INK, elinewidth=1, capsize=3)
    for xv, val, h, n in zip(xs, vals, hi, n3[cat].values):
        ax.text(xv, h + 5, f'{val:.0f}%', ha='center', fontsize=8, color=INK)
        ax.text(xv, -14, f'n={n}', ha='center', fontsize=6.8, color=INK2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x, BIN_LABELS); ax.tick_params(axis='x', pad=14); ax.set_xlabel('Mức chiết khấu (depth)')
ax.set_ylabel('% lift so với đợt giả cùng ngành hàng')
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper left')
ax.set_title('Discount depth vs Sales lift – có đối chứng đợt giả (E3, KTC 95% HC3)', loc='left',
             fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_reg_ep_depth_bins.png'); plt.close(fig)

# Hình 2: hệ số E1 (% thay đổi lift)
items = [('depth10', '+10đ% chiết khấu – Food', C_FOOD),
         ('depth10 (household = depth10 + depth10:household)', '+10đ% chiết khấu – Household', C_HH),
         ('household', 'Household vs Food (tại depth = 0)', C_HH),
         ("C(store, Treatment('1'))[T.2]", 'Store 2 vs Store 1', NEU),
         ("C(store, Treatment('1'))[T.3]", 'Store 3 vs Store 1', NEU),
         ("C(store, Treatment('1'))[T.4]", 'Store 4 vs Store 1', NEU),
         ("C(ptype, Treatment('5'))[T.6]", 'Promo type 6 vs 5', NEU),
         ("C(ptype, Treatment('5'))[T.other]", 'Promo type khác vs 5', NEU),
         ('log_dur', 'Độ dài đợt ×2.7 (ln duration +1)', NEU),
         ("C(hol, Treatment('normal'))[T.holiday]", 'Đợt chứa ngày lễ', NEU),
         ("C(hol, Treatment('normal'))[T.pre_holiday]", 'Đợt trước lễ', NEU),
         ("C(hol, Treatment('normal'))[T.post_holiday]", 'Đợt sau lễ', NEU)]
fig, ax = plt.subplots(figsize=(8.4, 5.6))
yy = np.arange(len(items))[::-1]
vals = []; los = []; his = []
for term, _, _ in items:
    a, b, c = row(t1, term); vals.append(a); los.append(b); his.append(c)
vals, los, his = map(np.array, (vals, los, his))
ax.barh(yy, vals, color=[c for _, _, c in items], height=0.62, edgecolor='white', linewidth=2)
ax.errorbar(vals, yy, xerr=[vals - los, his - vals], fmt='none', ecolor=INK, elinewidth=1, capsize=3)
span = his.max() - min(los.min(), 0)
for v_, y_, h in zip(vals, yy, his):
    ax.text(max(h, 0) + 0.02 * span, y_, f'{v_:+.1f}%', va='center', fontsize=8.5, color=INK)
ax.set_xlim(min(los.min(), 0) - 0.03 * span, his.max() + 0.2 * span)
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(yy, [l for _, l, _ in items])
ax.set_xlabel('% thay đổi của lift (KTC 95%, HC3)')
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('Các yếu tố liên quan đến Sales Lift (E1, đợt KM thật)', loc='left',
             fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_reg_ep_coefs.png'); plt.close(fig)
print('\nĐã lưu bảng và hình vào', OUT)
