"""Inferential statistics – kiểm định 3 khác biệt mà phần Descriptive (Hoàng) đã chỉ ra.

Mẫu: giống hệt phần Descriptive – đợt trong data/episodes.csv có bán đủ cả 3 ngày trước KM (pre_days = 3):
  3,564 đợt KM thật (Food 3,311, Household 253) + 2,740 đợt giả (Food 2,054, Household 686).
Thang đo: y_k = ln(q_k + 1/3), log ratio ℓ_k = y_k − y_pre (đúng định nghĩa log_lift / log_post13 / log_post47).

  T1  Trước vs trong KM          → Paired t-test (+ Wilcoxon); KM thật so với đợt giả: Welch t-test
  T2  Khác biệt giữa các phase    → One-way repeated-measures ANOVA (Greenhouse–Geisser) + post-hoc Holm + Friedman
  T3  Phase × Category            → Two-way mixed ANOVA (phase trong đợt × category giữa các đợt)
                                    + Two-way ANOVA category × source (KM thật/đợt giả) trên log_post, HC3
Hiệu chỉnh nhiều kiểm định: Benjamini–Hochberg (BH) trong mỗi họ kiểm định. Robustness: chạy lại trên toàn bộ đợt.

Kết quả: outputs/inf_*.csv, outputs/fig_inf_*.png, outputs/inferential_log.txt
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EP_PATH = HERE.parent / 'data' / 'episodes.csv'
OUT = HERE / 'outputs'; OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
C_SMOOTH = 1 / 3   # hằng số làm trơn của pipeline (không đặt tên C: trùng hàm C() của patsy)
PHASES = ['pre', 'during', 'post13', 'post47']
PH_LABEL = {'pre': '1–3 days before', 'during': 'During', 'post13': '1–3 days after', 'post47': '4–7 days after'}
C_FOOD, C_HH, INK, INK2, GRID, NEU = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#8a8986'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})
pct = lambda x: 100 * np.expm1(x)          # log ratio → % thay đổi (trung bình nhân)

ep_all = pd.read_csv(EP_PATH)
for k, col in [('pre', 'pre_qty'), ('during', 'during_qty'), ('post13', 'post13_qty'), ('post47', 'post47_qty')]:
    ep_all[f'y_{k}'] = np.log(ep_all[col] + C_SMOOTH)
for k in PHASES[1:]:
    ep_all[f'l_{k}'] = ep_all[f'y_{k}'] - ep_all['y_pre']
ep_all['cat'] = ep_all['category'].str.capitalize()


def mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return m, m - q * se, m + q * se


# ======================================================================
# T1 – Paired t-test: trước vs trong KM
def paired_tests(ep):
    rows = []
    for src in ['promo', 'placebo']:
        for cat in ['All', 'Food', 'Household']:
            d = ep[(ep.source == src) & ((ep.cat == cat) | (cat == 'All'))]
            diff = d['l_during'].to_numpy()
            t, p = stats.ttest_rel(d['y_during'], d['y_pre'])
            m, lo, hi = mean_ci(diff)
            w = stats.wilcoxon(diff[diff != 0])
            rows.append({'source': src, 'category': cat, 'n': len(d),
                         'mean_y_pre': d['y_pre'].mean(), 'mean_y_during': d['y_during'].mean(),
                         'mean_diff': m, 'ci_low': lo, 'ci_high': hi, 'sd_diff': diff.std(ddof=1),
                         't': t, 'df': len(d) - 1, 'p': p, 'cohen_dz': m / diff.std(ddof=1),
                         'pct_change': pct(m), 'pct_ci_low': pct(lo), 'pct_ci_high': pct(hi),
                         'share_up': (diff > 0).mean(), 'wilcoxon_p': w.pvalue})
    t1 = pd.DataFrame(rows)
    t1['p_BH'] = multipletests(t1['p'], method='fdr_bh')[1]
    return t1

def welch_promo_vs_placebo(ep, col):
    rows = []
    for cat in ['All', 'Food', 'Household']:
        d = ep[(ep.cat == cat) | (cat == 'All')]
        a = d.loc[d.source == 'promo', col].to_numpy(); b = d.loc[d.source == 'placebo', col].to_numpy()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        dfw = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        q = stats.t.ppf(0.975, dfw)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        g = diff / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))         # Hedges' g
        rows.append({'outcome': col, 'category': cat, 'n_promo': len(a), 'n_placebo': len(b),
                     'mean_promo': a.mean(), 'mean_placebo': b.mean(), 'diff': diff,
                     'ci_low': diff - q * se, 'ci_high': diff + q * se, 't': t, 'df': dfw, 'p': p,
                     'hedges_g': g, 'pct_diff': pct(diff), 'mannwhitney_p': stats.mannwhitneyu(a, b).pvalue})
    return pd.DataFrame(rows)


# ======================================================================
# T2 – One-way repeated-measures ANOVA (4 phase, chủ thể = đợt)
def gg_epsilon(Y):
    """Greenhouse–Geisser ε và Mauchly W từ ma trận n × k (đã bỏ hiệu ứng nhóm nếu có)."""
    n, k = Y.shape
    S = np.cov(Y, rowvar=False)
    # ma trận tương phản trực chuẩn (k−1) × k
    Cm = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:].T
    Sc = Cm @ S @ Cm.T
    eps = np.trace(Sc) ** 2 / ((k - 1) * np.trace(Sc @ Sc))
    W = np.linalg.det(Sc) / (np.trace(Sc) / (k - 1)) ** (k - 1)
    d = 1 - (2 * (k - 1) ** 2 + (k - 1) + 2) / (6 * (k - 1) * (n - 1))
    chi2 = -(n - 1) * d * np.log(W); dfm = (k - 1) * k / 2 - 1
    return eps, W, chi2, dfm, stats.chi2.sf(chi2, dfm)

def rm_anova(ep_src):
    Y = ep_src[[f'y_{k}' for k in PHASES]].to_numpy()
    n, k = Y.shape
    gm = Y.mean()
    ss_phase = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_tot = ((Y - gm) ** 2).sum()
    ss_err = ss_tot - ss_phase - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_phase / df1) / (ss_err / df2)
    eps, W, chi2, dfm, p_m = gg_epsilon(Y)
    fr = stats.friedmanchisquare(*[Y[:, j] for j in range(k)])
    return {'n': n, 'SS_phase': ss_phase, 'SS_subject': ss_subj, 'SS_error': ss_err, 'df1': df1, 'df2': df2,
            'F': F, 'p': stats.f.sf(F, df1, df2), 'GG_epsilon': eps, 'p_GG': stats.f.sf(F, df1 * eps, df2 * eps),
            'partial_eta2': ss_phase / (ss_phase + ss_err),
            'generalized_eta2': ss_phase / (ss_phase + ss_subj + ss_err),
            'mauchly_W': W, 'mauchly_chi2': chi2, 'mauchly_p': p_m,
            'friedman_chi2': fr.statistic, 'friedman_p': fr.pvalue}

def posthoc(ep_src, label):
    rows = []
    for i, a in enumerate(PHASES):
        for b in PHASES[i + 1:]:
            d = (ep_src[f'y_{b}'] - ep_src[f'y_{a}']).to_numpy()
            t, p = stats.ttest_1samp(d, 0)
            m, lo, hi = mean_ci(d)
            rows.append({'sample': label, 'comparison': f'{PH_LABEL[b]} vs {PH_LABEL[a]}', 'mean_diff': m,
                         'ci_low': lo, 'ci_high': hi, 'pct_change': pct(m), 't': t, 'p': p, 'cohen_dz': m / d.std(ddof=1)})
    r = pd.DataFrame(rows); r['p_Holm'] = multipletests(r['p'], method='holm')[1]
    return r


# ======================================================================
# T3 – Two-way mixed ANOVA: Phase (trong) × Category (giữa)
def mixed_anova(ep_src):
    Y = ep_src[[f'y_{k}' for k in PHASES]].to_numpy()
    g = (ep_src['cat'] == 'Household').to_numpy().astype(int)
    n, k = Y.shape; G = 2
    m_i = Y.mean(1); gm = m_i.mean()
    # Giữa các đợt
    ss_cat = k * sum((g == j).sum() * (m_i[g == j].mean() - gm) ** 2 for j in range(G))
    ss_sg = k * sum(((m_i[g == j] - m_i[g == j].mean()) ** 2).sum() for j in range(G))
    df_cat, df_sg = G - 1, n - G
    # Trong đợt: hồi quy y đã trừ trung bình đợt lên phase (effect coding) và phase × category (effect coding)
    Yc = (Y - m_i[:, None]).ravel()
    ph = np.tile(np.arange(k), n); gg = np.repeat(np.where(g == 1, 1.0, -1.0), k)
    P = np.zeros((n * k, k - 1))
    for j in range(k - 1):
        P[:, j] = (ph == j).astype(float) - (ph == k - 1).astype(float)
    I = P * gg[:, None]
    def sse(X):
        b = np.linalg.lstsq(X, Yc, rcond=None)[0]; e = Yc - X @ b
        return e @ e
    sse_full = sse(np.hstack([P, I]))
    ss_phase = sse(I) - sse_full
    ss_int = sse(P) - sse_full
    df_ph, df_err = k - 1, (n - G) * (k - 1)
    ms_err = sse_full / df_err
    # ε từ hiệp phương sai gộp trong nhóm
    Yw = np.vstack([Y[g == j] - Y[g == j].mean(0) for j in range(G)])
    eps, W, chi2, dfm, p_m = gg_epsilon(Yw)
    rows = []
    for name, ss, df1, df2, ms_e, within in [('Category', ss_cat, df_cat, df_sg, ss_sg / df_sg, False),
                                             ('Phase', ss_phase, df_ph, df_err, ms_err, True),
                                             ('Phase × Category', ss_int, df_ph, df_err, ms_err, True)]:
        F = (ss / df1) / ms_e
        e = eps if within else 1.0
        rows.append({'effect': name, 'SS': ss, 'df1': df1, 'df2': df2, 'F': F, 'p': stats.f.sf(F, df1, df2),
                     'GG_epsilon': e if within else np.nan, 'p_GG': stats.f.sf(F, df1 * e, df2 * e),
                     'partial_eta2': ss / (ss + ms_e * df2)})
    res = pd.DataFrame(rows)
    res.attrs.update({'n': n, 'mauchly_W': W, 'mauchly_p': p_m,
                      'levene_p': stats.levene(m_i[g == 0], m_i[g == 1]).pvalue})
    return res

def source_anova(ep, outcome):
    """Two-way ANOVA category × source trên log_post, sai số chuẩn HC3 (thiết kế của pipeline)."""
    m = smf.ols(f'{outcome} ~ C(cat, Sum) * C(source, Sum)', ep).fit()
    a = anova_lm(m, typ=3, robust='hc3')
    a = a.drop(index='Intercept').reset_index().rename(columns={'index': 'effect', 'PR(>F)': 'p'})
    a['effect'] = a['effect'].map({'C(cat, Sum)': 'Category', 'C(source, Sum)': 'Source (promo vs placebo)',
                                   'C(cat, Sum):C(source, Sum)': 'Category × Source', 'Residual': 'Residual'})
    ss_res = float(a.loc[a.effect == 'Residual', 'sum_sq'].iloc[0])
    a['partial_eta2'] = a['sum_sq'] / (a['sum_sq'] + ss_res)
    a.loc[a.effect == 'Residual', 'partial_eta2'] = np.nan
    a.insert(0, 'outcome', outcome)
    return a


# ======================================================================
def run(ep, tag):
    t1 = paired_tests(ep)
    did = pd.concat([welch_promo_vs_placebo(ep, c) for c in ['l_during', 'l_post13', 'l_post47']], ignore_index=True)
    did['p_BH'] = multipletests(did['p'], method='fdr_bh')[1]
    rm = pd.DataFrame([{'sample': f'{s} – {c}', **rm_anova(ep[(ep.source == s) & ((ep.cat == c) | (c == 'All'))])}
                       for s in ['promo', 'placebo'] for c in ['All', 'Food', 'Household']])
    ph = pd.concat([posthoc(ep[ep.source == 'promo'], 'promo – All'),
                    posthoc(ep[ep.source == 'placebo'], 'placebo – All')], ignore_index=True)
    mx = []
    for s in ['promo', 'placebo']:
        r = mixed_anova(ep[ep.source == s].reset_index(drop=True)); r.insert(0, 'sample', s)
        r['n'] = r.attrs['n']; r['mauchly_W'] = r.attrs['mauchly_W']; r['mauchly_p'] = r.attrs['mauchly_p']
        r['levene_p_subject_means'] = r.attrs['levene_p']; mx.append(r)
    mx = pd.concat(mx, ignore_index=True)
    sa = pd.concat([source_anova(ep, o) for o in ['l_post13', 'l_post47']], ignore_index=True)
    for d in (t1, did, rm, ph, mx, sa):
        d.insert(0, 'data', tag)
    return t1, did, rm, ph, mx, sa

ep_main = ep_all[ep_all.pre_days == 3].reset_index(drop=True)
print('Mẫu chính (pre_days = 3):'); print(ep_main.groupby(['source', 'cat']).size())
main = run(ep_main, 'main: sold on all 3 pre days')
rob = run(ep_all, 'robustness: all episodes')
names = ['inf_paired_ttest', 'inf_promo_vs_placebo', 'inf_oneway_rm_anova', 'inf_oneway_posthoc',
         'inf_twoway_mixed_anova', 'inf_twoway_category_source']
for nm, a, b in zip(names, main, rob):
    pd.concat([a, b], ignore_index=True).to_csv(OUT / f'{nm}.csv', index=False)
    print(f'\n===== {nm} (mẫu chính) ====='); print(a.drop(columns='data').round(4).to_string(index=False))
for nm, b in zip(names, rob):
    print(f'\n===== {nm} (robustness: toàn bộ đợt) ====='); print(b.drop(columns='data').round(4).to_string(index=False))

# Kiểm tra giả định: phân phối của các log ratio (mẫu chính)
ass = []
for s in ['promo', 'placebo']:
    for c in ['Food', 'Household']:
        d = ep_main[(ep_main.source == s) & (ep_main.cat == c)]
        for col in ['l_during', 'l_post13', 'l_post47']:
            x = d[col]
            ass.append({'source': s, 'category': c, 'variable': col, 'n': len(x), 'skew': stats.skew(x),
                        'excess_kurtosis': stats.kurtosis(x), 'dagostino_p': stats.normaltest(x).pvalue,
                        'shapiro_p': stats.shapiro(x.sample(min(len(x), 5000), random_state=42)).pvalue})
ass = pd.DataFrame(ass); ass.to_csv(OUT / 'inf_assumptions.csv', index=False)
print('\n===== Kiểm tra phân phối ====='); print(ass.round(4).to_string(index=False))

# ======================================================================
# Biểu đồ
t1, did, rm, ph, mx, sa = main
gm_units = lambda y: np.exp(y) - C_SMOOTH          # trung bình trên thang log → đơn vị/ngày (trung bình nhân)
STY = {('promo', 'Food'): dict(color=C_FOOD, ls='-', marker='o'), ('promo', 'Household'): dict(color=C_HH, ls='-', marker='o'),
       ('placebo', 'Food'): dict(color=C_FOOD, ls=(0, (4, 3)), marker='s', alpha=0.55),
       ('placebo', 'Household'): dict(color=C_HH, ls=(0, (4, 3)), marker='s', alpha=0.55)}
NAME = {'promo': 'promotion', 'placebo': 'placebo'}

# Hình T1: paired plot (trước → trong) + mean difference plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axes[0]
for (s, c), st in STY.items():
    d = ep_main[(ep_main.source == s) & (ep_main.cat == c)]
    ms = [mean_ci(d[f'y_{k}']) for k in ['pre', 'during']]
    v = [gm_units(m[0]) for m in ms]
    ax.plot([0, 1], v, lw=2, ms=7, markeredgecolor='white', markeredgewidth=1.5, label=f'{c} – {NAME[s]}', **st)
    for x_, m in zip([0, 1], ms):
        ax.plot([x_, x_], [gm_units(m[1]), gm_units(m[2])], color=st['color'], lw=1.2, alpha=st.get('alpha', 1))
ax.set_yscale('log'); ax.set_xticks([0, 1], ['1–3 days before', 'During promotion']); ax.set_xlim(-0.3, 1.3)
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 3, 4, 6, 8, 10]))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel('Geometric mean units / day (log scale)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('(a) Paired means, before vs during', loc='left', fontsize=10.5, fontweight='bold', color=INK)
ax = axes[1]
rows = [(s, c) for s in ['promo', 'placebo'] for c in ['Food', 'Household']]
yy = np.arange(len(rows))[::-1]
for y_, (s, c) in zip(yy, rows):
    r = t1[(t1.source == s) & (t1.category == c)].iloc[0]
    st = STY[(s, c)]
    v, lo, hi = r['pct_change'], r['pct_ci_low'], r['pct_ci_high']    # r.pct_change trùng method của Series
    ax.errorbar(v, y_, xerr=[[v - lo], [hi - v]], fmt=st['marker'],
                color=st['color'], ms=8, capsize=3, alpha=st.get('alpha', 1), markeredgecolor='white')
    ax.text(hi + 3, y_, f"{v:+.0f}%  (t = {r['t']:.1f}, p {'< 0.001' if r['p_BH'] < 0.001 else f'= {r.p_BH:.3f}'})",
            va='center', fontsize=8.3, color=INK)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, [f'{c} – {NAME[s]}' for s, c in rows])
xl = ax.get_xlim(); ax.set_xlim(xl[0], xl[1] + 70)
ax.set_xlabel('Change during vs before, % (geometric mean, 95% CI)'); ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title('(b) Mean difference, paired t-test', loc='left', fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_inf_paired.png'); plt.close(fig)

# Hình T2: boxplot theo phase + mean plot 95% CI
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
pr = ep_main[ep_main.source == 'promo']
data = [pr[f'y_{k}'] for k in PHASES]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, showmeans=True,
                medianprops=dict(color=INK, lw=1.5), meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, ms=6),
                whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for b in bp['boxes']:
    b.set_facecolor('#c9c8c3'); b.set_edgecolor(INK2)
ax.set_xticks(range(1, 5), [PH_LABEL[k] for k in PHASES], fontsize=8.5)
ax.set_ylabel('ln(units/day + 1/3)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('(a) Boxplot by phase, promotion episodes', loc='left', fontsize=10.5, fontweight='bold', color=INK)
ax = axes[1]
x = np.arange(4)
for s, col, name, ls, mk in [('promo', INK, 'Promotion episodes', '-', 'o'), ('placebo', NEU, 'Placebo episodes', (0, (4, 3)), 's')]:
    d = ep_main[ep_main.source == s]
    ms = [(0.0, 0.0, 0.0)] + [mean_ci(d[f'l_{k}']) for k in PHASES[1:]]
    v = np.array([pct(m[0]) for m in ms]); lo = np.array([pct(m[1]) for m in ms]); hi = np.array([pct(m[2]) for m in ms])
    ax.errorbar(x, v, yerr=[v - lo, hi - v], color=col, ls=ls, marker=mk, ms=7, lw=2, capsize=3, label=name,
                markeredgecolor='white')
    dy = 4 if s == 'promo' else -4                       # promo: nhãn phía trên, placebo: phía dưới
    for xv, val in zip(x[1:], v[1:]):
        ax.text(xv, val + dy, f'{val:+.0f}%', fontsize=8, color=col, ha='center', va='bottom' if dy > 0 else 'top')
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x, [PH_LABEL[k] for k in PHASES], fontsize=8.5)
ax.set_ylabel('Change vs 1–3 days before, % (95% CI)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title('(b) Mean by phase with 95% CI', loc='left', fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_inf_oneway.png'); plt.close(fig)

# Hình T3: interaction plot Phase × Category + hiệu ứng đã trừ đợt giả
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for (s, c), st in STY.items():
    d = ep_main[(ep_main.source == s) & (ep_main.cat == c)]
    ms = [(0.0, 0.0, 0.0)] + [mean_ci(d[f'l_{k}']) for k in PHASES[1:]]
    v = np.array([pct(m[0]) for m in ms]); lo = np.array([pct(m[1]) for m in ms]); hi = np.array([pct(m[2]) for m in ms])
    off = {'Food': -0.05, 'Household': 0.05}[c]
    ax.errorbar(x + off, v, yerr=[v - lo, hi - v], lw=2, ms=6, capsize=2.5, label=f'{c} – {NAME[s]}',
                markeredgecolor='white', **st)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x, [PH_LABEL[k] for k in PHASES], fontsize=8.5)
ax.set_ylabel('Change vs 1–3 days before, % (95% CI)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.set_title('(a) Interaction plot: Phase × Category', loc='left', fontsize=10.5, fontweight='bold', color=INK)
ax = axes[1]
w = 0.36
for i, (c, col) in enumerate([('Food', C_FOOD), ('Household', C_HH)]):
    r = did[did.category == c].set_index('outcome').loc[['l_during', 'l_post13', 'l_post47']]
    v = pct(r['diff'].to_numpy()); lo = pct(r['ci_low'].to_numpy()); hi = pct(r['ci_high'].to_numpy())
    xs = np.arange(3) + (i - 0.5) * w
    ax.bar(xs, v, w - 0.04, color=col, label=c, edgecolor='white', linewidth=2)
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt='none', ecolor=INK, capsize=3, elinewidth=1)
    for xv, val, h, l in zip(xs, v, hi, lo):
        ax.text(xv, (h + 4) if val >= 0 else (l - 4), f'{val:+.0f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=8, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(3), ['During', '1–3 days after', '4–7 days after'])
ax.set_ylabel('Promotion minus placebo, % (95% CI)'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
yl = ax.get_ylim(); ax.set_ylim(yl[0] - 8, yl[1] + 8)
ax.legend(frameon=False, fontsize=8.5, loc='upper right')
ax.set_title('(b) Effect net of placebo (Welch)', loc='left', fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_inf_interaction.png'); plt.close(fig)
print('\nĐã lưu bảng và hình vào', OUT)
