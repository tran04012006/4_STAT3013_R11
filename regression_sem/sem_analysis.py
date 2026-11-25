"""SEM – Chiết khấu, Sales Lift và Post-Promotion Dip liên hệ với nhau qua cơ chế nào?

Làm theo EDA và pipeline chung của nhóm (bao_cao_pipeline_tien_xu_ly.md):
  - Dữ liệu: data/episodes.csv – 10,044 đợt KM thật + 9,700 đợt giả (placebo, không KM, dài 14 ngày).
  - Cửa sổ: pre = 1–3 ngày trước, during, post 1–3 ngày, post 4–7 ngày sau đợt; hằng số c = 1/3.
  - log_lift = ln((during + c)/(pre + c)), log_post13 / log_post47 = ln((post + c)/(pre + c)).
  - "Luôn so KM thật với đợt giả" (EDA trang 7, pipeline mục 7) và kiểm soát store, promo_type, lễ (EDA trang 3, 9, 12).

Biến trong SEM (mỗi đợt):
  DEPTH   mức chiết khấu, đơn vị 10 điểm %                       (chỉ có ở đợt KM thật)
  LIFT    log_lift                                               (biến quan sát)
  DIP     biến ẩn, đo bằng dip13 = −log_post13 và dip47 = −log_post47  (DIP > 0: sau KM bán thấp hơn trước KM)
  Kiểm soát: FOOD, DUR = ln(duration), BASE = ln(pre_qty + 1/3), cửa hàng (S2, S3, S4), promo_type (PT6, PTO), HOL

Vì sao phải kiểm soát BASE: LIFT = ln(during+c) − BASE và DIP = BASE − ln(post+c) dùng CHUNG cửa sổ pre 3 ngày,
nên nhiễu của pre tạo ra tương quan âm giả giữa LIFT và DIP. Đưa BASE vào cả phương trình LIFT và DIP triệt tiêu
đúng thành phần chung đó (kiểm tra ở Robustness R1: bỏ BASE).

Mô hình:
  DIP   =~ dip13 + dip47
  DEPTH ~ FOOD + DUR + BASE + ctrl
  LIFT  ~ a·DEPTH + FOOD + DUR + BASE + ctrl
  DIP   ~ b·LIFT + c'·DEPTH + FOOD + DUR + BASE + ctrl
  Gián tiếp a·b, tổng c' + a·b. Đa nhóm Food / Household.
  Đối chứng đợt giả: cùng mô hình không có DEPTH → b_placebo = mức LIFT→DIP "tự nhiên" khi không KM
  (cú sốc nhu cầu chung). b_promo − b_placebo = phần gắn với KM.
KTC 95%: cluster bootstrap theo item_id, mỗi lần lấy mẫu lại item một lần cho TẤT CẢ mô hình (so sánh giữa nhóm đúng).

Kết quả: outputs/sem_*.csv, outputs/fig_sem_*.png, outputs/sem_log.txt
"""
import os
from pathlib import Path
import warnings
import numpy as np, pandas as pd
import semopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
EP_PATH = HERE.parent / 'data' / 'episodes.csv'
OUT = HERE / 'outputs'; OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 200)
SEED = 42
N_BOOT = int(os.environ.get('N_BOOT', 500))   # số lần bootstrap (đặt N_BOOT=5 để chạy thử nhanh)
C = 1 / 3                                      # hằng số làm trơn của pipeline

C_FOOD, C_HH, INK, INK2, GRID, NEU = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#8a8986'
C_PLACEBO = '#a9a8a3'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})

# ======================================================================
# 1. Dữ liệu
ep = pd.read_csv(EP_PATH)
ep['DEPTH'] = ep['depth'] * 10
ep['LIFT'] = ep['log_lift']
ep['dip13'] = -ep['log_post13']
ep['dip47'] = -ep['log_post47']
ep['FOOD'] = (ep['category'] == 'food').astype(float)
ep['DUR'] = np.log(ep['duration'])
ep['BASE'] = np.log(ep['pre_qty'] + C)
for s in (2, 3, 4):
    ep[f'S{s}'] = (ep['store_id'] == s).astype(float)
ep['PT6'] = (ep['promo_type'] == 6).astype(float)
ep['PTO'] = (ep['source'].eq('promo') & ~ep['promo_type'].isin([5, 6])).astype(float)
ep['HOL'] = (ep['hol_status'] != 'normal').astype(float)
ep = ep.reset_index(drop=True)

VARS = ['DEPTH', 'LIFT', 'dip13', 'dip47', 'FOOD', 'DUR', 'BASE']
desc = ep.groupby(['source', 'category'])[VARS + ['pre_qty', 'during_qty', 'post13_qty', 'post47_qty', 'duration']] \
         .agg(['mean', 'std', 'median']).T
desc.to_csv(OUT / 'sem_descriptive.csv')
print('Số đợt:'); print(ep.groupby(['source', 'category']).size())
print('\nTương quan (đợt KM thật):'); print(ep[ep.source == 'promo'][VARS].corr().round(3))
print('\nTương quan (đợt giả):'); print(ep[ep.source == 'placebo'][['LIFT', 'dip13', 'dip47', 'FOOD', 'BASE']].corr().round(3))

# ======================================================================
# 2. Đặc tả mô hình
def spec(promo=True, food=True, base=True):
    ctrl = (['FOOD'] if food else []) + (['DUR'] if promo else []) + (['BASE'] if base else []) \
           + ['S2', 'S3', 'S4', 'HOL'] + (['PT6', 'PTO'] if promo else [])
    c = ' + '.join(ctrl)
    s = 'DIP =~ dip13 + dip47\n'
    if promo:
        s += f'DEPTH ~ {c}\nLIFT ~ DEPTH + {c}\nDIP ~ LIFT + DEPTH + {c}\n'
    else:
        s += f'LIFT ~ {c}\nDIP ~ LIFT + {c}\n'
    return s

RUNS = {  # tên: (điều kiện lọc, đặc tả)
    'Promo – gộp':      (lambda d: d.source == 'promo', spec(True, True)),
    'Promo – Food':     (lambda d: (d.source == 'promo') & (d.FOOD == 1), spec(True, False)),
    'Promo – Household': (lambda d: (d.source == 'promo') & (d.FOOD == 0), spec(True, False)),
    'Placebo – gộp':    (lambda d: d.source == 'placebo', spec(False, True)),
    'Placebo – Food':   (lambda d: (d.source == 'placebo') & (d.FOOD == 1), spec(False, False)),
    'Placebo – Household': (lambda d: (d.source == 'placebo') & (d.FOOD == 0), spec(False, False)),
}

def fit(data, desc_):
    m = semopy.Model(desc_)
    m.fit(data, obj='MLW')
    return m

def get(t, l, r, col='Estimate'):
    x = t[(t.lval == l) & (t.op == '~') & (t.rval == r)]
    return float(x[col].iloc[0]) if len(x) else np.nan

def key_paths(m):
    t = m.inspect(std_est=True)
    out = {}
    for suffix, col in [('', 'Estimate'), ('std_', 'Est. Std')]:
        a, b, c = get(t, 'LIFT', 'DEPTH', col), get(t, 'DIP', 'LIFT', col), get(t, 'DIP', 'DEPTH', col)
        out.update({f'{suffix}a': a, f'{suffix}b': b, f'{suffix}c_direct': c,
                    f'{suffix}indirect_ab': a * b, f'{suffix}total': c + a * b})
    return out

STAT_COLS = ['DoF', 'chi2', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'GFI', 'AGFI', 'AIC', 'BIC']
def fit_stats(m, label, n):
    s = semopy.calc_stats(m).T.iloc[:, 0]
    return {'model': label, 'N': n, **{c: float(s[c]) for c in STAT_COLS if c in s.index}}

# ======================================================================
# 3. Ước lượng trên mẫu đầy đủ
params, stats_rows, point = [], [], {}
for label, (cond, desc_) in RUNS.items():
    data = ep[cond(ep)].reset_index(drop=True)
    m = fit(data, desc_)
    t = m.inspect(std_est=True); t.insert(0, 'model', label); params.append(t)
    stats_rows.append(fit_stats(m, label, len(data)))
    point[label] = key_paths(m)
    print(f'\n===== {label} (N = {len(data)}) =====')
    print(t[t.op == '~'][['lval', 'op', 'rval', 'Estimate', 'Est. Std', 'Std. Err', 'p-value']].round(4).to_string(index=False))

# Robustness (không bootstrap)
rob = {'R1: bỏ kiểm soát BASE (baseline chung → tương quan giả)': (lambda d: d.source == 'promo', spec(True, True, base=False)),
       'R1: bỏ kiểm soát BASE – Placebo': (lambda d: d.source == 'placebo', spec(False, True, base=False))}
for label, (cond, desc_) in rob.items():
    data = ep[cond(ep)].reset_index(drop=True)
    m = fit(data, desc_)
    t = m.inspect(std_est=True); t.insert(0, 'model', label); params.append(t)
    stats_rows.append(fit_stats(m, label, len(data))); point[label] = key_paths(m)
# R2: path analysis với DIP quan sát (TB 2 chỉ báo), không có biến ẩn
ep['DIP_obs'] = ep[['dip13', 'dip47']].mean(axis=1)
ctrl = 'FOOD + DUR + BASE + S2 + S3 + S4 + HOL + PT6 + PTO'
m = fit(ep[ep.source == 'promo'].reset_index(drop=True),
        f'DEPTH ~ {ctrl}\nLIFT ~ DEPTH + {ctrl}\nDIP_obs ~ LIFT + DEPTH + {ctrl}\n')
t = m.inspect(std_est=True); t.insert(0, 'model', 'R2: DIP quan sát (không biến ẩn)'); params.append(t)
pr2 = {}
for suffix, col in [('', 'Estimate'), ('std_', 'Est. Std')]:
    a, b, c = get(t, 'LIFT', 'DEPTH', col), get(t, 'DIP_obs', 'LIFT', col), get(t, 'DIP_obs', 'DEPTH', col)
    pr2.update({f'{suffix}a': a, f'{suffix}b': b, f'{suffix}c_direct': c, f'{suffix}indirect_ab': a * b,
                f'{suffix}total': c + a * b})
point['R2: DIP quan sát (không biến ẩn)'] = pr2

# ======================================================================
# 4. Cluster bootstrap theo item (lấy mẫu chung cho mọi mô hình)
rng = np.random.default_rng(SEED)
groups = {k: v.to_numpy() for k, v in ep.groupby('item_id').groups.items()}
keys = np.array(list(groups))
boot_rows = []
for r in range(N_BOOT):
    idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
    bs = ep.loc[idx].reset_index(drop=True)
    row = {'rep': r}
    for label, (cond, desc_) in RUNS.items():
        try:
            kp = key_paths(fit(bs[cond(bs)].reset_index(drop=True), desc_))
            row.update({f'{label}|{k}': v for k, v in kp.items()})
        except Exception:
            pass
    boot_rows.append(row)
    if (r + 1) % 50 == 0:
        print(f'bootstrap {r + 1}/{N_BOOT}', flush=True)
boot = pd.DataFrame(boot_rows)

def ci(series):
    s = series.dropna()
    return s.std(), s.quantile(0.025), s.quantile(0.975), min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean())), len(s)

EFFECTS = ['a', 'b', 'c_direct', 'indirect_ab', 'total']
rows = []
for label in point:
    for k in EFFECTS:
        est, std = point[label][k], point[label][f'std_{k}']
        if not np.isfinite(est):
            continue
        rec = {'model': label, 'effect': k, 'estimate': est, 'std_estimate': std}
        if f'{label}|{k}' in boot:
            se, lo, hi, p, n = ci(boot[f'{label}|{k}'])
            _, slo, shi, _, _ = ci(boot[f'{label}|std_{k}'])
            rec.update({'boot_se': se, 'ci_low': lo, 'ci_high': hi, 'std_ci_low': slo, 'std_ci_high': shi,
                        'p_boot': p, 'n_boot': n})
        rows.append(rec)
res = pd.DataFrame(rows)

# So sánh: Food − Household (KM thật) và KM thật − đợt giả (đường b)
cmp_rows = []
def add_cmp(name, l1, l2, k):
    d = boot[f'{l1}|{k}'] - boot[f'{l2}|{k}']
    se, lo, hi, p, n = ci(d)
    cmp_rows.append({'comparison': name, 'effect': k, 'first': point[l1][k], 'second': point[l2][k],
                     'diff': point[l1][k] - point[l2][k], 'ci_low': lo, 'ci_high': hi, 'p_boot': p, 'n_boot': n})
for k in EFFECTS:
    add_cmp('Food − Household (KM thật)', 'Promo – Food', 'Promo – Household', k)
for g in ['gộp', 'Food', 'Household']:
    add_cmp(f'KM thật − đợt giả ({g})', f'Promo – {g}', f'Placebo – {g}', 'b')
cmp = pd.DataFrame(cmp_rows)

res.to_csv(OUT / 'sem_effects.csv', index=False)
cmp.to_csv(OUT / 'sem_comparisons.csv', index=False)
P = pd.concat(params, ignore_index=True); P.to_csv(OUT / 'sem_parameters.csv', index=False)
fstats = pd.DataFrame(stats_rows); fstats.to_csv(OUT / 'sem_fit_stats.csv', index=False)
pd.DataFrame([{'source': s, 'category': c, 'n': n} for (s, c), n in ep.groupby(['source', 'category']).size().items()]) \
  .to_csv(OUT / 'sem_sample.csv', index=False)
print('\n===== Hiệu ứng (KTC 95% cluster bootstrap) ====='); print(res.round(4).to_string(index=False))
print('\n===== So sánh ====='); print(cmp.round(4).to_string(index=False))
print('\n===== Chỉ số phù hợp ====='); print(fstats.round(4).to_string(index=False))

# ======================================================================
# 5. SEM Path Diagram
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

CNAMES = {'FOOD': 'Category (Food = 1)', 'DUR': 'ln(độ dài KM)', 'BASE': 'ln(pre_qty + 1/3)'}

def draw_path_diagram(label, fname, title):
    t = P[P.model == label]
    def gs(l, r):
        row = t[(t.lval == l) & (t.op == '~') & (t.rval == r)].iloc[0]
        p = pd.to_numeric(row['p-value'], errors='coerce')
        return float(row['Est. Std']), (0.0 if not np.isfinite(p) else float(p))   # loading cố định = 1 không có p

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 7.4); ax.axis('off')
    Y, W, H, EW, EH = 4.55, 1.55, 0.72, 2.1, 1.2
    pos = {'dip13': (7.8, 6.35), 'dip47': (9.8, 6.35), 'DEPTH': (1.2, Y), 'LIFT': (4.6, Y), 'DIP': (8.8, Y)}
    names = {'dip13': '−log_post13\n(1–3 ngày sau)', 'dip47': '−log_post47\n(4–7 ngày sau)',
             'LIFT': 'SALES LIFT\n(log_lift)', 'DIP': 'POST-PROMO\nDIP', 'DEPTH': 'DISCOUNT\nDEPTH'}
    for k, (x, y) in pos.items():
        if k == 'DIP':
            ax.add_patch(Ellipse((x, y), EW, EH, fc='#f3f2ee', ec=INK, lw=1.6))
        else:
            w_ = 1.9 if k == 'LIFT' else W
            ax.add_patch(FancyBboxPatch((x - w_ / 2, y - H / 2), w_, H, boxstyle='round,pad=0.02,rounding_size=0.08',
                                        fc='white', ec=INK, lw=1.6 if k in ('DEPTH', 'LIFT') else 1.1))
        main = k in ('LIFT', 'DIP', 'DEPTH')
        ax.text(x, y, names[k], ha='center', va='center', fontsize=9.5 if main else 8.3,
                fontweight='bold' if main else 'normal', color=INK)

    def label_box(x, y, s, fs=9.5, bold=True):
        ax.text(x, y, s, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

    def arr(p0, p1, sig, main=True, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15 if main else 10,
                                     lw=2.2 if main else 1.0, color=INK if main else NEU,
                                     linestyle='-' if sig else (0, (4, 3)), connectionstyle=f'arc3,rad={rad}',
                                     shrinkA=0, shrinkB=1))

    for i in ['dip13', 'dip47']:                       # loading: "chỉ báo ~ DIP"
        v, p = gs(i, 'DIP')
        (x0, y0), (x1, y1) = pos['DIP'], pos[i]
        dx, dy = x1 - x0, y1 - y0; n = np.hypot(dx / (EW / 2), dy / (EH / 2))
        p0 = (x0 + dx / n, y0 + dy / n); p1 = (x1 - dx * (H / 2) / dy, y1 - H / 2)
        arr(p0, p1, p < 0.05, main=False)
        label_box((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, f'{v:.2f}{stars(p)}', fs=8.5, bold=False)

    a, pa = gs('LIFT', 'DEPTH'); b, pb = gs('DIP', 'LIFT'); c, pc = gs('DIP', 'DEPTH')
    arr((pos['DEPTH'][0] + W / 2, Y), (pos['LIFT'][0] - 0.95, Y), pa < 0.05)
    arr((pos['LIFT'][0] + 0.95, Y), (pos['DIP'][0] - EW / 2, Y), pb < 0.05)
    label_box(2.8, Y + 0.3, f'a = {a:.2f}{stars(pa)}')
    label_box(6.65, Y + 0.3, f'b = {b:.2f}{stars(pb)}')
    p0, p1, rad = (pos['DEPTH'][0] + 0.4, Y - H / 2), (pos['DIP'][0], Y - EH / 2), 0.25
    arr(p0, p1, pc < 0.05, rad=rad)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    label_box((p0[0] + p1[0]) / 2 + rad * dy / 2, (p0[1] + p1[1]) / 2 - rad * dx / 2, f"c' = {c:.2f}{stars(pc)}  (trực tiếp)")

    ctrls = [k for k in ('FOOD', 'DUR', 'BASE') if (t.rval == k).any()]
    cx, cy, cw, ch = 5.75, 1.95, 8.4, 0.62
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc='white', ec=NEU, lw=1.0))
    ax.text(cx, cy, 'Biến kiểm soát: ' + ' · '.join(CNAMES[k] for k in ctrls) + ' · cửa hàng · promo_type · lễ',
            ha='center', va='center', fontsize=8.3, color=INK2)
    for dst, xs in [('DEPTH', 0.85), ('LIFT', 4.6), ('DIP', 8.8)]:
        top = Y - (EH / 2 if dst == 'DIP' else H / 2)
        x_from = min(max(xs, cx - cw / 2 + 0.3), cx + cw / 2 - 0.3)
        arr((x_from, cy + ch / 2), (xs, top), True, main=False)

    rows = [f"{'':<20}{'→ Depth':>10}{'→ Lift':>10}{'→ Dip':>10}"]
    for k in ctrls:
        cells = []
        for dst in ('DEPTH', 'LIFT', 'DIP'):
            v, p = gs(dst, k); cells.append(f'{v:.2f}{stars(p)}')
        rows.append(f'{CNAMES[k]:<20}' + ''.join(f'{s:>10}' for s in cells))
    ax.text(0.15, 1.3, '\n'.join(rows), fontsize=7.8, color=INK2, va='top', family='DejaVu Sans Mono')

    r = res[res.model == label].set_index('effect')
    ind, tot = r.loc['indirect_ab'], r.loc['total']
    s = fstats[fstats.model == label].iloc[0]
    txt = (f"Gián tiếp a×b = {ind.std_estimate:.3f}  [{ind.std_ci_low:.3f}; {ind.std_ci_high:.3f}]\n"
           f"Trực tiếp c'  = {c:.3f}\n"
           f"Tổng            = {tot.std_estimate:.3f}  [{tot.std_ci_low:.3f}; {tot.std_ci_high:.3f}]\n"
           f"N = {int(s.N)} đợt · χ²({int(s.DoF)}) = {s.chi2:.1f} · CFI = {s.CFI:.3f} · RMSEA = {s.RMSEA:.3f}")
    ax.text(11.35, 1.3, txt, ha='right', va='top', multialignment='left', fontsize=8.3, color=INK,
            bbox=dict(boxstyle='round,pad=0.4', fc='#f3f2ee', ec='none'))
    ax.text(0.15, 7.3, title, fontsize=12, fontweight='bold', color=INK, va='top')
    ax.text(0.15, 0.05, f'Hệ số chuẩn hóa. *** p<0.001, ** p<0.01, * p<0.05; nét đứt = không có ý nghĩa thống kê. '
                        f'KTC 95% (trong ngoặc): cluster bootstrap theo item, {N_BOOT} lần.',
            fontsize=7.5, color=INK2, va='bottom')
    fig.savefig(OUT / fname, bbox_inches='tight'); plt.close(fig)

draw_path_diagram('Promo – gộp', 'fig_sem_path_diagram.png', 'SEM Path Diagram – Discount → Sales Lift → Post-Promotion Dip')
draw_path_diagram('Promo – Food', 'fig_sem_path_food.png', 'SEM Path Diagram – Food')
draw_path_diagram('Promo – Household', 'fig_sem_path_household.png', 'SEM Path Diagram – Household')

# So sánh đa nhóm Food vs Household (hệ số chưa chuẩn hóa)
def errbars(ax, v, yv, lo, hi):
    ax.errorbar(v, yv, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], fmt='none', ecolor=INK,
                elinewidth=1, capsize=2.5)   # KTC percentile có thể không chứa ước lượng điểm → clip

fig, ax = plt.subplots(figsize=(8, 4.2))
effects = [('a', 'a: Depth → Lift'), ('b', 'b: Lift → Dip'), ('c_direct', "c': Depth → Dip (trực tiếp)"),
           ('indirect_ab', 'a×b: Depth → Lift → Dip'), ('total', 'Tổng: Depth → Dip')]
yy = np.arange(len(effects))[::-1]; h = 0.36
for i, (grp, col) in enumerate([('Promo – Food', C_FOOD), ('Promo – Household', C_HH)]):
    r = res[res.model == grp].set_index('effect')
    v = np.array([r.loc[k, 'estimate'] for k, _ in effects])
    lo = np.array([r.loc[k, 'ci_low'] for k, _ in effects]); hi = np.array([r.loc[k, 'ci_high'] for k, _ in effects])
    yv = yy + (h / 2 if i == 0 else -h / 2)
    ax.barh(yv, v, h - 0.04, color=col, label=grp.split('– ')[1], edgecolor='white', linewidth=2)
    errbars(ax, v, yv, lo, hi)
    for val, y_, h_ in zip(v, yv, hi):
        ax.text(max(h_, 0) + 0.008, y_, f'{val:+.3f}', va='center', fontsize=8, color=INK)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, [l for _, l in effects]); ax.set_xlabel('Hệ số chưa chuẩn hóa (KTC 95% bootstrap)')
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
xl = ax.get_xlim(); ax.set_xlim(xl[0], xl[1] + 0.06)
ax.legend(frameon=False, loc='best')
ax.set_title('So sánh đa nhóm: Food vs Household (đợt KM thật)', loc='left', fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_sem_group_comparison.png'); plt.close(fig)

# Đường b: KM thật vs đợt giả
fig, ax = plt.subplots(figsize=(7.6, 3.4))
grps = ['gộp', 'Food', 'Household']; yy = np.arange(len(grps))[::-1]; h = 0.36
for i, (src, col, name) in enumerate([('Promo', INK2, 'KM thật'), ('Placebo', C_PLACEBO, 'Đợt giả (đối chứng)')]):
    r = res[(res.effect == 'b')].set_index('model')
    v = np.array([r.loc[f'{src} – {g}', 'estimate'] for g in grps])
    lo = np.array([r.loc[f'{src} – {g}', 'ci_low'] for g in grps]); hi = np.array([r.loc[f'{src} – {g}', 'ci_high'] for g in grps])
    yv = yy + (h / 2 if i == 0 else -h / 2)
    ax.barh(yv, v, h - 0.04, color=col, label=name, edgecolor='white', linewidth=2)
    errbars(ax, v, yv, lo, hi)
    for val, y_, l_ in zip(v, yv, lo):
        ax.text(min(l_, 0) - 0.01, y_, f'{val:+.3f}', va='center', ha='right', fontsize=8, color=INK)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, grps); ax.set_xlabel('b: Lift → Dip (chưa chuẩn hóa, KTC 95% bootstrap)')
xl = ax.get_xlim(); ax.set_xlim(xl[0] - 0.08, max(xl[1], 0.05))
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.5)
ax.set_title('Đường Lift → Dip: KM thật so với đợt giả', loc='left', fontsize=10.5, fontweight='bold', color=INK,
             pad=22)
fig.tight_layout(); fig.savefig(OUT / 'fig_sem_placebo_b.png'); plt.close(fig)
print('\nĐã lưu bảng và hình vào', OUT)
