"""SEM – Chiết khấu, Sales Lift và Post-Promotion Dip liên hệ với nhau qua cơ chế nào?

Đơn vị phân tích: một ĐỢT khuyến mãi = chuỗi ngày liên tiếp promo_status = 1 của một cặp item×store.

Biến (mỗi đợt):
  DEPTH       mức chiết khấu trung bình trong đợt, đơn vị 10 điểm % (DEPTH = 2.3 ↔ giảm 23%)
  FOOD        1 = food, 0 = household
  DUR         ln(số ngày của đợt)
  BASE        ln(baseline_A + c): mức bán nền (tốc độ bán) của cặp item×store trước đợt
  lift_early, lift_late  ln((lượng bán TB nửa đầu / nửa sau đợt + c) / (baseline_A + c))  → chỉ báo của LIFT
  dip_13, dip_47         ln((baseline_B + c) / (lượng bán TB ngày +1..+3 / +4..+7 sau đợt + c)) → chỉ báo của DIP
                         (DIP > 0: bán thấp hơn mức nền sau KM, tức có "dip")

Baseline: ngày thường (phase = 'other', promo_status = 0) trong 28 ngày trước đợt. Chia theo ngày chẵn/lẻ:
  baseline_A (offset chẵn) dùng cho LIFT, baseline_B (offset lẻ) dùng cho DIP. Mỗi nửa chứa đủ mỗi thứ trong tuần
  2 lần. Nếu dùng chung một baseline, nhiễu của baseline đi vào cả LIFT lẫn DIP và tạo ra tương quan LIFT → DIP
  giả (kiểm tra ở mục Robustness R1).

Mô hình cấu trúc (semopy, ML):
  Đo lường:  LIFT =~ lift_early + lift_late ;  DIP =~ dip_13 + dip_47
  Cấu trúc:  DEPTH ~ FOOD + DUR + BASE
             LIFT  ~ a·DEPTH + FOOD + DUR + BASE
             DIP   ~ b·LIFT + c'·DEPTH + FOOD + DUR + BASE
  Hiệu ứng gián tiếp DEPTH → LIFT → DIP = a·b; tổng = c' + a·b. KTC 95% bằng cluster bootstrap theo item_id.
  Đa nhóm: ước lượng riêng cho Food và Household (bỏ biến FOOD) và so sánh a, b, c', a·b.

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
DATA_PATH = HERE.parent / 'data' / 'daily_panel.parquet'
OUT = HERE / 'outputs'; OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 200)
SEED = 42
N_BOOT = int(os.environ.get('N_BOOT', 500))   # số lần bootstrap (đặt N_BOOT=5 để chạy thử nhanh)
C = 0.1                    # hằng số cộng thêm trong log-ratio (đơn vị lượng bán/ngày), tránh chia cho 0
BASE_DAYS, POST_DAYS = 28, 7
MIN_DUR, MAX_DUR = 4, 35   # đợt quá ngắn không chia được nửa đầu/nửa sau; đợt rất dài (tới 604 ngày) là giá cố định
MIN_BASE_DAYS = 6          # số ngày thường tối thiểu trong MỖI nửa baseline (tối đa 14)

C_FOOD, C_HH, INK, INK2, GRID, NEU = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#8a8986'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})

# ======================================================================
# 1. Xác định các đợt khuyến mãi
df = pd.read_parquet(DATA_PATH, columns=['date', 'item_id', 'store_id', 'category', 'quantity',
                                         'promo_status', 'depth', 'phase'])
df['pair'] = df['item_id'].astype(str) + '_' + df['store_id'].astype(str)
df = df.sort_values(['pair', 'date']).reset_index(drop=True)
df['dnum'] = (df['date'] - df['date'].min()).dt.days
is_promo = df['promo_status'].eq(1)
new_run = is_promo & ~(is_promo.shift(fill_value=False) & df['pair'].eq(df['pair'].shift())
                       & df['dnum'].diff().eq(1))
df['ep'] = np.where(is_promo, new_run.cumsum(), 0)

ev = (df[is_promo].groupby('ep')
        .agg(pair=('pair', 'first'), item_id=('item_id', 'first'), store_id=('store_id', 'first'),
             category=('category', 'first'), start=('dnum', 'min'), end=('dnum', 'max'),
             n_days=('dnum', 'size'), depth=('depth', 'mean'))
        .reset_index())
print('Tổng số đợt KM:', len(ev))

# Lượng bán TB nửa đầu / nửa sau đợt
d_in = df[is_promo][['ep', 'dnum', 'quantity']].merge(ev[['ep', 'start', 'n_days']], on='ep')
d_in['early'] = (d_in['dnum'] - d_in['start']) < d_in['n_days'] / 2
q_in = d_in.pivot_table(index='ep', columns='early', values='quantity', aggfunc='mean')
ev['q_early'] = ev['ep'].map(q_in[True]); ev['q_late'] = ev['ep'].map(q_in[False])
ev['q_during'] = ev['ep'].map(d_in.groupby('ep')['quantity'].mean())

# Cửa sổ trước (baseline) và sau đợt: ghép theo (pair, ngày)
offs = np.r_[np.arange(-BASE_DAYS, 0), np.arange(1, POST_DAYS + 1)]
win = ev[['ep', 'pair', 'start', 'end']].loc[ev.index.repeat(len(offs))].reset_index(drop=True)
win['off'] = np.tile(offs, len(ev))
win['dnum'] = np.where(win['off'] < 0, win['start'] + win['off'], win['end'] + win['off'])
win = win.merge(df[['pair', 'dnum', 'quantity', 'promo_status', 'phase']], on=['pair', 'dnum'], how='left')

pre = win[(win['off'] < 0) & win['phase'].eq('other') & win['promo_status'].eq(0)].copy()
pre['half'] = np.where(pre['off'] % 2 == 0, 'A', 'B')
b = pre.groupby(['ep', 'half'])['quantity'].agg(['mean', 'size']).unstack('half')
ev = ev.merge(pd.DataFrame({'base_A': b[('mean', 'A')], 'nA': b[('size', 'A')],
                            'base_B': b[('mean', 'B')], 'nB': b[('size', 'B')]}).reset_index(), on='ep', how='left')
ev['base_all'] = ev['ep'].map(pre.groupby('ep')['quantity'].mean())

post = win[win['off'] > 0]
post_ok = post.groupby('ep').apply(lambda g: (g['promo_status'].eq(0).sum() == POST_DAYS), include_groups=False)
ev['post_clean'] = ev['ep'].map(post_ok).fillna(False)
ev['q_post13'] = ev['ep'].map(post[post['off'] <= 3].groupby('ep')['quantity'].mean())
ev['q_post47'] = ev['ep'].map(post[post['off'] >= 4].groupby('ep')['quantity'].mean())

# ======================================================================
# 2. Lọc đợt và tạo biến
flt = [('Tổng số đợt KM', np.ones(len(ev), bool)),
       (f'Độ dài {MIN_DUR}–{MAX_DUR} ngày', ev['n_days'].between(MIN_DUR, MAX_DUR)),
       (f'Đủ baseline (≥{MIN_BASE_DAYS} ngày thường mỗi nửa)', (ev['nA'] >= MIN_BASE_DAYS) & (ev['nB'] >= MIN_BASE_DAYS)),
       ('Có bán trong baseline (cả 2 nửa > 0)', (ev['base_A'] > 0) & (ev['base_B'] > 0)),
       (f'{POST_DAYS} ngày sau KM đủ dữ liệu, không có KM mới', ev['post_clean'])]
keep = np.ones(len(ev), bool); flow = []
for name, m in flt:
    keep &= np.asarray(m, bool); flow.append({'bước': name, 'số đợt còn lại': int(keep.sum())})
flow = pd.DataFrame(flow); print(flow.to_string(index=False))
ev = ev[keep].copy()

lr = lambda num, den: np.log((num + C) / (den + C))
ev['DEPTH'] = ev['depth'] * 10
ev['FOOD'] = (ev['category'] == 'food').astype(float)
ev['DUR'] = np.log(ev['n_days'])
ev['BASE'] = np.log(ev['base_A'] + C)
ev['lift_early'] = lr(ev['q_early'], ev['base_A'])
ev['lift_late'] = lr(ev['q_late'], ev['base_A'])
ev['dip_13'] = lr(ev['base_B'], ev['q_post13'])
ev['dip_47'] = lr(ev['base_B'], ev['q_post47'])
# R1 (robustness): dùng chung một baseline cho LIFT và DIP
ev['lift_early_sh'] = lr(ev['q_early'], ev['base_all']); ev['lift_late_sh'] = lr(ev['q_late'], ev['base_all'])
ev['dip_13_sh'] = lr(ev['base_all'], ev['q_post13']); ev['dip_47_sh'] = lr(ev['base_all'], ev['q_post47'])
ev['BASE_sh'] = np.log(ev['base_all'] + C)
IND = ['lift_early', 'lift_late', 'dip_13', 'dip_47']
IND_SH = [f'{c}_sh' for c in IND]
for c in IND + IND_SH:                                   # winsorize 1%–99% để giảm ảnh hưởng của ngoại lai
    lo, hi = ev[c].quantile([0.01, 0.99]); ev[c] = ev[c].clip(lo, hi)
ev = ev.reset_index(drop=True)

# Đọc hiểu dạng % (trung vị): Lift = lượng bán trong KM / baseline − 1 ; Dip = sau KM / baseline − 1
ev['lift_pct'] = 100 * ((ev['q_during'] + C) / (ev['base_A'] + C) - 1)
ev['post_pct'] = 100 * (((ev['q_post13'] * 3 + ev['q_post47'] * 4) / 7 + C) / (ev['base_B'] + C) - 1)
VARS = ['DEPTH', 'FOOD', 'DUR', 'BASE'] + IND
desc = ev.groupby('category')[VARS + ['n_days', 'lift_pct', 'post_pct']].agg(['mean', 'std', 'median']).T
desc.to_csv(OUT / 'sem_descriptive.csv')   # chỉ lưu số liệu tổng hợp; dữ liệu từng đợt suy ra từ data/ (không commit)
flow.to_csv(OUT / 'sem_sample_flow.csv', index=False)
print(f'\nSố đợt dùng cho SEM: {len(ev)} (food {int(ev.FOOD.sum())}, household {int((1 - ev.FOOD).sum())}),'
      f' {ev.item_id.nunique()} item')
print(ev.groupby('category')[['depth', 'n_days', 'lift_pct', 'post_pct']].median().round(3))
print('\nTương quan giữa các biến:'); print(ev[VARS].corr().round(3))

# ======================================================================
# 3. Mô hình SEM
def spec(food=True, suffix=''):
    ctrl = ' + '.join((['FOOD'] if food else []) + ['DUR', f'BASE{suffix}'])
    le, ll, d1, d4 = [f'{c}{suffix}' for c in IND]
    return (f'LIFT =~ {le} + {ll}\n'
            f'DIP =~ {d1} + {d4}\n'
            f'DEPTH ~ {ctrl}\n'
            f'LIFT ~ DEPTH + {ctrl}\n'
            f'DIP ~ LIFT + DEPTH + {ctrl}\n')

def fit(data, desc_):
    m = semopy.Model(desc_)
    m.fit(data, obj='MLW')
    return m

def key_paths(m, std=False):
    t = m.inspect(std_est=std)
    g = lambda l, r: float(t[(t.lval == l) & (t.op == '~') & (t.rval == r)]['Est. Std' if std else 'Estimate'].iloc[0])
    a, b, c = g('LIFT', 'DEPTH'), g('DIP', 'LIFT'), g('DIP', 'DEPTH')
    return {'a': a, 'b': b, 'c_direct': c, 'indirect_ab': a * b, 'total': c + a * b}

def bootstrap(data, desc_, n=N_BOOT, seed=SEED):
    """Cluster bootstrap theo item_id: lấy mẫu lại item (kèm mọi đợt của item đó)."""
    rng = np.random.default_rng(seed)
    groups = {k: v.to_numpy() for k, v in data.groupby('item_id').groups.items()}
    keys = np.array(list(groups))
    res = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        try:
            m = fit(data.loc[idx].reset_index(drop=True), desc_)
            res.append({**key_paths(m), **{f'std_{k}': v for k, v in key_paths(m, std=True).items()}})
        except Exception:
            continue
    return pd.DataFrame(res)

def summarize(m, boot, label):
    est, std = key_paths(m), key_paths(m, std=True)
    rows = []
    for k in est:
        rows.append({'model': label, 'effect': k, 'estimate': est[k], 'std_estimate': std[k],
                     'boot_se': boot[k].std(), 'ci_low': boot[k].quantile(0.025), 'ci_high': boot[k].quantile(0.975),
                     'std_ci_low': boot[f'std_{k}'].quantile(0.025), 'std_ci_high': boot[f'std_{k}'].quantile(0.975),
                     'p_boot': min(1.0, 2 * min((boot[k] <= 0).mean(), (boot[k] >= 0).mean())),
                     'n_boot': len(boot)})
    return pd.DataFrame(rows)

STAT_COLS = ['DoF', 'chi2', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'GFI', 'AGFI', 'AIC', 'BIC']
def fit_stats(m, label, n):
    s = semopy.calc_stats(m).T.iloc[:, 0]
    return {'model': label, 'N': n, **{c: float(s[c]) for c in STAT_COLS if c in s.index}}

results, params, stats_rows, boots = [], [], [], {}
runs = [('Full (pooled)', ev, spec(True)),
        ('Food', ev[ev.FOOD == 1].reset_index(drop=True), spec(False)),
        ('Household', ev[ev.FOOD == 0].reset_index(drop=True), spec(False)),
        ('R1: shared baseline', ev, spec(True, '_sh'))]
for label, data, desc_ in runs:
    m = fit(data, desc_)
    t = m.inspect(std_est=True); t.insert(0, 'model', label); params.append(t)
    stats_rows.append(fit_stats(m, label, len(data)))
    boot = bootstrap(data, desc_)
    boots[label] = boot
    results.append(summarize(m, boot, label))
    print(f'\n===== {label} (N = {len(data)}, bootstrap thành công {len(boot)}/{N_BOOT}) =====')
    # semopy ghi cả hệ số cấu trúc lẫn factor loading (dạng "chỉ báo ~ BIẾN_ẨN") với op = '~'
    print(t[t.op == '~'][['lval', 'op', 'rval', 'Estimate', 'Est. Std', 'Std. Err', 'p-value']].round(4).to_string(index=False))

# R2 (robustness): path analysis với biến quan sát (trung bình 2 chỉ báo), không có biến ẩn
ev['LIFT_obs'] = ev[['lift_early', 'lift_late']].mean(axis=1); ev['DIP_obs'] = ev[['dip_13', 'dip_47']].mean(axis=1)
desc_obs = ('DEPTH ~ FOOD + DUR + BASE\nLIFT_obs ~ DEPTH + FOOD + DUR + BASE\n'
            'DIP_obs ~ LIFT_obs + DEPTH + FOOD + DUR + BASE\n')
m_obs = fit(ev, desc_obs)
t = m_obs.inspect(std_est=True); t.insert(0, 'model', 'R2: observed path model'); params.append(t)
g = lambda l, r, col='Estimate': float(t[(t.lval == l) & (t.op == '~') & (t.rval == r)][col].iloc[0])
a_, b_, c_ = g('LIFT_obs', 'DEPTH'), g('DIP_obs', 'LIFT_obs'), g('DIP_obs', 'DEPTH')
sa, sb, sc = g('LIFT_obs', 'DEPTH', 'Est. Std'), g('DIP_obs', 'LIFT_obs', 'Est. Std'), g('DIP_obs', 'DEPTH', 'Est. Std')
results.append(pd.DataFrame([{'model': 'R2: observed path model', 'effect': k, 'estimate': v, 'std_estimate': s}
                             for k, v, s in [('a', a_, sa), ('b', b_, sb), ('c_direct', c_, sc),
                                             ('indirect_ab', a_ * b_, sa * sb), ('total', c_ + a_ * b_, sc + sa * sb)]]))

res = pd.concat(results, ignore_index=True)
res.to_csv(OUT / 'sem_effects.csv', index=False)
pd.concat(params, ignore_index=True).to_csv(OUT / 'sem_parameters.csv', index=False)
fstats = pd.DataFrame(stats_rows); fstats.to_csv(OUT / 'sem_fit_stats.csv', index=False)

# Khác biệt Food − Household (bootstrap độc lập hai nhóm)
bf, bh = boots['Food'], boots['Household']
n_ = min(len(bf), len(bh))
diff_rows = []
for k in ['a', 'b', 'c_direct', 'indirect_ab', 'total']:
    d = bf[k].iloc[:n_].to_numpy() - bh[k].iloc[:n_].to_numpy()
    ef = res[(res.model == 'Food') & (res.effect == k)].estimate.iloc[0]
    eh = res[(res.model == 'Household') & (res.effect == k)].estimate.iloc[0]
    diff_rows.append({'effect': k, 'food': ef, 'household': eh, 'diff_food_minus_household': ef - eh,
                      'ci_low': np.quantile(d, 0.025), 'ci_high': np.quantile(d, 0.975),
                      'p_boot': min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
diff = pd.DataFrame(diff_rows); diff.to_csv(OUT / 'sem_group_difference.csv', index=False)

print('\n===== Hiệu ứng (KTC 95% cluster bootstrap) ====='); print(res.round(4).to_string(index=False))
print('\n===== Chỉ số phù hợp ====='); print(fstats.round(4).to_string(index=False))
print('\n===== Khác biệt Food − Household ====='); print(diff.round(4).to_string(index=False))

# ======================================================================
# 4. SEM Path Diagram
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def draw_path_diagram(param_tab, label, fname, title):
    t = param_tab[param_tab.model == label]
    def gs(l, r):
        row = t[(t.lval == l) & (t.op == '~') & (t.rval == r)].iloc[0]
        p = pd.to_numeric(row['p-value'], errors='coerce')
        return float(row['Est. Std']), (0.0 if not np.isfinite(p) else float(p))   # loading cố định = 1 không có p

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 7.4); ax.axis('off')
    Y = 4.55                                               # hàng biến chính
    pos = {'lift_early': (3.4, 6.35), 'lift_late': (5.4, 6.35), 'dip_13': (7.8, 6.35), 'dip_47': (9.8, 6.35),
           'DEPTH': (1.2, Y), 'LIFT': (4.4, Y), 'DIP': (8.8, Y)}
    names = {'lift_early': 'Lift nửa đầu', 'lift_late': 'Lift nửa sau', 'dip_13': 'Dip ngày 1–3',
             'dip_47': 'Dip ngày 4–7', 'LIFT': 'SALES LIFT', 'DIP': 'POST-PROMO\nDIP', 'DEPTH': 'DISCOUNT\nDEPTH'}
    W, H = 1.55, 0.72
    EW, EH = 2.1, 1.2
    for k, (x, y) in pos.items():
        if k in ('LIFT', 'DIP'):
            ax.add_patch(Ellipse((x, y), EW, EH, fc='#f3f2ee', ec=INK, lw=1.6))
        else:
            ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle='round,pad=0.02,rounding_size=0.08',
                                        fc='white', ec=INK, lw=1.6 if k == 'DEPTH' else 1.1))
        main = k in ('LIFT', 'DIP', 'DEPTH')
        ax.text(x, y, names[k], ha='center', va='center', fontsize=9.5 if main else 8.5,
                fontweight='bold' if main else 'normal', color=INK)

    def label_box(x, y, s, fs=9.5, bold=True):
        ax.text(x, y, s, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

    def arr(p0, p1, sig, main=True, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15 if main else 10,
                                     lw=2.2 if main else 1.0, color=INK if main else NEU,
                                     linestyle='-' if sig else (0, (4, 3)), connectionstyle=f'arc3,rad={rad}',
                                     shrinkA=0, shrinkB=1))

    # Đo lường: biến ẩn → chỉ báo
    for lat, inds in [('LIFT', ['lift_early', 'lift_late']), ('DIP', ['dip_13', 'dip_47'])]:
        for i in inds:
            v, p = gs(i, lat)                              # loading: "chỉ báo ~ biến ẩn"
            (x0, y0), (x1, y1) = pos[lat], pos[i]
            dx = x1 - x0; dy = y1 - y0; n = np.hypot(dx / (EW / 2), dy / (EH / 2))
            p0 = (x0 + dx / n, y0 + dy / n); p1 = (x1 - dx * (H / 2) / dy, y1 - H / 2)
            arr(p0, p1, p < 0.05, main=False)
            label_box((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, f'{v:.2f}{stars(p)}', fs=8.5, bold=False)

    # Cấu trúc chính: a, b (thẳng) và c' (cung phía dưới LIFT)
    a, pa = gs('LIFT', 'DEPTH'); b, pb = gs('DIP', 'LIFT'); c, pc = gs('DIP', 'DEPTH')
    arr((pos['DEPTH'][0] + W / 2, Y), (pos['LIFT'][0] - EW / 2, Y), pa < 0.05)
    arr((pos['LIFT'][0] + EW / 2, Y), (pos['DIP'][0] - EW / 2, Y), pb < 0.05)
    label_box(2.65, Y + 0.3, f'a = {a:.2f}{stars(pa)}')
    label_box(6.6, Y + 0.3, f'b = {b:.2f}{stars(pb)}')
    p0, p1, rad = (pos['DEPTH'][0] + 0.4, Y - H / 2), (pos['DIP'][0], Y - EH / 2), 0.25
    arr(p0, p1, pc < 0.05, rad=rad)
    # điểm giữa của cung arc3: lệch rad·dx/2 theo phương vuông góc (dx > 0, dy ≈ 0 → đi xuống)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    label_box((p0[0] + p1[0]) / 2 + rad * dy / 2, (p0[1] + p1[1]) / 2 - rad * dx / 2, f"c' = {c:.2f}{stars(pc)}  (trực tiếp)")

    # Biến kiểm soát: một khối chung, mũi tên mảnh tới DEPTH, LIFT, DIP
    ctrls = [k for k in ('FOOD', 'DUR', 'BASE') if (t.rval == k).any()]
    cnames = {'FOOD': 'Category (Food = 1)', 'DUR': 'ln(độ dài KM)', 'BASE': 'ln(mức bán nền)'}
    cx, cy, cw, ch = 5.75, 1.95, 5.6, 0.62
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc='white', ec=NEU, lw=1.0))
    ax.text(cx, cy, 'Biến kiểm soát: ' + ' · '.join(cnames[k] for k in ctrls), ha='center', va='center',
            fontsize=8.5, color=INK2)
    for dst, xs in [('DEPTH', 0.85), ('LIFT', 4.4), ('DIP', 8.8)]:
        top = Y - (H / 2 if dst == 'DEPTH' else EH / 2)
        x_from = min(max(xs, cx - cw / 2 + 0.3), cx + cw / 2 - 0.3)
        arr((x_from, cy + ch / 2), (xs, top), True, main=False)

    # Bảng hệ số chuẩn hóa của biến kiểm soát
    rows = [f"{'':<20}{'→ Depth':>10}{'→ Lift':>10}{'→ Dip':>10}"]
    for k in ctrls:
        cells = []
        for dst in ('DEPTH', 'LIFT', 'DIP'):
            v, p = gs(dst, k); cells.append(f'{v:.2f}{stars(p)}')
        rows.append(f'{cnames[k]:<20}' + ''.join(f'{s:>10}' for s in cells))
    ax.text(0.15, 1.3, '\n'.join(rows), fontsize=7.8, color=INK2, va='top', family='DejaVu Sans Mono')

    # Hộp hiệu ứng gián tiếp / tổng + chỉ số phù hợp
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

P = pd.concat(params, ignore_index=True)
draw_path_diagram(P, 'Full (pooled)', 'fig_sem_path_diagram.png', 'SEM Path Diagram – Discount → Sales Lift → Post-Promotion Dip')
draw_path_diagram(P, 'Food', 'fig_sem_path_food.png', 'SEM Path Diagram – Food')
draw_path_diagram(P, 'Household', 'fig_sem_path_household.png', 'SEM Path Diagram – Household')

# So sánh đa nhóm: a, b, c', a×b (chưa chuẩn hóa, đơn vị log / 10 điểm %)
fig, ax = plt.subplots(figsize=(8, 4.2))
effects = [('a', 'a: Depth → Lift'), ('b', 'b: Lift → Dip'), ('c_direct', "c': Depth → Dip (trực tiếp)"),
           ('indirect_ab', 'a×b: Depth → Lift → Dip'), ('total', 'Tổng: Depth → Dip')]
yy = np.arange(len(effects))[::-1]; h = 0.36
for i, (grp, col) in enumerate([('Food', C_FOOD), ('Household', C_HH)]):
    r = res[res.model == grp].set_index('effect')
    v = np.array([r.loc[k, 'estimate'] for k, _ in effects])
    lo = np.array([r.loc[k, 'ci_low'] for k, _ in effects]); hi = np.array([r.loc[k, 'ci_high'] for k, _ in effects])
    yv = yy + (h / 2 if i == 0 else -h / 2)
    ax.barh(yv, v, h - 0.04, color=col, label=grp, edgecolor='white', linewidth=2)
    # KTC percentile có thể không chứa ước lượng điểm → clip để độ dài error bar không âm
    ax.errorbar(v, yv, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], fmt='none', ecolor=INK,
                elinewidth=1, capsize=2.5)
    for val, y_, h_ in zip(v, yv, hi):
        ax.text(max(h_, 0) + 0.008, y_, f'{val:+.3f}', va='center', fontsize=8, color=INK)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, [l for _, l in effects]); ax.set_xlabel('Hệ số chưa chuẩn hóa (KTC 95% bootstrap)')
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
xl = ax.get_xlim(); ax.set_xlim(xl[0], xl[1] + 0.06)
ax.legend(frameon=False, loc='upper left')
ax.set_title('So sánh đa nhóm: Food vs Household', loc='left', fontsize=10.5, fontweight='bold', color=INK)
fig.tight_layout(); fig.savefig(OUT / 'fig_sem_group_comparison.png'); plt.close(fig)
print('\nĐã lưu bảng và hình vào', OUT)
