"""Đánh giá mô hình tốt nhất của mỗi objective theo: tổng thể, category (food/household), phase khuyến mãi.

Dựng lại feature y hệt xgboost_train.py (chạy phần 1–3 của script, KHÔNG train lại), nạp checkpoint và tính metric
trên valid và test. Kết quả: outputs/breakdown_metrics.csv
"""
import re
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

HERE = Path(__file__).resolve().parent
src = (HERE / 'xgboost_train.py').read_text(encoding='utf-8')
prep = src.split('# 4. Tuning')[0]                      # phần đọc dữ liệu + feature engineering + metric
prep = prep.replace('__file__', repr(str(HERE / 'xgboost_train.py')))
ns = {}
exec(compile(prep, 'xgboost_train.py[prep]', 'exec'), ns)
df, FEATURES, TARGET, OUT = ns['df'], ns['FEATURES'], ns['TARGET'], ns['OUT']

tuning = pd.read_csv(f'{OUT}/tuning_results.csv')
tuning['group'] = tuning['experiment'].str.replace(r'_(base|rs\d+)$', '', regex=True)
best = tuning.sort_values('best_val_mse').groupby('group').head(1)
MODELS = dict(zip(best['group'], best['experiment']))   # objective -> experiment tốt nhất
print('Mô hình đánh giá:', MODELS)

PHASES = ['pre_1_3', 'during', 'post_1_3', 'post_4_7', 'other']

def score(y, p):
    e = np.abs(y - p)
    return {'n': len(y), 'MAE': e.mean(), 'RMSE': np.sqrt(((y - p) ** 2).mean()), 'WAPE': e.sum() / y.sum()}

rows = []
for split in ['valid', 'test']:
    part = df[df.split == split]
    d = xgb.DMatrix(part[FEATURES], enable_categorical=True)
    y = part[TARGET].values
    for obj, exp in MODELS.items():
        bst = xgb.Booster(); bst.load_model(f'{OUT}/checkpoints/{exp}.json')
        p = np.clip(bst.predict(d), 0, None)
        rows.append({'split': split, 'objective': obj, 'experiment': exp, 'by': 'overall', 'group': 'all', **score(y, p)})
        for col, groups in [('category', ['food', 'household']), ('phase', PHASES)]:
            v = part[col].astype(str).values
            for g in groups:
                m = v == g
                rows.append({'split': split, 'objective': obj, 'experiment': exp, 'by': col, 'group': g,
                             **score(y[m], p[m])})

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/breakdown_metrics.csv', index=False)
pd.set_option('display.width', 200)
for split in ['valid', 'test']:
    r = res[res.split == split]
    print(f'\n===== {split.upper()} =====')
    print(r[r.by == 'overall'].pivot_table(index='objective', values=['MAE', 'RMSE', 'WAPE']).round(4))
    print(r[r.by == 'category'].pivot_table(index='objective', columns='group', values='WAPE').round(4))
    for m in ['MAE', 'WAPE']:
        print(m, 'theo phase'); print(r[r.by == 'phase'].pivot_table(index='objective', columns='group', values=m)[PHASES].round(4))
    print(r[r.by != 'overall'].drop_duplicates(['by', 'group'])[['by', 'group', 'n']].to_string(index=False))
