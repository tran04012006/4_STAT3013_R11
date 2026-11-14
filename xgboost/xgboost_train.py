
import os, time, warnings
import numpy as np, pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
from pathlib import Path
# Đường dẫn tính theo vị trí file script → chạy từ thư mục nào cũng được
HERE = Path(__file__).resolve().parent              # <repo>/xgboost
DATA_PATH = HERE.parent / 'data' / 'daily_panel.parquet'
OUT = str(HERE / 'outputs'); os.makedirs(OUT, exist_ok=True)
SEED = 42
print('xgboost', xgb.__version__)


# ======================================================================
# 1. Đọc dữ liệu

df = pd.read_parquet(DATA_PATH)
df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
print(df.shape)
print(df.groupby('split').date.agg(['min', 'max', 'count']))
print('\nNgày bắt đầu có dữ liệu theo store:')
print(df.groupby('store_id').date.min())


# ======================================================================
# 2. Feature engineering
#
# Theo gợi ý trong STATISTIC + ML_DL.pdf:
#
# | Nhóm | Feature |
# |---|---|
# | Cửa hàng / cơ cấu | store_id, store4_period, format, area, density |
# | Sản phẩm | category, group4, class_name, subclass_name, item_type, trọng lượng |
# | Mức bán lịch sử (expanding mean đến t−7) | item_store_mean, item_mean, subclass_store_mean, pair_age (xử lý sản phẩm mới) |
# | Khuyến mãi | promo_status, depth (discount depth), promo_type, price_ratio, day_in_promo, days_to_promo_end, promo_duration |
# | Hậu khuyến mãi | days_since_promotion_end (tự tạo, đang KM = 0), days_since_last_promo, phase, status6, is_post_promo |
# | Tương tác | depth_x_food, depth_x_household |
# | Lịch tuần | dow, is_friday, is_weekend, lag_7, lag_14, lag_28, roll_mean_7/28, roll_std_28 |
# | Mùa vụ / lễ | month, week, day_of_month, holiday_status, is_holiday, minor_holiday, days_to_holiday, days_after_holiday, ny_break, pheic |
#
# Loại bỏ để tránh leak: quantity (target), qty_norm, revenue.

df['is_friday'] = (df['dow'] == 4).astype(int)            # dow: 0 = Thứ Hai
df['is_food'] = (df['category'] == 'food').astype(int)
df['depth_x_food'] = df['depth'] * df['is_food']
df['depth_x_household'] = df['depth'] * (1 - df['is_food'])
df['is_post_promo'] = df['phase'].isin(['post_1_3', 'post_4_7']).astype(int)
# days_since_last_promo trong panel = số ngày kể từ ngày KM gần nhất (=1 ngay sau ngày KM cuối, nhưng cũng =1
# trong lúc đang KM). Tạo days_since_promotion_end rõ nghĩa: đang KM = 0, sau KM = 1,2,3..., chưa từng KM = NaN
df['days_since_promotion_end'] = np.where(df['promo_status'].eq(1), 0, df['days_since_last_promo'])
df['lag7_ratio_roll28'] = df['lag_7'] / (df['roll_mean_28'] + 1e-3)

# --- Mức bán lịch sử (expanding mean đến ngày t-7) – không leak, dùng được cho mọi split ---
# Lưu ý: KHÔNG dùng mean encoding tĩnh tính trên cả tập train: (1) leak trong train (chứa chính target),
# (2) mọi dòng train đều có giá trị nên mô hình không học được trường hợp sản phẩm mới (NaN) → dự báo
# sản phẩm mới cao gấp ~11 lần thực tế.
def expanding_mean(frame, keys, lag=7):
    d = (frame.groupby(keys + ['date'], observed=True)['quantity'].agg(['sum', 'count'])
              .reset_index().sort_values(keys + ['date']))
    g = d.groupby(keys, observed=True)
    cs = g['sum'].cumsum() - d['sum']; cc = g['count'].cumsum() - d['count']      # lũy kế đến t-1
    by = [d[k] for k in keys]
    d['cs'] = cs.groupby(by).shift(lag - 1); d['cc'] = cc.groupby(by).shift(lag - 1)  # lũy kế đến t-7
    return d.assign(v=d['cs'] / d['cc'].replace(0, np.nan))[keys + ['date', 'v']]

for name, keys in [('item_store_mean', ['item_id', 'store_id']),        # mức bán của chính cặp item×store
                   ('item_mean', ['item_id']),                          # mức bán của item ở mọi store (dự phòng)
                   ('subclass_store_mean', ['subclass_name', 'store_id'])]:  # mức bán của subclass (sản phẩm mới)
    df = df.merge(expanding_mean(df, keys).rename(columns={'v': name}), on=keys + ['date'], how='left')
df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

# Tuổi của cặp item×store (số ngày kể từ lần đầu xuất hiện) – nhận diện sản phẩm mới / cold-start
first_date = df.groupby(['item_id', 'store_id']).date.transform('min')
df['pair_age'] = (df.date - first_date).dt.days.clip(upper=365)
# Các cặp xuất hiện ngay từ ngày đầu của dữ liệu (store 1–3) không phải sản phẩm mới mà chỉ là dữ liệu bị cắt trái
# → 35 ngày đầu thiếu lag/rolling một cách "giả tạo", loại khỏi tập huấn luyện để mô hình không học sai.
store_start = df.groupby('store_id').date.transform('min')
df['left_censored_warmup'] = (first_date == store_start) & (df.store_id != 4) & (df.pair_age < 35)

CAT_COLS = ['store_id', 'category', 'group4', 'class_name', 'subclass_name',
            'item_type', 'phase', 'status6', 'format', 'city', 'division']
NUM_COLS = ['store4_period', 'area', 'density',
            'weight_volume', 'weight_netto',
            'item_store_mean', 'item_mean', 'subclass_store_mean', 'pair_age',
            'promo_status', 'depth', 'promo_type', 'price_ratio', 'regular_price', 'price',
            'day_in_promo', 'days_to_promo_end', 'promo_duration',
            'days_since_last_promo', 'days_since_promotion_end', 'is_post_promo',
            'depth_x_food', 'depth_x_household',
            'dow', 'is_friday', 'is_weekend',
            'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_28', 'roll_std_28', 'lag7_ratio_roll28',
            'month', 'week', 'day_of_month', 'year',
            'holiday_status', 'is_holiday', 'minor_holiday', 'days_to_holiday', 'days_after_holiday',
            'ny_break', 'pheic']
FEATURES = CAT_COLS + NUM_COLS
TARGET = 'quantity'

for c in CAT_COLS:
    df[c] = df[c].astype(str).astype('category')

train = df[(df.split == 'train') & ~df.left_censored_warmup]; valid = df[df.split == 'valid'].copy()
print(len(FEATURES), 'features | train', len(train), '| valid', len(valid), '| loại warm-up:', int(df.left_censored_warmup.sum()))
print('Tỉ lệ NaN mức bán lịch sử (train / valid):')
tv = df[df.split.isin(['train', 'valid'])]
print(tv[['item_store_mean', 'item_mean', 'subclass_store_mean']].isna().groupby(tv.split).mean().round(4))


# ======================================================================
# 3. Hàm metric dùng chung

def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def wape(y, p): return float(np.sum(np.abs(y - p)) / np.sum(y))

def metrics(y, p):
    return {'MAE': mae(y, p), 'RMSE': rmse(y, p), 'WAPE': wape(y, p), 'n': len(y)}


# ======================================================================
# 4. Tuning trên tập Valid (cùng quy trình với leader / DLinear)
#
# Quy trình thống nhất với evaluate.py của leader:
# 1. Chạy nhiều experiment, mỗi experiment train trên train, early stopping trên valid theo MSE (RMSE) và lưu checkpoint checkpoints/{experiment}.json (chỉ giữ các cây tới best iteration).
# 2. Ghi tuning_results.csv với cột experiment, best_val_mse (MSE của dự báo trên valid, đã clip ≥ 0).
# 3. Chọn experiment có best_val_mse thấp nhất; thiếu file / thiếu checkpoint thì báo lỗi, không tự dùng cấu hình mặc định.
# 4. Checkpoint được chọn sẽ dùng thẳng cho bước đánh giá test sau này (không train lại trên train+valid).
#
# Không gian tìm kiếm: 4 objective (reg:tweedie power 1.2/1.4/1.6, reg:squarederror) × siêu tham số cây.
# Gồm 4 experiment *_base (tham số cũ) + 16 experiment random search (seed cố định).

CKPT_DIR = f'{OUT}/checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
TUNING_RESULTS_PATH = f'{OUT}/tuning_results.csv'

dtrain = xgb.DMatrix(train[FEATURES], label=train[TARGET], enable_categorical=True)
dvalid = xgb.DMatrix(valid[FEATURES], label=valid[TARGET], enable_categorical=True)
y_valid = valid[TARGET].values

BASE_PARAMS = dict(tree_method='hist', max_cat_to_onehot=8, max_bin=256, seed=SEED, nthread=-1,
                   eval_metric='rmse')   # early stopping theo MSE trên valid – cùng tiêu chí chọn model với leader

OBJECTIVES = {
    'tweedie1.2': dict(objective='reg:tweedie', tweedie_variance_power=1.2),
    'tweedie1.4': dict(objective='reg:tweedie', tweedie_variance_power=1.4),
    'tweedie1.6': dict(objective='reg:tweedie', tweedie_variance_power=1.6),
    'mse':        dict(objective='reg:squarederror'),
}
SEARCH_SPACE = dict(learning_rate=[0.03, 0.05, 0.1], max_depth=[6, 8, 10], min_child_weight=[5, 20, 50],
                    subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], reg_lambda=[0.5, 1.0, 5.0])
OLD_HP = dict(learning_rate=0.05, max_depth=8, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
N_TRIALS = 16

import random
rng = random.Random(SEED)
EXPERIMENTS = {f'{o}_base': {**OBJECTIVES[o], **OLD_HP} for o in OBJECTIVES}
for i in range(N_TRIALS):
    o = rng.choice(list(OBJECTIVES))
    EXPERIMENTS[f'{o}_rs{i:02d}'] = {**OBJECTIVES[o], **{k: rng.choice(v) for k, v in SEARCH_SPACE.items()}}

rows = []
for exp, hp in EXPERIMENTS.items():
    t0 = time.time()
    bst = xgb.train({**BASE_PARAMS, **hp}, dtrain, num_boost_round=3000,
                    evals=[(dvalid, 'valid')], early_stopping_rounds=100, verbose_eval=False)
    ckpt = bst[: bst.best_iteration + 1]            # checkpoint = model tại best iteration
    ckpt.save_model(f'{CKPT_DIR}/{exp}.json')
    p = np.clip(ckpt.predict(dvalid), 0, None)
    m = metrics(y_valid, p)
    rows.append({'experiment': exp, 'best_val_mse': float(np.mean((y_valid - p) ** 2)),
                 'best_iter': bst.best_iteration + 1, 'val_MAE': m['MAE'], 'val_WAPE': m['WAPE'],
                 **hp, 'sec': round(time.time() - t0, 1)})
    print({k: rows[-1][k] for k in ['experiment', 'best_val_mse', 'val_WAPE', 'best_iter', 'sec']})

tuning = pd.DataFrame(rows).sort_values('best_val_mse')
tuning.to_csv(TUNING_RESULTS_PATH, index=False)
print(tuning)


# ======================================================================
def choose_checkpoint():
    """Chọn experiment có Validation MSE thấp nhất trong tuning_results.csv (giống evaluate.py của leader).
    Không tự động dùng cấu hình mặc định nếu thiếu file."""
    if not os.path.exists(TUNING_RESULTS_PATH):
        raise FileNotFoundError(f'Tuning results not found: {TUNING_RESULTS_PATH}')
    results = pd.read_csv(TUNING_RESULTS_PATH)
    if not {'experiment', 'best_val_mse'}.issubset(results.columns):
        raise ValueError('Missing required columns in tuning_results.csv')
    if results.empty:
        raise ValueError('Tuning results are empty')
    results['best_val_mse'] = pd.to_numeric(results['best_val_mse'], errors='coerce')
    results = results.dropna(subset=['best_val_mse'])
    results = results[np.isfinite(results['best_val_mse'])]
    if results.empty:
        raise ValueError('No valid Validation MSE found')
    best_row = results.sort_values('best_val_mse', ascending=True).iloc[0]
    experiment, expected_mse = str(best_row['experiment']), float(best_row['best_val_mse'])
    checkpoint_path = f'{CKPT_DIR}/{experiment}.json'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Best checkpoint not found: {checkpoint_path}')
    print('========== MODEL SELECTION ==========')
    print('Selected experiment:', experiment)
    print('Expected Validation MSE:', expected_mse)
    print('Checkpoint path:', checkpoint_path)
    return checkpoint_path, experiment, expected_mse

ckpt_path, BEST, expected_mse = choose_checkpoint()
BEST_PARAMS = EXPERIMENTS[BEST]
print(BEST_PARAMS)

# 5. Kết quả
#
# - outputs_xgb/tuning_results.csv: kết quả mọi experiment trên valid.
# - outputs_xgb/checkpoints/{experiment}.json: checkpoint từng experiment; checkpoint được chọn in ở cell trên.
# - Chưa đụng tới tập test.
