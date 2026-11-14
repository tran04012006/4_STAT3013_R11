# LSTM Demand Forecasting Report

**Project:** Sales analysis before, during and after promotions; demand forecasting; business recommendations
**Data:** `data_model/daily_panel.parquet` (table C: item × store × day, 1,197,327 rows, 4,063 pairs, 2022-08-28 – 2024-09-26)
**Script:** `src/train_lstm.py` · **Outputs:** `output/lstm/` · **Run date:** 2026-11-14 (Kaggle, Tesla T4 GPU)
**Tuning method (agreed for all models):** Manual Grid Search + Hold-out Validation + Early Stopping

---

## 1. Summary

| | LSTM | 7-day moving average (MA7) | Seasonal naive |
|---|---|---|---|
| **Test WAPE** | **0.544** | 0.605 | 0.696 |
| Validation WAPE | **0.508** | 0.582 | 0.691 |
| Test MAE | **2.43** | 2.71 | 3.11 |
| Test RMSE | 12.22 | **11.98** | 14.68 |
| Test bias (total forecast / total actual − 1) | −15.6% | **+0.1%** | −0.3% |

- The LSTM was tuned with a **full hyperparameter grid**, selected by WAPE on the **hold-out validation set** (Apr–Jun 2024), and every configuration used **early stopping** on validation WAPE. The test set (Jul–Sep 2024) was evaluated **only once**, for the selected configuration.
- The LSTM **reduces WAPE by about 10% (relative)** versus MA7 and by 22% versus seasonal naive. It has the lowest WAPE in **every product group, every store, every forecast horizon (1–7 days) and every phase before / during / after promotions**.
- The trade-off: the LSTM **under-forecasts total volume by about 16%** (−36% for Household goods). Because it is trained on absolute error (matching WAPE), it forecasts close to the median, and for sparse series the median is often 0 (Section 6).
- Sales history is by far the most important feature group, followed by **promotion schedule and discount depth**, store / product attributes, **post-promotion status** and day of week.

---

## 2. Problem setup

| Item | Choice | Rationale |
|---|---|---|
| Forecast level | Item × store × day | Agreed in `STATISTIC + ML_DL.md` |
| Target | `quantity` (net daily units sold; days without sales = 0) | Section 5 of `bao_cao_feature.md` |
| Horizon | Next 7 days | Lag features in the panel are shifted by ≥ 7 days, so the panel supports horizons up to 7 days |
| Data split (time-based hold-out) | **train** ≤ 2024-03-31 · **validation** 2024-04-01 – 06-30 · **test** 2024-07-01 – 09-26 | Chronological split, no shuffling, as in the pipeline |
| Role of validation | Hyperparameter selection (grid search) **and** early-stopping decisions | |
| Role of test | Evaluated once for the selected configuration; never used for any choice | |
| Evaluation scheme | Non-overlapping weekly blocks: forecast origins 7 days apart, so each validation / test day is forecast exactly once (horizon 1–7) | 204,663 validation day-rows; 177,713 test day-rows (matches the panel row counts exactly) |
| Metrics | WAPE (primary, used for tuning), MAE, RMSE, bias; broken down by promo / non-promo day, promotion phase, product group, store and horizon | Section 5 of `bao_cao_feature.md` |
| Baselines | MA7 (mean of the last 7 days), seasonal naive (same weekday last week) | Minimum bar the model must beat |

---

## 3. Architecture and features

**Architecture:** a global encoder–decoder LSTM (one model for all 4,063 series). The selected configuration has 128 hidden units per layer, a 56-day lookback window and 221,533 parameters.

```
 L past days                                   next 7 days
 [log1p(qty), observed flag,                   [planned promotions, price,
  promo, price, calendar, holidays, static      calendar, holidays, static
  embeddings]                                   embeddings]
        │                                              │
   LSTM encoder ── state (h, c) ───────────────▶  LSTM decoder
                                                       │
                                     MLP (+ skip connection of target-day features)
                                                       │
                       encoder base level + Δ  →  log1p(qtŷ)  →  qtŷ ≥ 0
```

**Features (following the suggestions in `STATISTIC + ML_DL.md`):**

| Group | Features | Corresponding suggestion |
|---|---|---|
| Sales history (encoder only) | log1p(`quantity`), observed / missing flag | "7-day lag": the LSTM reads the last L days directly, so lags 1–L are included |
| Planned promotion + depth | `promo_disc`, `promo_nodisc` (promotion without discount), `depth`, **`depth × food`**, `price_ratio`, promotion progress, `days_to_promo_end`, `promo_duration`, `promo_type` (5, 6, 10, 8, other) | `promo_status`, `discount_depth`, discount depth × product group interaction |
| Post-promotion | `days_since_promo` (≈ days since the promotion ended), `never_promo`, `post_promo_1_7`, `post_promo_8_14` | `days_since_promotion_end`, post-promotion status |
| Price level | log regular price (standardised on train), missing-price flag | |
| Day of week | one-hot `dow`, `is_weekend`, `is_friday` | `day_of_week`, `is_friday`, `is_weekend` |
| Month / day | sin/cos of month and day of month | `month` |
| Holidays | `is_holiday`, pre-holiday, post-holiday, minor holiday, Tet break, `days_to_holiday`, `days_after_holiday` | `holiday_status`, `days_to_holiday` |
| Structural | `pheic` (COVID period), `store4_period` (since store 4 opened) | When stores start having data, structural changes |
| Static (embeddings) | `store_id`, `group4`, `dept_name`, `class_name`, `format` | `store_id`, `category` |

**Not used:** `revenue`, `qty_norm`, `density` (contain future information), `phase`, `status6` (analysis labels), `price_base`. All normalisation is fitted on train only.

---

## 4. Tuning: Manual Grid Search + Hold-out Validation + Early Stopping

### 4.1 Procedure

1. Define a **full grid** (Cartesian product) of hyperparameter values in advance.
2. For **each configuration**: train on the training set and measure WAPE on validation after every epoch. **Stop early** when validation WAPE has not improved by at least 0.0001 for 4 consecutive epochs, then restore the weights of the best epoch.
3. Select the configuration with the **lowest validation WAPE**.
4. Evaluate that configuration on the test set **once**.

### 4.2 Grid and results: 8 configurations

| Hyperparameter | Grid | Fixed |
|---|---|---|
| Lookback L | 28, 56 days | Loss: MAE on raw quantity |
| Hidden units | 64, 128 | Dropout 0.1; batch 1,024; Adam |
| Learning rate | 0.001, 0.002 | 150,000 random windows per epoch (out of 816,119) |
| Early stopping | patience 4 epochs, max 40 epochs, restore best-epoch weights | LR halved when validation does not improve for 1 epoch |

| Lookback | Hidden | LR | Validation WAPE | Best epoch / epochs run | Early-stopped |
|---|---|---|---|---|---|
| **56** | **128** | **0.002** | **0.5082** | 22 / 26 | ✓ |
| 56 | 128 | 0.001 | 0.5098 | 13 / 17 | ✓ |
| 28 | 64 | 0.001 | 0.5099 | 28 / 32 | ✓ |
| 28 | 128 | 0.002 | 0.5099 | 12 / 16 | ✓ |
| 56 | 64 | 0.002 | 0.5099 | 19 / 23 | ✓ |
| 28 | 128 | 0.001 | 0.5100 | 12 / 16 | ✓ |
| 28 | 64 | 0.002 | 0.5100 | 19 / 23 | ✓ |
| 56 | 64 | 0.001 | 0.5153 | 13 / 17 | ✓ |

- **All 8 configurations stopped early**; none reached the 40-epoch limit, so all converged on validation.
- 7 of 8 configurations fall within 0.508–0.510: **the model is insensitive to hyperparameters in this region**. The selected configuration beats the rest by only about 0.002 WAPE, a small gap that could change with the random seed. Following the procedure, the configuration with the lowest validation WAPE is still selected.
- The loss is fixed to MAE on raw quantity because WAPE = Σ|error| / Σ actual, so MAE optimises exactly the evaluated quantity. In the earlier grid (loss × lookback), MSE on log1p was about 3 WAPE points worse at both L = 28 and L = 56 (`output/lstm_v1_backup/variant_selection.csv`).

Full tables: `variant_selection.csv` (per configuration), `training_log.csv` (per epoch).

---

## 5. Test results (2024-07-01 – 09-26)

Overfitting check: LSTM WAPE is 0.477 on train, 0.508 on validation and 0.544 on test. The gap grows moderately over time.

### 5.1 By promotion status

| Day type | Rows | LSTM WAPE | MA7 WAPE | Seasonal naive WAPE |
|---|---|---|---|---|
| No promotion | 146,371 | **0.548** | 0.619 | 0.717 |
| Promotion with discount | 24,468 | **0.539** | 0.580 | 0.666 |
| Promotion without discount | 6,874 | **0.513** | 0.538 | 0.580 |

On promotion days, the history-based baselines (MA7, seasonal naive) under-forecast by 19–27% because they do not know a promotion is running. The LSTM receives the promotion schedule, so it does not fall short specifically on promotion days: its bias on promotion days (−17%) is similar to normal days (−16%). This is direct evidence that **including promotions as a feature group improves forecasts**.

### 5.2 By promotion phase

| Phase | Rows | LSTM WAPE | MA7 WAPE | Seasonal naive WAPE |
|---|---|---|---|---|
| 1–3 days before promotion | 4,889 | **0.625** | 0.717 | 0.825 |
| During promotion | 23,725 | **0.533** | 0.574 | 0.660 |
| 1–3 days after promotion | 5,014 | **0.541** | 0.774 | 0.898 |
| 4–7 days after promotion | 6,323 | **0.526** | 0.819 | 1.069 |
| Other days | 137,762 | **0.545** | 0.593 | 0.677 |

Right after a promotion, the history-based baselines are badly wrong (WAPE 0.77–1.07) because they use the high promotional sales as the base level. The LSTM keeps errors at the level of normal days thanks to the post-promotion features. The 1–3 days before a promotion are the hardest to forecast for every model.

### 5.3 By product group

| Group | Rows | LSTM WAPE | MA7 WAPE | Seasonal naive WAPE | LSTM bias |
|---|---|---|---|---|---|
| **Food** | 87,185 | **0.516** | 0.569 | 0.653 | −13% |
| F1 Fresh produce | 48,319 | **0.530** | 0.570 | 0.655 | −14% |
| F2 Fresh and processed meat | 38,866 | **0.483** | 0.569 | 0.647 | −9% |
| **Household** | 90,528 | **0.747** | 0.870 | 1.015 | −36% |
| H1 Personal care, paper | 71,786 | **0.722** | 0.832 | 0.979 | −33% |
| H2 Laundry, cleaning | 18,742 | **0.961** | 1.210 | 1.332 | −65% |

Household goods are much harder to forecast. These series are sparse (1.04 units/day on average, 57% zero-sales days; Food: 8.04 units/day, 31%), so WAPE is high for every model. The LSTM's advantage is largest here (13–21% lower WAPE than MA7), at the cost of a large negative bias (Section 6).

### 5.4 By forecast horizon

| Horizon (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| LSTM | **0.511** | **0.523** | **0.523** | **0.531** | **0.539** | **0.566** | **0.616** |
| MA7 | 0.552 | 0.586 | 0.577 | 0.591 | 0.606 | 0.638 | 0.686 |
| Seasonal naive | 0.697 | 0.709 | 0.684 | 0.691 | 0.651 | 0.710 | 0.738 |

Error increases with the horizon, as expected. The LSTM beats MA7 at every horizon, and the gap stays stable from horizon 1 to 7 (about 0.04–0.07 WAPE).

### 5.5 By store

| Store | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| LSTM WAPE | **0.539** | **0.533** | **0.569** | **0.547** |
| MA7 WAPE | 0.611 | 0.588 | 0.621 | 0.601 |
| LSTM MAE | **2.56** | **1.45** | **1.63** | **2.64** |
| MA7 MAE | 2.90 | 1.60 | 1.78 | 2.90 |

The LSTM is better in all 4 stores, including store 4 (which only appears from December 2023).

---

## 6. Weakness: under-forecasting

| | Total actual | Total LSTM | Difference |
|---|---|---|---|
| Whole test set | 795,204 | 671,149 | −15.6% |
| Food | 701,242 | 611,074 | −12.9% |
| Household | 93,961 | 60,075 | −36.1% |

- System-wide daily average (88 test days): actual 9,036, LSTM 7,627.
- **Cause:** MAE pulls the model towards the median. For sparse series (57% of Household day-rows in the test set have zero sales, versus 31% for Food), the median is much lower than the mean. WAPE rewards this behaviour, but total volume falls short. The larger configuration selected by the grid search (hidden 128, L = 56) lowers WAPE but deepens the negative bias (−10.4% for the earlier 64 / 28 configuration): minimising WAPE and matching total volume are different objectives.
- **Practical implication:** for **ordering / replenishment** (where total volume matters), the current LSTM would under-order, especially for Household goods. If the smallest day-level error is what matters, the LSTM is the best choice.
- **Possible fixes (not yet done):** (a) Poisson / Tweedie loss to forecast the mean; (b) quantile forecasts (P50 for accuracy, P80–P90 for safety stock); (c) per-group scaling factors learned on validation.

---

## 7. Feature importance

Method: break the link of one feature group on the validation set and measure how much WAPE increases (average of 3 runs, base WAPE 0.508). Series-level groups are permuted across samples. Calendar groups are replaced with the calendar of another random day, because within one evaluation block all series share the same calendar.

| Feature group | WAPE increase | Std. dev. | Interpretation |
|---|---|---|---|
| Sales history | **+0.950** | 0.016 | Indispensable |
| **Promotion schedule + discount depth + promotion type** | **+0.140** | 0.006 | Most important known-in-advance group |
| Static (store, group, department, class, format) | +0.084 | 0.004 | Distinguishes series scale and behaviour |
| **Post-promotion** (days since promotion ended) | **+0.058** | 0.013 | Clear effect, consistent with Section 5.2 |
| **Day of week / weekend / Friday** | **+0.035** | 0.001 | Clear weekly seasonality |
| Regular price level | +0.025 | 0.002 | |
| Holidays | +0.007 | 0.000 | Low, as expected: the test period (Jul–Sep) has no major holidays |
| Month / day of month | +0.006 | 0.001 | Test covers only 3 summer months, so little variation |
| Structural (COVID, store 4) | +0.002 | 0.000 | |

Compared with the suggestions in `STATISTIC + ML_DL.md`: every proposed group contributes positively. Promotion / discount depth, post-promotion and day of week matter most; month and holidays have small effects. The selected model (L = 56, hidden 128) relies much more on promotion and post-promotion features than the earlier configuration (L = 28, hidden 64: +0.054 and +0.007), probably because a 56-day window contains entire previous promotions.

---

## 8. Discount depth simulation (for Insight 3)

The simulation runs on the 24,468 discounted promotion days in the test set. Each run changes only `depth` (and `price_ratio = 1 − depth`) and keeps every other feature fixed. Revenue = forecast quantity × regular price × (1 − depth). Index = 100 at a 20% discount (the observed median).

| Discount | 5% | 10% | 15% | **20%** | 25% | 30% | 35% | 40% |
|---|---|---|---|---|---|---|---|---|
| Quantity – Food | 89 | 92 | 96 | 100 | 104 | 108 | 113 | 118 |
| Quantity – Household | 84 | 89 | 94 | 100 | 106 | 112 | 119 | 126 |
| Revenue – Food | **105** | 104 | 102 | 100 | 98 | 95 | 92 | 89 |
| Revenue – Household | 98 | 99 | 100 | **100** | 100 | 99 | 98 | 96 |

- The model has learned that **deeper discounts increase quantity**, but not enough to offset the lower price. Implied elasticity is about −0.6 (Food) and −0.9 (Household). **Revenue on promotion days does not increase beyond a discount of about 20%.**
- **Read with caution:**
  1. The LSTM's elasticities are lower than the promotion-level estimates in `bao_cao_feature.md` (−1.34 / −0.83), and the ordering of the two groups is reversed. A forecasting model is not designed to measure causal effects: discount depth is chosen by the stores (selection bias), and the effect may be "shared" with other features (promotion type, sales history during earlier promotions).
  2. The simulation covers **promotion days only**, not the post-promotion dip or carry-over.
  3. There is no cost data, so conclusions are about revenue, not profit.
  4. Do not extrapolate outside 5–40%.
  5. The LSTM under-forecasts (Section 6), so absolute levels are too low; only the relative index should be read.
- **Conclusion for Insight 3:** the LSTM supports the view that "the revenue-optimal discount is **not** the deepest one". The specific optimal level should come from the promotion-level elasticity model combined with `incr_rev`, with the LSTM used as a cross-check.

---

## 9. Limitations and next steps

1. **Negative bias** (Section 6): try a Poisson / Tweedie loss or quantile forecasts before using the model for inventory decisions.
2. **Single hold-out, single seed:** the validation WAPE gap among the top configurations is only about 0.002, likely smaller than seed-to-seed variation. The ranking within the top group should not be over-interpreted.
3. **Limited grid:** dropout, number of layers and batch size were fixed. The loss was fixed for the reason given in Section 4.2.
4. **Summer test period** (Jul–Sep 2024) with no major holidays: forecasting around Tet / March 8 has not been evaluated.
5. **Online channel not used yet** (lagged `online.quantity` for stores 1 and 4), and TFT has not been tried as planned.

---

## 10. Reproduction

```
pip install torch                           # GPU if available; CPU works but is much slower
python src/train_lstm.py                    # ~90 min on Kaggle (T4 + 4 CPUs); the LSTM grid alone takes ~10 min
python src/train_lstm.py --importance-only  # recompute Section 7 from the saved model only
python src/train_lstm.py --smoke            # quick test run, about 1–2 min
```

Optional environment variables: `PANEL_PATH` (panel path), `OUT_DIR` (output folder). By default the script reads and writes inside the project. GPU and CPU results may differ in the 4th decimal (smoke run: 0.5879 on GPU vs 0.5881 on CPU).

| File in `output/lstm/` | Contents |
|---|---|
| `lstm_best.pt` | Selected LSTM weights, hyperparameters and feature lists |
| `variant_selection.csv`, `training_log.csv` | LSTM grid search: per-configuration results, per-epoch log |
| `metrics_*.csv`, `metrics.txt` | WAPE / MAE / RMSE / bias for each breakdown |
| `test_predictions.parquet` | Row-level test forecasts from the LSTM, MA7 and seasonal naive |
| `feature_group_importance.csv` | Feature group importance (Section 7) |
| `depth_simulation.csv` | Discount depth simulation (Section 8) |
| `run.log`, `kaggle_kernel.log`, `summary.json` | Full logs and configuration summary |

*Note:* the script also trains a HistGradientBoosting tree model as an internal sanity check (results in `hgb_*.csv` and the `hgb*` columns of `metrics_*.csv`). It is not included in this report because tree-based models are covered by the team's LightGBM branch.

Earlier runs: `output/lstm_v1_backup/` (loss × lookback grid, 20 epochs), `output/lstm_v2_backup/` (40 epochs).
