# Retail Promotion Analysis and Demand Forecasting

**STAT3013.R11.CTTT — Group R11**

This project analyzes how retail sales change before, during, and after promotions across Food and Household products. It also compares five models for forecasting daily unit sales over a seven-day horizon: DLinear, NLinear-X, XGBoost, LightGBM, and LSTM.

## Dataset and Demo Video

| Resource | Direct link |
| --- | --- |
| Processed dataset (`daily_panel.parquet`) | https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data/discussion?sort=recent-comments&fbclid=IwY2xjawV3TlNleHRuA2FlbQIxMQBwZG9mA3NydGMGYXBwX2lkATAAAR8ttYAdJBtjzniQs0JIKMvuQNtkpb7IJGa_QyducDUaInx_ZwIjZfMKMsn8lw_aem_iL3kMF6cOBQ0fFNBcHg7Ng |
| 15-minute project demo video | `REPLACE_WITH_DIRECT_DEMO_VIDEO_URL` |

The processed dataset contains daily item–store observations, including units sold, product category, promotion phase, discount depth, and the train, validation, and test split. Use the dataset to rerun model training and evaluation.

## Environment Requirements

| Workflow | Requirements |
| --- | --- |
| Compare reported model results | Git, Python 3, JupyterLab, NumPy, pandas, Matplotlib, and internet access to read the published CSV files from GitHub. The comparison notebook records Python 3.13.18. A GPU is unnecessary for this workflow. |
| Retrain individual models | Python 3 and the dependencies imported by the selected model branch. The forecasting workflows use packages such as PyArrow for Parquet input, PyTorch for the neural models, and XGBoost or LightGBM for the corresponding tree models. |

## Setup and Execution

### Reproduce the model comparison

```bash
git clone --branch CompareModel https://github.com/tran04012006/4_STAT3013_R11.git
cd 4_STAT3013_R11
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install jupyterlab numpy pandas matplotlib
python -m jupyterlab compare-models.ipynb
```

In JupyterLab, select **Run → Run All Cells**. The notebook loads published metrics for NLinear-X, LightGBM, and LSTM from GitHub and uses reported values included in the notebook for DLinear and XGBoost. It writes comparison tables and figures to `comparison_figures/`. This workflow uses the reported results and does not require downloading `daily_panel.parquet`.

### Rerun an individual model

1. Download `daily_panel.parquet` from the dataset link above.
2. Switch to the branch for the model you want to run.
3. Open the model notebook, install the dependencies used by that notebook, and set its input path to the downloaded Parquet file. In Kaggle, attach the dataset as a notebook input and use its mounted path.
4. Run the model notebook from top to bottom to prepare the data, train the model, and evaluate it on the test split.

| Workflow | Git branch |
| --- | --- |
| DLinear | [`dlinear-forecasting`](https://github.com/tran04012006/4_STAT3013_R11/tree/dlinear-forecasting) |
| NLinear-X | [`NLinear-X_forecasting`](https://github.com/tran04012006/4_STAT3013_R11/tree/NLinear-X_forecasting) |
| XGBoost | [`XGBoost`](https://github.com/tran04012006/4_STAT3013_R11/tree/XGBoost) |
| LightGBM | [`Hoang_LightGBM_GridSearch`](https://github.com/tran04012006/4_STAT3013_R11/tree/Hoang_LightGBM_GridSearch) |
| LSTM | [`LSTM_GridSearch`](https://github.com/tran04012006/4_STAT3013_R11/tree/LSTM_GridSearch) |
| Model comparison | [`CompareModel`](https://github.com/tran04012006/4_STAT3013_R11/tree/CompareModel) |

## License

No software license is currently declared on the `main` branch. Dataset access and reuse are subject to the dataset provider’s terms.
