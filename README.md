# Malawi Inflation Forecasting Lab

A  project comparing four ways of forecasting Malawi's monthly inflation one month ahead:

1. **Seasonal naive**: this month's forecast is the value from the same month last year (the baseline to beat)
2. **SARIMA**: a classic statistical time-series model with a yearly seasonal pattern
3. **XGBoost**: a machine-learning model fed lagged values, rolling statistics and calendar months
4. **Hybrid**: SARIMA forecast plus an XGBoost forecast of SARIMA's errors

The three series forecast are headline inflation, food inflation and non-food inflation, using data from the Reserve Bank of Malawi (RBM).

This is a learning project, not a published study, so treat the results as exploratory.

## Project structure

```
forecasting/
├─ data/
│   ├─ Malawi_rbm_data.csv                      # raw RBM data (long format: date, indicator, value)
│   ├─ cleaned_malawi_inflation_2003_2026.csv   # earlier cleaned series, kept for reference
│   └─ cleaned_inflation_prepped.csv            # wide monthly dataset (created by the notebook / prepare_data.py)
├─ notebooks/
│   ├─ 01_data_inspection.ipynb                 # checks on the raw files
│   └─ 01_data.ipynb                            # main notebook: preparation, EDA, all four models, evaluation
├─ outputs/
│   ├─ forecasts/                               # one CSV of forecasts per model
│   ├─ tables/                                  # model_comparison.csv
│   └─ figures/                                 # PNG and interactive HTML (Plotly) charts
├─ prepare_data.py                              # script version of the data-preparation step
├─ requirements.txt
└─ README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

The notebook needs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `scikit-learn`, `xgboost` and `nbformat` (Plotly needs it to display charts in Jupyter).

If a package is "not found" inside the notebook even though it's installed, run `%pip install <package>` from a notebook cell and restart the kernel, so it installs into the kernel's own environment.

## How to run

Open `notebooks/01_data.ipynb` and **run the cells in order**, because each cell depends on the ones before it.

| Cell | What it does |
|---|---|
| 1 | Imports, helper functions, creates the `outputs/` folders |
| 2 | Loads the raw RBM data |
| 3 | Pivots to wide format, fills in a complete monthly index, adds the CPI rebase flag, saves `cleaned_inflation_prepped.csv` |
| 4 | Exploratory analysis: line plots, seasonal box plots, histograms, rolling statistics, ACF/PACF, ADF and KPSS stationarity tests, high-inflation episodes (>15%) |
| 5 | Defines the evaluation window and produces seasonal naive forecasts |
| 6 | SARIMA forecasts |
| 7 | XGBoost forecasts |
| 8 | Hybrid SARIMA + XGBoost forecasts |
| 9 | Compares all models: metrics table, RMSE bar chart, forecast-vs-actual charts |

`prepare_data.py` does the same job as cell 3 from the command line (run it from the repository root: `python prepare_data.py`). It also prints duplicate rows, missing values per indicator and the years where a CPI rebase was flagged.

**Runtime:** the SARIMA cell is the slow one. It runs a grid search over 144 model configurations per series, then refits the model for every evaluation month. On a low-spec laptop (4 GB RAM) expect it to take a while, and don't worry that it prints nothing during the refits. The other cells finish in minutes or less. All results are saved to CSV, so you only need to run the slow cell once.

## Method

**Evaluation scheme.** Expanding-window, one-step-ahead forecasts.
- Estimation period: Jan 2003 to Dec 2022
- Evaluation period: Jan 2023 to Oct 2026
- For each month in the evaluation period, models only see data up to the previous month.

**SARIMA.** The order `(p,d,q)(P,D,Q,12)` is chosen once per series by lowest AIC, using data up to Dec 2022 only. The model is then refit with that fixed order at every evaluation month. Interior missing months are left as gaps and handled by `statsmodels`. Series with fewer than 36 valid observations are skipped.

**XGBoost.** Refit every month on:
- the previous 12 values of the series
- 12-month rolling mean and standard deviation (shifted by one month, so the month being forecast is never included)
- month-of-year dummies
- the CPI rebase flag (also shifted by one month)

**Hybrid.** SARIMA's one-step errors are modelled with XGBoost using the same features. The history for this model comes from SARIMA's in-sample residuals before 2023, then its real out-of-sample errors after that. Final forecast = SARIMA forecast + predicted error. Where the error model has too little data to forecast, the hybrid falls back to the plain SARIMA forecast.

**Scoring.** Every model is scored on the same months, meaning only months where the actual value and all four forecasts exist. The notebook prints how many months that is for each series.

Metrics:
- RMSE and MAE (lower is better)
- MAPE (lower is better; unreliable when inflation is close to zero)
- Directional accuracy: how often the forecast gets the direction of change from last month's actual value right (higher is better)
- Bias: average forecast minus actual (closer to 0 is better; positive means over-forecasting)

## Results
 
RMSE on the evaluation period (Jan 2023 to Oct 2026, one-step-ahead, lower is better). Values are approximate, read from the RMSE chart; the exact figures are in `outputs/tables/model_comparison.csv`.
 
| Series | Seasonal naive | SARIMA | XGBoost | Hybrid |
|---|---|---|---|---|
| Headline inflation | ~6.8 | ~1.8 | ~2.4 | ~1.9 |
| Food inflation | ~11.5 | ~2.6 | ~3.4 | ~2.7 |
| Non-food inflation | ~7.0 | ~1.3 | ~1.9 | ~1.35 |
 
What stands out:
 
- **All three real models beat the seasonal naive baseline by a wide margin.** Their errors are roughly 3 to 4 times smaller.
- **SARIMA is the best or tied for best on every series.** XGBoost on its own is the weakest of the three.
- **The hybrid does not improve on SARIMA.** It is about the same, so the XGBoost model of SARIMA's errors found little nonlinear structure to add. With only about 46 evaluation months, the small gaps between SARIMA and the hybrid shouldn't be read as one being better.
- **Food inflation is the hardest series to forecast** for every model.
These results are exploratory. Seasonal naive is a weak baseline for a series as persistent as inflation, so a fairer test is comparing against simply using last month's value.

## Data notes and limitations

- **Missing months:** a few months are missing for each inflation series. They are kept as `NaN`, not filled in.
- **CPI rebasing:** large jumps in CPI (more than 15% month to month) are flagged as `cpi_rebase_flag`. Check the output of `prepare_data.py` to see which years were flagged in your copy of the data.
- **One fixed set of XGBoost settings:** the hyperparameters aren't tuned.
- **XGBoost and levels:** tree models can't predict values beyond the range they were trained on, so XGBoost can underpredict if inflation moves above its historical range. Look at the Bias column for signs of this.
- **Hybrid residuals:** the in-sample residuals used to start the hybrid are slightly smaller than true out-of-sample errors, which is standard for this kind of hybrid but worth knowing.
- **Short evaluation window:** about 46 months per series, so small differences in RMSE between models may not mean much.

## Data source

The data was compiled from Reserve Bank of Malawi (RBM) statistics through
[AfriQuant](https://afriquant-app.vercel.app), a Malawian macroeconomic data
platform we're building. The original figures belong to the RBM; this repo uses
them for educational and exploratory purposes only.

`data/Malawi_rbm_data.csv` is in long format (`date`, `indicator`, `value`).
Check RBM's own publications for official figures before relying on anything
here.

## Ideas for later

- Forecasting the monthly change in inflation instead of the level
- Multi-step horizons (3, 6 and 12 months ahead)
- Prediction intervals
- Exogenous variables such as exchange rates and policy rates
- Residual diagnostics and a separate score for high-inflation episodes
- Splitting the main notebook into one notebook per stage and moving the helpers into a `src/utils.py`
