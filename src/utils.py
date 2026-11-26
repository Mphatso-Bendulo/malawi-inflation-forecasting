"""
Utility functions shared across notebooks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_prepared_data(path: str = "../data/cleaned_inflation_prepped.csv") -> pd.DataFrame:
    """Load the wide‑format, pre‑processed dataset."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def expanding_window_split(
    series: pd.Series, train_end: pd.Timestamp, forecast_horizon: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """Return training slice up to `train_end` (inclusive) and the next `forecast_horizon` observations."""
    train = series.loc[:train_end]
    test = series.loc[train_end + pd.offsets.MonthBegin(): train_end + pd.DateOffset(months=forecast_horizon)]
    return train, test


def rolling_forecast(
    series: pd.Series,
    model_func,
    train_end_dates: List[pd.Timestamp],
    **model_kwargs,
) -> pd.DataFrame:
    """Generic expanding‑window forecast driver.
    `model_func` must accept a training series and return a fitted model with a `.forecast(steps)` method.
    Returns a DataFrame with columns `forecast` and `actual`.
    """
    forecasts = []
    actuals = []
    for end_date in train_end_dates:
        train, test = expanding_window_split(series, end_date)
        model = model_func(train, **model_kwargs)
        pred = model.forecast(1).iloc[0]
        forecasts.append(pred)
        actuals.append(test.iloc[0] if not test.empty else np.nan)
    return pd.DataFrame({"forecast": forecasts, "actual": actuals}, index=train_end_dates)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, MAPE, directional accuracy, and bias. NaNs are ignored."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    safe_true = np.where(y_true == 0, np.nan, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_true)) * 100
    true_change = np.diff(y_true, prepend=y_true[0])
    pred_change = np.diff(y_pred, prepend=y_pred[0])
    dir_acc = np.mean(np.sign(true_change) == np.sign(pred_change)) * 100
    bias = np.mean(y_pred - y_true)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "DirectionalAccuracy": dir_acc, "Bias": bias}
