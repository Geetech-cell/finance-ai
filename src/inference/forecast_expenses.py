import joblib
import pandas as pd
from pathlib import Path
import numpy as np


MODEL_PATH = "models/forecast_model.pkl"


def load_forecast_model(model_path: str = MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Forecast model not found at {model_path}. Train it first."
        )
    return joblib.load(model_path)


def forecast_future(days: int = 30, model=None) -> pd.DataFrame:
    """
    Forecast spending for the next N days.
    Returns Prophet forecast dataframe.
    """
    if model is None:
        model = load_forecast_model()

    if isinstance(model, dict) and model.get("model_type") == "moving_average_v1":
        start = pd.to_datetime(model.get("train_ds_max"))
        if pd.isna(start):
            start = pd.Timestamp.today().normalize()

        ds = pd.date_range(start=start, periods=days + 1, freq="D")
        yhat = float(model.get("yhat", 0.0))
        sigma = float(model.get("sigma", 0.0))
        z = 1.96

        forecast = pd.DataFrame({
            "ds": ds,
            "yhat": yhat,
            "yhat_lower": yhat - z * sigma,
            "yhat_upper": yhat + z * sigma,
        })
        return forecast

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast


def extract_forecast_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful columns from Prophet output.
    """
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
