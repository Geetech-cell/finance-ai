import joblib
from pathlib import Path
import pandas as pd
import numpy as np


MODEL_DIR = Path("models/forecasting_by_category")


def list_available_category_models():
    if not MODEL_DIR.exists():
        return []

    available = []
    for f in MODEL_DIR.glob("*_forecast.pkl"):
        try:
            obj = joblib.load(f)
            if isinstance(obj, dict) and obj.get("category"):
                available.append(str(obj["category"]))
            else:
                available.append(f.stem.replace("_forecast", ""))
        except Exception:
            available.append(f.stem.replace("_forecast", ""))
    return sorted(set(available))


def load_category_forecast_model(category: str):
    def _safe_category_filename(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
        safe = safe.strip("_")
        return safe or "category"

    candidates = [
        MODEL_DIR / f"{category}_forecast.pkl",
        MODEL_DIR / f"{_safe_category_filename(category)}_forecast.pkl",
    ]

    for path in candidates:
        if path.exists():
            return joblib.load(path)

    # Last resort: scan all models for matching metadata
    for path in MODEL_DIR.glob("*_forecast.pkl"):
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and str(obj.get("category")) == str(category):
                return obj
        except Exception:
            continue

    available = list_available_category_models()
    raise FileNotFoundError(
        f"No forecast model found for category '{category}'. "
        f"Available models: {available}"
    )


def forecast_category_spending(category: str, days: int = 30) -> pd.DataFrame:
    model = load_category_forecast_model(category)

    if isinstance(model, dict) and model.get("model_type") == "moving_average_v1":
        start = pd.to_datetime(model.get("train_ds_max"))
        if pd.isna(start):
            start = pd.Timestamp.today().normalize()

        ds = pd.date_range(start=start, periods=days + 1, freq="D")
        yhat = float(model.get("yhat", 0.0))
        sigma = float(model.get("sigma", 0.0))
        z = 1.96

        result = pd.DataFrame({
            "ds": ds,
            "yhat": yhat,
            "yhat_lower": yhat - z * sigma,
            "yhat_upper": yhat + z * sigma,
        })
        result["category"] = category
        return result

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["category"] = category

    return result
