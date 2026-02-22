import pandas as pd
import joblib
from pathlib import Path
import numpy as np

from src.features.forecast_features import prepare_forecast_data


def train_forecasting_model(
    dataset_path: str = "data/processed/categorized_transactions.csv",
    model_output_path: str = "models/forecast_model.pkl"
):
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        results_dir = Path("results")
        exported = sorted(
            results_dir.glob("categorized_transactions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if exported:
            dataset_file = exported[0]
        else:
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            days = 365
            dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
            base = rng.normal(loc=50, scale=15, size=days).clip(min=5)
            weekend_boost = np.where(dates.dayofweek >= 5, 1.25, 1.0)
            spends = base * weekend_boost
            df_synth = pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "amount": -np.round(spends, 2),
            })
            df_synth.to_csv(dataset_file, index=False)
            print(f"⚠️ No categorized dataset found; generated synthetic dataset at: {dataset_file}")

    df = pd.read_csv(dataset_file)

    required_cols = {"date", "amount"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    prophet_df = prepare_forecast_data(df)

    if len(prophet_df) < 30:
        raise ValueError("Need at least 30 days of data for forecasting.")

    try:
        from prophet import Prophet

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(prophet_df)
        model_to_save = model
    except Exception:
        y = prophet_df['y'].to_numpy(dtype=float)
        window = 7
        if len(y) < window:
            window = max(1, len(y))

        recent = y[-window:]
        yhat = float(np.mean(recent))
        sigma = float(np.std(recent, ddof=0))
        model_to_save = {
            "model_type": "moving_average_v1",
            "train_ds_max": str(pd.to_datetime(prophet_df['ds']).max().date()),
            "window": int(window),
            "yhat": yhat,
            "sigma": sigma,
        }

    # Save model
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_to_save, model_output_path)

    print(f"✅ Forecasting model saved at: {model_output_path}")
    return model_to_save


if __name__ == "__main__":
    train_forecasting_model()
