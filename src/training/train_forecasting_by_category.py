import pandas as pd
import joblib
from pathlib import Path
import numpy as np

from src.features.forecast_features import prepare_forecast_data_by_category


def train_forecasting_models_by_category(
    dataset_path: str = "data/processed/categorized_transactions.csv",
    output_dir: str = "models/forecasting_by_category"
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
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. Also couldn't find any categorized exports in '{results_dir}'."
            )

    df = pd.read_csv(dataset_file)

    required_cols = {"date", "amount"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    if "predicted_category" not in df.columns:
        df = df.copy()
        df["predicted_category"] = "all"

    categories = df["predicted_category"].dropna().astype(str).unique().tolist()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trained_models = []

    def _safe_category_filename(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        safe = safe.strip("_")
        return safe or "category"

    for category in categories:
        prophet_df = prepare_forecast_data_by_category(df, category)

        # Prophet needs enough time points
        if len(prophet_df) < 30:
            print(f"⚠️ Skipping '{category}' (not enough data points: {len(prophet_df)})")
            continue

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
                "category": str(category),
                "train_ds_max": str(pd.to_datetime(prophet_df['ds']).max().date()),
                "window": int(window),
                "yhat": yhat,
                "sigma": sigma,
            }

        safe_category = _safe_category_filename(str(category))
        model_file = output_path / f"{safe_category}_forecast.pkl"
        joblib.dump(model_to_save, model_file)

        trained_models.append(str(category))
        print(f"✅ Trained forecast model for '{category}' saved at: {model_file}")

    print("\n🎉 Training Complete")
    print("Models trained for categories:", trained_models)

    return trained_models


if __name__ == "__main__":
    train_forecasting_models_by_category()
