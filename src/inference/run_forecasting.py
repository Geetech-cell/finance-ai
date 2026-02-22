from pathlib import Path
import pandas as pd

from src.inference.forecast_expenses import load_forecast_model, forecast_future, extract_forecast_summary


def run(days: int = 30):
    model = load_forecast_model()

    forecast = forecast_future(days=days, model=model)
    summary = extract_forecast_summary(forecast)

    output_path = Path("data/processed/forecast_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path, index=False)

    print(f"📈 Forecast results saved to: {output_path}")


if __name__ == "__main__":
    run(days=30)
