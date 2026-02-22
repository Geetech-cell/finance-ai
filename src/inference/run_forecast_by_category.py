from pathlib import Path
from src.inference.forecast_by_category import forecast_category_spending


def run(category: str = "food", days: int = 30):
    forecast_df = forecast_category_spending(category=category, days=days)

    output_dir = Path("data/processed/category_forecasts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{category}_forecast.csv"
    forecast_df.to_csv(output_path, index=False)

    print(f"📈 Forecast saved to: {output_path}")


if __name__ == "__main__":
    run(category="food", days=30)
