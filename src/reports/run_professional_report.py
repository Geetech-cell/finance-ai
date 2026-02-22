import pandas as pd

from src.inference.forecast_expenses import load_forecast_model, forecast_future, extract_forecast_summary
from src.inference.forecast_by_category import list_available_category_models, forecast_category_spending
from src.reports.professional_report import generate_professional_forecast_report


def run(days: int = 30):

    # TOTAL forecast
    total_model = load_forecast_model()
    total_forecast = forecast_future(days=days, model=total_model)
    total_summary = extract_forecast_summary(total_forecast)

    # CATEGORY forecasts
    categories = list_available_category_models()
    category_forecasts_dict = {}
    category_summary = []

    for cat in categories:
        fc = forecast_category_spending(cat, days=days)
        next_days = fc.tail(days)

        total_spend = next_days["yhat"].sum()

        category_summary.append({
            "category": cat,
            "predicted_spend": total_spend
        })

        category_forecasts_dict[cat] = fc

    category_summary_df = pd.DataFrame(category_summary)

    # ANOMALY placeholder (replace with real anomaly data if available)
    anomaly_df = pd.DataFrame()

    generate_professional_forecast_report(
        total_forecast_df=total_summary,
        category_summary_df=category_summary_df,
        category_forecasts_dict=category_forecasts_dict,
        anomaly_df=anomaly_df,
        days=days,
        output_file=f"professional_forecast_report_{days}_days.pdf"
    )


if __name__ == "__main__":
    run(days=30)
