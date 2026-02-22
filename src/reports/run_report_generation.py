import pandas as pd
from pathlib import Path

from src.inference.forecast_expenses import load_forecast_model, forecast_future, extract_forecast_summary
from src.inference.forecast_by_category import list_available_category_models, forecast_category_spending
from src.reports.generate_pdf_report import generate_forecast_pdf_report


def run(days: int = 30):
    # TOTAL forecast
    total_model = load_forecast_model()
    total_forecast = forecast_future(days=days, model=total_model)
    total_summary = extract_forecast_summary(total_forecast)

    # CATEGORY summary
    categories = list_available_category_models()
    category_forecasts = []

    for cat in categories:
        try:
            fc = forecast_category_spending(cat, days=days)
            next_days = fc.tail(days)
            total_cat_spend = next_days["yhat"].sum()

            category_forecasts.append({
                "category": cat,
                "predicted_spend": total_cat_spend
            })
        except:
            continue

    category_df = pd.DataFrame(category_forecasts)

    # Generate PDF
    report_name = f"forecast_report_{days}_days.pdf"
    pdf_path = generate_forecast_pdf_report(
        total_forecast_df=total_summary,
        category_summary_df=category_df,
        days=days,
        output_file=report_name
    )

    return pdf_path


if __name__ == "__main__":
    run(days=30)
