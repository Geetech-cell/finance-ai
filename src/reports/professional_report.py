from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


def save_total_forecast_chart(forecast_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(forecast_df["ds"], forecast_df["yhat"])
    ax.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.2
    )

    ax.set_title("Total Spending Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Spend")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_category_chart(category_name: str, forecast_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(forecast_df["ds"], forecast_df["yhat"])
    ax.set_title(f"{category_name} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spend")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_professional_forecast_report(
    total_forecast_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    category_forecasts_dict: dict,
    anomaly_df: pd.DataFrame = None,
    days: int = 30,
    output_file: str = "professional_forecast_report.pdf"
):
    output_dir = Path("reports/monthly_reports")
    figures_dir = Path("reports/figures")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / output_file

    # Prepare charts
    total_chart_path = figures_dir / "total_forecast.png"
    save_total_forecast_chart(total_forecast_df, total_chart_path)

    # Compute metrics
    next_period = total_forecast_df.tail(days)
    predicted_total = next_period["yhat"].sum()

    risk_category = "N/A"
    if not category_summary_df.empty:
        risk_category = (
            category_summary_df
            .sort_values("predicted_spend", ascending=False)
            .iloc[0]["category"]
        )

    # Create document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # --------------------------------
    # COVER PAGE
    # --------------------------------
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        elements.append(Image(str(logo_path), width=2 * inch, height=2 * inch))
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Finance Assistant", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Executive Forecast Report", styles["Heading2"]))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"Forecast Horizon: {days} Days", styles["Normal"]))
    elements.append(Spacer(1, 100))

    elements.append(Paragraph("Confidential Financial Analytics Report", styles["Italic"]))
    elements.append(PageBreak())

    # --------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------
    elements.append(Paragraph("Executive Summary", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    summary_text = f"""
    The projected total spending for the next {days} days is {predicted_total:,.2f}.
    The highest forecasted spending category is '{risk_category.upper()}'.
    Monitoring this category is recommended to control financial risk.
    """

    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # --------------------------------
    # TOTAL FORECAST SECTION
    # --------------------------------
    elements.append(Paragraph("Total Spending Forecast", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Image(str(total_chart_path), width=6 * inch, height=3 * inch))
    elements.append(PageBreak())

    # --------------------------------
    # CATEGORY SECTION
    # --------------------------------
    elements.append(Paragraph("Category Forecast Breakdown", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for category, forecast_df in category_forecasts_dict.items():
        chart_path = figures_dir / f"{category}_chart.png"
        save_category_chart(category, forecast_df, chart_path)

        elements.append(Paragraph(category.upper(), styles["Heading2"]))
        elements.append(Spacer(1, 8))
        elements.append(Image(str(chart_path), width=5 * inch, height=2.5 * inch))
        elements.append(Spacer(1, 20))

    elements.append(PageBreak())

    # --------------------------------
    # ANOMALY SECTION
    # --------------------------------
    elements.append(Paragraph("Anomaly Detection Summary", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    if anomaly_df is not None and not anomaly_df.empty:
        elements.append(Paragraph(
            f"{len(anomaly_df)} anomalies were detected in recent transactions.",
            styles["Normal"]
        ))
    else:
        elements.append(Paragraph(
            "No significant anomalies detected.",
            styles["Normal"]
        ))

    elements.append(Spacer(1, 30))

    # --------------------------------
    # AI INSIGHT SECTION
    # --------------------------------
    elements.append(Paragraph("AI Insight Summary", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    ai_summary = """
    Based on the predictive modeling and anomaly detection,
    spending patterns show seasonal and category-based fluctuations.
    Strategic monitoring and budgeting adjustments are advised
    for categories exhibiting growth trends.
    """

    elements.append(Paragraph(ai_summary, styles["Normal"]))

    doc.build(elements)

    print(f"✅ Professional report generated: {pdf_path}")
    return pdf_path
