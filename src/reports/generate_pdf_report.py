from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def plot_forecast_chart(forecast_df: pd.DataFrame, output_path: Path):
    """
    Save forecast chart image for PDF embedding.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast")
    ax.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.2,
        label="Confidence Interval"
    )

    ax.set_title("Total Spending Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Spend")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_forecast_pdf_report(
    total_forecast_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    days: int = 30,
    output_file: str = "forecast_report.pdf"
):
    """
    Generate a PDF report containing:
    - Total forecast chart
    - Total spending metrics
    - Top categories table
    - Risk category
    """

    output_dir = Path("reports/monthly_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / output_file

    # Chart image path
    chart_dir = Path("reports/figures")
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / "forecast_chart.png"

    plot_forecast_chart(total_forecast_df, chart_path)

    # Compute total forecast stats
    next_period = total_forecast_df.tail(days)

    predicted_total = next_period["yhat"].sum()
    lower_total = next_period["yhat_lower"].sum()
    upper_total = next_period["yhat_upper"].sum()

    # Risk category
    risk_category = "N/A"
    risk_amount = 0

    if not category_summary_df.empty:
        top_row = category_summary_df.sort_values("predicted_spend", ascending=False).iloc[0]
        risk_category = top_row["category"]
        risk_amount = top_row["predicted_spend"]

    # PDF setup
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("AI Finance Assistant - Forecast Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph(f"<b>Forecast Period:</b> Next {days} days", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Expected Total Spend:</b> {predicted_total:,.2f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Lower Estimate:</b> {lower_total:,.2f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Upper Estimate:</b> {upper_total:,.2f}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    # Risk category warning
    elements.append(Paragraph(
        f"<b>⚠ Risk Category:</b> {risk_category.upper()} (Expected: {risk_amount:,.2f})",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # Chart
    elements.append(Paragraph("Total Spending Forecast Chart", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Image(str(chart_path), width=500, height=250))
    elements.append(Spacer(1, 20))

    # Top categories table
    elements.append(Paragraph("Top Forecasted Categories", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    if category_summary_df.empty:
        elements.append(Paragraph("No category forecast data available.", styles["Normal"]))
    else:
        top_categories = category_summary_df.sort_values("predicted_spend", ascending=False).head(10)

        table_data = [["Category", "Predicted Spend"]]
        for _, row in top_categories.iterrows():
            table_data.append([row["category"], f"{row['predicted_spend']:,.2f}"])

        table = Table(table_data, hAlign="LEFT")

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)

    elements.append(Spacer(1, 20))

    # Footer note
    elements.append(Paragraph(
        "Generated automatically by AI Finance Assistant.",
        styles["Italic"]
    ))

    doc.build(elements)

    print(f"✅ PDF report generated: {pdf_path}")
    return pdf_path
