import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.inference.forecast_expenses import load_forecast_model, forecast_future, extract_forecast_summary
from src.inference.forecast_by_category import list_available_category_models, forecast_category_spending


st.set_page_config(page_title="Forecast Dashboard", page_icon="📈")

st.title("📈 Forecast Dashboard (Total + Categories)")
st.write("A full spending forecast dashboard with category breakdown and growth insights.")

days = st.slider("Forecast how many days ahead?", 7, 180, 30)

# -------------------------------
# TOTAL SPENDING FORECAST SECTION
# -------------------------------
st.header("💰 Total Spending Forecast")

try:
    total_model = load_forecast_model()
    total_forecast = forecast_future(days=days, model=total_model)
    total_summary = extract_forecast_summary(total_forecast)

    next_period = total_summary.tail(days)

    predicted_total = next_period["yhat"].sum()
    lower_total = next_period["yhat_lower"].sum()
    upper_total = next_period["yhat_upper"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Spend", f"{predicted_total:,.2f}")
    col2.metric("Lower Estimate", f"{lower_total:,.2f}")
    col3.metric("Upper Estimate", f"{upper_total:,.2f}")

    fig, ax = plt.subplots()
    ax.plot(total_summary["ds"], total_summary["yhat"], label="Forecast")
    ax.fill_between(
        total_summary["ds"],
        total_summary["yhat_lower"],
        total_summary["yhat_upper"],
        alpha=0.2,
        label="Confidence Interval"
    )
    ax.set_title("Total Spending Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spending")
    st.pyplot(fig)

except Exception as e:
    st.error("❌ Total forecast model not found or error occurred.")
    st.exception(e)

# -------------------------------
# CATEGORY FORECAST SECTION
# -------------------------------
st.header("📊 Category Forecast Insights")

categories = list_available_category_models()

if not categories:
    st.warning("⚠️ No category forecast models found. Train them first:")
    st.code("python -m src.training.train_forecasting_by_category")
else:
    category_forecasts = []

    with st.spinner("Generating category forecasts..."):
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

    if category_df.empty:
        st.warning("⚠️ Could not generate forecasts. Ensure you have enough data per category.")
    else:
        category_df = category_df.sort_values("predicted_spend", ascending=False)

        st.subheader("🏆 Top Forecasted Spending Categories")
        st.dataframe(category_df.head(10))

        # Identify risk category (highest predicted spending)
        risk_category = category_df.iloc[0]["category"]
        risk_amount = category_df.iloc[0]["predicted_spend"]

        st.warning(f"🚨 Risk Category: **{risk_category.upper()}** (Expected Spend: {risk_amount:,.2f})")

        # Plot Top 5 Categories
        top5 = category_df.head(5)

        st.subheader("📈 Top 5 Categories Comparison Chart")
        fig2, ax2 = plt.subplots()
        ax2.bar(top5["category"], top5["predicted_spend"])
        ax2.set_title(f"Top 5 Category Forecast (Next {days} Days)")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Predicted Spending")
        st.pyplot(fig2)

        # Save dashboard summary results
        output_dir = Path("data/processed/dashboard_forecasts")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"forecast_dashboard_{days}_days.csv"
        category_df.to_csv(output_path, index=False)

        st.success(f"📁 Dashboard summary saved to `{output_path}`")
from src.reports.run_report_generation import run as generate_pdf

st.subheader("📄 Download Forecast PDF Report")

if st.button("Generate PDF Report"):
    with st.spinner("Generating PDF report..."):
        pdf_path = generate_pdf(days=days)

    st.success(f"✅ PDF Report saved at: {pdf_path}")


