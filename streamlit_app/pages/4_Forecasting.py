import streamlit as st
import pandas as pd
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from src.inference.forecast_expenses import (
    load_forecast_model,
    forecast_future,
    extract_forecast_summary
)


st.set_page_config(page_title="Forecasting", page_icon="📈")

st.title("📈 Spending Forecasting")
st.write("Predict future spending trends using Prophet Time Series Forecasting.")

sys.path.append(str(Path(__file__).parent.parent.parent))

default_input_path = Path("data/processed/categorized_transactions.csv")
results_dir = Path("results")

df = None
data_source = None

if 'categorized_df' in st.session_state and isinstance(st.session_state.categorized_df, pd.DataFrame):
    df = st.session_state.categorized_df.copy()
    data_source = "Streamlit session (Expense Categorization page)"
elif default_input_path.exists():
    df = pd.read_csv(default_input_path)
    data_source = str(default_input_path)
else:
    exported = sorted(results_dir.glob("categorized_transactions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if exported:
        df = pd.read_csv(exported[0])
        data_source = str(exported[0])

if df is None:
    st.warning("⚠️ categorized transactions not found.")
    st.info("Run the Expense Categorization page and export/save results, or upload a categorized CSV here.")
    uploaded = st.file_uploader("Upload categorized transactions CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = "Uploaded file"

if df is not None:
    st.caption(f"Data source: {data_source}")
    days = st.slider("Forecast how many days ahead?", 7, 180, 30)

    if st.button("Generate Forecast"):
        try:
            model = load_forecast_model()
            forecast_df = forecast_future(days=days, model=model)
            summary_df = extract_forecast_summary(forecast_df)

            st.subheader("📊 Forecast Table")
            st.dataframe(summary_df.tail(20))

            st.subheader("📈 Forecast Plot")

            fig, ax = plt.subplots()
            ax.plot(summary_df["ds"], summary_df["yhat"])
            ax.fill_between(
                summary_df["ds"],
                summary_df["yhat_lower"],
                summary_df["yhat_upper"],
                alpha=0.2
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Spending")
            ax.set_title("Forecasted Spending Trend")
            st.pyplot(fig)

            # Save results
            output_path = Path("data/processed/forecast_results.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_path, index=False)
            st.success(f"Forecast saved to `{output_path}`")

        except Exception as e:
            st.error("❌ Error generating forecast")
            st.exception(e)
