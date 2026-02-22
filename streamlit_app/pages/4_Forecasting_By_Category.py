import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.inference.forecast_by_category import (
    list_available_category_models,
    forecast_category_spending
)


st.set_page_config(page_title="Forecast by Category", page_icon="📊")

st.title("📊 Forecast Spending by Category")
st.write("Select a category and forecast future spending using Prophet.")

categories = list_available_category_models()

if not categories:
    st.warning("⚠️ No category forecast models found. Train them first.")
    st.code("python -m src.training.train_forecasting_by_category")
else:
    selected_category = st.selectbox("Select Category", categories)
    days = st.slider("Forecast how many days ahead?", 7, 180, 30)

    if st.button("Generate Category Forecast"):
        try:
            forecast_df = forecast_category_spending(selected_category, days=days)

            st.subheader("📄 Forecast Table (Last 20 Days)")
            st.dataframe(forecast_df.tail(20))

            st.subheader("📈 Forecast Plot")

            fig, ax = plt.subplots()
            ax.plot(forecast_df["ds"], forecast_df["yhat"])
            ax.fill_between(
                forecast_df["ds"],
                forecast_df["yhat_lower"],
                forecast_df["yhat_upper"],
                alpha=0.2
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Spending")
            ax.set_title(f"Forecast for Category: {selected_category}")

            st.pyplot(fig)

            # Save forecast
            output_dir = Path("data/processed/category_forecasts")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{selected_category}_forecast.csv"
            forecast_df.to_csv(output_path, index=False)

            st.success(f"Forecast saved to `{output_path}`")

        except Exception as e:
            st.error("❌ Error generating forecast")
            st.exception(e)
