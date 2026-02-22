import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.detect_anomaly import load_anomaly_model, detect_anomalies


st.set_page_config(page_title="Anomaly Detection", page_icon="🚨")

st.title("🚨 Anomaly Detection")
st.write("This module detects unusual transactions using Isolation Forest.")

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

    st.subheader("📄 Categorized Transactions Preview")
    st.dataframe(df.head(20))

    if st.button("Detect Anomalies"):
        try:
            try:
                model, scaler = load_anomaly_model()
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Train or provide the anomaly model artifacts in the `models/` folder, then retry.")
                st.stop()

            feature_df = df.select_dtypes(include=[np.number]).copy()
            if feature_df.shape[1] == 0:
                raise ValueError("No numeric columns found to run anomaly detection on.")

            anomalies, reconstruction_errors, threshold = detect_anomalies(
                feature_df,
                model,
                scaler,
                threshold_percentile=95,
            )

            st.caption(f"Threshold used: {threshold:.6f}")

            results_df = df.copy()
            results_df['is_anomaly'] = anomalies
            results_df['reconstruction_error'] = reconstruction_errors
            results_df['anomaly_label'] = np.where(results_df['is_anomaly'], 'anomaly', 'normal')

            st.subheader("✅ Full Results")
            st.dataframe(results_df.head(50))

            anomalies_df = results_df[results_df["anomaly_label"] == "anomaly"]

            st.subheader("🚨 Suspicious Transactions")
            st.dataframe(anomalies_df)

            st.write(f"Total Transactions: **{len(results_df)}**")
            st.write(f"Suspicious Transactions: **{len(anomalies_df)}**")

            # Save results
            output_path = Path("data/processed/anomaly_transactions.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)

            suspicious_path = Path("data/processed/suspicious_transactions.csv")
            suspicious_path.parent.mkdir(parents=True, exist_ok=True)
            anomalies_df.to_csv(suspicious_path, index=False)

            st.success(f"Saved full results to `{output_path}`")
            st.success(f"Saved suspicious transactions to `{suspicious_path}`")

        except Exception as e:
            st.error("❌ Error running anomaly detection")
            st.exception(e)
