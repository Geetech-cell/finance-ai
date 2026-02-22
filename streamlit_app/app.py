"""
Main Streamlit App - AI Finance Assistant
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from api_client import get_api_client

st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Finance Assistant")
st.markdown("### Your Personal Financial Intelligence Platform")

# API Connection Status
api_client = get_api_client()
health = api_client.health_check()

if "error" in health:
    st.error("🔴 Cannot connect to backend API")
    st.info("Make sure the API service is running on port 8000")
else:
    st.success("🟢 Connected to backend API")

st.markdown("""
Welcome! This app helps you:
- 📤 Upload and manage financial data
- 📊 Analyze spending patterns
- 🤖 Get AI-powered insights
- 💡 Receive personalized recommendations
""")

st.info("👈 Use the sidebar to navigate between pages")

# Quick stats from API
if "error" not in health:
    try:
        transactions = api_client.get_transactions()
        users = api_client.get_users()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(transactions))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            if transactions:
                total_amount = sum(t.get("amount", 0) for t in transactions)
                st.metric("Total Amount", f"${total_amount:,.2f}")
            else:
                st.metric("Total Amount", "$0")
    except Exception as e:
        st.warning(f"Could not load stats: {str(e)}")
        # Fallback placeholder
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", "0")
        with col2:
            st.metric("This Month", "$0")
        with col3:
            st.metric("Categories", "0")