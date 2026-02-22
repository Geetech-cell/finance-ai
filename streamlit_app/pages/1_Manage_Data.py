"""
Data Management Page - Connect to Backend API
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import pandas as pd
from api_client import get_api_client

st.set_page_config(page_title="Manage Data", page_icon="📊", layout="wide")

st.title("📊 Manage Financial Data")
st.markdown("Add, view, and manage transactions and users")

api_client = get_api_client()

# Tabs for different operations
tab1, tab2, tab3 = st.tabs(["➕ Add Transaction", "👥 Users", "📋 Transactions"])

with tab1:
    st.header("Add New Transaction")
    
    # Get users for dropdown
    users = api_client.get_users()
    if users:
        user_options = {f"{user['email']} (ID: {user['id']})": user['id'] for user in users}
        selected_user = st.selectbox("Select User", options=list(user_options.keys()))
        user_id = user_options[selected_user]
    else:
        st.warning("No users found. Please create a user first.")
        user_id = None
    
    col1, col2 = st.columns(2)
    with col1:
        description = st.text_input("Description")
        amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
    with col2:
        category = st.text_input("Category (optional)")
    
    if st.button("Add Transaction", type="primary") and user_id:
        if amount > 0:
            result = api_client.create_transaction(
                user_id=user_id,
                description=description,
                amount=amount,
                category=category
            )
            if "error" not in result:
                st.success("Transaction added successfully!")
                st.rerun()
            else:
                st.error(f"Failed to add transaction: {result.get('error')}")
        else:
            st.error("Amount must be greater than 0")

with tab2:
    st.header("Users")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Existing Users")
        users = api_client.get_users()
        if users:
            user_df = pd.DataFrame(users)
            st.dataframe(user_df, use_container_width=True)
        else:
            st.info("No users found")
    
    with col2:
        st.subheader("Create New User")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        
        if st.button("Create User", type="primary"):
            if new_email and new_password:
                result = api_client.create_user(new_email, new_password)
                if "error" not in result:
                    st.success("User created successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to create user: {result.get('error')}")
            else:
                st.error("Email and password are required")

with tab3:
    st.header("Transactions")
    
    # Filter options
    users = api_client.get_users()
    if users:
        user_options = {"All Users": None}
        user_options.update({f"{user['email']} (ID: {user['id']})": user['id'] for user in users})
        selected_filter = st.selectbox("Filter by User", options=list(user_options.keys()))
        filter_user_id = user_options[selected_filter]
    else:
        filter_user_id = None
    
    transactions = api_client.get_transactions(filter_user_id)
    
    if transactions:
        # Convert to DataFrame for better display
        df = pd.DataFrame(transactions)
        
        # Format date and amount
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
        if 'amount' in df.columns:
            df['amount'] = df['amount'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(df, use_container_width=True)
        
        # Delete transaction option
        st.subheader("Delete Transaction")
        transaction_options = {f"{t.get('description', 'N/A')} - ${t.get('amount', 0):.2f} (ID: {t['id']})": t['id'] for t in transactions}
        selected_transaction = st.selectbox("Select Transaction to Delete", options=list(transaction_options.keys()))
        
        if st.button("Delete Transaction", type="secondary"):
            transaction_id = transaction_options[selected_transaction]
            result = api_client.delete_transaction(transaction_id)
            if "error" not in result:
                st.success("Transaction deleted successfully!")
                st.rerun()
            else:
                st.error(f"Failed to delete transaction: {result.get('error')}")
    else:
        st.info("No transactions found")
