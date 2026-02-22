"""
Upload Data Page - AI Finance Assistant
"""
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")

st.title("📤 Upload Financial Data")
st.markdown("Upload your bank statements or financial data in CSV/Excel format")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your financial transaction data"
)

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 🔧 FIX: Convert nullable types to standard types for Arrow compatibility
        df = df.convert_dtypes()  # Convert to best possible dtypes
        for col in df.columns:
            if df[col].dtype.name in ['Int64', 'Int32', 'Int16', 'Int8']:
                df[col] = df[col].astype('float64')  # Convert to float (handles NaN)
            elif df[col].dtype.name == 'string':
                df[col] = df[col].astype('object')  # Convert string to object
        
        # Display success message
        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
        
        # Show data preview
        st.subheader("📊 Data Preview")
        st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
        st.dataframe(df.head(10))
        
        # Show column info
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info)
        
        # Save to data folder
        if st.button("💾 Save to Database"):
            save_path = f"data/raw/{uploaded_file.name}"
            os.makedirs("data/raw", exist_ok=True)
            
            if uploaded_file.name.endswith('.csv'):
                df.to_csv(save_path, index=False)
            else:
                df.to_excel(save_path, index=False)
            
            st.success(f"Data saved to: {save_path}")
            st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
else:
    st.info("👆 Please upload a file to get started")
    
    # Show example format
    with st.expander("📝 Example Data Format"):
        example_df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Description': ['Salary', 'Groceries'],
            'Amount': [5000.0, -150.0],
            'Category': ['Income', 'Food']
        })
        st.dataframe(example_df)