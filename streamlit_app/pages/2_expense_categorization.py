"""
Expense Categorization Page
Automatically categorize transactions using trained ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="Expense Categorization",
    page_icon="🏷️",
    layout="wide"
)

# Constants
MODEL_PATH = Path("models/transaction_classifier.pkl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class TransactionCategorizer:
    """Handles transaction categorization using trained model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_dict = None
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.categories = []
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_dict = pickle.load(f)
            
            self.model = self.model_dict['model']
            self.label_encoder = self.model_dict['label_encoder']
            self.feature_columns = self.model_dict['feature_columns']
            
            if hasattr(self.label_encoder, 'classes_'):
                self.categories = self.label_encoder.classes_.tolist()
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features required by the model"""
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Basic features
        if 'amount' in df.columns:
            df['amount_abs'] = df['amount'].abs()
            df['is_income'] = (df['amount'] > 0).astype(int)
            df['is_expense'] = (df['amount'] < 0).astype(int)
        
        # Date features
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Description features
        if 'description' in df.columns:
            df['description_length'] = df['description'].fillna('').astype(str).str.len()
            df['description_word_count'] = df['description'].fillna('').astype(str).str.split().str.len()
        
        # Sort by date for rolling features
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Category-based features (if category exists - for training data compatibility)
        if 'description' in df.columns and 'amount' in df.columns:
            # Group by description prefix for category-like aggregation
            df['desc_prefix'] = df['description'].fillna('').astype(str).str[:10]
            cat_stats = df.groupby('desc_prefix')['amount'].agg(['mean', 'std', 'count'])
            df['cat_mean'] = df['desc_prefix'].map(cat_stats['mean']).fillna(0)
            df['cat_std'] = df['desc_prefix'].map(cat_stats['std']).fillna(0)
            df['cat_count'] = df['desc_prefix'].map(cat_stats['count']).fillna(0)
            df = df.drop('desc_prefix', axis=1)
        else:
            df['cat_mean'] = 0
            df['cat_std'] = 0
            df['cat_count'] = 0
        
        # Rolling window features
        if 'amount' in df.columns:
            df['rolling_7d_mean'] = df['amount'].rolling(window=7, min_periods=1).mean()
            df['rolling_7d_std'] = df['amount'].rolling(window=7, min_periods=1).std().fillna(0)
            df['rolling_30d_mean'] = df['amount'].rolling(window=30, min_periods=1).mean()
            df['rolling_30d_std'] = df['amount'].rolling(window=30, min_periods=1).std().fillna(0)
        
        # Days since last transaction
        if 'date' in df.columns:
            df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
        
        # Transaction count in last 7 days
        if 'date' in df.columns:
            df['transaction_count_7d'] = df.groupby(df['date'].dt.date).cumcount() + 1
        else:
            df['transaction_count_7d'] = 1
        
        # Monthly aggregates
        if 'date' in df.columns and 'amount' in df.columns:
            df['year_month'] = df['date'].dt.to_period('M')
            monthly_stats = df.groupby('year_month').agg({
                'amount': ['sum', 'mean', 'std', 'count']
            })
            monthly_stats.columns = ['monthly_total', 'monthly_mean', 'monthly_std', 'monthly_count']
            
            df = df.merge(monthly_stats, left_on='year_month', right_index=True, how='left')
            df['monthly_std'] = df['monthly_std'].fillna(0)
            
            # Additional monthly features
            monthly_abs = df.groupby('year_month')['amount_abs'].agg(['sum', 'mean'])
            monthly_abs.columns = ['monthly_abs_sum', 'monthly_abs_mean']
            df = df.merge(monthly_abs, left_on='year_month', right_index=True, how='left')
            
            # Income/expense by month
            df['monthly_income'] = df.groupby('year_month')['amount'].transform(
                lambda x: x[x > 0].sum()
            )
            df['monthly_expense'] = df.groupby('year_month')['amount'].transform(
                lambda x: x[x < 0].sum()
            )
            
            # Income/expense ratio
            df['income_expense_ratio'] = (
                df['monthly_income'].abs() / 
                (df['monthly_expense'].abs() + 1)  # Add 1 to avoid division by zero
            )
            
            # Drop year_month as it's not needed for prediction
            if 'year_month' in df.columns:
                df = df.drop('year_month', axis=1)
        else:
            # Create default values
            for col in ['monthly_total', 'monthly_mean', 'monthly_std', 'monthly_count',
                       'monthly_abs_sum', 'monthly_abs_mean', 'monthly_income', 
                       'monthly_expense', 'income_expense_ratio']:
                df[col] = 0
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Check for missing features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            st.warning(f"⚠️ Missing {len(missing_features)} features. Engineering them now...")
            # Engineer missing features
            df = self.engineer_features(df)
            
            # Check again after engineering
            still_missing = set(self.feature_columns) - set(df.columns)
            if still_missing:
                st.info(f"Filling remaining {len(still_missing)} features with 0")
                for col in still_missing:
                    df[col] = 0
        
        # Select features in correct order
        X = df[self.feature_columns].copy()
        
        # Handle NaN values
        if X.isnull().any().any():
            st.warning("Found NaN values in features, filling with 0")
            X = X.fillna(0)
        
        return X
    
    def categorize(self, df: pd.DataFrame, confidence_threshold: float = 0.5):
        """Categorize transactions"""
        X = self.prepare_features(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores
        confidence_scores = probabilities.max(axis=1)
        
        # Decode predictions
        predicted_categories = self.label_encoder.inverse_transform(predictions)
        
        # Add results to dataframe
        result = df.copy()
        result['predicted_category'] = predicted_categories
        result['confidence'] = confidence_scores
        result['needs_review'] = confidence_scores < confidence_threshold
        
        # Add probability columns for all categories
        for idx, category in enumerate(self.categories):
            result[f'prob_{category}'] = probabilities[:, idx]
        
        return result


def load_data_source(source_type: str, uploaded_file=None):
    """Load transaction data from various sources"""
    try:
        if source_type == "Upload CSV":
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} transactions from uploaded file")
                st.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
                return df
            else:
                st.warning("Please upload a CSV file")
                return None
                
        elif source_type == "Processed Data":
            path = Path("data/processed/featured_transactions.csv")
            if path.exists():
                df = pd.read_csv(path)
                st.success(f"✅ Loaded {len(df)} transactions from processed data")
                st.info(f"📋 Available columns: {', '.join(df.columns.tolist()[:10])}... ({len(df.columns)} total)")
                return df
            else:
                st.error(f"File not found: {path}")
                return None
                
        elif source_type == "Raw Export":
            path = Path("data/raw")
            if path.exists():
                csv_files = list(path.glob("*.csv"))
                if csv_files:
                    # Get most recent file
                    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                    df = pd.read_csv(latest_file)
                    st.success(f"✅ Loaded {len(df)} transactions from {latest_file.name}")
                    st.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
                    return df
            st.error("No raw export files found")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def display_categorization_results(df: pd.DataFrame, categorizer: TransactionCategorizer):
    """Display categorization results with visualizations"""
    
    # Summary metrics
    st.subheader("📊 Categorization Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(df))
    
    with col2:
        if 'confidence' in df.columns:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col3:
        if 'needs_review' in df.columns:
            needs_review = df['needs_review'].sum()
            st.metric("Needs Review", f"{needs_review} ({needs_review/len(df)*100:.1f}%)")
        else:
            st.metric("Needs Review", "N/A")
    
    with col4:
        if 'predicted_category' in df.columns:
            unique_categories = df['predicted_category'].nunique()
            st.metric("Categories Found", unique_categories)
        else:
            st.metric("Categories Found", "N/A")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        st.subheader("Category Distribution")
        if 'predicted_category' in df.columns:
            category_counts = df['predicted_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Transactions by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category information available")
    
    with col2:
        # Confidence distribution
        st.subheader("Confidence Distribution")
        if 'confidence' in df.columns:
            fig = px.histogram(
                df,
                x='confidence',
                nbins=30,
                title="Prediction Confidence Scores"
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                          annotation_text="Review Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence information available")
    
    # Category breakdown with confidence
    st.subheader("📋 Category Breakdown")
    
    if 'predicted_category' in df.columns:
        # Build aggregation based on available columns
        agg_dict = {}
        
        if 'amount' in df.columns:
            agg_dict['amount'] = ['count', 'sum', 'mean']
        
        if 'confidence' in df.columns:
            agg_dict['confidence'] = 'mean'
        
        if agg_dict:
            category_stats = df.groupby('predicted_category').agg(agg_dict).round(2)
            
            # Flatten column names
            if 'amount' in df.columns:
                category_stats.columns = ['Count', 'Total Amount', 'Avg Amount', 'Avg Confidence'] if 'confidence' in df.columns else ['Count', 'Total Amount', 'Avg Amount']
            else:
                category_stats.columns = ['Avg Confidence']
            
            category_stats = category_stats.sort_values('Count' if 'Count' in category_stats.columns else category_stats.columns[0], ascending=False)
            st.dataframe(category_stats, use_container_width=True)
        else:
            # Fallback: just show counts
            category_counts = df['predicted_category'].value_counts().to_frame('Count')
            st.dataframe(category_counts, use_container_width=True)
    else:
        st.info("No category information available")
    
    # Spending by category over time
    if 'date' in df.columns and 'amount' in df.columns and 'predicted_category' in df.columns:
        st.subheader("📈 Spending Trends by Category")
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        
        monthly_by_category = df_copy.groupby(['month', 'predicted_category'])['amount'].sum().reset_index()
        
        fig = px.bar(
            monthly_by_category,
            x='month',
            y='amount',
            color='predicted_category',
            title="Monthly Spending by Category",
            labels={'amount': 'Amount ($)', 'month': 'Month', 'predicted_category': 'Category'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing_cols = []
        if 'date' not in df.columns:
            missing_cols.append('date')
        if 'amount' not in df.columns:
            missing_cols.append('amount')
        if 'predicted_category' not in df.columns:
            missing_cols.append('predicted_category')
        
        if missing_cols:
            st.info(f"📅 Trend analysis unavailable. Missing columns: {', '.join(missing_cols)}")


def display_review_items(df: pd.DataFrame):
    """Display transactions that need review"""
    
    st.subheader("⚠️ Transactions Needing Review")
    
    review_df = df[df['needs_review'] == True].copy()
    
    if len(review_df) == 0:
        st.success("🎉 No transactions need review! All predictions are confident.")
        return
    
    st.write(f"Found {len(review_df)} transactions with low confidence scores")
    
    # Add category selector for manual correction
    for idx, row in review_df.iterrows():
        # Build title from available columns
        title_parts = []
        if 'date' in row:
            title_parts.append(str(row['date']))
        if 'description' in row:
            title_parts.append(str(row['description']))
        if 'amount' in row:
            title_parts.append(f"${row['amount']:.2f}")
        
        title = " - ".join(title_parts) if title_parts else f"Transaction {idx}"
        
        with st.expander(title):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'predicted_category' in row:
                    st.write(f"**Predicted:** {row['predicted_category']}")
                if 'confidence' in row:
                    st.write(f"**Confidence:** {row['confidence']:.1%}")
                
                # Show top 3 predictions with probabilities
                prob_cols = [col for col in review_df.columns if col.startswith('prob_')]
                if prob_cols:
                    probs = {col.replace('prob_', ''): row[col] for col in prob_cols}
                    top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    st.write("**Top Predictions:**")
                    for cat, prob in top_3:
                        st.write(f"  - {cat}: {prob:.1%}")
            
            with col2:
                st.write("**Transaction Details:**")
                if 'amount' in row:
                    st.write(f"Amount: ${row['amount']:.2f}")
                if 'merchant' in row and pd.notna(row['merchant']):
                    st.write(f"Merchant: {row['merchant']}")
                if 'account' in row and pd.notna(row['account']):
                    st.write(f"Account: {row['account']}")


def main():
    """Main application"""
    
    st.title("🏷️ Expense Categorization")
    st.write("Automatically categorize your transactions using machine learning")
    
    # Check if model exists
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("💡 Please train a model first using the Training page")
        return
    
    # Initialize categorizer
    if 'categorizer' not in st.session_state:
        categorizer = TransactionCategorizer(MODEL_PATH)
        if categorizer.load_model():
            st.session_state.categorizer = categorizer
            st.success(f"✅ Model loaded successfully! Categories: {', '.join(categorizer.categories)}")
        else:
            return
    else:
        categorizer = st.session_state.categorizer
    
    # Data source selection
    st.sidebar.header("⚙️ Settings")
    
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Upload CSV", "Processed Data", "Raw Export"]
    )
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload a CSV file with transaction data"
        )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Transactions below this confidence level will be flagged for review"
    )
    
    # Load and categorize button
    if st.sidebar.button("🚀 Load & Categorize", type="primary"):
        with st.spinner("Loading data..."):
            df = load_data_source(data_source, uploaded_file)
        
        if df is not None:
            # Show data preview before categorization
            with st.expander("🔍 Data Preview (Before Categorization)"):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                st.dataframe(df.head(), use_container_width=True)
            
            with st.spinner("Categorizing transactions..."):
                categorized_df = categorizer.categorize(df, confidence_threshold)
                st.session_state.categorized_df = categorized_df
                
                # Show what was added
                new_cols = set(categorized_df.columns) - set(df.columns)
                if new_cols:
                    st.success(f"✅ Added columns: {', '.join(new_cols)}")
                
                st.success("✅ Categorization complete!")
    
    # Display results if available
    if 'categorized_df' in st.session_state:
        df = st.session_state.categorized_df
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📋 All Transactions", 
            "⚠️ Review Items",
            "💾 Export"
        ])
        
        with tab1:
            display_categorization_results(df, categorizer)
        
        with tab2:
            st.subheader("All Categorized Transactions")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_categories = st.multiselect(
                    "Filter by Category",
                    options=df['predicted_category'].unique(),
                    default=df['predicted_category'].unique()
                )
            
            with col2:
                min_confidence = st.slider(
                    "Min Confidence",
                    0.0, 1.0, 0.0
                )
            
            with col3:
                show_review_only = st.checkbox("Show Review Items Only")
            
            # Apply filters
            filtered_df = df[df['predicted_category'].isin(selected_categories)]
            filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
            if show_review_only:
                filtered_df = filtered_df[filtered_df['needs_review'] == True]
            
            # Display filtered data
            # Build display columns based on what's actually available
            display_columns = []
            
            # Required columns that should always be present
            for col in ['date', 'description', 'amount']:
                if col in filtered_df.columns:
                    display_columns.append(col)
            
            # Add optional columns
            for col in ['merchant', 'account']:
                if col in filtered_df.columns:
                    display_columns.append(col)
            
            # Add prediction columns
            for col in ['predicted_category', 'confidence', 'needs_review']:
                if col in filtered_df.columns:
                    display_columns.append(col)
            
            # Sort by date if available, otherwise by index
            if 'date' in filtered_df.columns:
                display_df = filtered_df[display_columns].sort_values('date', ascending=False)
            else:
                display_df = filtered_df[display_columns]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            st.write(f"Showing {len(filtered_df)} of {len(df)} transactions")
        
        with tab3:
            display_review_items(df)
        
        with tab4:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Options:**")
                include_probabilities = st.checkbox("Include probability columns", value=False)
                export_review_only = st.checkbox("Export review items only", value=False)
            
            with col2:
                st.write("**Preview:**")
                export_df = df[df['needs_review'] == True] if export_review_only else df
                st.write(f"Will export {len(export_df)} transactions")
            
            # Prepare export data
            if not include_probabilities:
                # Remove probability columns
                export_columns = [col for col in export_df.columns if not col.startswith('prob_')]
                export_df = export_df[export_columns]
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"categorized_transactions_{timestamp}.csv"
            
            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=filename,
                mime='text/csv',
                type="primary"
            )
            
            # Save to results directory
            if st.button("💾 Save to Results Directory"):
                output_path = RESULTS_DIR / filename
                export_df.to_csv(output_path, index=False)
                st.success(f"✅ Saved to {output_path}")
    
    else:
        st.info("👆 Select a data source and click 'Load & Categorize' to begin")
        
        # Show model info
        st.subheader("📋 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {st.session_state.categorizer.model_dict.get('model_type', 'Unknown')}")
            st.write(f"**Categories:** {len(st.session_state.categorizer.categories)}")
            
        with col2:
            st.write(f"**Features:** {len(st.session_state.categorizer.feature_columns)}")
            trained_date = st.session_state.categorizer.model_dict.get('trained_date', 'Unknown')
            st.write(f"**Trained:** {trained_date}")
        
        with st.expander("View Categories"):
            st.write(st.session_state.categorizer.categories)
        
        with st.expander("View Required Features"):
            st.write(st.session_state.categorizer.feature_columns)


if __name__ == "__main__":
    main()