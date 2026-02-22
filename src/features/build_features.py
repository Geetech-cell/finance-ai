"""
Feature Engineering for Financial Data
Build features from labeled transaction data for ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureBuilder:
    """Build features from transaction data"""
    
    def __init__(self):
        self.feature_names = []
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from transaction data
        
        Args:
            df: DataFrame with columns: date, amount, category, description
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Build different feature groups
        df = self.add_temporal_features(df)
        df = self.add_transaction_features(df)
        df = self.add_category_features(df)
        df = self.add_spending_patterns(df)
        df = self.add_aggregated_features(df)
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df['date'].dt.day_name()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        return df
    
    def add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-specific features"""
        df['amount_abs'] = df['amount'].abs()
        df['is_income'] = (df['amount'] > 0).astype(int)
        df['is_expense'] = (df['amount'] < 0).astype(int)
        df['transaction_size'] = pd.cut(
            df['amount_abs'], 
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['tiny', 'small', 'medium', 'large', 'huge']
        )
        
        # Description length
        if 'description' in df.columns:
            df['description_length'] = df['description'].str.len()
            df['description_word_count'] = df['description'].str.split().str.len()
        
        return df
    
    def add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category-based features"""
        if 'category' in df.columns:
            # One-hot encode categories
            category_dummies = pd.get_dummies(df['category'], prefix='cat')
            df = pd.concat([df, category_dummies], axis=1)
            
            # Category statistics
            category_stats = df.groupby('category')['amount_abs'].agg(['mean', 'std', 'count'])
            category_stats.columns = ['cat_mean', 'cat_std', 'cat_count']
            df = df.merge(category_stats, left_on='category', right_index=True, how='left')
        
        return df
    
    def add_spending_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spending pattern features"""
        df = df.sort_values('date').reset_index(drop=True)
        
        # Rolling statistics (7-day and 30-day windows by number of transactions)
        df['rolling_7d_mean'] = df['amount_abs'].rolling(window=7, min_periods=1).mean()
        df['rolling_7d_std'] = df['amount_abs'].rolling(window=7, min_periods=1).std().fillna(0)
        df['rolling_30d_mean'] = df['amount_abs'].rolling(window=30, min_periods=1).mean()
        df['rolling_30d_std'] = df['amount_abs'].rolling(window=30, min_periods=1).std().fillna(0)
        
        # Days since last transaction
        df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
        
        # Time-based rolling count (fixed version)
        try:
            # Set date as index temporarily for time-based rolling
            df_indexed = df.set_index('date').sort_index()
            transaction_count = df_indexed.rolling(window='7D')['amount'].count()
            df['transaction_count_7d'] = transaction_count.values
        except Exception as e:
            print(f"Warning: Could not calculate time-based rolling count: {e}")
            # Fallback: use simple rolling count by rows
            df['transaction_count_7d'] = 7  # Default value
        
        return df
    
    def add_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add aggregated features by month/week"""
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Monthly aggregations
        monthly_stats = df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'std', 'count'],
            'amount_abs': ['sum', 'mean']
        }).reset_index()
        monthly_stats.columns = ['year_month', 'monthly_total', 'monthly_mean', 
                                 'monthly_std', 'monthly_count', 'monthly_abs_sum', 'monthly_abs_mean']
        
        df = df.merge(monthly_stats, on='year_month', how='left')
        
        # Income vs Expense ratio
        df['monthly_income'] = df.groupby('year_month')['amount'].transform(
            lambda x: x[x > 0].sum()
        )
        df['monthly_expense'] = df.groupby('year_month')['amount'].transform(
            lambda x: abs(x[x < 0].sum())
        )
        df['income_expense_ratio'] = df['monthly_income'] / (df['monthly_expense'] + 1)
        
        return df


def build_features_from_file(input_path: str, output_path: str = None):
    """
    Build features from a CSV file
    
    Args:
        input_path: Path to labeled transactions CSV
        output_path: Path to save featured data (optional)
        
    Returns:
        DataFrame with features
    """
    # Load data
    print(f"📖 Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Build features
    print("🔧 Building features...")
    builder = FeatureBuilder()
    df_featured = builder.build_all_features(df)
    
    # Save if output path provided
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_featured.to_csv(output_path, index=False)
        print(f"✅ Saved featured data to {output_path}")
    
    print(f"📊 Features built: {len(df_featured.columns)} columns")
    print(f"📝 Sample features: {list(df_featured.columns[:10])}")
    
    return df_featured


if __name__ == "__main__":
    # Example usage
    input_file = 'data/processed/labeled_transactions.csv'
    output_file = 'data/processed/featured_transactions.csv'
    
    try:
        df_featured = build_features_from_file(input_file, output_file)
        
        # Show feature summary
        print("\n" + "="*50)
        print("FEATURE SUMMARY")
        print("="*50)
        print(f"Total transactions: {len(df_featured)}")
        print(f"Total features: {len(df_featured.columns)}")
        print(f"\nNumeric features:")
        numeric_features = df_featured.select_dtypes(include=[np.number]).columns.tolist()
        for i, feat in enumerate(numeric_features[:20], 1):  # Show first 20
            print(f"  {i}. {feat}")
        
        if len(numeric_features) > 20:
            print(f"  ... and {len(numeric_features) - 20} more")
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("💡 First run: python src/data/create_sample_data.py")