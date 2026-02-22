"""
Simple Feature Extraction Functions
"""
import pandas as pd
import numpy as np

def extract_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract basic features quickly"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['amount_abs'] = df['amount'].abs()
    df['is_income'] = (df['amount'] > 0).astype(int)
    
    return df

def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get monthly spending summary"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    summary = df.groupby('year_month').agg({
        'amount': lambda x: x[x < 0].sum(),  # Total expenses
        'transaction_id': 'count'  # Transaction count
    }).reset_index()
    
    summary.columns = ['month', 'total_spent', 'transaction_count']
    summary['total_spent'] = summary['total_spent'].abs()
    
    return summary

def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Get spending by category"""
    expenses = df[df['amount'] < 0].copy()
    expenses['amount_abs'] = expenses['amount'].abs()
    
    breakdown = expenses.groupby('category').agg({
        'amount_abs': ['sum', 'mean', 'count']
    }).reset_index()
    
    breakdown.columns = ['category', 'total', 'avg_transaction', 'count']
    breakdown = breakdown.sort_values('total', ascending=False)
    
    return breakdown