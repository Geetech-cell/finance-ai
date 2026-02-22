"""
Create Sample Labeled Transaction Data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_transactions(n_transactions=500):
    """Generate sample labeled financial transactions"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Categories and their typical descriptions
    categories = {
        'Food & Dining': ['McDonald\'s', 'Starbucks', 'Uber Eats', 'Restaurant', 'Grocery Store'],
        'Transportation': ['Uber', 'Lyft', 'Gas Station', 'Parking', 'Metro Card'],
        'Shopping': ['Amazon', 'Target', 'Walmart', 'Best Buy', 'Online Store'],
        'Bills & Utilities': ['Electric Bill', 'Water Bill', 'Internet', 'Phone Bill', 'Rent'],
        'Entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Concert', 'Gaming'],
        'Healthcare': ['Pharmacy', 'Doctor Visit', 'Health Insurance', 'Gym Membership'],
        'Income': ['Salary', 'Freelance Payment', 'Refund', 'Bonus'],
        'Transfer': ['Bank Transfer', 'ATM Withdrawal', 'Venmo', 'PayPal'],
        'Other': ['Misc Purchase', 'Unknown', 'Cash Withdrawal']
    }
    
    # Generate transactions
    transactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_transactions):
        # Random category
        category = np.random.choice(list(categories.keys()), p=[0.20, 0.10, 0.15, 0.15, 0.10, 0.05, 0.15, 0.05, 0.05])
        
        # Random description from category
        description = np.random.choice(categories[category])
        
        # Generate amount based on category
        if category == 'Income':
            amount = np.random.uniform(1000, 5000)  # Positive for income
        elif category == 'Bills & Utilities':
            amount = -np.random.uniform(50, 500)
        elif category == 'Shopping':
            amount = -np.random.uniform(20, 500)
        elif category == 'Food & Dining':
            amount = -np.random.uniform(5, 100)
        else:
            amount = -np.random.uniform(10, 300)
        
        # Random date
        transaction_date = start_date + timedelta(days=np.random.randint(0, 365))
        
        # Create transaction
        transactions.append({
            'transaction_id': f'TXN{i+1:05d}',
            'date': transaction_date.strftime('%Y-%m-%d'),
            'description': description,
            'amount': round(amount, 2),
            'category': category,
            'account': np.random.choice(['Checking', 'Savings', 'Credit Card']),
            'merchant': description,
            'is_recurring': np.random.choice([True, False], p=[0.2, 0.8])
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate sample data
    print("Generating sample labeled transactions...")
    df = create_sample_transactions(500)
    
    # Save to CSV
    output_path = 'data/processed/labeled_transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created {len(df)} transactions")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Summary:")
    print(df['category'].value_counts())
    print(f"\n💰 Total Income: ${df[df['amount'] > 0]['amount'].sum():.2f}")
    print(f"💸 Total Expenses: ${abs(df[df['amount'] < 0]['amount'].sum()):.2f}")