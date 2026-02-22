"""
Quick Prediction Utility
Simple functions for making predictions
"""
from src.inference.predict_category import CategoryPredictor

# Initialize predictor (singleton)
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = CategoryPredictor()
    return _predictor

def predict(description: str, amount: float, date: str = None) -> str:
    """
    Quick prediction function
    
    Args:
        description: Transaction description
        amount: Transaction amount (negative for expense)
        date: Transaction date (YYYY-MM-DD), defaults to today
        
    Returns:
        Predicted category
    """
    if date is None:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
    
    transaction = {
        'description': description,
        'amount': amount,
        'date': date
    }
    
    predictor = get_predictor()
    return predictor.predict_single(transaction)

def predict_with_confidence(description: str, amount: float, date: str = None):
    """
    Quick prediction with confidence scores
    
    Returns:
        Tuple of (category, confidence_dict)
    """
    if date is None:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
    
    transaction = {
        'description': description,
        'amount': amount,
        'date': date
    }
    
    predictor = get_predictor()
    return predictor.predict_single(transaction, return_proba=True)


# Example usage
if __name__ == "__main__":
    # Test predictions
    test_transactions = [
        ("Starbucks Coffee", -5.50),
        ("Salary Deposit", 3000.00),
        ("Uber Ride", -15.75),
        ("Amazon Purchase", -45.99),
        ("Electric Bill", -120.00)
    ]
    
    print("🔮 Quick Prediction Test\n")
    for desc, amt in test_transactions:
        category, probs = predict_with_confidence(desc, amt)
        confidence = probs[category]
        print(f"{desc:25s} → {category:20s} ({confidence:.1%})")