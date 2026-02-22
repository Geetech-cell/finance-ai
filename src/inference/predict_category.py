"""
Transaction Category Prediction
Make predictions on new transactions using trained classifier
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.build_features import FeatureBuilder


class CategoryPredictor:
    """Predict transaction categories using trained model"""
    
    def __init__(self, model_path='models/transaction_classifier.pkl'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.feature_builder = FeatureBuilder()
        self.feature_defaults = {}  # Store default values for missing features
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using: python src/training/train_classifier.py"
            )
        
        print(f"📦 Loading model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.feature_defaults = model_data.get('feature_defaults', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"📅 Trained on: {model_data.get('trained_date', 'Unknown')}")
        print(f"📊 Categories: {', '.join(self.label_encoder.classes_)}")
    
    def prepare_transaction(self, transaction: dict) -> pd.DataFrame:
        """
        Prepare a single transaction for prediction
        
        Args:
            transaction: Dict with keys: date, amount, description
            
        Returns:
            DataFrame with features
        """
        # Create DataFrame from transaction
        df = pd.DataFrame([transaction])
        
        # Build features
        df_featured = self.feature_builder.build_all_features(df)
        
        return df_featured
    
    def _ensure_features(self, df_featured: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required features are present, add defaults for missing ones
        
        Args:
            df_featured: Featured dataframe
            
        Returns:
            DataFrame with all required features
        """
        # Find missing features
        missing_features = set(self.feature_columns) - set(df_featured.columns)
        
        if missing_features:
            # Add missing features with default values
            for feature in missing_features:
                if feature in self.feature_defaults:
                    df_featured[feature] = self.feature_defaults[feature]
                elif 'cat_' in feature:
                    # Category-based features default to 0
                    df_featured[feature] = 0.0
                else:
                    # Other features default to 0
                    df_featured[feature] = 0.0
        
        return df_featured
    
    def predict_single(self, transaction: dict, return_proba=False):
        """
        Predict category for a single transaction
        
        Args:
            transaction: Dict with keys: date, amount, description
            return_proba: Whether to return prediction probabilities
            
        Returns:
            Predicted category (and probabilities if return_proba=True)
        """
        # Prepare features
        df_featured = self.prepare_transaction(transaction)
        
        # Ensure all required features are present
        df_featured = self._ensure_features(df_featured)
        
        # Extract relevant features in correct order
        X = df_featured[self.feature_columns].fillna(0)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        if return_proba:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            proba_dict = {
                cat: prob 
                for cat, prob in zip(self.label_encoder.classes_, probabilities)
            }
            # Sort by probability
            proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
            
            return category, proba_dict
        
        return category
    
    def predict_batch(self, transactions: pd.DataFrame, return_proba=False):
        """
        Predict categories for multiple transactions
        
        Args:
            transactions: DataFrame with columns: date, amount, description
            return_proba: Whether to return prediction probabilities
            
        Returns:
            DataFrame with predictions (and probabilities if return_proba=True)
        """
        print(f"🔮 Predicting categories for {len(transactions)} transactions...")
        
        # Build features
        df_featured = self.feature_builder.build_all_features(transactions)
        
        # Ensure all required features are present
        df_featured = self._ensure_features(df_featured)
        
        # Extract relevant features
        X = df_featured[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        categories = self.label_encoder.inverse_transform(predictions)
        
        # Add predictions to original dataframe
        result = transactions.copy()
        result['predicted_category'] = categories
        
        if return_proba:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)
            
            # Add confidence score (max probability)
            result['confidence'] = probabilities.max(axis=1)
            
            # Add top 3 predictions with probabilities
            for i in range(min(3, len(self.label_encoder.classes_))):
                top_idx = probabilities.argsort(axis=1)[:, -(i+1)]
                result[f'top_{i+1}_category'] = self.label_encoder.inverse_transform(top_idx)
                result[f'top_{i+1}_prob'] = probabilities[np.arange(len(probabilities)), top_idx]
        
        print(f"✅ Predictions completed!")
        return result
    
    def predict_from_csv(self, input_path: str, output_path: str = None):
        """
        Predict categories for transactions in a CSV file
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"📖 Reading transactions from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Make predictions
        result = self.predict_batch(df, return_proba=True)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to: {output_path}")
        
        return result


def predict_interactive():
    """Interactive prediction mode"""
    print("="*60)
    print("INTERACTIVE TRANSACTION CATEGORY PREDICTOR")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = CategoryPredictor()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    print("\n💡 Enter transaction details (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*60)
        
        # Get user input
        date = input("Date (YYYY-MM-DD) [default: today]: ").strip()
        if date.lower() == 'quit':
            break
        if not date:
            from datetime import datetime
            date = datetime.now().strftime('%Y-%m-%d')
        
        description = input("Description: ").strip()
        if description.lower() == 'quit':
            break
        if not description:
            print("❌ Description is required!")
            continue
        
        amount = input("Amount (negative for expense): ").strip()
        if amount.lower() == 'quit':
            break
        try:
            amount = float(amount)
        except ValueError:
            print("❌ Invalid amount!")
            continue
        
        # Create transaction
        transaction = {
            'date': date,
            'description': description,
            'amount': amount
        }
        
        # Predict
        print("\n🔮 Predicting category...")
        try:
            category, probabilities = predictor.predict_single(transaction, return_proba=True)
            
            # Display results
            print(f"\n{'='*60}")
            print(f"📊 PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"\n🎯 Predicted Category: {category}")
            print(f"\n📈 Confidence Scores:")
            for i, (cat, prob) in enumerate(list(probabilities.items())[:5], 1):
                bar = '█' * int(prob * 50)
                print(f"  {i}. {cat:20s} {prob:6.1%} {bar}")
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()


def predict_from_file(input_file: str, output_file: str = None):
    """
    Predict categories for transactions in a file
    
    Args:
        input_file: Path to input CSV
        output_file: Path to save predictions
    """
    print("="*60)
    print("BATCH TRANSACTION CATEGORY PREDICTION")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = CategoryPredictor()
        
        # Make predictions
        result = predictor.predict_from_csv(input_file, output_file)
        
        # Display summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"\nTotal transactions: {len(result)}")
        print(f"\nCategory distribution:")
        print(result['predicted_category'].value_counts())
        
        print(f"\nAverage confidence: {result['confidence'].mean():.1%}")
        
        # Show low confidence predictions
        low_confidence = result[result['confidence'] < 0.5]
        if len(low_confidence) > 0:
            print(f"\n⚠️  {len(low_confidence)} predictions with low confidence (<50%):")
            print(low_confidence[['description', 'predicted_category', 'confidence']].head(10))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict transaction categories')
    parser.add_argument('--mode', choices=['interactive', 'file'], default='interactive',
                       help='Prediction mode: interactive or file')
    parser.add_argument('--input', type=str, help='Input CSV file path (for file mode)')
    parser.add_argument('--output', type=str, help='Output CSV file path (for file mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        predict_interactive()
    elif args.mode == 'file':
        if not args.input:
            print("❌ --input is required for file mode")
            sys.exit(1)
        
        output = args.output or args.input.replace('.csv', '_predicted.csv')
        predict_from_file(args.input, output)