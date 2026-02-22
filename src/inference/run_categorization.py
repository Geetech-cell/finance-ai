"""
Transaction Categorization Script
Uses trained model to predict categories for new transactions
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransactionCategorizer:
    """Categorizes transactions using a trained model"""
    
    def __init__(self, model_path: str):
        """
        Initialize categorizer with trained model
        
        Args:
            model_path: Path to saved model pickle file
        """
        self.model_path = model_path
        self.model_dict = None
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.categories = []
        
    def load_model(self):
        """Load the trained model from pickle file"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_dict = pickle.load(f)
            
            # Extract components from model dictionary
            self.model = self.model_dict['model']
            self.label_encoder = self.model_dict['label_encoder']
            self.feature_columns = self.model_dict['feature_columns']
            
            # Get categories from label encoder
            if hasattr(self.label_encoder, 'classes_'):
                self.categories = self.label_encoder.classes_.tolist()
            
            logger.info("Model loaded successfully")
            logger.info(f"Expected features ({len(self.feature_columns)}): {self.feature_columns}")
            logger.info(f"Categories: {self.categories}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction by selecting only the columns the model expects
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with only the required features in the correct order
        """
        logger.info("Preparing features...")
        
        # Check which expected features are missing
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features (will be filled with 0): {missing_features}")
            # Add missing columns with default value 0
            for col in missing_features:
                df[col] = 0
        
        # Select only the features the model expects, in the correct order
        X = df[self.feature_columns].copy()
        
        # Handle any NaN values
        if X.isnull().any().any():
            logger.warning("Found NaN values in features, filling with 0")
            X = X.fillna(0)
        
        logger.info(f"Prepared features shape: {X.shape}")
        
        return X
    
    def categorize_transactions(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Categorize transactions
        
        Args:
            df: DataFrame with transaction data
            confidence_threshold: Minimum confidence for prediction (0.0 to 1.0)
            
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Categorizing {len(df)} transactions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores (max probability for each prediction)
        confidence_scores = probabilities.max(axis=1)
        
        # Decode predictions to category names
        predicted_categories = self.label_encoder.inverse_transform(predictions)
        
        # Create result dataframe
        result = df.copy()
        result['predicted_category'] = predicted_categories
        result['confidence'] = confidence_scores
        
        # Mark low confidence predictions
        if confidence_threshold > 0:
            low_confidence_mask = confidence_scores < confidence_threshold
            result.loc[low_confidence_mask, 'needs_review'] = True
            result.loc[~low_confidence_mask, 'needs_review'] = False
            
            logger.info(f"Predictions below threshold ({confidence_threshold}): "
                       f"{low_confidence_mask.sum()} ({low_confidence_mask.sum()/len(df)*100:.1f}%)")
        else:
            result['needs_review'] = False
        
        # Add probabilities for all categories if requested
        return result, probabilities
    
    def add_probability_columns(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Add probability columns for each category
        
        Args:
            df: Result DataFrame
            probabilities: Probability matrix from predict_proba
            
        Returns:
            DataFrame with probability columns added
        """
        result = df.copy()
        
        # Add probability for each category
        for idx, category in enumerate(self.categories):
            col_name = f'prob_{category}'
            result[col_name] = probabilities[:, idx]
        
        return result


def load_transactions(input_path: str) -> pd.DataFrame:
    """
    Load transactions from CSV file
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        DataFrame with transaction data
    """
    logger.info(f"Loading transactions from {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Convert date column if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading transactions: {e}")
        raise


def save_results(
    df: pd.DataFrame,
    output_path: str,
    review_output_path: Optional[str] = None
):
    """
    Save categorization results
    
    Args:
        df: DataFrame with results
        output_path: Path to save all results
        review_output_path: Optional path to save only items needing review
    """
    logger.info(f"Saving results to {output_path}")
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all results
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved {len(df)} categorized transactions to {output_path}")
        
        # Save review items if specified
        if review_output_path and 'needs_review' in df.columns:
            review_df = df[df['needs_review'] == True]
            if len(review_df) > 0:
                review_df.to_csv(review_output_path, index=False)
                logger.info(f"✅ Saved {len(review_df)} transactions needing review to {review_output_path}")
            else:
                logger.info("No transactions need review")
                
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def print_summary(df: pd.DataFrame):
    """Print summary of categorization results"""
    
    print("\n" + "="*80)
    print("CATEGORIZATION SUMMARY")
    print("="*80)
    
    print(f"\nTotal transactions: {len(df)}")
    
    if 'predicted_category' in df.columns:
        print("\nCategory Distribution:")
        print(df['predicted_category'].value_counts().to_string())
        
    if 'confidence' in df.columns:
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {df['confidence'].mean():.3f}")
        print(f"  Median: {df['confidence'].median():.3f}")
        print(f"  Min: {df['confidence'].min():.3f}")
        print(f"  Max: {df['confidence'].max():.3f}")
        
    if 'needs_review' in df.columns:
        review_count = df['needs_review'].sum()
        print(f"\nTransactions needing review: {review_count} ({review_count/len(df)*100:.1f}%)")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Categorize transactions using trained model'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file with transactions'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save categorized transactions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Confidence threshold for flagging review (0.0 to 1.0)'
    )
    parser.add_argument(
        '--review-output',
        help='Optional path to save transactions needing review'
    )
    parser.add_argument(
        '--include-probabilities',
        action='store_true',
        help='Include probability columns for all categories'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Load transactions
        transactions = load_transactions(args.input)
        
        # Initialize categorizer and load model
        categorizer = TransactionCategorizer(args.model)
        categorizer.load_model()
        
        # Categorize transactions
        categorized_transactions, probabilities = categorizer.categorize_transactions(
            transactions,
            confidence_threshold=args.threshold
        )
        
        # Add probability columns if requested
        if args.include_probabilities:
            logger.info("Adding probability columns for all categories")
            categorized_transactions = categorizer.add_probability_columns(
                categorized_transactions,
                probabilities
            )
        
        # Save results
        save_results(
            categorized_transactions,
            args.output,
            args.review_output
        )
        
        # Print summary
        if args.verbose:
            print_summary(categorized_transactions)
        
        logger.info("✅ Categorization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during categorization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()