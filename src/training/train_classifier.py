"""
Train Transaction Category Classifier
Trains a machine learning model to classify transactions into categories
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class TransactionClassifier:
    """Train and evaluate transaction classification model"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'category'
        
    def prepare_data(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            df: Featured transaction dataframe
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("📊 Preparing data for training...")
        
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = [self.target_column, 'transaction_id', 'year_month']
        self.feature_columns = [col for col in numeric_features if col not in exclude_cols]
        
        # Prepare features and target
        X = df[self.feature_columns].fillna(0)
        y = df[self.target_column]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"✅ Training set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"✅ Features: {len(self.feature_columns)}")
        print(f"✅ Categories: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for the model
        """
        print(f"\n🎯 Training {self.model_type} classifier...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        print("✅ Training completed!")
        
    def evaluate(self, X_test, y_test, show_plots=True):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            show_plots: Whether to show evaluation plots
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n📈 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get category names
        categories = self.label_encoder.classes_
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=categories,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
        print(f"\n📊 Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=categories))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot confusion matrix if requested
        if show_plots:
            self._plot_confusion_matrix(cm, categories)
            if hasattr(self.model, 'feature_importances_'):
                self._plot_feature_importance(feature_importance)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None
        }
    
    def _plot_confusion_matrix(self, cm, categories):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=categories, yticklabels=categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n📊 Confusion matrix saved to: reports/figures/confusion_matrix.png")
        plt.close()
    
    def _plot_feature_importance(self, feature_importance, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        print("📊 Feature importance plot saved to: reports/figures/feature_importance.png")
        plt.close()
    
    def save_model(self, filepath='models/transaction_classifier.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'trained_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath='models/transaction_classifier.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = TransactionClassifier(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        
        print(f"✅ Model loaded from: {filepath}")
        print(f"📅 Trained on: {model_data['trained_date']}")
        
        return classifier
    
    def predict(self, X):
        """Make predictions on new data"""
        X_features = X[self.feature_columns].fillna(0)
        predictions = self.model.predict(X_features)
        return self.label_encoder.inverse_transform(predictions)


def train_model_pipeline(
    data_path='data/processed/featured_transactions.csv',
    model_path='models/transaction_classifier.pkl',
    test_size=0.2,
    random_state=42
):
    """
    Complete training pipeline
    
    Args:
        data_path: Path to featured transaction data
        model_path: Path to save trained model
        test_size: Test set proportion
        random_state: Random seed
    """
    print("="*60)
    print("TRANSACTION CLASSIFIER TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print(f"\n📖 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} transactions")
    
    # Initialize classifier
    classifier = TransactionClassifier(model_type='random_forest')
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Train model
    classifier.train(X_train, y_train, n_estimators=100, max_depth=20, random_state=random_state)
    
    # Evaluate model
    results = classifier.evaluate(X_test, y_test, show_plots=True)
    
    # Save model
    classifier.save_model(model_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return classifier, results


if __name__ == "__main__":
    # Run the training pipeline
    try:
        classifier, results = train_model_pipeline()
        
        print("\n🎉 Model is ready to use!")
        print("💡 To use the model, run:")
        print("   from src.training.train_classifier import TransactionClassifier")
        print("   classifier = TransactionClassifier.load_model('models/transaction_classifier.pkl')")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have run:")
        print("   1. python src/data/create_sample_data.py")
        print("   2. python src/features/build_features.py")