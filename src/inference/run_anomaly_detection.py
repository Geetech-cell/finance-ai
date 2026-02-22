"""
Run Anomaly Detection on Financial Data
"""
import pandas as pd
import numpy as np
from src.inference.detect_anomaly import load_anomaly_model, detect_anomalies, analyze_anomalies, get_feature_contributions
import sys
import os


def main():
    """Main function to run anomaly detection."""
    
    print("="*80)
    print("Financial Anomaly Detection System")
    print("="*80)
    
    # Configuration
    model_path = 'models/anomaly_detector.h5'
    scaler_path = 'models/anomaly_scaler.pkl'
    data_path = 'data/processed/financial_data.csv'  # Update this path as needed
    
    # Step 1: Load the model and scaler
    print("\n[1/4] Loading model and scaler...")
    try:
        model, scaler = load_anomaly_model(model_path, scaler_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the model files exist:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        sys.exit(1)
    
    # Step 2: Load the data
    print("\n[2/4] Loading financial data...")
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        print("Generating sample data for demonstration...")
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
        for idx in anomaly_indices:
            data.iloc[idx] = data.iloc[idx] * 3 + 5
        
        data.index = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Features: {list(data.columns)}")
    
    # Step 3: Detect anomalies
    print("\n[3/4] Detecting anomalies...")
    anomalies, reconstruction_errors, threshold = detect_anomalies(
        data, 
        model, 
        scaler,
        threshold_percentile=95  # Adjust this to control sensitivity
    )
    
    # Step 4: Analyze results
    print("\n[4/4] Analyzing anomalies...")
    anomaly_analysis = analyze_anomalies(
        data, 
        anomalies, 
        reconstruction_errors,
        top_n=10
    )
    
    # Display detailed results
    if len(anomaly_analysis) > 0:
        print("\nAnomaly Details:")
        print(anomaly_analysis.to_string())
        
        # Analyze top anomaly
        print("\n" + "="*80)
        print("Feature Contribution Analysis for Top Anomaly")
        print("="*80)
        
        top_anomaly_idx = anomaly_analysis.index[0]
        original_idx = data.index.get_loc(top_anomaly_idx)
        
        contributions = get_feature_contributions(
            data,
            model,
            scaler,
            original_idx
        )
        
        print(f"\nSample index: {top_anomaly_idx}")
        print(f"Reconstruction error: {reconstruction_errors[original_idx]:.6f}")
        print("\nTop contributing features:")
        print(contributions.head(10).to_string())
        
        # Save results
        output_dir = 'results/anomaly_detection'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save anomaly analysis
        output_file = os.path.join(output_dir, 'detected_anomalies.csv')
        anomaly_analysis.to_csv(output_file)
        print(f"\n✓ Anomaly analysis saved to: {output_file}")
        
        # Save all results with scores
        full_results = data.copy()
        full_results['is_anomaly'] = anomalies
        full_results['reconstruction_error'] = reconstruction_errors
        full_output_file = os.path.join(output_dir, 'full_results.csv')
        full_results.to_csv(full_output_file)
        print(f"✓ Full results saved to: {full_output_file}")
        
    else:
        print("\n✓ No anomalies detected in the dataset.")
    
    print("\n" + "="*80)
    print("Anomaly detection completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()