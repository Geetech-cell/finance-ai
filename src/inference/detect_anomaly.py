"""
Anomaly Detection Module for Financial Data
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import os


class _DenseCompat(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = dict(config)
            config.pop('quantization_config', None)
        return super().from_config(config)


def load_anomaly_model(model_path='models/anomaly_detector.h5', 
                       scaler_path='models/anomaly_scaler.pkl'):
    """
    Load the trained anomaly detection model and scaler.
    
    Args:
        model_path: Path to the saved Keras model
        scaler_path: Path to the saved scaler
        
    Returns:
        model: Loaded Keras model
        scaler: Loaded StandardScaler
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    # Load the model
    model = keras.models.load_model(
        model_path,
        custom_objects={"Dense": _DenseCompat},
        compile=False,
    )
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")
    
    return model, scaler


def detect_anomalies(data, model, scaler, threshold=None, threshold_percentile=95):
    """
    Detect anomalies in financial data using the trained autoencoder model.
    
    Args:
        data: DataFrame or numpy array of financial data
        model: Trained Keras autoencoder model
        scaler: Fitted StandardScaler
        threshold: Fixed threshold for anomaly detection (optional)
        threshold_percentile: Percentile to use for threshold if not provided
        
    Returns:
        anomalies: Boolean array indicating anomalies
        reconstruction_errors: Array of reconstruction errors
        threshold_used: The threshold value used
    """
    expected_n_features = getattr(scaler, 'n_features_in_', None)
    if expected_n_features is None and hasattr(scaler, 'mean_'):
        expected_n_features = int(np.asarray(scaler.mean_).shape[0])

    # Convert to numpy array if DataFrame (and align columns if possible)
    if isinstance(data, pd.DataFrame):
        if expected_n_features is not None:
            if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                expected_cols = list(scaler.feature_names_in_)
                data = data.reindex(columns=expected_cols, fill_value=0)
            else:
                default_cols = [f'feature_{i}' for i in range(expected_n_features)]
                if all(c in data.columns for c in default_cols):
                    data = data[default_cols].copy()
                else:
                    numeric_df = data.select_dtypes(include=[np.number]).copy()
                    if numeric_df.shape[1] >= expected_n_features:
                        data = numeric_df.iloc[:, :expected_n_features].copy()
                    else:
                        data = numeric_df.copy()
                        for i in range(expected_n_features - numeric_df.shape[1]):
                            data[f'__pad_{i}'] = 0

        feature_names = data.columns.tolist()
        data_array = data.values
    else:
        feature_names = None
        data_array = np.asarray(data)

        if expected_n_features is not None and data_array.ndim == 2:
            current_n = int(data_array.shape[1])
            if current_n > expected_n_features:
                data_array = data_array[:, :expected_n_features]
            elif current_n < expected_n_features:
                pad = np.zeros((data_array.shape[0], expected_n_features - current_n), dtype=data_array.dtype)
                data_array = np.concatenate([data_array, pad], axis=1)
    
    # Scale the data
    data_scaled = scaler.transform(data_array)
    
    # Get predictions (reconstructions)
    reconstructions = model.predict(data_scaled, verbose=0)
    
    # Calculate reconstruction error (MSE for each sample)
    reconstruction_errors = np.mean(np.square(data_scaled - reconstructions), axis=1)
    
    # Determine threshold
    if threshold is None:
        threshold_used = np.percentile(reconstruction_errors, threshold_percentile)
    else:
        threshold_used = threshold
    
    # Identify anomalies
    anomalies = reconstruction_errors > threshold_used
    
    print(f"Threshold used: {threshold_used:.6f}")
    print(f"Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} samples ({100*np.mean(anomalies):.2f}%)")
    
    return anomalies, reconstruction_errors, threshold_used


def analyze_anomalies(data, anomalies, reconstruction_errors, top_n=10):
    """
    Analyze detected anomalies and return detailed information.
    
    Args:
        data: Original DataFrame
        anomalies: Boolean array of anomalies
        reconstruction_errors: Array of reconstruction errors
        top_n: Number of top anomalies to return
        
    Returns:
        DataFrame with anomaly analysis
    """
    if isinstance(data, pd.DataFrame):
        result_df = data.copy()
    else:
        result_df = pd.DataFrame(data)
    
    result_df['is_anomaly'] = anomalies
    result_df['reconstruction_error'] = reconstruction_errors
    
    # Get top anomalies
    anomaly_df = result_df[result_df['is_anomaly'] == True].copy()
    anomaly_df = anomaly_df.nlargest(top_n, 'reconstruction_error')
    
    print(f"\nTop {len(anomaly_df)} anomalies:")
    print("="*80)
    
    return anomaly_df


def get_feature_contributions(data, model, scaler, sample_idx):
    """
    Calculate feature contributions to anomaly score for a specific sample.
    
    Args:
        data: DataFrame or array of financial data
        model: Trained autoencoder model
        scaler: Fitted scaler
        sample_idx: Index of the sample to analyze
        
    Returns:
        DataFrame with feature contributions
    """
    # Convert to array if needed
    if isinstance(data, pd.DataFrame):
        feature_names = data.columns.tolist()
        data_array = data.values
    else:
        feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        data_array = data
    
    # Get the sample
    sample = data_array[sample_idx:sample_idx+1]
    
    # Scale
    sample_scaled = scaler.transform(sample)
    
    # Get reconstruction
    reconstruction = model.predict(sample_scaled, verbose=0)
    
    # Calculate per-feature squared errors
    feature_errors = np.square(sample_scaled - reconstruction)[0]
    
    # Create result DataFrame
    contribution_df = pd.DataFrame({
        'feature': feature_names,
        'original_value': sample[0],
        'scaled_value': sample_scaled[0],
        'reconstructed_scaled': reconstruction[0],
        'squared_error': feature_errors,
        'contribution_pct': 100 * feature_errors / np.sum(feature_errors)
    })
    
    contribution_df = contribution_df.sort_values('squared_error', ascending=False)
    
    return contribution_df