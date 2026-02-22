"""
Train Anomaly Detection Model (Autoencoder)
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import os


def create_autoencoder_model(input_dim, encoding_dim=10):
    """
    Create an autoencoder model for anomaly detection.
    
    Args:
        input_dim: Number of input features
        encoding_dim: Dimension of the encoding layer
        
    Returns:
        Compiled Keras model
    """
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(encoder_input)
    encoded = layers.Dense(16, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(decoded)
    decoder_output = layers.Dense(input_dim, activation='linear')(decoded)
    
    # Autoencoder model
    autoencoder = keras.Model(encoder_input, decoder_output)
    
    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return autoencoder


def train_anomaly_detector(data, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train an autoencoder for anomaly detection.
    
    Args:
        data: Training data (DataFrame or numpy array)
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        history: Training history
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    print(f"Training data shape: {data_array.shape}")
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)
    
    # Create model
    input_dim = data_array.shape[1]
    encoding_dim = max(3, input_dim // 3)  # Adaptive encoding dimension
    
    print(f"Creating autoencoder (input_dim={input_dim}, encoding_dim={encoding_dim})...")
    model = create_autoencoder_model(input_dim, encoding_dim)
    
    print(model.summary())
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        data_scaled,
        data_scaled,  # Autoencoder targets same as inputs
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    return model, scaler, history


def main():
    """Main training function."""
    
    print("="*80)
    print("Training Anomaly Detection Model")
    print("="*80)
    
    # Configuration
    data_path = 'data/processed/financial_data.csv'
    model_output_path = 'models/anomaly_detector.h5'
    scaler_output_path = 'models/anomaly_scaler.pkl'
    
    # Load or generate training data
    if os.path.exists(data_path):
        print(f"\nLoading training data from {data_path}...")
        data = pd.read_csv(data_path, index_col=0)
        # Remove target column if present
        if 'target' in data.columns:
            data = data.drop('target', axis=1)
    else:
        print(f"\nWarning: Data file not found at {data_path}")
        print("Generating sample data for demonstration...")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 5000
        n_features = 10
        
        # Normal data
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add correlations to make it more realistic
        data['feature_1'] = data['feature_0'] * 0.8 + np.random.randn(n_samples) * 0.2
        data['feature_3'] = data['feature_2'] * 0.6 + np.random.randn(n_samples) * 0.4
    
    print(f"Training data: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Train the model
    model, scaler, history = train_anomaly_detector(
        data,
        epochs=100,
        batch_size=64,
        validation_split=0.2
    )
    
    # Create output directory if needed
    os.makedirs('models', exist_ok=True)
    
    # Save the model and scaler
    print(f"\nSaving model to {model_output_path}...")
    model.save(model_output_path)
    
    print(f"Saving scaler to {scaler_output_path}...")
    joblib.dump(scaler, scaler_output_path)
    
    # Print final metrics
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"\n✓ Model saved to: {model_output_path}")
    print(f"✓ Scaler saved to: {scaler_output_path}")
    print("\nYou can now run anomaly detection using:")
    print("  python -m src.inference.run_anomaly_detection")
    print("="*80)


if __name__ == "__main__":
    main()