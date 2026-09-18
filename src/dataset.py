"""
Dataset Loading, Preprocessing, Scaler Management, and Temporal Sequence Generation
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = {'rssi', 'snr', 'plr', 'success'}

def generate_synthetic_channel_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Fallback generator: Produces synthetic WSN link dataset following the physical
    path-loss and log-normal shadowing distributions matching Rahul's Module 1.
    """
    rng = np.random.default_rng(seed)
    # Generate distances from 1m to 45m
    distances = rng.uniform(1.0, 45.0, size=n_samples)
    comm_range = 40.0
    d_norm = np.clip(distances / comm_range, 1e-3, 1.3)

    # Physical link parameters
    ple = 2.5
    rssi = -40.0 - 60.0 * (d_norm ** ple) + rng.normal(0, 3.0, size=n_samples)
    rssi = np.clip(rssi, -105.0, -30.0)

    snr = 40.0 * np.maximum(0.0, 1.0 - d_norm / 1.3) + rng.normal(0, 2.0, size=n_samples)
    snr = np.clip(snr, -5.0, 45.0)

    plr_raw = 1.0 / (1.0 + np.exp(-6.0 * (d_norm - 0.75))) + rng.normal(0, 0.03, size=n_samples)
    plr = np.clip(plr_raw, 0.0, 1.0)

    # Success probability: logistic model based on SNR and PLR
    prob_success = 1.0 / (1.0 + np.exp(-(0.15 * snr - 4.0 * plr + 0.05 * (rssi + 80.0))))
    success = (rng.uniform(0, 1, size=n_samples) < prob_success).astype(int)

    df = pd.DataFrame({'rssi': rssi, 'snr': snr, 'plr': plr, 'success': success})
    return df

def load_or_create_dataset(data_path: str, seed: int = 42) -> pd.DataFrame:
    """Load dataset from path or fall back to synthetic generator if file not found."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Check alternative common locations
        alt_paths = [
            'ZIP/Dataset/module1_synthetic_dataset_balanced.csv',
            './module1_synthetic_dataset_balanced.csv',
            '/kaggle/input/datasets/rahulanand1618/dataset-wsn/module1_synthetic_dataset_balanced.csv',
            '/content/module1_synthetic_dataset_balanced.csv',
        ]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                df = pd.read_csv(alt)
                found = True
                break
        if not found:
            print(f"Dataset not found at {data_path}. Generating realistic physical dataset...")
            df = generate_synthetic_channel_dataset(10000, seed)

    # Normalize columns
    df.columns = df.columns.str.lower().str.strip()
    for col in ['rssi', 'snr', 'plr', 'success']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['rssi', 'snr', 'plr', 'success']).reset_index(drop=True)
    df['plr'] = df['plr'].clip(0.0, 1.0)
    df['success'] = df['success'].astype(int)
    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.20, val_size: float = 0.10, seed: int = 42):
    """Returns X_train, X_val, X_test, y_train, y_val, y_test as float32 arrays."""
    X = df[['rssi', 'snr', 'plr']].values.astype(np.float32)
    y = df['success'].values.astype(np.float32)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=seed, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_scaler_npz(scaler: StandardScaler, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)

def load_scaler_npz(path: str) -> StandardScaler:
    data = np.load(path)
    sc = StandardScaler()
    sc.mean_ = data['mean']
    sc.scale_ = data['scale']
    sc.var_ = data['var']
    sc.n_features_in_ = len(data['mean'])
    return sc

def generate_temporal_sequences(X_scaled: np.ndarray, y: np.ndarray, seq_len: int = 10,
                                rho: float = 0.82, seed: int = 42):
    """
    Synthesize temporal time-series sequences of length seq_len using an AR(1)
    Gauss-Markov process on top of scaled physical link metrics.

    x_t = rho * x_{t-1} + sqrt(1 - rho^2) * e_t

    Output:
      X_seq: shape (N - seq_len + 1, seq_len, 3)
      y_seq: shape (N - seq_len + 1,) corresponding to success at future step
    """
    rng = np.random.default_rng(seed)
    N, D = X_scaled.shape
    
    # Generate continuous AR(1) trajectory over N steps
    ar_series = np.zeros_like(X_scaled)
    ar_series[0] = X_scaled[0]
    std_innov = np.sqrt(max(1e-6, 1.0 - rho ** 2))
    
    for t in range(1, N):
        # Drift slightly along the dataset manifold + AR(1) persistence
        target = X_scaled[t]
        ar_series[t] = rho * ar_series[t - 1] + (1.0 - rho) * target + rng.normal(0, std_innov * 0.2, size=D)

    # Slice into rolling windows
    n_windows = N - seq_len
    X_seq = np.zeros((n_windows, seq_len, D), dtype=np.float32)
    y_seq = np.zeros((n_windows,), dtype=np.float32)

    for i in range(n_windows):
        X_seq[i] = ar_series[i:i + seq_len]
        y_seq[i] = y[i + seq_len]

    return X_seq, y_seq
