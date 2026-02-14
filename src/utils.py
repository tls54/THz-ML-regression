"""
Utility functions for THz parameter extraction.
"""

import torch
import numpy as np
from pathlib import Path
from training_config import Config


# ============================================================================
# DATALAKE LOADING
# ============================================================================

def load_dataset_from_datalake(dataset_name, split='train', datalake_root=None):
    """
    Load dataset from datalake structure.
    
    Args:
        dataset_name: Name of dataset in datalake
        split: 'train', 'val', or 'test'
        datalake_root: Root directory of datalake (auto-detected if None)
        
    Returns:
        dict with keys:
            'X': Input tensors [n_samples, 2, n_freq]
            'y': Ground truth parameters [n_samples, 3]
            'T': Transmission coefficients [n_samples, n_freq] (complex)
            'frequencies': Frequency array [n_freq]
            'metadata': Dataset metadata dict
    """
    # Auto-detect datalake root
    if datalake_root is None:
        # Assume datalake is in parent directory
        datalake_root = Path(__file__).parent.parent / 'datalake' / 'datasets'
    else:
        datalake_root = Path(datalake_root)
    
    dataset_path = datalake_root / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading {split} split from datalake: {dataset_name}")
    
    # Load split data
    split_file = dataset_path / f"{split}.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    data = np.load(split_file, allow_pickle=True)

    # Load metadata
    metadata_file = dataset_path / 'metadata.json'
    metadata = None
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Extract data
    # Handle nested structure from your dataset generator
    if 'parameters' in data.files:
        # Parameters stored as a pickled dict
        params = data['parameters'].item()
        n = params['n']
        kappa = params['kappa']
        d = params['d']
    elif 'parameters/n' in data.files:
        # New format with nested parameters
        n = data['parameters/n']
        kappa = data['parameters/kappa']
        d = data['parameters/d']
    else:
        # Try old format with direct arrays
        n = data['n']
        kappa = data['kappa']
        d = data['d']
    
    # Stack parameters [n_samples, 3]
    y = np.stack([n, kappa, d], axis=1)
    
    # Get frequencies
    frequencies = data['frequencies']
    
    # Get ML input - reconstruct from T_complex to save 50% storage
    if 'T_complex' in data.files:
        # Reconstruct ML input from complex transmission (efficient numpy operation)
        T_complex = data['T_complex']
        # Stack real and imaginary parts: [N, M+1] complex64 → [N, 2, M+1] float32
        X = np.stack([T_complex.real, T_complex.imag], axis=1).astype(np.float32)
    elif 'ml_input' in data.files:
        # Legacy datasets with pre-computed ml_input (deprecated, wastes 50% storage)
        X = data['ml_input']
    else:
        raise ValueError("Dataset must contain either 'T_complex' or 'ml_input'")
    
    # Get transmission coefficients if available
    T = data.get('T_complex', None)
    
    print(f"✓ Loaded {split} split:")
    print(f"  Samples: {len(y)}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    if metadata:
        print(f"  Dataset: {metadata.get('dataset_name', 'Unknown')}")
        print(f"  Generated: {metadata.get('generation_date', 'Unknown')}")
    
    return {
        'X': X,
        'y': y,
        'T': T,
        'frequencies': frequencies,
        'metadata': metadata
    }


def list_available_datasets(datalake_root=None):
    """
    List all available datasets in datalake.
    
    Args:
        datalake_root: Root directory of datalake
        
    Returns:
        List of dataset names
    """
    if datalake_root is None:
        datalake_root = Path(__file__).parent.parent / 'datalake' / 'datasets'
    else:
        datalake_root = Path(datalake_root)
    
    if not datalake_root.exists():
        print(f"Datalake not found at: {datalake_root}")
        return []
    
    datasets = []
    for item in datalake_root.iterdir():
        if item.is_dir():
            # Check if it has the required split files
            if (item / 'train.npz').exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def get_dataset_info(dataset_name, datalake_root=None):
    """
    Get information about a dataset without loading it.
    
    Args:
        dataset_name: Name of dataset
        datalake_root: Root directory of datalake
        
    Returns:
        Dict with dataset info
    """
    if datalake_root is None:
        datalake_root = Path(__file__).parent.parent / 'datalake' / 'datasets'
    else:
        datalake_root = Path(datalake_root)
    
    dataset_path = datalake_root / dataset_name
    metadata_file = dataset_path / 'metadata.json'
    
    if not metadata_file.exists():
        return None
    
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class THz_Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for THz spectroscopy."""

    def __init__(self, X, y, T=None, dtype=torch.float32):
        """
        Args:
            X: Input tensors [n_samples, 2, n_freq]
            y: Ground truth parameters [n_samples, 3]
            T: Optional transmission coefficients [n_samples, n_freq]
            dtype: Tensor dtype (torch.float32 or torch.float64)
        """
        self.dtype = dtype
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.X = torch.from_numpy(X).to(dtype) if isinstance(X, np.ndarray) else X.to(dtype)
        self.y = torch.from_numpy(y).to(dtype) if isinstance(y, np.ndarray) else y.to(dtype)

        if T is not None:
            self.T = torch.from_numpy(T).to(complex_dtype) if isinstance(T, np.ndarray) else T.to(complex_dtype)
        else:
            self.T = None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.T is not None:
            return self.X[idx], self.y[idx], self.T[idx]
        else:
            return self.X[idx], self.y[idx]


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth [n_samples, 3]
        y_pred: Predictions [n_samples, 3]
        
    Returns:
        dict with metrics for each parameter
    """
    # Separate parameters
    n_true, kappa_true, d_true = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    n_pred, kappa_pred, d_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    def param_metrics(true, pred, name):
        mae = torch.mean(torch.abs(pred - true)).item()
        mse = torch.mean((pred - true) ** 2).item()
        rmse = np.sqrt(mse)
        
        # Relative error (avoid division by zero)
        rel_error = torch.mean(torch.abs((pred - true) / (torch.abs(true) + 1e-10))).item() * 100
        
        # R² score
        ss_res = torch.sum((true - pred) ** 2).item()
        ss_tot = torch.sum((true - torch.mean(true)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            f'{name}_mae': mae,
            f'{name}_rmse': rmse,
            f'{name}_relative_error_%': rel_error,
            f'{name}_r2': r2
        }
    
    metrics = {}
    metrics.update(param_metrics(n_true, n_pred, 'n'))
    metrics.update(param_metrics(kappa_true, kappa_pred, 'kappa'))
    metrics.update(param_metrics(d_true * 1e6, d_pred * 1e6, 'd_um'))  # Convert to μm
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if 'r2' in key:
            print(f"{key:30s}: {value:8.4f}")
        elif '%' in key:
            print(f"{key:30s}: {value:7.2f}%")
        elif 'd_um' in key:
            print(f"{key:30s}: {value:8.2f} μm")
        else:
            print(f"{key:30s}: {value:8.5f}")
    
    print(f"{'='*60}\n")