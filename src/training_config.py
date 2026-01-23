"""
Configuration file for THz parameter extraction.
"""

from pathlib import Path


class Config:
    """Base configuration."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATALAKE_ROOT = PROJECT_ROOT / 'datalake'
    DATASETS_DIR = DATALAKE_ROOT / 'datasets'
    MODELS_DIR = DATALAKE_ROOT / 'models'
    RESULTS_DIR = DATALAKE_ROOT / 'results'
    
    # Dataset to use (from datalake)
    DATASET_NAME = 'default_dataset'  # Override this to use different datasets
    
    # Simulation parameters (for physics loss, should match dataset)
    L = 4096  # 2**12 time points
    DELTAT = 0.1e-12  # 100 fs → ~5 THz bandwidth
    
    # Parameter ranges (should match dataset)
    N_MIN = 1.1
    N_MAX = 7.0
    KAPPA_MIN = -0.1
    KAPPA_MAX = 0.01
    D_MIN = 100e-6  # meters
    D_MAX = 1000e-6  # meters
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    
    # Loss function
    LOSS_TYPE = 'hybrid'  # 'supervised', 'physics', 'hybrid'
    ALPHA = 0.5  # Weight for hybrid loss (1.0=supervised only, 0.0=physics only)

    # Loss scaling configuration
    USE_NORMALIZED_SUPERVISED_LOSS = True  # Normalize each parameter to [0,1] for equal contribution
    PHYSICS_LOSS_SCALE = 1.0  # Scaling factor for physics loss component
    PARAM_WEIGHTS = {  # Per-parameter weights (after normalization)
        'n': 1.0,
        'kappa': 1.0,
        'd': 1.0
    }

    # Network Architecture
    # CNN/ResNet scaling parameters
    CNN_WIDTH_MULT = 1.0  # Channel multiplier (0.5=half channels, 2.0=double channels)
    CNN_NUM_BLOCKS = 4  # Number of convolutional blocks (2-6 recommended)
    CNN_BASE_CHANNELS = 32  # Base channel count for first layer
    CNN_DROPOUT = 0.2  # Dropout rate

    # ResNet scaling parameters
    RESNET_WIDTH_MULT = 1.0  # Channel multiplier
    RESNET_NUM_RES_BLOCKS = [2, 2, 2]  # Number of residual blocks per stage
    RESNET_BASE_CHANNELS = 64  # Base channel count
    RESNET_DROPOUT = 0.2

    # FC layer configuration
    FC_HIDDEN_DIMS = [128, 64]  # Hidden layer dimensions for FC layers

    # Hardware
    DEVICE = 'cuda'  # or 'cpu'
    NUM_WORKERS = 4
    
    # Logging
    LOG_INTERVAL = 100  # Log every N batches
    SAVE_INTERVAL = 10  # Save checkpoint every N epochs
    
    # Early stopping
    PATIENCE = 15  # Stop if no improvement for N epochs
    MIN_DELTA = 1e-6  # Minimum improvement to count
    
    @classmethod
    def get_freq_info(cls):
        """Get frequency array information."""
        M = 2 * cls.L
        deltaf = 1.0 / (M * 2 * cls.DELTAT)
        n_freq = M + 1
        return n_freq, deltaf
    
    @classmethod
    def get_model_save_path(cls, network_name, run_name=None):
        """Get path for saving models in datalake."""
        if run_name is None:
            from datetime import datetime
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_dir = cls.MODELS_DIR / cls.DATASET_NAME / network_name / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    @classmethod
    def get_results_save_path(cls, network_name, run_name=None):
        """Get path for saving results in datalake."""
        if run_name is None:
            from datetime import datetime
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_dir = cls.RESULTS_DIR / cls.DATASET_NAME / network_name / run_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir