# THz-ML Source Code

Machine learning pipeline for terahertz (THz) parameter extraction using physics-informed neural networks.

## Overview

This codebase implements a complete ML workflow for extracting material parameters (refractive index, extinction coefficient, thickness) from THz transmission spectroscopy data. The approach combines supervised learning with physics-informed constraints using a differentiable forward model based on the transfer matrix method.

## Architecture

### Core Modules

#### [simulate.py](simulate.py)
Physics simulation engine using the transfer matrix method for THz wave propagation.

**Key Components:**
- **Reference Pulse Generation**: Creates THz pulses for air propagation
- **Transfer Matrix Method**: Computes reflection/transmission coefficients through layered materials
- **Forward Model**: `simulate_transmission_ml()` - Differentiable forward model for ML training
  - Inputs: `n` (refractive index), `κ` (extinction coefficient), `d` (thickness)
  - Outputs: Complex transmission coefficients in frequency domain
  - Supports batched operations for efficient training
- **ML Input Preparation**: Converts complex transmission to real/imaginary channels avoiding phase wrapping
- **Visualization**: Tools for plotting magnitude and phase spectra

**Physics:**
- Sign convention: `n_complex = n + i*κ` where `κ < 0` indicates absorption
- Uses conjugate symmetry for real-valued time domain signals
- Frequency range configurable via time sampling parameters

#### [generate_dataset.py](generate_dataset.py)
Dataset generation script with configurable parameter ranges and noise models.

**Features:**
- Configurable parameter ranges (uniform/log-uniform sampling)
- Train/validation/test split generation with separate random seeds
- Noise injection (Gaussian by default)
- Automatic metadata tracking and documentation
- Saves to datalake structure with README and visualization
- Batch processing for memory efficiency

**Usage:**
```bash
# Generate default dataset
python generate_dataset.py

# Custom configuration
python generate_dataset.py --config my_config.json

# Quick test dataset
python generate_dataset.py --quick
```

**Output Format:**
- `train.npz`, `val.npz`, `test.npz`: Dataset splits
- `metadata.json`: Complete generation parameters
- `README.md`: Auto-generated documentation
- `parameter_distributions.png`: Histogram visualizations

#### [networks.py](networks.py)
Neural network architectures for THz parameter extraction.

**Available Architectures:**

1. **THz_Encoder_CNN** (Recommended)
   - 1D convolutional encoder with progressive downsampling
   - 4 conv layers (32 → 64 → 128 → 256 channels)
   - Batch normalization and dropout for regularization
   - Output activation: sigmoid mapping to physical parameter ranges
   - Best for: General purpose, good balance of performance and efficiency

2. **THz_Encoder_ResNet**
   - ResNet-style with skip connections
   - Residual blocks to enable deeper networks
   - Use when: Simple CNN underfits the data

3. **THz_Encoder_MultiScale**
   - Parallel branches with different kernel sizes (3, 7, 15)
   - Captures features at multiple scales
   - Use for: Data with varying oscillation frequencies

4. **THz_Encoder_MLP**
   - Simple fully-connected baseline
   - For comparison only, not recommended

**Input/Output:**
- Input: `[batch, 2, n_freq]` - Real and imaginary parts of transmission
- Output: `[batch, 3]` - Predicted `[n, κ, d]` scaled to physical ranges

#### [train.py](train.py)
Training script with multiple loss functions and datalake integration.

**Loss Functions:**
- **Supervised**: Direct MSE on parameters (fastest)
- **Physics-Informed**: Reconstruction loss via forward model (physics consistency)
- **Hybrid**: Weighted combination (`α * supervised + (1-α) * physics`)

**Features:**
- Automatic datalake dataset loading
- Checkpoint saving and resume capability
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping with patience
- Comprehensive training history tracking
- Progress bars and metric reporting

**Usage:**
```bash
# Train with default settings
python train.py --network cnn

# Use specific dataset
python train.py --network resnet --dataset my_dataset

# Resume training
python train.py --resume path/to/checkpoint.pt

# List available datasets
python train.py --list-datasets
```

#### [evaluate.py](evaluate.py)
Model evaluation and visualization tools.

**Capabilities:**
- Load trained models from checkpoints
- Compute comprehensive metrics (MAE, RMSE, R², relative error)
- Generate prediction scatter plots
- Error distribution histograms
- Physics consistency checks (compare predicted vs. true transmission spectra)

**Metrics Computed:**
- Per-parameter: MAE, RMSE, R², relative error
- Separate tracking for `n`, `κ`, and `d` (converted to μm)

**Usage:**
```bash
python evaluate.py path/to/checkpoint.pt --network cnn
```

#### [configs.py](configs.py)
Centralized configuration management.

**Configuration Sections:**
- **Paths**: Project root, datalake structure (datasets, models, results)
- **Simulation**: Time/frequency parameters (L=4096, Δt=0.1ps → ~5 THz bandwidth)
- **Parameter Ranges**: Physical bounds for `n`, `κ`, `d`
- **Training**: Batch size, learning rate, epochs, weight decay
- **Loss**: Type selection (supervised/physics/hybrid) and α weighting
- **Hardware**: Device (cuda/cpu), workers
- **Early Stopping**: Patience and minimum improvement threshold

**Utility Methods:**
- `get_freq_info()`: Compute frequency array parameters
- `get_model_save_path()`: Generate timestamped model directories
- `get_results_save_path()`: Generate results directories

#### [utils.py](utils.py)
Utility functions for data loading and metrics.

**Datalake Functions:**
- `load_dataset_from_datalake()`: Load datasets with automatic format detection
- `list_available_datasets()`: Discover all datasets in datalake
- `get_dataset_info()`: Read metadata without loading data

**Dataset Class:**
- `THz_Dataset`: PyTorch Dataset wrapper
- Handles optional transmission coefficients for physics loss
- Automatic numpy/torch conversion

**Metrics:**
- `compute_metrics()`: Calculate MAE, RMSE, R², relative error per parameter
- `print_metrics()`: Pretty-printed metric tables

## Workflow

### 1. Dataset Generation
```bash
# Create a dataset with custom parameters
python generate_dataset.py --config dataset_config.json
```

The dataset will be saved to `datalake/datasets/<dataset_name>/` with:
- Train/val/test splits
- Metadata and documentation
- Parameter distribution plots

### 2. Training
```bash
# Train a CNN model
python train.py --network cnn --dataset my_dataset
```

Models are saved to `datalake/models/<dataset_name>/<network>/<timestamp>/`:
- `best_model.pt`: Best validation checkpoint
- `final_model.pt`: Final epoch checkpoint
- `checkpoint_epoch_*.pt`: Periodic checkpoints
- `run_config.json`: Complete training configuration

Results are saved to `datalake/results/<dataset_name>/<network>/<timestamp>/`:
- `training_history.json`: Loss curves and metrics per epoch

### 3. Evaluation
```bash
# Evaluate trained model
python evaluate.py datalake/models/.../best_model.pt --network cnn
```

Generates:
- Prediction scatter plots (true vs. predicted)
- Error distribution histograms
- Physics consistency visualizations

## Physics Background

### Transfer Matrix Method
The forward model computes THz wave propagation through a sample using:

1. **Complex Refractive Index**: `ñ = n + iκ`
   - `n`: Real refractive index (typically 1.2-7.0)
   - `κ`: Extinction coefficient (`< 0` for absorption)

2. **Fresnel Coefficients**: Reflection and transmission at interfaces

3. **Phase Accumulation**: `exp(i*k*ñ*d)` through material of thickness `d`

4. **Multiple Reflections**: Handled via recursive transfer matrix relations

### Frequency Domain Approach
- Works in frequency domain for computational efficiency
- Uses FFT with conjugate symmetry for real-valued time signals
- Typical bandwidth: 0-5 THz (configurable via `L` and `Δt`)

### Differentiability
The entire forward model is implemented in PyTorch with full gradient support, enabling:
- Physics-informed loss functions
- Gradient-based parameter optimization
- Hybrid supervised + physics training

## Configuration Tips

### Simulation Parameters
- **L** (time samples): 4096 typical, increase for higher resolution
- **Δt** (time step): 0.1 ps → 5 THz bandwidth
  - Smaller Δt → higher frequency range
  - Ensure sample round-trip time < time window

### Parameter Ranges
Match your experimental conditions:
```python
N_MIN = 1.1      # Minimum refractive index
N_MAX = 7.0      # Maximum refractive index
KAPPA_MIN = -0.1 # Strong absorption
KAPPA_MAX = 0.01 # Weak absorption
D_MIN = 100e-6   # 100 μm
D_MAX = 1000e-6  # 1000 μm (1 mm)
```

### Loss Function Selection
- **Supervised** (`α=1.0`): Fastest, requires ground truth labels
- **Physics** (`α=0.0`): No labels needed, slower due to forward model
- **Hybrid** (`α=0.5`): Best of both worlds, recommended for real data

### Network Selection
1. Start with **THz_Encoder_CNN** (default)
2. If underfitting → try **THz_Encoder_ResNet**
3. For complex spectral features → **THz_Encoder_MultiScale**

## Data Format

### Datalake Structure
```
datalake/
├── datasets/
│   └── <dataset_name>/
│       ├── train.npz
│       ├── val.npz
│       ├── test.npz
│       ├── metadata.json
│       ├── README.md
│       └── parameter_distributions.png
├── models/
│   └── <dataset_name>/
│       └── <network>/
│           └── <timestamp>/
│               ├── best_model.pt
│               ├── final_model.pt
│               └── run_config.json
└── results/
    └── <dataset_name>/
        └── <network>/
            └── <timestamp>/
                └── training_history.json
```

### NPZ File Contents
Each split file contains:
- `parameters/n`: Refractive indices `[N]`
- `parameters/kappa`: Extinction coefficients `[N]`
- `parameters/d`: Thicknesses in meters `[N]`
- `frequencies`: Frequency grid in Hz `[M+1]`
- `ml_input`: ML-ready input `[N, 2, M+1]` (real, imag channels)
- `T_complex`: Complex transmission coefficients `[N, M+1]` (optional)

## Dependencies

Required packages:
- PyTorch (with CUDA support for GPU training)
- NumPy
- Matplotlib
- tqdm (progress bars)

## Development Notes

### Adding New Networks
1. Create network class in [networks.py](networks.py) inheriting from `nn.Module`
2. Implement `forward()` with sigmoid output scaling
3. Add `count_parameters()` method
4. Register in `NETWORKS` dictionary

### Adding New Loss Functions
1. Create loss class in [train.py](train.py) inheriting from `nn.Module`
2. Update `get_loss_function()` factory
3. Handle in train/validation loops

### Extending Dataset Generation
1. Modify `DEFAULT_CONFIG` in [generate_dataset.py](generate_dataset.py)
2. Update `sample_parameters()` for new distributions
3. Add visualization functions as needed

## Troubleshooting

### Memory Issues
- Reduce `BATCH_SIZE` in config
- Use smaller `L` (time samples)
- Process datasets in smaller batches during generation

### Slow Training
- Switch from physics/hybrid to supervised loss
- Reduce network size (use CNN instead of ResNet/MultiScale)
- Enable GPU with `--device cuda`

### Poor Predictions
- Check parameter ranges match data
- Increase dataset size
- Try hybrid loss (`α=0.5`) for physics constraints
- Use data augmentation (add noise during training)

### Gradient Issues
- Check forward model input ranges (sigmoid outputs should be valid)
- Reduce learning rate if loss is unstable
- Enable gradient clipping if needed

## References

The transfer matrix method implementation follows standard THz-TDS analysis procedures. The real/imaginary input representation avoids phase wrapping ambiguities common in magnitude/phase approaches.

## License

See project root for license information.
