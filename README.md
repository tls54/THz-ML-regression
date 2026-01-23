# THz-ML

A machine learning framework for extracting material parameters from terahertz (THz) transmission spectroscopy data using physics-informed neural networks.

## Overview

THz-ML predicts material properties (refractive index `n`, extinction coefficient `κ`, and thickness `d`) from THz transmission spectra. The system combines supervised learning with physics-based constraints through a differentiable forward model based on the transfer matrix method.

## Features

- **Multiple neural network architectures**: CNN, ResNet, MultiScale, and scalable variants
- **Physics-informed training**: Optional physics loss using differentiable THz simulation
- **Hybrid loss functions**: Combine supervised and physics-based learning
- **Flexible configuration**: JSON configs for datasets and training presets
- **Organized data management**: Datalake structure for datasets, models, and results

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a quick training test (5 epochs)
python test_train.py

# View results
open datalake/results/quick_test/cnn/*/training_plots.png
```

## Project Structure

```
THz-ML/
├── src/                    # Core ML pipeline
│   ├── simulate.py         # Physics simulation (transfer matrix method)
│   ├── networks.py         # Neural network architectures
│   ├── train.py            # Training script
│   ├── evaluate.py         # Model evaluation
│   ├── generate_dataset.py # Dataset generation
│   └── training_config.py  # Configuration
├── configs/
│   ├── training/           # Network presets (cnn_tiny, cnn_base, resnet_base, etc.)
│   └── datasets/           # Dataset generation configs
├── datalake/               # Data storage (gitignored)
│   ├── datasets/           # Generated datasets
│   ├── models/             # Trained checkpoints
│   └── results/            # Training history and plots
├── docs/                   # Documentation
└── *.ipynb                 # Jupyter notebooks
```

## Usage

### Generate a Dataset

```bash
# Quick test dataset
python src/generate_dataset.py --quick

# From config file
python src/generate_dataset.py --config configs/datasets/production_dataset.json
```

### Train a Model

```bash
# Basic training
python src/train.py --network cnn --dataset quick_test

# With specific device
python src/train.py --network resnet --dataset production_v1 --device cuda

# List available datasets
python src/train.py --list-datasets
```

### Evaluate

```bash
python src/evaluate.py datalake/models/<dataset>/<network>/*/best_model.pt --network cnn
```

## Network Architectures

| Network | Parameters | Description |
|---------|-----------|-------------|
| `cnn` | ~200K | 1D CNN, recommended for most cases |
| `resnet` | ~250K | ResNet with skip connections |
| `multiscale` | ~250K | Multi-branch for varying frequencies |
| `cnn_scalable` | 10K-2M | Configurable width and depth |
| `resnet_scalable` | 20K-3M | Configurable residual blocks |

## Loss Functions

Configure in `src/training_config.py`:

- **`supervised`**: Direct MSE on parameters (requires ground truth)
- **`physics`**: Reconstruction loss via forward model (no labels needed)
- **`hybrid`**: Weighted combination (recommended)

```python
LOSS_TYPE = 'hybrid'
ALPHA = 0.5  # Balance between supervised and physics loss
```

## Configuration

### Training Config (`src/training_config.py`)

```python
# Key parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100

# Network scaling
CNN_WIDTH_MULT = 1.0      # Channel multiplier
CNN_NUM_BLOCKS = 4        # Depth

# Parameter ranges
N_MIN, N_MAX = 1.1, 7.0
KAPPA_MIN, KAPPA_MAX = -0.1, 0.01
D_MIN, D_MAX = 100e-6, 1000e-6
```

### Network Presets

Pre-configured options in `configs/training/`:
- `cnn_tiny.json` (~10K params) - rapid prototyping
- `cnn_base.json` (~200K params) - recommended
- `cnn_large.json` (~800K params) - complex patterns
- `resnet_base.json`, `resnet_deep.json`

## Physical Model

The forward model simulates THz wave propagation using the transfer matrix method:

- **Input**: Complex transmission spectrum `T(ω)` as real/imaginary channels
- **Output**: Material parameters `[n, κ, d]`
- **Physics**: Fresnel coefficients, phase accumulation, multiple reflections

Parameter conventions:
- `n`: Refractive index (1.1–7.0)
- `κ`: Extinction coefficient (negative for absorption)
- `d`: Sample thickness (100–1000 μm)

## Documentation

See the `docs/` folder:

- [QUICK_START.md](docs/QUICK_START.md) - Get running in 5 minutes
- [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) - Configuration reference
- [NETWORK_SCALING_GUIDE.md](docs/NETWORK_SCALING_GUIDE.md) - Tune model capacity
- [LEARNABILITY_GUIDE.md](docs/LEARNABILITY_GUIDE.md) - Parameter learnability analysis
- [DISPERSIVE_GUIDE.md](docs/DISPERSIVE_GUIDE.md) - Frequency-dependent materials

## Notebooks

- `train_notebook.ipynb` - Interactive training
- `explore_dataset.ipynb` - Dataset exploration
- `test_learnability.ipynb` - Learnability analysis

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+

Install with:
```bash
pip install -r requirements.txt
```

## Device Support

```python
# In training_config.py or via --device flag
DEVICE = 'cpu'   # Default
DEVICE = 'cuda'  # NVIDIA GPU
DEVICE = 'mps'   # Apple Silicon
```
