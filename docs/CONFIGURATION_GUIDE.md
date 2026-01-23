# Configuration Guide

This project uses two separate configuration systems with distinct purposes.

## Configuration Structure

```
THz-ML/
├── configs/                     # Dataset generation configurations
│   └── datasets/
│       ├── production_dataset.json
│       ├── small_test.json
│       └── *.json               # Other dataset configs
│
└── src/
    └── training_config.py       # Training, loss, and network configurations
```

## 1. Dataset Configurations (`configs/datasets/*.json`)

**Purpose:** Define parameters for generating synthetic THz transmission datasets.

**Location:** `configs/datasets/`

**Format:** JSON files

**Used by:** `src/generate_dataset.py`

**When to use:** When you want to generate a new dataset with specific parameter ranges, noise levels, or sample sizes.

### Example: `configs/datasets/production_dataset.json`

```json
{
  "dataset_name": "production_v1",
  "description": "Production THz transmission dataset",

  "simulation": {
    "L": 4096,
    "deltat": 1e-13,
    "device": "mps"
  },

  "parameter_ranges": {
    "n": {
      "min": 1.1,
      "max": 7.0,
      "distribution": "uniform"
    },
    "kappa": {
      "min": -0.1,
      "max": 0.01,
      "distribution": "uniform"
    },
    "d": {
      "min": 100e-6,
      "max": 1000e-6,
      "distribution": "uniform"
    }
  },

  "dataset_size": {
    "train": 50000,
    "val": 10000,
    "test": 10000
  }
}
```

### Key Parameters:

- `simulation.L` - Number of time points for simulation (typically 4096)
- `simulation.deltat` - Time step in seconds (1e-13 = 100 fs)
- `parameter_ranges` - Min/max values for n, κ, d
- `dataset_size` - Number of samples for train/val/test splits
- `noise.level` - Gaussian noise level to add to transmission data

### Creating a New Dataset Config:

1. Copy an existing JSON file from `configs/datasets/`
2. Modify parameters as needed
3. Change `dataset_name` to something unique
4. Generate the dataset:

```bash
python configs/datasets/make_large_dataset.py your_config.json
```

---

## 2. Training Configuration (`src/training_config.py`)

**Purpose:** Define parameters for model training, network architecture, loss functions, and optimization.

**Location:** `src/training_config.py`

**Format:** Python class (`Config`)

**Used by:** All training and evaluation scripts

**When to use:** When you want to adjust training hyperparameters, network architecture, or loss function settings.

### Key Configuration Sections:

#### A. Dataset Parameters

```python
DATASET_NAME = 'production_v1'  # Which dataset to use from datalake
```

#### B. Simulation Parameters (for physics loss)

```python
L = 4096           # Time points
DELTAT = 0.1e-12   # Time step
```

#### C. Parameter Ranges (must match dataset)

```python
N_MIN = 1.1
N_MAX = 7.0
KAPPA_MIN = -0.1
KAPPA_MAX = 0.01
D_MIN = 100e-6
D_MAX = 1000e-6
```

#### D. Training Hyperparameters

```python
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-5
```

#### E. Loss Function Configuration

```python
LOSS_TYPE = 'hybrid'  # 'supervised', 'physics', or 'hybrid'
ALPHA = 0.5  # Weight for hybrid loss (1.0=supervised, 0.0=physics)

# Loss scaling (NEW)
USE_NORMALIZED_SUPERVISED_LOSS = True  # Normalize parameters to [0,1]
PHYSICS_LOSS_SCALE = 1.0  # Scaling factor for physics loss
PARAM_WEIGHTS = {  # Per-parameter weights
    'n': 1.0,
    'kappa': 1.0,
    'd': 1.0
}
```

#### F. Network Architecture (NEW)

```python
# CNN Scalable
CNN_WIDTH_MULT = 1.0      # Channel multiplier (0.5=half, 2.0=double)
CNN_NUM_BLOCKS = 4        # Number of conv blocks (2-6)
CNN_BASE_CHANNELS = 32    # Base channel count
CNN_DROPOUT = 0.2         # Dropout rate

# ResNet Scalable
RESNET_WIDTH_MULT = 1.0
RESNET_NUM_RES_BLOCKS = [2, 2, 2]  # Blocks per stage
RESNET_BASE_CHANNELS = 64
RESNET_DROPOUT = 0.2

# FC layers
FC_HIDDEN_DIMS = [128, 64]  # Hidden layer dimensions
```

#### G. Early Stopping

```python
PATIENCE = 15      # Stop if no improvement for N epochs
MIN_DELTA = 1e-6   # Minimum improvement to count
```

### Modifying Training Config:

**Option 1: Edit `src/training_config.py` directly**

```python
# src/training_config.py
class Config:
    CNN_WIDTH_MULT = 1.5  # Increase model capacity
    LEARNING_RATE = 5e-4  # Lower learning rate
```

**Option 2: Override in training script**

```python
# In train_notebook.ipynb or train.py
from training_config import Config

config = Config()
config.CNN_WIDTH_MULT = 1.5
config.LEARNING_RATE = 5e-4
```

**Option 3: Command line (for CLI training)**

```python
import argparse
config = Config()
args = parser.parse_args()
config.LEARNING_RATE = args.lr
```

---

## Workflow Examples

### Example 1: Generate and Train on New Dataset

**Step 1:** Create dataset config

```bash
# Create configs/datasets/my_dataset.json
{
  "dataset_name": "my_experiment",
  "parameter_ranges": {
    "n": {"min": 1.5, "max": 4.0},
    "kappa": {"min": -0.05, "max": 0.001},
    "d": {"min": 200e-6, "max": 800e-6}
  },
  "dataset_size": {
    "train": 10000,
    "val": 2000,
    "test": 2000
  }
}
```

**Step 2:** Generate dataset

```bash
python configs/datasets/make_large_dataset.py my_dataset.json
```

**Step 3:** Update training config

```python
# src/training_config.py
DATASET_NAME = 'my_experiment'
N_MIN = 1.5
N_MAX = 4.0
# ... (match dataset ranges)
```

**Step 4:** Train

```bash
python src/train.py --network cnn_scalable
```

### Example 2: Experiment with Different Model Sizes

Keep the same dataset, just modify network architecture:

```python
# src/training_config.py

# Experiment 1: Tiny model
CNN_WIDTH_MULT = 0.5
CNN_NUM_BLOCKS = 2

# Experiment 2: Large model
CNN_WIDTH_MULT = 1.5
CNN_NUM_BLOCKS = 5
```

### Example 3: Tune Loss Function

```python
# src/training_config.py

# Emphasize physics loss
ALPHA = 0.3  # 30% supervised, 70% physics
PHYSICS_LOSS_SCALE = 2.0  # Boost physics loss

# Emphasize kappa learning
PARAM_WEIGHTS = {
    'n': 1.0,
    'kappa': 2.0,  # 2x weight on kappa
    'd': 1.0
}
```

---

## Quick Reference

| What do you want to do? | Which config to modify? |
|-------------------------|------------------------|
| Generate new synthetic data with different parameter ranges | `configs/datasets/*.json` |
| Change dataset sample count | `configs/datasets/*.json` |
| Add noise to dataset | `configs/datasets/*.json` |
| Switch which dataset to use for training | `src/training_config.py` → `DATASET_NAME` |
| Change network architecture (size, depth) | `src/training_config.py` → `CNN_*` or `RESNET_*` |
| Adjust learning rate, batch size, epochs | `src/training_config.py` → Training params |
| Change loss function type | `src/training_config.py` → `LOSS_TYPE`, `ALPHA` |
| Balance parameter learning (n, κ, d) | `src/training_config.py` → `PARAM_WEIGHTS` |
| Scale physics vs supervised loss | `src/training_config.py` → `PHYSICS_LOSS_SCALE` |

---

## Important Notes

1. **Parameter ranges must match:** If you generate a dataset with specific ranges (e.g., n: 1.1-7.0), make sure `training_config.py` has matching `N_MIN` and `N_MAX` values.

2. **Simulation parameters must match:** If your dataset was generated with `L=4096, deltat=1e-13`, use the same values in `training_config.py` for physics-informed loss.

3. **Don't confuse the two:**
   - Dataset configs (JSON) are for **data generation** (one-time setup)
   - Training config (Python) is for **model training** (frequent experimentation)

4. **Dataset config is immutable after generation:** Once you generate a dataset, changing its JSON config won't affect the already-generated data. You'd need to regenerate.

5. **Training config is live:** Changes to `training_config.py` take effect immediately on the next training run.

---

## Migration Note

Previously, training configuration was in `src/configs.py`. It has been renamed to `src/training_config.py` to avoid confusion with the `configs/` directory used for dataset generation.

All imports have been updated from:
```python
from configs import Config
```
to:
```python
from training_config import Config
```
