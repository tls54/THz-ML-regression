# Training Configuration Presets

This directory contains preset training configurations for the THz parameter extraction models.

## Available Configs

### CNN Scalable Architectures

| Config | Description | Parameters | Use Case |
|--------|-------------|------------|----------|
| `cnn_tiny.json` | Tiny CNN for rapid prototyping | ~10K | Fast experiments, debugging |
| `cnn_small.json` | Small CNN baseline | ~50K | Quick baseline, limited data |
| `cnn_base.json` | Base CNN - **recommended starting point** | ~200K | General purpose, good balance |
| `cnn_large.json` | Large CNN for high capacity | ~800K | When base underfits, lots of data |

### ResNet Architectures

| Config | Description | Parameters | Use Case |
|--------|-------------|------------|----------|
| `resnet_base.json` | Base ResNet with skip connections | ~250K | When CNN plateaus, complex patterns |
| `resnet_deep.json` | Deep ResNet for maximum capacity | ~1M | Very complex data, underfitting |

## Usage

### In Jupyter Notebook

```python
from config_loader import load_training_config, print_config_summary

# Load a preset config
config = load_training_config('cnn_base')

# Print summary
print_config_summary(config)

# Use with model
from networks import get_network
model = get_network('cnn_scalable', config=config)
```

### In Python Script

```python
import sys
sys.path.insert(0, 'src')

from config_loader import load_training_config
from train import train

# Load config
config = load_training_config('cnn_large')

# Train with it
train(config, network_name='cnn_scalable', run_name='my_experiment')
```

### Command Line (with custom wrapper)

```bash
python train_with_config.py --config cnn_base --run-name my_experiment
```

## Configuration Structure

Each JSON file contains:

```json
{
  "name": "config_identifier",
  "description": "Human-readable description",

  "network": {
    "type": "cnn_scalable | resnet_scalable",
    "width_mult": 1.0,        // Channel multiplier
    "num_blocks": 4,          // For CNN
    "num_res_blocks": [2,2,2], // For ResNet
    "base_channels": 32,
    "fc_hidden_dims": [128, 64],
    "dropout": 0.2
  },

  "training": {
    "num_epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 15
  },

  "loss": {
    "type": "hybrid",         // supervised | physics | hybrid
    "alpha": 0.5,             // Weight for hybrid loss
    "use_normalized_supervised_loss": true,
    "physics_loss_scale": 1.0,
    "param_weights": {
      "n": 1.0,
      "kappa": 1.0,
      "d": 1.0
    }
  },

  "dataset": {
    "name": "production_v1"   // Dataset to use
  }
}
```

## Creating Custom Configs

1. Copy an existing config file
2. Modify parameters as needed
3. Save with a descriptive name (e.g., `cnn_custom_v2.json`)
4. Load it: `config = load_training_config('cnn_custom_v2')`

## Recommended Workflow

### 1. Start with Base

```python
config = load_training_config('cnn_base')
```

Train and evaluate. Check R² scores for n, κ, and d.

### 2. If Underfitting (All R² < 0.85)

Scale up the model:

```python
config = load_training_config('cnn_large')
# OR
config = load_training_config('resnet_base')
```

### 3. If Overfitting (Train R² > Val R²)

Scale down or increase regularization:

```python
config = load_training_config('cnn_small')

# OR manually adjust
config = load_training_config('cnn_base')
config.CNN_DROPOUT = 0.3       # Increase dropout
config.WEIGHT_DECAY = 1e-4     # Increase weight decay
```

### 4. Fine-tune Loss Function

```python
config = load_training_config('cnn_base')

# Emphasize physics loss
config.ALPHA = 0.3
config.PHYSICS_LOSS_SCALE = 2.0

# Emphasize κ learning (if R² is low)
config.PARAM_WEIGHTS = {'n': 1.0, 'kappa': 2.0, 'd': 1.0}
```

## Quick Reference

**Fastest experiments:** `cnn_tiny`
**Best starting point:** `cnn_base` ⭐
**Maximum capacity:** `resnet_deep`
**Custom experiment:** Copy and modify any config

## Notes

- All configs use the **normalized supervised loss** by default (ensures balanced learning)
- All configs use **hybrid loss** (supervised + physics) by default
- Parameter ranges are defined in `src/training_config.py` and should match your dataset
- Adjust `dataset.name` to use different datasets from `datalake/datasets/`
