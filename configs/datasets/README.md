# Dataset Generation Configs

This directory contains configurations for generating THz transmission datasets.

## Usage

### Method 1: JSON Config File (Recommended)
```bash
python src/generate_dataset.py --config configs/datasets/production_dataset.json
```

### Method 2: Python Script
```bash
python configs/datasets/make_large_dataset.py --variant standard
```

### Method 3: Quick Test
```bash
python src/generate_dataset.py --quick
```

## Available Configs

### Pre-defined JSON Configs

| Config File | Samples | Description |
|------------|---------|-------------|
| `small_test.json` | 1,000 | Small dataset for quick testing |
| `production_dataset.json` | 70,000 | Production dataset with good coverage |

### Python Script Configs

**`make_large_dataset.py`** - Generates large production datasets with variants:
- `--variant standard` (100k samples) - Standard large dataset
- `--variant noisy` (100k samples) - Higher noise level (0.5%)
- `--variant log_thickness` (100k samples) - Log-uniform thickness distribution

## Config Structure

```json
{
  "dataset_name": "my_dataset",
  "description": "Description of dataset",

  "simulation": {
    "L": 4096,           // Time samples (power of 2)
    "deltat": 1e-13,     // Time step (seconds)
    "device": "mps"      // "cpu", "cuda", or "mps"
  },

  "parameter_ranges": {
    "n": {
      "min": 1.1,
      "max": 7.0,
      "distribution": "uniform"  // or "loguniform"
    },
    "kappa": {
      "min": -0.1,
      "max": 0.01,
      "distribution": "uniform"
    },
    "d": {
      "min": 100e-6,     // meters
      "max": 1000e-6,
      "distribution": "uniform"
    }
  },

  "dataset_size": {
    "train": 50000,
    "val": 10000,
    "test": 10000
  },

  "noise": {
    "enabled": true,
    "level": 1e-3,       // Standard deviation
    "type": "gaussian"
  },

  "output": {
    "save_time_domain": false,
    "save_frequency_domain": true,
    "save_ml_input": true,
    "compression": true
  },

  "random_seed": 42
}
```

## Parameter Guidelines

### Simulation Parameters
- **L**: Use power of 2 (2048, 4096, 8192) for FFT efficiency
- **deltat**: Smaller = higher frequency resolution
  - `1e-13` (0.1 ps) → ~5 THz bandwidth
  - `5e-14` (0.05 ps) → ~10 THz bandwidth
- **device**: Use "mps" on Mac, "cuda" on NVIDIA GPU, "cpu" otherwise

### Parameter Ranges
Based on typical THz materials:
- **n** (refractive index): 1.1 - 7.0 covers most materials
- **κ** (extinction coefficient): -0.1 to 0.01 (negative = absorption)
- **d** (thickness): 50-1000 μm typical for THz samples

### Dataset Size
- **Small test**: 1k samples (quick iteration)
- **Medium**: 10-20k samples (initial training)
- **Production**: 50-100k samples (full training)
- **Split ratio**: 80/10/10 or 70/15/15 for train/val/test

### Noise
- **Low noise**: 1e-3 (0.1%) - clean experimental conditions
- **Medium noise**: 5e-3 (0.5%) - typical lab conditions
- **High noise**: 1e-2 (1%) - challenging conditions

## Creating Custom Configs

### Option 1: Copy and modify existing JSON
```bash
cp configs/datasets/production_dataset.json configs/datasets/my_dataset.json
# Edit my_dataset.json with your parameters
python src/generate_dataset.py --config configs/datasets/my_dataset.json
```

### Option 2: Write Python script for complex configs
See `make_large_dataset.py` as a template for configs that need:
- Calculated values
- Multiple variants
- Conditional logic
- Parameter sweeps

## Tips

1. **Start small**: Test with `small_test.json` before generating large datasets
2. **Use descriptive names**: Include version numbers (e.g., `production_v1`)
3. **Document changes**: Update dataset description when modifying configs
4. **Version control**: Commit config files alongside code
5. **Reproducibility**: Keep random_seed consistent for reproducible datasets
