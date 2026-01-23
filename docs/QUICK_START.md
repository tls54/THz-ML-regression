# Quick Start Guide - Training Test

This guide will help you run a quick training test to verify everything works.

## TL;DR - Fastest Way to Test

```bash
cd /Users/theosmith/Documents/Projects/Python/THz-ML
python test_train.py
```

This runs 5 epochs on the quick_test dataset (~5-10 minutes on CPU).

## What Gets Tested

- ✓ Dataset loading from datalake
- ✓ Neural network initialization
- ✓ Forward pass
- ✓ Loss computation (hybrid: supervised + physics)
- ✓ Backward pass and optimization
- ✓ Validation metrics computation
- ✓ Model checkpointing
- ✓ Training history tracking
- ✓ Automatic plot generation

## Step-by-Step Instructions

### 1. Prerequisites

Make sure you have the required packages:

```bash
python3 -c "import torch, numpy, matplotlib; print('✓ Dependencies OK')"
```

If this fails, install the missing packages:

```bash
pip install torch numpy matplotlib tqdm
```

### 2. Run the Quick Test

**Option A: Use the test script (recommended)**

```bash
python test_train.py
```

**Option B: Direct command line**

```bash
python src/train.py --network cnn --dataset quick_test --device cpu
```

Option A is faster (5 epochs), Option B runs the full 100 epochs.

### 3. Monitor Progress

You'll see output like:

```
Loading dataset: quick_test
✓ Loaded 100 samples

Initializing cnn network...
Model parameters: 1,234,567

Starting Training
============================================================

Epoch 1/5
------------------------------------------------------------
Training: 100%|████████████████| 4/4 [00:15<00:00]

Epoch 1 Summary:
  Train Loss: 0.123456
  Val Loss:   0.234567
  
Validation Metrics
============================================================
n_mae                         : 0.12345
n_rmse                        : 0.23456
n_r2                          : 0.8765
...
```

### 4. Check Results

After training completes:

```bash
# View the training plot
open datalake/results/quick_test/cnn/*/training_plots.png

# Or on Linux
xdg-open datalake/results/quick_test/cnn/*/training_plots.png
```

The plot shows:
- Loss curves (train vs validation)
- Accuracy metrics (MAE, RMSE, R²) for all 3 parameters
- Final performance summary

## Understanding the Output

### Directory Structure Created

```
datalake/
├── models/quick_test/cnn/quick_test_run/
│   ├── best_model.pt          # Best model checkpoint
│   ├── final_model.pt         # Final epoch model
│   ├── checkpoint_epoch_2.pt  # Periodic checkpoint
│   ├── checkpoint_epoch_4.pt
│   └── run_config.json        # Training configuration
│
└── results/quick_test/cnn/quick_test_run/
    ├── training_history.json  # Loss and metrics per epoch
    └── training_plots.png     # Visualization
```

### What to Look For

**Good signs:**
- ✓ Loss decreases over epochs
- ✓ R² scores increase (toward 1.0)
- ✓ MAE/RMSE decrease
- ✓ Train and val loss track together (no overfitting)

**Warning signs:**
- ✗ Loss increases or plateaus immediately
- ✗ R² stays near 0 or becomes negative
- ✗ Train loss << val loss (overfitting)

For the quick_test dataset with only 5 epochs, expect:
- R² around 0.5-0.8 (not fully converged)
- Losses still decreasing (needs more epochs)

## Next Steps

### If Test Succeeds ✓

1. **Generate a larger dataset** for better results:
   ```bash
   python src/generate_dataset.py
   ```

2. **Train for more epochs** (edit `test_train.py` or use Option B)

3. **Try different networks**:
   ```bash
   python src/train.py --network resnet --dataset quick_test --device cpu
   ```

4. **Evaluate on test set**:
   ```bash
   python src/evaluate.py datalake/models/quick_test/cnn/*/best_model.pt --network cnn
   ```

5. **Explore the dataset** using the Jupyter notebook:
   ```bash
   jupyter notebook explore_dataset.ipynb
   ```

### If Test Fails ✗

1. **Check error message** - most common issues:
   - Dataset not found → Run `python src/generate_dataset.py --quick`
   - Import errors → Make sure you're in project root
   - CUDA error → Add `--device cpu`
   - Memory error → Reduce batch size in config

2. **Verify dataset exists**:
   ```bash
   ls datalake/datasets/quick_test/
   # Should show: train.npz, val.npz, test.npz, metadata.json
   ```

3. **Check Python path**:
   ```bash
   pwd  # Should be .../THz-ML
   ls src/  # Should show train.py, configs.py, etc.
   ```

## Advanced Options

### Use GPU (if available)

Edit `test_train.py`:
```python
config.DEVICE = 'cuda'  # Instead of 'cpu'
```

### Longer training

Edit `test_train.py`:
```python
config.NUM_EPOCHS = 20  # Or more
config.PATIENCE = 5
```

### Different loss function

Edit `test_train.py`:
```python
config.LOSS_TYPE = 'supervised'  # Faster
# or
config.LOSS_TYPE = 'physics'  # Slower but no labels needed
```

### Different network

```bash
python src/train.py --network resnet --dataset quick_test --device cpu
```

Available networks: `cnn`, `resnet`, `multiscale`, `mlp`

## Benchmarks

Expected performance on quick_test dataset (100 train, 20 val samples):

| Epochs | Time (CPU) | Time (GPU) | Typical R² |
|--------|------------|------------|------------|
| 5      | ~5 min     | ~1 min     | 0.5-0.7    |
| 20     | ~20 min    | ~4 min     | 0.7-0.85   |
| 50     | ~50 min    | ~10 min    | 0.85-0.95  |

Note: Results vary based on random initialization and data splits.

## Getting Help

- Check [src/README.md](src/README.md) for detailed documentation
- Review training plots to diagnose issues
- Look at training_history.json for epoch-by-epoch metrics
- Use the explore_dataset.ipynb notebook to inspect data

## Summary

Quick test command:
```bash
python test_train.py
```

Check results:
```bash
open datalake/results/quick_test/cnn/*/training_plots.png
```

That's it! You should see training progress and a nice plot at the end showing model performance.
