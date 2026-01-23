# Network Scaling Guide

This guide explains how to scale CNN and ResNet architectures for the THz parameter extraction models.

## Quick Start

### Using Scalable Networks in Notebook

```python
# In train_notebook.ipynb, cell 6:
NETWORK_NAME = 'cnn_scalable'  # or 'resnet_scalable'

# The network will automatically use the scaling parameters from configs.py
```

### Using Scalable Networks in CLI

```bash
python src/train.py --network cnn_scalable --dataset production_v1
```

## Configuration Parameters

### CNN Scalable (`configs.py`)

```python
CNN_WIDTH_MULT = 1.0       # Channel multiplier
CNN_NUM_BLOCKS = 4         # Number of conv blocks (2-6 recommended)
CNN_BASE_CHANNELS = 32     # Base channel count
CNN_DROPOUT = 0.2          # Dropout rate
FC_HIDDEN_DIMS = [128, 64] # FC layer dimensions
```

### ResNet Scalable (`configs.py`)

```python
RESNET_WIDTH_MULT = 1.0           # Channel multiplier
RESNET_NUM_RES_BLOCKS = [2, 2, 2] # Res blocks per stage
RESNET_BASE_CHANNELS = 64         # Base channel count
RESNET_DROPOUT = 0.2              # Dropout rate
FC_HIDDEN_DIMS = [64]             # FC layer dimensions
```

## Scaling Strategies

### Width Scaling (Channel Count)

Control model capacity by adjusting channel counts:

```python
# Tiny model (1/4 parameters)
CNN_WIDTH_MULT = 0.5
CNN_BASE_CHANNELS = 32
# Channels: [16, 32, 64, 128]

# Base model (default)
CNN_WIDTH_MULT = 1.0
CNN_BASE_CHANNELS = 32
# Channels: [32, 64, 128, 256]

# Large model (4x parameters)
CNN_WIDTH_MULT = 2.0
CNN_BASE_CHANNELS = 32
# Channels: [64, 128, 256, 512]

# XLarge model
CNN_WIDTH_MULT = 2.0
CNN_BASE_CHANNELS = 64
# Channels: [128, 256, 512, 1024]
```

### Depth Scaling (Number of Layers)

Control model depth by adjusting number of blocks:

```python
# Shallow CNN (fewer parameters, faster)
CNN_NUM_BLOCKS = 2
# Conv blocks: 2 layers

# Medium CNN (default)
CNN_NUM_BLOCKS = 4
# Conv blocks: 4 layers

# Deep CNN (more capacity)
CNN_NUM_BLOCKS = 6
# Conv blocks: 6 layers
```

### ResNet Depth Scaling

```python
# Shallow ResNet
RESNET_NUM_RES_BLOCKS = [1, 1, 1]  # 3 residual blocks total

# Base ResNet (default)
RESNET_NUM_RES_BLOCKS = [2, 2, 2]  # 6 residual blocks total

# Deep ResNet
RESNET_NUM_RES_BLOCKS = [3, 4, 3]  # 10 residual blocks total

# Very Deep ResNet
RESNET_NUM_RES_BLOCKS = [3, 4, 6, 3]  # 16 residual blocks, 4 stages
```

## Preset Configurations

### Tiny Models (Fast Training, Limited Capacity)

```python
# Tiny CNN (~10K parameters)
CNN_WIDTH_MULT = 0.5
CNN_NUM_BLOCKS = 2
CNN_BASE_CHANNELS = 16
FC_HIDDEN_DIMS = [64, 32]

# Tiny ResNet (~20K parameters)
RESNET_WIDTH_MULT = 0.5
RESNET_NUM_RES_BLOCKS = [1, 1]
RESNET_BASE_CHANNELS = 32
FC_HIDDEN_DIMS = [32]
```

### Small Models (Good Baseline)

```python
# Small CNN (~50K parameters)
CNN_WIDTH_MULT = 0.75
CNN_NUM_BLOCKS = 3
CNN_BASE_CHANNELS = 32
FC_HIDDEN_DIMS = [128, 64]

# Small ResNet (~80K parameters)
RESNET_WIDTH_MULT = 0.75
RESNET_NUM_RES_BLOCKS = [2, 2]
RESNET_BASE_CHANNELS = 64
FC_HIDDEN_DIMS = [64]
```

### Base Models (Default, Recommended Starting Point)

```python
# Base CNN (~200K parameters)
CNN_WIDTH_MULT = 1.0
CNN_NUM_BLOCKS = 4
CNN_BASE_CHANNELS = 32
FC_HIDDEN_DIMS = [128, 64]

# Base ResNet (~250K parameters)
RESNET_WIDTH_MULT = 1.0
RESNET_NUM_RES_BLOCKS = [2, 2, 2]
RESNET_BASE_CHANNELS = 64
FC_HIDDEN_DIMS = [64]
```

### Large Models (High Capacity)

```python
# Large CNN (~800K parameters)
CNN_WIDTH_MULT = 1.5
CNN_NUM_BLOCKS = 5
CNN_BASE_CHANNELS = 48
FC_HIDDEN_DIMS = [256, 128]

# Large ResNet (~1M parameters)
RESNET_WIDTH_MULT = 1.5
RESNET_NUM_RES_BLOCKS = [3, 4, 3]
RESNET_BASE_CHANNELS = 64
FC_HIDDEN_DIMS = [128]
```

### XLarge Models (Maximum Capacity)

```python
# XLarge CNN (~2M parameters)
CNN_WIDTH_MULT = 2.0
CNN_NUM_BLOCKS = 6
CNN_BASE_CHANNELS = 64
FC_HIDDEN_DIMS = [512, 256, 128]

# XLarge ResNet (~3M parameters)
RESNET_WIDTH_MULT = 2.0
RESNET_NUM_RES_BLOCKS = [3, 4, 6, 3]
RESNET_BASE_CHANNELS = 64
FC_HIDDEN_DIMS = [256, 128]
```

## Experimentation Workflow

### 1. Start with Base Model

```python
# configs.py
CNN_WIDTH_MULT = 1.0
CNN_NUM_BLOCKS = 4
```

Train and check validation metrics.

### 2. If Underfitting (Low Train & Val Performance)

The model lacks capacity. Scale up:

```python
# Option A: Increase width
CNN_WIDTH_MULT = 1.5

# Option B: Increase depth
CNN_NUM_BLOCKS = 5

# Option C: Both
CNN_WIDTH_MULT = 1.5
CNN_NUM_BLOCKS = 5
```

### 3. If Overfitting (Good Train, Poor Val Performance)

The model has too much capacity. Scale down or regularize:

```python
# Option A: Reduce width
CNN_WIDTH_MULT = 0.75

# Option B: Reduce depth
CNN_NUM_BLOCKS = 3

# Option C: Increase regularization
CNN_DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
```

### 4. If Training is Too Slow

Reduce model size:

```python
CNN_WIDTH_MULT = 0.5
CNN_NUM_BLOCKS = 2
```

## Architecture Comparison

| Network | Typical Params | Speed | Capacity | Best For |
|---------|---------------|-------|----------|----------|
| `cnn` | ~200K | Fast | Medium | General purpose, baseline |
| `resnet` | ~250K | Medium | High | Complex patterns, needs depth |
| `cnn_scalable` (tiny) | ~10K | Very Fast | Low | Rapid prototyping |
| `cnn_scalable` (base) | ~200K | Fast | Medium | Production baseline |
| `cnn_scalable` (large) | ~800K | Medium | High | When base underfits |
| `resnet_scalable` (base) | ~250K | Medium | High | Skip connections help |
| `resnet_scalable` (deep) | ~1M | Slow | Very High | Very complex data |

## Tips

1. **Start small, scale up**: Begin with base or small config, then increase if needed
2. **Width before depth**: Increasing width is usually more effective than depth for CNNs
3. **Watch parameter count**: Use `model.count_parameters()` to check model size
4. **Balance is key**: Very large models may overfit; very small may underfit
5. **Use early stopping**: Let the model train with patience, don't stop too early
6. **Compare on same data**: Always use the same dataset split for fair comparison

## Advanced: Custom Configurations in Notebook

You can override config parameters directly in the notebook:

```python
# In train_notebook.ipynb, after loading config

# Override with custom architecture
config.CNN_WIDTH_MULT = 1.5
config.CNN_NUM_BLOCKS = 5
config.FC_HIDDEN_DIMS = [256, 128, 64]

# Then create model with get_network
from networks import get_network
model = get_network('cnn_scalable', config=config).to(device)

print(f"Model parameters: {model.count_parameters():,}")
```

## Monitoring Model Size

Check parameter count before training:

```python
model = get_network('cnn_scalable', config=config)
print(f"Parameters: {model.count_parameters():,}")

# Compare different configs
configs_to_test = [
    (0.5, 2),   # tiny
    (1.0, 4),   # base
    (1.5, 5),   # large
]

for width, depth in configs_to_test:
    config.CNN_WIDTH_MULT = width
    config.CNN_NUM_BLOCKS = depth
    model = get_network('cnn_scalable', config=config)
    print(f"Width={width}, Depth={depth}: {model.count_parameters():,} params")
```

## Example Training Runs

```python
# Experiment 1: Tiny model (fast baseline)
NETWORK_NAME = 'cnn_scalable'
config.CNN_WIDTH_MULT = 0.5
config.CNN_NUM_BLOCKS = 2

# Experiment 2: Base model
NETWORK_NAME = 'cnn_scalable'
config.CNN_WIDTH_MULT = 1.0
config.CNN_NUM_BLOCKS = 4

# Experiment 3: Large model (if base underfits)
NETWORK_NAME = 'cnn_scalable'
config.CNN_WIDTH_MULT = 1.5
config.CNN_NUM_BLOCKS = 5

# Experiment 4: ResNet (if CNN plateaus)
NETWORK_NAME = 'resnet_scalable'
config.RESNET_WIDTH_MULT = 1.0
config.RESNET_NUM_RES_BLOCKS = [2, 2, 2]
```
