"""
Utility for loading training configurations from JSON files.
"""

import json
from pathlib import Path
from training_config import Config


def load_training_config(config_name, config_dir=None):
    """
    Load a training configuration from JSON file and apply to Config object.

    Args:
        config_name: Name of config file (with or without .json extension)
        config_dir: Directory containing config files (default: configs/training/)

    Returns:
        Config object with settings loaded from JSON

    Example:
        >>> config = load_training_config('cnn_base')
        >>> config = load_training_config('cnn_large.json')
        >>> config = load_training_config('my_custom', config_dir='my_configs/')
    """
    # Default config directory
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'configs' / 'training'
    else:
        config_dir = Path(config_dir)

    # Add .json extension if not present
    if not config_name.endswith('.json'):
        config_name = f'{config_name}.json'

    config_path = config_dir / config_name

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs in {config_dir}:\n" +
            "\n".join(f"  - {f.stem}" for f in config_dir.glob('*.json'))
        )

    # Load JSON
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create base Config object
    config = Config()

    # Apply network settings
    if 'network' in config_dict:
        net = config_dict['network']

        config._network_type = net.get('type', 'cnn_scalable')

        if net.get('type') == 'cnn_scalable':
            config.CNN_WIDTH_MULT = net.get('width_mult', 1.0)
            config.CNN_NUM_BLOCKS = net.get('num_blocks', 4)
            config.CNN_BASE_CHANNELS = net.get('base_channels', 32)
            config.CNN_DROPOUT = net.get('dropout', 0.2)
            if 'fc_hidden_dims' in net:
                config.FC_HIDDEN_DIMS = net['fc_hidden_dims']

        elif net.get('type') == 'resnet_scalable':
            config.RESNET_WIDTH_MULT = net.get('width_mult', 1.0)
            config.RESNET_NUM_RES_BLOCKS = net.get('num_res_blocks', [2, 2, 2])
            config.RESNET_BASE_CHANNELS = net.get('base_channels', 64)
            config.RESNET_DROPOUT = net.get('dropout', 0.2)
            if 'fc_hidden_dims' in net:
                config.FC_HIDDEN_DIMS = net['fc_hidden_dims']

    # Apply training settings
    if 'training' in config_dict:
        train = config_dict['training']
        config.NUM_EPOCHS = train.get('num_epochs', config.NUM_EPOCHS)
        config.BATCH_SIZE = train.get('batch_size', config.BATCH_SIZE)
        config.LEARNING_RATE = train.get('learning_rate', config.LEARNING_RATE)
        config.WEIGHT_DECAY = train.get('weight_decay', config.WEIGHT_DECAY)
        config.PATIENCE = train.get('patience', config.PATIENCE)

        # LR scheduler settings
        config.LR_SCHEDULER = train.get('lr_scheduler', config.LR_SCHEDULER)
        config.SCHEDULER_PATIENCE = train.get('scheduler_patience', config.SCHEDULER_PATIENCE)
        config.SCHEDULER_FACTOR = train.get('scheduler_factor', config.SCHEDULER_FACTOR)
        config.COSINE_T_MAX = train.get('cosine_t_max', config.COSINE_T_MAX)
        config.COSINE_ETA_MIN = train.get('cosine_eta_min', config.COSINE_ETA_MIN)

        # Gradient clipping
        if 'grad_clip_norm' in train:
            config.GRAD_CLIP_NORM = train['grad_clip_norm']

    # Apply loss settings
    if 'loss' in config_dict:
        loss = config_dict['loss']
        config.LOSS_TYPE = loss.get('type', config.LOSS_TYPE)
        config.ALPHA = loss.get('alpha', config.ALPHA)
        config.USE_NORMALIZED_SUPERVISED_LOSS = loss.get(
            'use_normalized_supervised_loss',
            config.USE_NORMALIZED_SUPERVISED_LOSS
        )
        config.PHYSICS_LOSS_SCALE = loss.get('physics_loss_scale', config.PHYSICS_LOSS_SCALE)
        config.PHYSICS_LOSS_MODE = loss.get('physics_loss_mode', config.PHYSICS_LOSS_MODE)
        if 'physics_combined_weight' in loss:
            config.PHYSICS_COMBINED_WEIGHT = loss['physics_combined_weight']
        if 'physics_freq_min' in loss:
            config.PHYSICS_FREQ_MIN = loss['physics_freq_min']
        if 'physics_freq_max' in loss:
            config.PHYSICS_FREQ_MAX = loss['physics_freq_max']
        if 'param_weights' in loss:
            config.PARAM_WEIGHTS = loss['param_weights']

        # Alpha scheduling settings
        if 'alpha_schedule' in loss:
            sched = loss['alpha_schedule']
            config.ALPHA_SCHEDULE_ENABLED = sched.get('enabled', False)
            config.ALPHA_WARMUP_EPOCHS = sched.get('warmup_epochs', config.ALPHA_WARMUP_EPOCHS)
            config.ALPHA_FINAL = sched.get('final_alpha', config.ALPHA_FINAL)
            config.ALPHA_DECAY_EPOCHS = sched.get('decay_epochs', config.ALPHA_DECAY_EPOCHS)

    # Apply dataset settings
    if 'dataset' in config_dict:
        dataset = config_dict['dataset']
        config.DATASET_NAME = dataset.get('name', config.DATASET_NAME)

    # Store config metadata
    config._config_name = config_dict.get('name', config_name)
    config._config_description = config_dict.get('description', '')
    config._config_source = str(config_path)

    return config


def list_available_configs(config_dir=None):
    """
    List all available training configuration files.

    Args:
        config_dir: Directory containing config files (default: configs/training/)

    Returns:
        List of config names (without .json extension)
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'configs' / 'training'
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        return []

    return sorted([f.stem for f in config_dir.glob('*.json')])


def print_config_summary(config):
    """
    Print a summary of the loaded configuration.

    Args:
        config: Config object (ideally loaded via load_training_config)
    """
    print("=" * 70)
    print(f"Configuration Summary".center(70))
    print("=" * 70)

    if hasattr(config, '_config_name'):
        print(f"\nConfig: {config._config_name}")
        if hasattr(config, '_config_description'):
            print(f"  {config._config_description}")

    print(f"\nNetwork:")
    network_type = getattr(config, '_network_type', 'cnn_scalable')
    if network_type == 'resnet_scalable':
        print(f"  Type: ResNet Scalable")
        print(f"  Width multiplier: {config.RESNET_WIDTH_MULT}")
        print(f"  Residual blocks: {config.RESNET_NUM_RES_BLOCKS}")
        print(f"  Base channels: {config.RESNET_BASE_CHANNELS}")
        print(f"  FC hidden dims: {config.FC_HIDDEN_DIMS}")
        print(f"  Dropout: {config.RESNET_DROPOUT}")
    else:
        print(f"  Type: Scalable CNN")
        print(f"  Width multiplier: {config.CNN_WIDTH_MULT}")
        print(f"  Number of blocks: {config.CNN_NUM_BLOCKS}")
        print(f"  Base channels: {config.CNN_BASE_CHANNELS}")
        print(f"  FC hidden dims: {config.FC_HIDDEN_DIMS}")
        print(f"  Dropout: {config.CNN_DROPOUT}")

    print(f"\nTraining:")
    print(f"  Dataset: {config.DATASET_NAME}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print(f"  Early stop patience: {config.PATIENCE}")

    # LR Scheduler
    lr_scheduler = getattr(config, 'LR_SCHEDULER', 'plateau')
    print(f"  LR scheduler: {lr_scheduler}")
    if lr_scheduler == 'cosine':
        t_max = getattr(config, 'COSINE_T_MAX', None) or config.NUM_EPOCHS
        eta_min = getattr(config, 'COSINE_ETA_MIN', 1e-6)
        print(f"    T_max: {t_max}, eta_min: {eta_min:.2e}")
    else:
        print(f"    Patience: {config.SCHEDULER_PATIENCE}, Factor: {config.SCHEDULER_FACTOR}")

    print(f"\nLoss Function:")
    print(f"  Type: {config.LOSS_TYPE}")
    print(f"  Alpha: {config.ALPHA}")
    print(f"  Normalized supervised loss: {config.USE_NORMALIZED_SUPERVISED_LOSS}")
    print(f"  Physics loss scale: {config.PHYSICS_LOSS_SCALE}")
    print(f"  Physics loss mode: {getattr(config, 'PHYSICS_LOSS_MODE', 'real_imag')}")
    print(f"  Parameter weights: {config.PARAM_WEIGHTS}")

    # Alpha scheduling
    if getattr(config, 'ALPHA_SCHEDULE_ENABLED', False):
        print(f"\nAlpha Scheduling:")
        print(f"  Enabled: True")
        print(f"  Warmup epochs: {config.ALPHA_WARMUP_EPOCHS}")
        print(f"  Final alpha: {config.ALPHA_FINAL}")
        print(f"  Decay epochs: {config.ALPHA_DECAY_EPOCHS}")

    print("=" * 70)


if __name__ == '__main__':
    # Demo usage
    print("Available training configurations:")
    for name in list_available_configs():
        print(f"  - {name}")

    print("\nLoading 'cnn_base' config...")
    config = load_training_config('cnn_base')
    print_config_summary(config)
