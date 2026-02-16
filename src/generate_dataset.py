#!/usr/bin/env python3
"""
Dataset Generation Script for THz-ML

Generates training datasets using the forward model from simulate.py.
Supports configurable parameter ranges, dataset sizes, noise levels, and metadata tracking.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import argparse

from .simulate import simulate_transmission_ml, prepare_ml_input


# ============================================================================
# CONFIGURATION SCHEMA
# ============================================================================

DEFAULT_CONFIG = {
    "dataset_name": "default_dataset",
    "description": "Default THz transmission dataset",

    # Simulation parameters
    "simulation": {
        "L": 4096,  # Number of time samples (2^12)
        "deltat": 0.1e-12,  # Time step in seconds (100 fs)
        "device": "cpu"
    },

    # Parameter ranges (uniform sampling)
    "parameter_ranges": {
        "n": {
            "min": 1.2,
            "max": 4.0,
            "distribution": "uniform"
        },
        "kappa": {
            "min": -0.05,  # Negative for absorption
            "max": -0.001,
            "distribution": "uniform"
        },
        "d": {
            "min": 50e-6,  # 50 microns
            "max": 1000e-6,  # 1000 microns
            "distribution": "uniform"
        }
    },

    # Dataset size
    "dataset_size": {
        "train": 10000,
        "val": 2000,
        "test": 2000
    },

    # Noise configuration
    "noise": {
        "enabled": True,
        "level": 5e-3,  # Standard deviation
        "type": "gaussian"
    },

    # Output configuration
    "output": {
        "save_time_domain": False,  # Whether to save time domain pulses
        "save_frequency_domain": True,  # Whether to save transmission coefficients
        "save_ml_input": True,  # Whether to save prepared ML inputs (real/imag)
        "compression": True  # Use numpy's compressed format
    },

    # Random seed for reproducibility
    "random_seed": 42
}


# ============================================================================
# DATASET GENERATION
# ============================================================================

def sample_parameters(config, n_samples, split='train'):
    """
    Sample parameters according to config specifications.

    Args:
        config: Dataset configuration dict
        n_samples: Number of samples to generate
        split: 'train', 'val', or 'test' (for different random states)

    Returns:
        n, kappa, d: Arrays of sampled parameters [n_samples]
    """
    # Set random seed based on split
    seed = config['random_seed']
    if split == 'val':
        seed += 1000
    elif split == 'test':
        seed += 2000

    np.random.seed(seed)
    torch.manual_seed(seed)

    params = {}
    for param_name, param_config in config['parameter_ranges'].items():
        if param_config['distribution'] == 'uniform':
            params[param_name] = np.random.uniform(
                param_config['min'],
                param_config['max'],
                size=n_samples
            )
        elif param_config['distribution'] == 'loguniform':
            # Log-uniform sampling
            log_min = np.log10(param_config['min'])
            log_max = np.log10(param_config['max'])
            params[param_name] = 10 ** np.random.uniform(log_min, log_max, size=n_samples)
        else:
            raise ValueError(f"Unknown distribution: {param_config['distribution']}")

    return params['n'], params['kappa'], params['d']


def generate_dataset_split(config, split='train', batch_size=100):
    """
    Generate a dataset split (train/val/test).

    Args:
        config: Dataset configuration dict
        split: 'train', 'val', or 'test'
        batch_size: Process data in batches to manage memory

    Returns:
        dict with generated data arrays
    """
    n_samples = config['dataset_size'][split]
    sim_config = config['simulation']
    noise_config = config['noise']

    print(f"\nGenerating {split} split ({n_samples} samples)...")

    # Sample parameters
    n_vals, kappa_vals, d_vals = sample_parameters(config, n_samples, split)

    # Storage for results
    results = {
        'parameters': {
            'n': n_vals,
            'kappa': kappa_vals,
            'd': d_vals
        }
    }

    # Initialize storage based on output config
    if config['output']['save_frequency_domain']:
        results['T_complex'] = []
        results['frequencies'] = None  # Will be set once

    if config['output']['save_ml_input']:
        results['ml_input'] = []

    if config['output']['save_time_domain']:
        results['time_domain'] = []
        results['reference_pulse'] = None  # Will be set once

    # Process in batches
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_n = n_vals[start_idx:end_idx]
        batch_kappa = kappa_vals[start_idx:end_idx]
        batch_d = d_vals[start_idx:end_idx]

        # Convert to tensors
        n_tensor = torch.tensor(batch_n, dtype=torch.float32, device=sim_config['device'])
        kappa_tensor = torch.tensor(batch_kappa, dtype=torch.float32, device=sim_config['device'])
        d_tensor = torch.tensor(batch_d, dtype=torch.float32, device=sim_config['device'])

        # Generate transmission data
        noise_level = noise_config['level'] if noise_config['enabled'] else None

        if config['output']['save_time_domain']:
            freq, T_complex, ref_pulse, time_domain = simulate_transmission_ml(
                n_tensor, kappa_tensor, d_tensor,
                sim_config['L'], sim_config['deltat'],
                device=sim_config['device'],
                return_time_domain=True,
                noise_level=noise_level
            )
            results['time_domain'].append(time_domain.cpu().numpy())
            if results['reference_pulse'] is None:
                results['reference_pulse'] = ref_pulse.cpu().numpy()
        else:
            freq, T_complex, ref_pulse = simulate_transmission_ml(
                n_tensor, kappa_tensor, d_tensor,
                sim_config['L'], sim_config['deltat'],
                device=sim_config['device'],
                return_time_domain=False
            )

        # Store frequency grid (same for all samples)
        if results.get('frequencies') is None:
            results['frequencies'] = freq.cpu().numpy()

        # Store transmission coefficients
        if config['output']['save_frequency_domain']:
            results['T_complex'].append(T_complex.cpu().numpy())

        # Prepare and store ML input
        if config['output']['save_ml_input']:
            ml_input = prepare_ml_input(T_complex)
            results['ml_input'].append(ml_input.cpu().numpy())

        # Progress update
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            progress = (end_idx / n_samples) * 100
            print(f"  Progress: {end_idx}/{n_samples} ({progress:.1f}%)")

    # Concatenate batches
    if config['output']['save_frequency_domain']:
        results['T_complex'] = np.concatenate(results['T_complex'], axis=0)

    if config['output']['save_ml_input']:
        results['ml_input'] = np.concatenate(results['ml_input'], axis=0)

    if config['output']['save_time_domain']:
        results['time_domain'] = np.concatenate(results['time_domain'], axis=0)

    print(f"  ✓ {split} split complete")

    return results


def plot_parameter_distributions(train_data, val_data, test_data, config, output_path):
    """
    Generate and save histogram plots of parameter distributions.

    Args:
        train_data, val_data, test_data: Generated data dicts
        config: Dataset configuration
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Combine all splits for overall distribution
    all_n = np.concatenate([
        train_data['parameters']['n'],
        val_data['parameters']['n'],
        test_data['parameters']['n']
    ])
    all_kappa = np.concatenate([
        train_data['parameters']['kappa'],
        val_data['parameters']['kappa'],
        test_data['parameters']['kappa']
    ])
    all_d = np.concatenate([
        train_data['parameters']['d'],
        val_data['parameters']['d'],
        test_data['parameters']['d']
    ])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Define consistent style
    hist_kwargs = {'bins': 50, 'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 0.5}

    # Plot 1: Refractive index (n)
    ax = axes[0]
    ax.hist(all_n, **hist_kwargs, color='steelblue')
    ax.axvline(all_n.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_n.mean():.3f}')
    ax.axvline(config['parameter_ranges']['n']['min'], color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Range')
    ax.axvline(config['parameter_ranges']['n']['max'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Refractive Index (n)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Refractive Index Distribution (N={len(all_n):,})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')

    # Add statistics text
    stats_text = f'Min: {all_n.min():.3f}\nMax: {all_n.max():.3f}\nStd: {all_n.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Extinction coefficient (kappa)
    ax = axes[1]
    ax.hist(all_kappa, **hist_kwargs, color='coral')
    ax.axvline(all_kappa.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_kappa.mean():.4f}')
    ax.axvline(config['parameter_ranges']['kappa']['min'], color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Range')
    ax.axvline(config['parameter_ranges']['kappa']['max'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Extinction Coefficient (κ)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Extinction Coefficient Distribution (N={len(all_kappa):,})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')

    stats_text = f'Min: {all_kappa.min():.4f}\nMax: {all_kappa.max():.4f}\nStd: {all_kappa.std():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Thickness (d) - convert to microns for readability
    ax = axes[2]
    d_microns = all_d * 1e6  # Convert to μm
    ax.hist(d_microns, **hist_kwargs, color='mediumseagreen')
    ax.axvline(d_microns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {d_microns.mean():.1f} μm')
    ax.axvline(config['parameter_ranges']['d']['min']*1e6, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Range')
    ax.axvline(config['parameter_ranges']['d']['max']*1e6, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Thickness (μm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Thickness Distribution (N={len(d_microns):,})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')

    stats_text = f'Min: {d_microns.min():.1f} μm\nMax: {d_microns.max():.1f} μm\nStd: {d_microns.std():.1f} μm'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    fig.suptitle(f'Parameter Distributions - {config["dataset_name"]}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_dataset(config, train_data, val_data, test_data, output_dir):
    """
    Save dataset to disk with metadata.

    Args:
        config: Dataset configuration
        train_data, val_data, test_data: Generated data dicts
        output_dir: Directory to save dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to {output_path}...")

    # Save each split
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_file = output_path / f"{split_name}.npz"

        if config['output']['compression']:
            np.savez_compressed(split_file, **split_data)
        else:
            np.savez(split_file, **split_data)

        print(f"  ✓ Saved {split_name}.npz")

    # Save metadata
    metadata = {
        'config': config,
        'generation_date': datetime.now().isoformat(),
        'dataset_name': config['dataset_name'],
        'description': config['description'],
        'splits': {
            'train': config['dataset_size']['train'],
            'val': config['dataset_size']['val'],
            'test': config['dataset_size']['test']
        },
        'total_samples': sum(config['dataset_size'].values()),
        'parameter_ranges': config['parameter_ranges'],
        'noise_config': config['noise'],
        'simulation_config': config['simulation']
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved metadata.json")

    # Create README
    readme_content = f"""# {config['dataset_name']}

{config['description']}

## Dataset Information

- **Generated:** {metadata['generation_date']}
- **Total Samples:** {metadata['total_samples']:,}
  - Train: {metadata['splits']['train']:,}
  - Validation: {metadata['splits']['val']:,}
  - Test: {metadata['splits']['test']:,}

## Parameter Ranges

- **Refractive Index (n):** {config['parameter_ranges']['n']['min']:.2f} - {config['parameter_ranges']['n']['max']:.2f}
- **Extinction Coefficient (κ):** {config['parameter_ranges']['kappa']['min']:.4f} - {config['parameter_ranges']['kappa']['max']:.4f}
- **Thickness (d):** {config['parameter_ranges']['d']['min']*1e6:.0f} - {config['parameter_ranges']['d']['max']*1e6:.0f} μm

## Simulation Parameters

- **Time Samples (L):** {config['simulation']['L']}
- **Time Step (Δt):** {config['simulation']['deltat']*1e12:.1f} fs
- **Frequency Range:** 0 - {0.5 / config['simulation']['deltat'] / 1e12:.2f} THz
- **Device:** {config['simulation']['device']}

## Noise Configuration

- **Enabled:** {config['noise']['enabled']}
- **Level:** {config['noise']['level']}
- **Type:** {config['noise']['type']}

## Visualizations

See [parameter_distributions.png](parameter_distributions.png) for histogram plots showing the distribution of all three parameters across the entire dataset.

## Data Format

Each split file (train.npz, val.npz, test.npz) contains:

- `parameters/n`: Refractive indices [N]
- `parameters/kappa`: Extinction coefficients [N]
- `parameters/d`: Thicknesses in meters [N]
- `frequencies`: Frequency grid in Hz [M+1]
"""

    if config['output']['save_ml_input']:
        readme_content += "- `ml_input`: ML-ready input [N, 2, M+1] (channels: real, imag)\n"

    if config['output']['save_frequency_domain']:
        readme_content += "- `T_complex`: Complex transmission coefficients [N, M+1]\n"

    if config['output']['save_time_domain']:
        readme_content += "- `time_domain`: Time domain pulses [N, 4*L]\n"
        readme_content += "- `reference_pulse`: Reference pulse [L]\n"

    readme_content += """
## Usage Example

```python
import numpy as np

# Load training data
data = np.load('train.npz')

# Access parameters
n = data['parameters/n']
kappa = data['parameters/kappa']
d = data['parameters/d']

# Access ML inputs
X = data['ml_input']  # Shape: [N, 2, M+1]

print(f"Dataset shape: {X.shape}")
print(f"Parameter ranges:")
print(f"  n: {n.min():.2f} - {n.max():.2f}")
print(f"  κ: {kappa.min():.4f} - {kappa.max():.4f}")
print(f"  d: {d.min()*1e6:.0f} - {d.max()*1e6:.0f} μm")
```
"""

    readme_file = output_path / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"  ✓ Saved README.md")

    # Generate and save distribution plots
    plot_file = output_path / 'parameter_distributions.png'
    plot_parameter_distributions(train_data, val_data, test_data, config, plot_file)
    print(f"  ✓ Saved parameter_distributions.png")

    print(f"\n✓ Dataset saved successfully to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_dataset(config_path=None, config_dict=None, output_dir=None):
    """
    Generate complete dataset from configuration.

    Args:
        config_path: Path to JSON config file (optional)
        config_dict: Config dictionary (optional, overrides config_path)
        output_dir: Output directory (optional, overrides config)
    """
    # Load configuration
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # Determine output directory
    if output_dir is None:
        # Use datalake structure
        datalake_root = Path(__file__).parent.parent / 'datalake' / 'datasets'
        output_dir = datalake_root / config['dataset_name']

    print("="*70)
    print(f"THz-ML Dataset Generation")
    print("="*70)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Description: {config['description']}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Generate splits
    train_data = generate_dataset_split(config, 'train')
    val_data = generate_dataset_split(config, 'val')
    test_data = generate_dataset_split(config, 'test')

    # Save dataset
    save_dataset(config, train_data, val_data, test_data, output_dir)

    print("\n" + "="*70)
    print("Dataset generation complete!")
    print("="*70)

    return output_dir


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate THz transmission datasets for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with default configuration
  python generate_dataset.py

  # Generate dataset from custom config
  python generate_dataset.py --config my_config.json

  # Generate dataset with custom output directory
  python generate_dataset.py --output /path/to/output

  # Quick small dataset for testing
  python generate_dataset.py --quick
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Generate small quick dataset for testing (100/20/20 samples)'
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        config = DEFAULT_CONFIG.copy()
        config['dataset_name'] = 'quick_test'
        config['description'] = 'Quick test dataset for validation'
        config['dataset_size'] = {
            'train': 100,
            'val': 20,
            'test': 20
        }
        generate_dataset(config_dict=config, output_dir=args.output)
    else:
        generate_dataset(config_path=args.config, output_dir=args.output)


if __name__ == '__main__':
    main()
