#!/usr/bin/env python3
"""
Dataset generation config builder for large production datasets.
"""

import sys
sys.path.insert(0, '../../src')

from generate_dataset import generate_dataset


def make_large_dataset():
    """Generate a large, production-ready dataset."""
    config = {
        "dataset_name": "large_production_v1",
        "description": "Large production dataset with 100k samples and comprehensive parameter coverage",

        "simulation": {
            "L": 4096,  # Good frequency resolution
            "deltat": 1e-13,  # 0.1 ps -> ~5 THz bandwidth
            "device": "mps"  # Use Metal on Mac (change to "cuda" on GPU systems)
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
                "min": 100e-6,  # 100 microns
                "max": 1000e-6,  # 1000 microns
                "distribution": "uniform"
            }
        },

        "dataset_size": {
            "train": 80000,
            "val": 10000,
            "test": 10000
        },

        "noise": {
            "enabled": True,
            "level": 1e-3,  # 0.1% noise
            "type": "gaussian"
        },

        "output": {
            "save_time_domain": False,  # Skip time domain to save space
            "save_frequency_domain": True,
            "save_ml_input": False,  # Reconstruct from T_complex on load (50% space savings)
            "compression": True
        },

        "random_seed": 42
    }

    return config


def make_high_noise_dataset():
    """Generate dataset with higher noise for robustness training."""
    config = make_large_dataset()
    config["dataset_name"] = "large_production_noisy_v1"
    config["description"] = "Large dataset with elevated noise level for robust training"
    config["noise"]["level"] = 5e-3  # 0.5% noise
    return config


def make_log_thickness_dataset():
    """Generate dataset with log-uniform thickness distribution."""
    config = make_large_dataset()
    config["dataset_name"] = "large_production_logd_v1"
    config["description"] = "Large dataset with log-uniform thickness sampling"
    config["parameter_ranges"]["d"]["distribution"] = "loguniform"
    return config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate large production datasets')
    parser.add_argument('--variant', type=str, default='standard',
                       choices=['standard', 'noisy', 'log_thickness'],
                       help='Dataset variant to generate')

    args = parser.parse_args()

    # Select config based on variant
    if args.variant == 'standard':
        config = make_large_dataset()
    elif args.variant == 'noisy':
        config = make_high_noise_dataset()
    elif args.variant == 'log_thickness':
        config = make_log_thickness_dataset()

    print(f"Generating dataset variant: {args.variant}")
    print(f"Dataset name: {config['dataset_name']}")
    print(f"Total samples: {sum(config['dataset_size'].values()):,}")
    print()

    # Generate dataset
    generate_dataset(config_dict=config)
