#!/usr/bin/env python3
"""
Quick training test script - runs 5 epochs for fast validation.
"""

import sys
sys.path.insert(0, 'src')

from training_config import Config
from train import train


if __name__ == '__main__':
    # Override config for quick test
    config = Config()
    config.DATASET_NAME = 'quick_test'
    config.NUM_EPOCHS = 5  # Just 5 epochs for quick test
    config.BATCH_SIZE = 32  # Smaller batch for speed
    config.DEVICE = 'mps'  # Use MPS on Mac (change to 'cuda' if you have GPU)
    config.NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Mac
    config.PATIENCE = 3  # Early stop after 3 epochs no improvement
    config.SAVE_INTERVAL = 2  # Save checkpoint every 2 epochs

    print("="*70)
    print("Quick Training Test")
    print("="*70)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Device: {config.DEVICE}")
    print(f"Loss type: {config.LOSS_TYPE}")
    print("="*70)
    print()

    try:
        model_dir = train(config, network_name='cnn', run_name='quick_test_run')

        # Compute results directory path
        results_dir = config.get_results_save_path('cnn', 'quick_test_run')

        print()
        print("="*70)
        print("✓ Training Test Complete!")
        print("="*70)
        print(f"Models saved to: {model_dir}")
        print(f"Results saved to: {results_dir}")
        print()
        print("Next steps:")
        print("  1. Check the training plots (training_plots.png)")
        print("  2. Review training_history.json")
        print("  3. Load and evaluate the model")
        print("="*70)

    except Exception as e:
        print()
        print("="*70)
        print("✗ Training failed!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
