#!/usr/bin/env python3
"""
Regenerate training plots from existing training history.
"""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from train import plot_training_history


if __name__ == '__main__':
    # Path to the training history
    results_dir = Path('datalake/results/default_dataset/cnn/quick_test_run')
    history_file = results_dir / 'training_history.json'

    if not history_file.exists():
        print(f"Error: Training history not found at {history_file}")
        sys.exit(1)

    # Load history
    print(f"Loading training history from {history_file}")
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Regenerate plots
    print("Regenerating training plots with fixed R² visualization...")
    plot_path = results_dir / 'training_plots.png'
    plot_training_history(history, save_path=plot_path)

    print(f"\n✓ Updated plots saved to: {plot_path}")
    print("\nNote: Negative R² values indicate the model is performing worse than")
    print("simply predicting the mean. This is expected in early training or with")
    print("difficult datasets. R² should improve and approach 1.0 with more training.")
