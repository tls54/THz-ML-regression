"""
Evaluation script for THz parameter extraction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from training_config import Config
from networks import get_network
from utils import (
    generate_dataset, load_dataset, save_dataset,
    THz_Dataset, compute_metrics, print_metrics
)
from simulate import simulate_transmission_ml, get_transmission_features


def load_trained_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get network name from checkpoint
    network_name = checkpoint['network_name']
    
    # Create model
    model = get_network(network_name).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded {network_name} model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, checkpoint


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate model on dataset.
    
    Returns:
        predictions: [n_samples, 3]
        targets: [n_samples, 3]
        metrics: dict of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            X, y = batch_data[:2]  # Handle both 2 and 3 item batches
            X = X.to(device)
            
            y_pred = model(X)
            
            all_preds.append(y_pred.cpu())
            all_targets.append(y)
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(targets, predictions)
    
    return predictions, targets, metrics


def plot_predictions(predictions, targets, save_path=None):
    """
    Plot predicted vs true parameters.
    
    Args:
        predictions: [n_samples, 3] - predicted [n, κ, d]
        targets: [n_samples, 3] - true [n, κ, d]
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    param_names = ['n (Refractive Index)', 'κ (Extinction)', 'd (Thickness, μm)']
    units = [1, 1, 1e6]  # Convert d to μm
    
    for i, (ax, name, unit) in enumerate(zip(axes, param_names, units)):
        pred = predictions[:, i].numpy() * unit
        true = targets[:, i].numpy() * unit
        
        # Scatter plot
        ax.scatter(true, pred, alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val, max_val = true.min(), true.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # R² score
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved predictions plot to {save_path}")
    
    plt.show()
    return fig


def plot_error_distribution(predictions, targets, save_path=None):
    """
    Plot error distributions for each parameter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    param_names = ['n', 'κ', 'd']
    units = [1, 1, 1e6]
    
    for i, (ax, name, unit) in enumerate(zip(axes, param_names, units)):
        errors = (predictions[:, i].numpy() - targets[:, i].numpy()) * unit
        
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(errors):.4f}')
        
        ax.set_xlabel(f'Prediction Error ({name})')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Error Distribution\nStd: {np.std(errors):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved error distribution to {save_path}")
    
    plt.show()
    return fig


def evaluate_physics_consistency(model, test_data, config, device, n_samples=10):
    """
    Evaluate physics consistency by comparing:
    1. True transmission from parameters
    2. Predicted transmission from network output
    
    Shows how well the network respects physical constraints.
    """
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(test_data['X']), n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            X = test_data['X'][idx:idx+1].to(device)
            y_true = test_data['y'][idx:idx+1]
            T_true = test_data['T'][idx:idx+1]
            
            # Network prediction
            y_pred = model(X).cpu()
            
            # Generate transmission from predicted parameters
            _, T_pred, _ = simulate_transmission_ml(
                y_pred[:, 0], y_pred[:, 1], y_pred[:, 2],
                config.L, config.DELTAT, device='cpu'
            )
            
            # Get magnitude and phase
            mag_true, phase_true = get_transmission_features(T_true[0])
            mag_pred, phase_pred = get_transmission_features(T_pred[0])
            
            freq = test_data['frequencies'].cpu().numpy() / 1e12
            
            # Plot magnitude
            axes[i, 0].plot(freq, mag_true.numpy(), 'b-', label='True', linewidth=2)
            axes[i, 0].plot(freq, mag_pred.numpy(), 'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[i, 0].set_xlabel('Frequency (THz)')
            axes[i, 0].set_ylabel('|T(ω)|')
            axes[i, 0].set_title(f'Sample {i+1}: Magnitude')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xlim([0, 5])
            
            # Plot phase
            axes[i, 1].plot(freq, phase_true.numpy(), 'b-', label='True', linewidth=2)
            axes[i, 1].plot(freq, phase_pred.numpy(), 'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[i, 1].set_xlabel('Frequency (THz)')
            axes[i, 1].set_ylabel('∠T(ω) (rad)')
            axes[i, 1].set_title(f'Sample {i+1}: Phase')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_xlim([0, 5])
            axes[i, 1].set_ylim([-np.pi, np.pi])
            
            # Add parameter info
            n_true_val, k_true_val, d_true_val = y_true[0]
            n_pred_val, k_pred_val, d_pred_val = y_pred[0]
            
            info_text = (
                f"True: n={n_true_val:.2f}, κ={k_true_val:.3f}, d={d_true_val*1e6:.0f}μm\n"
                f"Pred: n={n_pred_val:.2f}, κ={k_pred_val:.3f}, d={d_pred_val*1e6:.0f}μm"
            )
            axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes,
                          fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', 
                          facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    return fig


def main(checkpoint_path, network_name='cnn'):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        network_name: Name of network architecture
    """
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print(f"{'='*60}")
    print(f"{'Model Evaluation':^60}")
    print(f"{'='*60}\n")
    
    # Load model
    model, checkpoint = load_trained_model(checkpoint_path, device)
    
    # Generate or load test dataset
    test_path = Path(config.DATA_DIR) / 'test_dataset.pt'
    
    if test_path.exists():
        print("\nLoading existing test dataset...")
        test_data = load_dataset(test_path, device='cpu')
    else:
        print("\nGenerating new test dataset...")
        test_data = generate_dataset(config.TEST_SIZE, config, noise_level=None, device='cpu')
        save_dataset(test_data, test_path)
    
    # Create dataloader
    test_dataset = THz_Dataset(test_data['X'], test_data['y'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Evaluate
    print("\nEvaluating model on test set...")
    predictions, targets, metrics = evaluate_model(model, test_loader, device, config)
    
    # Print metrics
    print_metrics(metrics, title="Test Set Metrics")
    
    # Create results directory
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot predictions
    print("\nGenerating plots...")
    plot_predictions(
        predictions, targets,
        save_path=results_dir / f'{network_name}_predictions.png'
    )
    
    # Plot error distributions
    plot_error_distribution(
        predictions, targets,
        save_path=results_dir / f'{network_name}_errors.png'
    )
    
    # Evaluate physics consistency
    print("\nEvaluating physics consistency...")
    evaluate_physics_consistency(
        model, test_data, config, device, n_samples=5
    )
    
    print(f"\n{'='*60}")
    print(f"{'Evaluation Complete!':^60}")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained THz model')
    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--network', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'multiscale', 'mlp'],
                       help='Network architecture')
    
    args = parser.parse_args()
    
    main(args.checkpoint, args.network)