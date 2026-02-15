"""
Training script for THz parameter extraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from training_config import Config
from networks import get_network


def get_dtype(dtype_str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from {list(dtype_map.keys())}")
    return dtype_map[dtype_str]
from utils import (
    load_dataset_from_datalake, list_available_datasets,
    THz_Dataset, compute_metrics, print_metrics
)
from simulate import simulate_transmission_ml


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss using forward model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, predicted_params, true_T):
        """
        Compute reconstruction loss via forward model.
        
        Args:
            predicted_params: [B, 3] - predicted [n, κ, d]
            true_T: [B, M+1] - true transmission (complex)
            
        Returns:
            loss: scalar
        """
        # Extract predicted parameters
        n_pred = predicted_params[:, 0]
        kappa_pred = predicted_params[:, 1]
        d_pred = predicted_params[:, 2]
        
        # Forward model (differentiable!)
        _, T_pred, _ = simulate_transmission_ml(
            n_pred, kappa_pred, d_pred,
            self.config.L, self.config.DELTAT,
            device=predicted_params.device,
            return_time_domain=False
        )
        
        # MSE on real and imaginary parts
        loss_real = torch.mean((T_pred.real - true_T.real) ** 2)
        loss_imag = torch.mean((T_pred.imag - true_T.imag) ** 2)
        
        return loss_real + loss_imag


class NormalizedMSELoss(nn.Module):
    """MSE loss with per-parameter normalization for balanced learning."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Parameter ranges for normalization
        self.n_range = config.N_MAX - config.N_MIN
        self.kappa_range = config.KAPPA_MAX - config.KAPPA_MIN
        self.d_range = config.D_MAX - config.D_MIN

        # Per-parameter weights
        self.weight_n = config.PARAM_WEIGHTS['n']
        self.weight_kappa = config.PARAM_WEIGHTS['kappa']
        self.weight_d = config.PARAM_WEIGHTS['d']

    def forward(self, predicted_params, true_params):
        """
        Compute normalized MSE loss.

        Args:
            predicted_params: [B, 3] - [n, κ, d] in physical ranges
            true_params: [B, 3] - [n, κ, d] in physical ranges

        Returns:
            loss: scalar (average of per-parameter normalized MSE)
        """
        # Extract parameters
        n_pred, kappa_pred, d_pred = predicted_params[:, 0], predicted_params[:, 1], predicted_params[:, 2]
        n_true, kappa_true, d_true = true_params[:, 0], true_params[:, 1], true_params[:, 2]

        # Normalize to [0, 1] range
        n_pred_norm = (n_pred - self.config.N_MIN) / self.n_range
        n_true_norm = (n_true - self.config.N_MIN) / self.n_range

        kappa_pred_norm = (kappa_pred - self.config.KAPPA_MIN) / self.kappa_range
        kappa_true_norm = (kappa_true - self.config.KAPPA_MIN) / self.kappa_range

        d_pred_norm = (d_pred - self.config.D_MIN) / self.d_range
        d_true_norm = (d_true - self.config.D_MIN) / self.d_range

        # Compute per-parameter MSE on normalized values
        loss_n = self.weight_n * torch.mean((n_pred_norm - n_true_norm) ** 2)
        loss_kappa = self.weight_kappa * torch.mean((kappa_pred_norm - kappa_true_norm) ** 2)
        loss_d = self.weight_d * torch.mean((d_pred_norm - d_true_norm) ** 2)

        # Average (equal contribution from each parameter)
        return (loss_n + loss_kappa + loss_d) / 3.0


class NormalizedHuberLoss(nn.Module):
    """Huber loss with per-parameter normalization. Less sensitive to outliers than MSE."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.delta = getattr(config, 'HUBER_DELTA', 1.0)

        # Parameter ranges for normalization
        self.n_range = config.N_MAX - config.N_MIN
        self.kappa_range = config.KAPPA_MAX - config.KAPPA_MIN
        self.d_range = config.D_MAX - config.D_MIN

        # Per-parameter weights
        self.weight_n = config.PARAM_WEIGHTS['n']
        self.weight_kappa = config.PARAM_WEIGHTS['kappa']
        self.weight_d = config.PARAM_WEIGHTS['d']

        # Huber loss function
        self.huber = nn.SmoothL1Loss(beta=self.delta)

    def forward(self, predicted_params, true_params):
        """
        Compute normalized Huber loss.

        Args:
            predicted_params: [B, 3] - [n, κ, d] in physical ranges
            true_params: [B, 3] - [n, κ, d] in physical ranges

        Returns:
            loss: scalar (average of per-parameter normalized Huber loss)
        """
        # Extract parameters
        n_pred, kappa_pred, d_pred = predicted_params[:, 0], predicted_params[:, 1], predicted_params[:, 2]
        n_true, kappa_true, d_true = true_params[:, 0], true_params[:, 1], true_params[:, 2]

        # Normalize to [0, 1] range
        n_pred_norm = (n_pred - self.config.N_MIN) / self.n_range
        n_true_norm = (n_true - self.config.N_MIN) / self.n_range

        kappa_pred_norm = (kappa_pred - self.config.KAPPA_MIN) / self.kappa_range
        kappa_true_norm = (kappa_true - self.config.KAPPA_MIN) / self.kappa_range

        d_pred_norm = (d_pred - self.config.D_MIN) / self.d_range
        d_true_norm = (d_true - self.config.D_MIN) / self.d_range

        # Compute per-parameter Huber loss on normalized values
        loss_n = self.weight_n * self.huber(n_pred_norm, n_true_norm)
        loss_kappa = self.weight_kappa * self.huber(kappa_pred_norm, kappa_true_norm)
        loss_d = self.weight_d * self.huber(d_pred_norm, d_true_norm)

        # Average (equal contribution from each parameter)
        return (loss_n + loss_kappa + loss_d) / 3.0


class HybridLoss(nn.Module):
    """Hybrid loss: supervised + physics-informed."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.ALPHA
        self.physics_loss = PhysicsInformedLoss(config)

        # Use normalized or standard supervised loss
        supervised_loss_type = getattr(config, 'SUPERVISED_LOSS', 'mse')
        if config.USE_NORMALIZED_SUPERVISED_LOSS:
            if supervised_loss_type == 'huber':
                self.supervised_loss = NormalizedHuberLoss(config)
            else:
                self.supervised_loss = NormalizedMSELoss(config)
        else:
            if supervised_loss_type == 'huber':
                self.supervised_loss = nn.SmoothL1Loss(beta=getattr(config, 'HUBER_DELTA', 1.0))
            else:
                self.supervised_loss = nn.MSELoss()

        self.physics_scale = config.PHYSICS_LOSS_SCALE
        
    def forward(self, predicted_params, true_params, true_T):
        """
        Args:
            predicted_params: [B, 3]
            true_params: [B, 3]
            true_T: [B, M+1] complex
        """
        loss_supervised = self.supervised_loss(predicted_params, true_params)
        loss_physics = self.physics_loss(predicted_params, true_T)

        # Apply physics loss scaling
        loss_physics_scaled = self.physics_scale * loss_physics

        return self.alpha * loss_supervised + (1 - self.alpha) * loss_physics_scaled


def get_loss_function(config):
    """Get loss function based on config."""
    supervised_loss_type = getattr(config, 'SUPERVISED_LOSS', 'mse')

    if config.LOSS_TYPE == 'supervised':
        if config.USE_NORMALIZED_SUPERVISED_LOSS:
            if supervised_loss_type == 'huber':
                return NormalizedHuberLoss(config)
            else:
                return NormalizedMSELoss(config)
        else:
            if supervised_loss_type == 'huber':
                return nn.SmoothL1Loss(beta=getattr(config, 'HUBER_DELTA', 1.0))
            else:
                return nn.MSELoss()
    elif config.LOSS_TYPE == 'physics':
        return PhysicsInformedLoss(config)
    elif config.LOSS_TYPE == 'hybrid':
        return HybridLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch. Returns dict with loss components."""
    model.train()
    total_loss = 0
    total_supervised = 0
    total_physics = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            X, y, T = batch_data
            X, y, T = X.to(device), y.to(device), T.to(device)
        else:
            X, y = batch_data
            X, y = X.to(device), y.to(device)
            T = None

        optimizer.zero_grad()

        y_pred = model(X)

        if config.LOSS_TYPE == 'supervised':
            loss = criterion(y_pred, y)
            loss_supervised = loss.item()
            loss_physics = 0.0
        elif config.LOSS_TYPE == 'physics':
            loss = criterion(y_pred, T)
            loss_supervised = 0.0
            loss_physics = loss.item()
        else:  # hybrid - decompose the loss
            loss_sup = criterion.supervised_loss(y_pred, y)
            loss_phys = criterion.physics_loss(y_pred, T)
            loss_phys_scaled = criterion.physics_scale * loss_phys
            loss = criterion.alpha * loss_sup + (1 - criterion.alpha) * loss_phys_scaled
            loss_supervised = loss_sup.item()
            loss_physics = loss_phys.item()

        loss.backward()

        # Gradient clipping
        if getattr(config, 'GRAD_CLIP_NORM', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

        optimizer.step()

        total_loss += loss.item()
        total_supervised += loss_supervised
        total_physics += loss_physics
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return {
        'total': total_loss / num_batches,
        'supervised': total_supervised / num_batches,
        'physics': total_physics / num_batches
    }


def validate(model, dataloader, criterion, device, config):
    """Validate model. Returns (loss_dict, metrics)."""
    model.eval()
    total_loss = 0
    total_supervised = 0
    total_physics = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                X, y, T = batch_data
                X, y, T = X.to(device), y.to(device), T.to(device)
            else:
                X, y = batch_data
                X, y = X.to(device), y.to(device)
                T = None

            y_pred = model(X)

            if config.LOSS_TYPE == 'supervised':
                loss = criterion(y_pred, y)
                loss_supervised = loss.item()
                loss_physics = 0.0
            elif config.LOSS_TYPE == 'physics':
                loss = criterion(y_pred, T)
                loss_supervised = 0.0
                loss_physics = loss.item()
            else:  # hybrid - decompose the loss
                loss_sup = criterion.supervised_loss(y_pred, y)
                loss_phys = criterion.physics_loss(y_pred, T)
                loss_phys_scaled = criterion.physics_scale * loss_phys
                loss = criterion.alpha * loss_sup + (1 - criterion.alpha) * loss_phys_scaled
                loss_supervised = loss_sup.item()
                loss_physics = loss_phys.item()

            total_loss += loss.item()
            total_supervised += loss_supervised
            total_physics += loss_physics
            num_batches += 1
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_targets, all_preds)

    loss_dict = {
        'total': total_loss / num_batches,
        'supervised': total_supervised / num_batches,
        'physics': total_physics / num_batches
    }

    return loss_dict, metrics


def plot_training_history(history, save_path=None):
    """
    Create comprehensive training visualization plots.

    Args:
        history: Training history dict with train_loss, val_loss, val_metrics
        save_path: Path to save the figure (PNG)

    Returns:
        fig: matplotlib figure
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # Extract metrics per epoch
    n_mae = [m['n_mae'] for m in history['val_metrics']]
    n_rmse = [m['n_rmse'] for m in history['val_metrics']]
    n_r2 = [m['n_r2'] for m in history['val_metrics']]

    kappa_mae = [m['kappa_mae'] for m in history['val_metrics']]
    kappa_rmse = [m['kappa_rmse'] for m in history['val_metrics']]
    kappa_r2 = [m['kappa_r2'] for m in history['val_metrics']]

    d_mae = [m['d_um_mae'] for m in history['val_metrics']]
    d_rmse = [m['d_um_rmse'] for m in history['val_metrics']]
    d_r2 = [m['d_um_r2'] for m in history['val_metrics']]

    # Row 1, Col 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15,
            label=f'Best (epoch {best_epoch})')
    ax.legend(loc='upper right')

    # Row 1, Col 2: Refractive Index (n) Metrics
    ax = axes[0, 1]
    ax2 = ax.twinx()

    ln1 = ax.plot(epochs, n_mae, 'b-', linewidth=2, label='MAE', alpha=0.8)
    ln2 = ax.plot(epochs, n_rmse, 'r-', linewidth=2, label='RMSE', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE / RMSE', fontsize=11, color='black')
    ax.tick_params(axis='y', labelcolor='black')

    ln3 = ax2.plot(epochs, n_r2, 'g-', linewidth=2, label='R²', alpha=0.8)
    ax2.set_ylabel('R²', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # Auto-scale R² axis, but cap upper limit at 1.0
    r2_min = min(n_r2)
    r2_max = min(max(n_r2), 1.0)
    margin = (r2_max - r2_min) * 0.1 if r2_max != r2_min else 0.1
    ax2.set_ylim([r2_min - margin, r2_max + margin])

    ax.set_title('Refractive Index (n) Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Combined legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left', fontsize=9)

    # Row 1, Col 3: Extinction Coefficient (κ) Metrics
    ax = axes[0, 2]
    ax2 = ax.twinx()

    ln1 = ax.plot(epochs, kappa_mae, 'b-', linewidth=2, label='MAE', alpha=0.8)
    ln2 = ax.plot(epochs, kappa_rmse, 'r-', linewidth=2, label='RMSE', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE / RMSE', fontsize=11, color='black')
    ax.tick_params(axis='y', labelcolor='black')

    ln3 = ax2.plot(epochs, kappa_r2, 'g-', linewidth=2, label='R²', alpha=0.8)
    ax2.set_ylabel('R²', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # Auto-scale R² axis, but cap upper limit at 1.0
    r2_min = min(kappa_r2)
    r2_max = min(max(kappa_r2), 1.0)
    margin = (r2_max - r2_min) * 0.1 if r2_max != r2_min else 0.1
    ax2.set_ylim([r2_min - margin, r2_max + margin])

    ax.set_title('Extinction Coefficient (κ) Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left', fontsize=9)

    # Row 2, Col 1: Thickness (d) Metrics
    ax = axes[1, 0]
    ax2 = ax.twinx()

    ln1 = ax.plot(epochs, d_mae, 'b-', linewidth=2, label='MAE (μm)', alpha=0.8)
    ln2 = ax.plot(epochs, d_rmse, 'r-', linewidth=2, label='RMSE (μm)', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE / RMSE (μm)', fontsize=11, color='black')
    ax.tick_params(axis='y', labelcolor='black')

    ln3 = ax2.plot(epochs, d_r2, 'g-', linewidth=2, label='R²', alpha=0.8)
    ax2.set_ylabel('R²', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # Auto-scale R² axis, but cap upper limit at 1.0
    r2_min = min(d_r2)
    r2_max = min(max(d_r2), 1.0)
    margin = (r2_max - r2_min) * 0.1 if r2_max != r2_min else 0.1
    ax2.set_ylim([r2_min - margin, r2_max + margin])

    ax.set_title('Thickness (d) Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left', fontsize=9)

    # Row 2, Col 2: R² Comparison
    ax = axes[1, 1]
    ax.plot(epochs, n_r2, linewidth=2, label='n', alpha=0.8)
    ax.plot(epochs, kappa_r2, linewidth=2, label='κ', alpha=0.8)
    ax.plot(epochs, d_r2, linewidth=2, label='d', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('R² Comparison (All Parameters)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Auto-scale to show all R² values
    all_r2 = n_r2 + kappa_r2 + d_r2
    r2_min = min(all_r2)
    r2_max = min(max(all_r2), 1.0)
    margin = (r2_max - r2_min) * 0.1 if r2_max != r2_min else 0.1
    ax.set_ylim([r2_min - margin, r2_max + margin])

    # Add reference lines at 0 (baseline) and positive R² targets if in range
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Baseline (mean)')
    if r2_max > 0.5:
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Row 2, Col 3: Learning Rate Schedule
    ax = axes[1, 2]

    # Check if learning rate history is available
    if 'learning_rate' in history and len(history['learning_rate']) > 0:
        lr_history = history['learning_rate']
        ax.plot(epochs, lr_history, 'purple', linewidth=2, marker='o', markersize=3, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Mark LR reductions
        for i in range(1, len(lr_history)):
            if lr_history[i] < lr_history[i-1]:
                ax.axvline(x=i+1, color='red', linestyle='--', alpha=0.3, linewidth=1)

        # Add final LR annotation
        final_lr = lr_history[-1]
        initial_lr = lr_history[0]
        ax.annotate(f'Final: {final_lr:.2e}', xy=(epochs[-1], final_lr),
                   xytext=(-50, 10), textcoords='offset points',
                   fontsize=9, color='purple')
        if final_lr != initial_lr:
            reduction_factor = initial_lr / final_lr
            ax.set_title(f'Learning Rate Schedule ({reduction_factor:.1f}x reduction)',
                        fontsize=12, fontweight='bold')
    else:
        # Fallback: show text summary if no LR history
        ax.axis('off')
        final_metrics = history['val_metrics'][-1]

        summary_text = "Final Validation Metrics\n" + "="*30 + "\n\n"
        summary_text += f"Refractive Index (n):\n"
        summary_text += f"  MAE:  {final_metrics['n_mae']:.5f}\n"
        summary_text += f"  RMSE: {final_metrics['n_rmse']:.5f}\n"
        summary_text += f"  R²:   {final_metrics['n_r2']:.4f}\n\n"

        summary_text += f"Extinction Coeff (κ):\n"
        summary_text += f"  MAE:  {final_metrics['kappa_mae']:.6f}\n"
        summary_text += f"  RMSE: {final_metrics['kappa_rmse']:.6f}\n"
        summary_text += f"  R²:   {final_metrics['kappa_r2']:.4f}\n\n"

        summary_text += f"Thickness (d):\n"
        summary_text += f"  MAE:  {final_metrics['d_um_mae']:.2f} μm\n"
        summary_text += f"  RMSE: {final_metrics['d_um_rmse']:.2f} μm\n"
        summary_text += f"  R²:   {final_metrics['d_um_r2']:.4f}\n\n"

        summary_text += "="*30 + "\n"
        summary_text += f"Total Epochs: {len(epochs)}\n"
        summary_text += f"Best Epoch: {best_epoch}\n"
        summary_text += f"Best Val Loss: {best_val_loss:.6f}\n"
        summary_text += f"Final Val Loss: {history['val_loss'][-1]:.6f}"

        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training plots to {save_path}")

    return fig


def train(config, network_name='cnn', run_name=None, resume_from=None):
    """
    Main training function.
    
    Args:
        config: Config object
        network_name: Name of network architecture
        run_name: Name for this training run (auto-generated if None)
        resume_from: Path to checkpoint to resume from
    """
    # Generate run name if not provided
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup device
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config.DEVICE == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.DEVICE == 'cpu':
        device = torch.device('cpu')
    else:
        # Fallback to CPU if requested device not available
        print(f"Warning: {config.DEVICE} not available, falling back to CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Get save paths from datalake
    model_dir = config.get_model_save_path(network_name, run_name)
    results_dir = config.get_results_save_path(network_name, run_name)
    
    print(f"Model directory: {model_dir}")
    print(f"Results directory: {results_dir}")
    
    # List available datasets
    print("\nAvailable datasets in datalake:")
    available_datasets = list_available_datasets()
    for ds in available_datasets:
        marker = " (selected)" if ds == config.DATASET_NAME else ""
        print(f"  - {ds}{marker}")
    
    if config.DATASET_NAME not in available_datasets:
        raise ValueError(f"Dataset '{config.DATASET_NAME}' not found in datalake. "
                        f"Available: {available_datasets}")
    
    # Load datasets from datalake
    print(f"\nLoading dataset: {config.DATASET_NAME}")
    train_data = load_dataset_from_datalake(config.DATASET_NAME, 'train')
    val_data = load_dataset_from_datalake(config.DATASET_NAME, 'val')

    # Get dtype from config
    dtype = get_dtype(getattr(config, 'DTYPE', 'float32'))
    print(f"Using dtype: {dtype}")

    # Create datasets and dataloaders
    train_dataset = THz_Dataset(train_data['X'], train_data['y'], train_data['T'], dtype=dtype)
    val_dataset = THz_Dataset(val_data['X'], val_data['y'], val_data['T'], dtype=dtype)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    print(f"\nInitializing {network_name} network...")
    model = get_network(network_name).to(device=device, dtype=dtype)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = get_loss_function(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_supervised_loss': [],
        'train_physics_loss': [],
        'val_loss': [],
        'val_supervised_loss': [],
        'val_physics_loss': [],
        'val_metrics': [],
        'learning_rate': []
    }
    
    # Save config for this run
    run_config = {
        'network_name': network_name,
        'run_name': run_name,
        'dataset_name': config.DATASET_NAME,
        'dataset_metadata': train_data.get('metadata'),
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'start_time': datetime.now().isoformat()
    }
    
    config_file = model_dir / 'run_config.json'
    with open(config_file, 'w') as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"✓ Saved run config to {config_file}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"{'Starting Training':^60}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"{'-'*60}")
        
        # Train (returns dict with loss components)
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, config)

        # Validate (returns dict with loss components)
        val_losses, val_metrics = validate(model, val_loader, criterion, device, config)

        # Update scheduler
        scheduler.step(val_losses['total'])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_losses['total'])
        history['train_supervised_loss'].append(train_losses.get('supervised', 0.0))
        history['train_physics_loss'].append(train_losses.get('physics', 0.0))
        history['val_loss'].append(val_losses['total'])
        history['val_supervised_loss'].append(val_losses.get('supervised', 0.0))
        history['val_physics_loss'].append(val_losses.get('physics', 0.0))
        history['val_metrics'].append(val_metrics)
        history['learning_rate'].append(current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['total']:.6f} (supervised: {train_losses.get('supervised', 0.0):.6f}, physics: {train_losses.get('physics', 0.0):.6f})")
        print(f"  Val Loss:   {val_losses['total']:.6f} (supervised: {val_losses.get('supervised', 0.0):.6f}, physics: {val_losses.get('physics', 0.0):.6f})")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
        print_metrics(val_metrics, title="Validation Metrics")
        
        # Save best model
        val_loss = val_losses['total']
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total'],
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'config': run_config,
                'network_name': network_name,
                'run_name': run_name
            }

            save_path = model_dir / 'best_model.pt'
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model to {save_path}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total'],
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'config': run_config,
                'network_name': network_name,
                'run_name': run_name
            }

            save_path = model_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint to {save_path}")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {config.PATIENCE} epochs")
            break
        
        print(f"\n{'='*60}\n")
    
    # Save final model and history
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses['total'],
        'val_loss': val_losses['total'],
        'best_val_loss': best_val_loss,
        'config': run_config,
        'network_name': network_name,
        'run_name': run_name,
        'history': history
    }
    
    save_path = model_dir / 'final_model.pt'
    torch.save(final_checkpoint, save_path)
    print(f"✓ Saved final model to {save_path}")
    
    # Save training history
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert metrics to serializable format
        history_json = {
            'train_loss': history['train_loss'],
            'train_supervised_loss': history['train_supervised_loss'],
            'train_physics_loss': history['train_physics_loss'],
            'val_loss': history['val_loss'],
            'val_supervised_loss': history['val_supervised_loss'],
            'val_physics_loss': history['val_physics_loss'],
            'val_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in history['val_metrics']
            ],
            'learning_rate': history['learning_rate']
        }
        json.dump(history_json, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    # Generate and save training plots
    print("\nGenerating training plots...")
    plot_path = results_dir / 'training_plots.png'
    plot_training_history(history, save_path=plot_path)

    print(f"\n{'='*60}")
    print(f"{'Training Complete!':^60}")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {model_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"Training plots: {plot_path}")

    return model_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train THz parameter extraction network')
    parser.add_argument('--network', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'multiscale', 'mlp'],
                       help='Network architecture to use')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name from datalake (overrides config)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        print("Available datasets in datalake:")
        datasets = list_available_datasets()
        if not datasets:
            print("  (none found)")
        else:
            for ds in datasets:
                from utils import get_dataset_info
                info = get_dataset_info(ds)
                if info:
                    print(f"\n  {ds}:")
                    print(f"    Description: {info.get('description', 'N/A')}")
                    print(f"    Samples: {info.get('total_samples', 'N/A')}")
                    print(f"    Generated: {info.get('generation_date', 'N/A')}")
                else:
                    print(f"  {ds}")
        exit(0)
    
    # Create config
    config = Config()
    config.DEVICE = args.device
    
    if args.dataset:
        config.DATASET_NAME = args.dataset
    
    # Train
    train(config, network_name=args.network, run_name=args.run_name, resume_from=args.resume)