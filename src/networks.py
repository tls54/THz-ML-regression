"""
Neural network architectures for THz parameter extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training_config import Config


class THz_Encoder_CNN(nn.Module):
    """
    1D CNN encoder for THz parameter extraction.
    Recommended as starting architecture.
    """
    def __init__(self, n_freq=None, dropout=0.2):
        super().__init__()
        
        if n_freq is None:
            n_freq, _ = Config.get_freq_info()
        
        self.n_freq = n_freq
        
        # Input: [batch, 2, n_freq] where 2 = [real, imag]
        
        self.conv_layers = nn.Sequential(
            # Layer 1: Extract local features
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Layer 2: Build intermediate representations
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            
            # Layer 3: Higher-level features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            
            # Layer 4: Deep features
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # Output: [n, κ, d]
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, 2, n_freq] - real and imaginary parts
            
        Returns:
            params: [batch, 3] - [n, κ, d] scaled to physical ranges
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        # Apply output activations to map to physical ranges
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])
        
        return torch.stack([n, kappa, d], dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class THz_Encoder_ResNet(nn.Module):
    """
    ResNet-style encoder with skip connections.
    Use if simple CNN underfits.
    """
    def __init__(self, n_freq=None, dropout=0.2):
        super().__init__()
        
        if n_freq is None:
            n_freq, _ = Config.get_freq_info()
        
        self.n_freq = n_freq
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2, 2)
        
        self.res_block2 = ResidualBlock(64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2, 2)
        
        # Increase channels
        self.conv_expand = nn.Conv1d(64, 128, kernel_size=1)
        self.res_block3 = ResidualBlock(128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2, 2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        x = self.conv_init(x)
        
        x = self.res_block1(x)
        x = self.pool1(x)
        
        x = self.res_block2(x)
        x = self.pool2(x)
        
        x = self.conv_expand(x)
        x = self.res_block3(x)
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = self.fc_layers(x)
        
        # Output scaling
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])
        
        return torch.stack([n, kappa, d], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class THz_Encoder_MultiScale(nn.Module):
    """
    Multi-scale CNN with parallel branches.
    Use for capturing varying oscillation frequencies.
    """
    def __init__(self, n_freq=None, dropout=0.2):
        super().__init__()
        
        if n_freq is None:
            n_freq, _ = Config.get_freq_info()
        
        self.n_freq = n_freq
        
        # Branch 1: Fine features (small kernels)
        self.branch_fine = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Branch 2: Medium features
        self.branch_medium = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Branch 3: Coarse features (large kernels)
        self.branch_coarse = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),  # 64*3 = 192 channels
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # Parallel branches
        x_fine = self.branch_fine(x)
        x_medium = self.branch_medium(x)
        x_coarse = self.branch_coarse(x)
        
        # Concatenate
        x = torch.cat([x_fine, x_medium, x_coarse], dim=1)
        
        x = self.combine(x)
        x = self.fc_layers(x)
        
        # Output scaling
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])
        
        return torch.stack([n, kappa, d], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class THz_Encoder_MLP(nn.Module):
    """
    Simple MLP baseline.
    Use for comparison only - not recommended.
    """
    def __init__(self, n_freq=None, dropout=0.3):
        super().__init__()
        
        if n_freq is None:
            n_freq, _ = Config.get_freq_info()
        
        self.n_freq = n_freq
        input_size = 2 * n_freq
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        x = self.layers(x)
        
        # Output scaling
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])
        
        return torch.stack([n, kappa, d], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class THz_Encoder_CNN_Scalable(nn.Module):
    """
    Scalable 1D CNN encoder with configurable width and depth.

    Args:
        width_mult: Channel multiplier (default 1.0)
        num_blocks: Number of convolutional blocks (default 4, range 2-6)
        base_channels: Base channel count for first layer (default 32)
        fc_hidden_dims: List of FC hidden layer dimensions (default [128, 64])
        dropout: Dropout rate (default 0.2)
    """
    def __init__(self, n_freq=None, width_mult=1.0, num_blocks=4,
                 base_channels=32, fc_hidden_dims=[128, 64], dropout=0.2):
        super().__init__()

        if n_freq is None:
            n_freq, _ = Config.get_freq_info()

        self.n_freq = n_freq
        self.width_mult = width_mult
        self.num_blocks = num_blocks

        # Calculate channel progression
        channels = [int(base_channels * width_mult * (2 ** i)) for i in range(num_blocks)]

        # Build convolutional blocks dynamically
        conv_blocks = []
        in_channels = 2  # Real and imaginary input

        for i, out_channels in enumerate(channels):
            # First layer uses larger kernel, others use smaller
            kernel_size = 7 if i == 0 else 5 if i == 1 else 3

            conv_blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels

        # Add global pooling at the end
        conv_blocks.append(nn.AdaptiveAvgPool1d(1))

        self.conv_layers = nn.Sequential(*conv_blocks)

        # Build FC layers dynamically
        fc_blocks = [nn.Flatten()]
        in_features = channels[-1]  # Last conv channel count

        for hidden_dim in fc_hidden_dims:
            fc_blocks.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim

        # Final output layer
        fc_blocks.append(nn.Linear(in_features, 3))

        self.fc_layers = nn.Sequential(*fc_blocks)

    def forward(self, x):
        """
        Args:
            x: [batch, 2, n_freq] - real and imaginary parts

        Returns:
            params: [batch, 3] - [n, κ, d] scaled to physical ranges
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        # Apply output activations to map to physical ranges
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])

        return torch.stack([n, kappa, d], dim=1)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class THz_Encoder_ResNet_Scalable(nn.Module):
    """
    Scalable ResNet-style encoder with configurable width and depth.

    Args:
        width_mult: Channel multiplier (default 1.0)
        num_res_blocks: List of residual blocks per stage (default [2, 2, 2])
        base_channels: Base channel count (default 64)
        fc_hidden_dims: List of FC hidden layer dimensions (default [64])
        dropout: Dropout rate (default 0.2)
    """
    def __init__(self, n_freq=None, width_mult=1.0, num_res_blocks=[2, 2, 2],
                 base_channels=64, fc_hidden_dims=[64], dropout=0.2):
        super().__init__()

        if n_freq is None:
            n_freq, _ = Config.get_freq_info()

        self.n_freq = n_freq
        self.width_mult = width_mult

        # Calculate channel counts for each stage
        channels = [int(base_channels * width_mult * (2 ** i)) for i in range(len(num_res_blocks))]

        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(2, channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Build residual stages dynamically
        self.stages = nn.ModuleList()

        for stage_idx, (num_blocks, out_channels) in enumerate(zip(num_res_blocks, channels)):
            stage_blocks = []

            # If channels change, add a 1x1 conv for dimension matching
            if stage_idx > 0 and channels[stage_idx-1] != out_channels:
                stage_blocks.append(nn.Conv1d(channels[stage_idx-1], out_channels, kernel_size=1))

            # Add residual blocks for this stage
            for _ in range(num_blocks):
                stage_blocks.append(ResidualBlock(out_channels, kernel_size=5 if stage_idx == 0 else 3))

            # Add pooling after residual blocks (except last stage)
            if stage_idx < len(num_res_blocks) - 1:
                stage_blocks.append(nn.MaxPool1d(2, 2))

            self.stages.append(nn.Sequential(*stage_blocks))

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build FC layers
        fc_blocks = [nn.Flatten()]
        in_features = channels[-1]  # Last stage channel count

        for hidden_dim in fc_hidden_dims:
            fc_blocks.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim

        fc_blocks.append(nn.Linear(in_features, 3))

        self.fc_layers = nn.Sequential(*fc_blocks)

    def forward(self, x):
        x = self.conv_init(x)

        for stage in self.stages:
            x = stage(x)

        x = self.global_pool(x)
        x = self.fc_layers(x)

        # Output scaling
        n = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(x[:, 0])
        kappa = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(x[:, 1])
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(x[:, 2])

        return torch.stack([n, kappa, d], dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Dictionary for easy network selection
NETWORKS = {
    'cnn': THz_Encoder_CNN,
    'resnet': THz_Encoder_ResNet,
    'multiscale': THz_Encoder_MultiScale,
    'mlp': THz_Encoder_MLP,
    'cnn_scalable': THz_Encoder_CNN_Scalable,
    'resnet_scalable': THz_Encoder_ResNet_Scalable,
}


def get_network(name, config=None, **kwargs):
    """
    Factory function to get network by name.

    Args:
        name: Network name ('cnn', 'resnet', 'multiscale', 'mlp', 'cnn_scalable', 'resnet_scalable')
        config: Config object (optional, for scalable networks)
        **kwargs: Additional arguments for network (override config if provided)

    Returns:
        Network instance
    """
    if name not in NETWORKS:
        raise ValueError(f"Unknown network: {name}. Choose from {list(NETWORKS.keys())}")

    # For scalable networks, pull parameters from config if not provided in kwargs
    if config is not None:
        if name == 'cnn_scalable':
            kwargs.setdefault('width_mult', config.CNN_WIDTH_MULT)
            kwargs.setdefault('num_blocks', config.CNN_NUM_BLOCKS)
            kwargs.setdefault('base_channels', config.CNN_BASE_CHANNELS)
            kwargs.setdefault('fc_hidden_dims', config.FC_HIDDEN_DIMS)
            kwargs.setdefault('dropout', config.CNN_DROPOUT)
        elif name == 'resnet_scalable':
            kwargs.setdefault('width_mult', config.RESNET_WIDTH_MULT)
            kwargs.setdefault('num_res_blocks', config.RESNET_NUM_RES_BLOCKS)
            kwargs.setdefault('base_channels', config.RESNET_BASE_CHANNELS)
            kwargs.setdefault('fc_hidden_dims', config.FC_HIDDEN_DIMS)
            kwargs.setdefault('dropout', config.RESNET_DROPOUT)

    return NETWORKS[name](**kwargs)


def model_summary(model, print_output=True):
    """
    Print a clean summary of the model architecture.

    Args:
        model: PyTorch model instance
        print_output: If True, prints the summary. If False, returns as string.

    Returns:
        Summary string if print_output=False
    """
    lines = []
    model_name = model.__class__.__name__

    # Header
    lines.append("=" * 60)
    lines.append(f"{model_name}".center(60))
    lines.append("=" * 60)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"\nParameters: {trainable_params:,} trainable / {total_params:,} total")

    # Architecture details based on model type
    if isinstance(model, THz_Encoder_ResNet_Scalable):
        lines.append(f"\nArchitecture: ResNet-style with skip connections")
        lines.append(f"  Width multiplier: {model.width_mult}")

        # Extract stage info
        lines.append(f"\nConvolutional Backbone:")
        lines.append(f"  Input: [batch, 2, n_freq] (real + imag)")

        # Initial conv
        init_conv = model.conv_init[0]
        lines.append(f"  Initial: Conv1d(2 → {init_conv.out_channels}, k={init_conv.kernel_size[0]}) → MaxPool")

        # Stages
        for i, stage in enumerate(model.stages):
            res_blocks = sum(1 for m in stage if isinstance(m, ResidualBlock))
            # Find channel count from first ResidualBlock
            for m in stage:
                if isinstance(m, ResidualBlock):
                    channels = m.conv1.out_channels
                    kernel = m.conv1.kernel_size[0]
                    break
            has_pool = any(isinstance(m, nn.MaxPool1d) for m in stage)
            pool_str = " → MaxPool" if has_pool else ""
            lines.append(f"  Stage {i+1}: {res_blocks}x ResBlock({channels}ch, k={kernel}){pool_str}")

        lines.append(f"  Global: AdaptiveAvgPool1d(1)")

        # FC head
        lines.append(f"\nFC Head:")
        fc_dims = []
        for m in model.fc_layers:
            if isinstance(m, nn.Linear):
                fc_dims.append((m.in_features, m.out_features))
        for i, (in_f, out_f) in enumerate(fc_dims):
            if i < len(fc_dims) - 1:
                lines.append(f"  Linear({in_f} → {out_f}) → ReLU → Dropout")
            else:
                lines.append(f"  Linear({in_f} → {out_f}) → Output")

    elif isinstance(model, THz_Encoder_CNN_Scalable):
        lines.append(f"\nArchitecture: Scalable CNN")
        lines.append(f"  Width multiplier: {model.width_mult}")
        lines.append(f"  Num blocks: {model.num_blocks}")

        lines.append(f"\nConvolutional Backbone:")
        lines.append(f"  Input: [batch, 2, n_freq] (real + imag)")

        # Extract conv layers
        block_idx = 0
        for i, m in enumerate(model.conv_layers):
            if isinstance(m, nn.Conv1d):
                block_idx += 1
                lines.append(f"  Block {block_idx}: Conv1d({m.in_channels} → {m.out_channels}, k={m.kernel_size[0]}) → BN → ReLU → MaxPool")

        lines.append(f"  Global: AdaptiveAvgPool1d(1)")

        # FC head
        lines.append(f"\nFC Head:")
        fc_dims = []
        for m in model.fc_layers:
            if isinstance(m, nn.Linear):
                fc_dims.append((m.in_features, m.out_features))
        for i, (in_f, out_f) in enumerate(fc_dims):
            if i < len(fc_dims) - 1:
                lines.append(f"  Linear({in_f} → {out_f}) → ReLU → Dropout")
            else:
                lines.append(f"  Linear({in_f} → {out_f}) → Output")

    elif isinstance(model, (THz_Encoder_CNN, THz_Encoder_ResNet, THz_Encoder_MultiScale, THz_Encoder_MLP)):
        # Generic summary for non-scalable models
        lines.append(f"\nArchitecture: {model_name.replace('THz_Encoder_', '')}")
        lines.append(f"  (Use print(model) for detailed layer info)")

    # Output info
    lines.append(f"\nOutput: [batch, 3] → [n, κ, d] (sigmoid-scaled to physical ranges)")
    lines.append("=" * 60)

    summary = "\n".join(lines)

    if print_output:
        print(summary)
    return summary