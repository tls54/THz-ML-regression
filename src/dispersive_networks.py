"""
Neural network architectures for extracting frequency-dependent n(ω) and κ(ω).

This extends the constant parameter extraction to handle dispersive materials.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_config import Config


class THz_Dispersive_Polynomial(nn.Module):
    """
    Extract dispersive parameters using polynomial expansion.

    Model:
        n(ω) = n₀ + n₁·ω + n₂·ω² + ...
        κ(ω) = κ₀ + κ₁·ω + κ₂·ω² + ...

    This is the simplest dispersive model - good starting point.
    """
    def __init__(self, n_freq=None, n_terms_n=3, n_terms_kappa=2, dropout=0.2):
        """
        Args:
            n_freq: Number of frequency points (for input)
            n_terms_n: Number of polynomial terms for n(ω)
            n_terms_kappa: Number of polynomial terms for κ(ω)
            dropout: Dropout probability
        """
        super().__init__()

        if n_freq is None:
            n_freq, _ = Config.get_freq_info()

        self.n_freq = n_freq
        self.n_terms_n = n_terms_n
        self.n_terms_kappa = n_terms_kappa

        # CNN encoder (reuse architecture from base model)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.AdaptiveAvgPool1d(1)
        )

        # FC layers to predict polynomial coefficients + thickness
        n_params = n_terms_n + n_terms_kappa + 1  # n coeffs + κ coeffs + d

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_params)
        )

    def evaluate_polynomial(self, coeffs, omega_normalized):
        """
        Evaluate polynomial: f(ω) = c₀ + c₁·ω + c₂·ω² + ...

        Args:
            coeffs: [B, n_terms] polynomial coefficients
            omega_normalized: [n_freq] normalized frequencies in [0, 1]

        Returns:
            values: [B, n_freq] evaluated polynomial
        """
        B = coeffs.shape[0]
        n_freq = len(omega_normalized)

        # Compute powers of omega: [n_freq, n_terms]
        omega_powers = torch.stack([
            omega_normalized ** i
            for i in range(coeffs.shape[1])
        ], dim=1)  # [n_freq, n_terms]

        # Matrix multiply: [B, n_terms] @ [n_terms, n_freq] = [B, n_freq]
        values = torch.matmul(coeffs, omega_powers.T)

        return values

    def forward(self, x, frequencies=None):
        """
        Args:
            x: [B, 2, n_freq] - real and imaginary transmission
            frequencies: [n_freq] - frequency array in Hz (optional)

        Returns:
            n_freq: [B, n_freq] - refractive index at each frequency
            kappa_freq: [B, n_freq] - extinction at each frequency
            d: [B] - thickness
        """
        B = x.shape[0]

        # If frequencies not provided, use default
        if frequencies is None:
            from training_config import Config
            freq_info = Config.get_freq_info()
            frequencies = torch.linspace(0, freq_info[1], self.n_freq,
                                        device=x.device)

        # Normalize frequencies to [0, 1] for numerical stability
        omega_normalized = frequencies / frequencies.max()

        # Encode input
        features = self.conv_layers(x)
        params = self.fc_layers(features)

        # Split into n coefficients, κ coefficients, and d
        n_coeffs = params[:, :self.n_terms_n]
        kappa_coeffs = params[:, self.n_terms_n:self.n_terms_n + self.n_terms_kappa]
        d_raw = params[:, -1]

        # Evaluate polynomials
        n_centered = self.evaluate_polynomial(n_coeffs, omega_normalized)
        kappa_centered = self.evaluate_polynomial(kappa_coeffs, omega_normalized)

        # Apply output activations to map to physical ranges
        # For n(ω): center around mean and add polynomial variation
        n_mean = Config.N_MIN + (Config.N_MAX - Config.N_MIN) * torch.sigmoid(n_coeffs[:, 0:1])
        n_variation_scale = 0.2 * (Config.N_MAX - Config.N_MIN)  # Allow ±20% variation
        n_freq = n_mean + n_variation_scale * torch.tanh(n_centered)

        # For κ(ω): similar approach
        kappa_mean = Config.KAPPA_MIN + (Config.KAPPA_MAX - Config.KAPPA_MIN) * torch.sigmoid(kappa_coeffs[:, 0:1])
        kappa_variation_scale = 0.2 * (Config.KAPPA_MAX - Config.KAPPA_MIN)
        kappa_freq = kappa_mean + kappa_variation_scale * torch.tanh(kappa_centered)

        # Thickness
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(d_raw)

        return n_freq, kappa_freq, d

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class THz_Dispersive_Debye(nn.Module):
    """
    Extract Debye model parameters for polar dielectrics.

    Model:
        ε(ω) = ε∞ + (εs - ε∞)/(1 - iωτ)
        n̂(ω) = √ε(ω) = n(ω) + iκ(ω)

    Use for: Water, polar polymers, ceramics, ice
    """
    def __init__(self, n_freq=None, dropout=0.2):
        super().__init__()

        if n_freq is None:
            n_freq, _ = Config.get_freq_info()

        self.n_freq = n_freq

        # CNN encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.AdaptiveAvgPool1d(1)
        )

        # FC layers to predict Debye parameters [ε∞, εs, τ, d]
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # 4 Debye parameters
        )

    def debye_to_nk(self, eps_inf, eps_s, tau, omega):
        """
        Convert Debye parameters to n(ω) and κ(ω).

        Args:
            eps_inf: [B] - high-frequency permittivity
            eps_s: [B] - static permittivity
            tau: [B] - relaxation time (seconds)
            omega: [n_freq] - angular frequency (rad/s)

        Returns:
            n_freq: [B, n_freq]
            kappa_freq: [B, n_freq]
        """
        B = eps_inf.shape[0]
        n_freq = len(omega)

        # Reshape for broadcasting
        eps_inf = eps_inf.view(B, 1)
        eps_s = eps_s.view(B, 1)
        tau = tau.view(B, 1)
        omega = omega.view(1, n_freq)

        # Debye model: ε(ω) = ε∞ + (εs - ε∞)/(1 - iωτ)
        delta_eps = eps_s - eps_inf
        denominator = 1.0 + (omega * tau)**2

        eps_real = eps_inf + delta_eps / denominator
        eps_imag = delta_eps * omega * tau / denominator

        # Complex permittivity
        eps_complex = eps_real + 1j * eps_imag

        # Complex refractive index: n̂ = √ε
        n_complex = torch.sqrt(eps_complex)

        n_freq = n_complex.real
        kappa_freq = n_complex.imag

        return n_freq, kappa_freq

    def forward(self, x, frequencies=None):
        """
        Args:
            x: [B, 2, n_freq] - transmission spectrum
            frequencies: [n_freq] - frequencies in Hz

        Returns:
            n_freq: [B, n_freq]
            kappa_freq: [B, n_freq]
            d: [B]
        """
        # Get frequencies
        if frequencies is None:
            from training_config import Config
            freq_info = Config.get_freq_info()
            frequencies = torch.linspace(0, freq_info[1], self.n_freq,
                                        device=x.device)

        omega = 2 * np.pi * frequencies  # Convert to angular frequency

        # Encode and predict parameters
        features = self.conv_layers(x)
        params = self.fc_layers(features)

        # Apply activations to map to physical ranges
        eps_inf_raw = params[:, 0]
        eps_s_raw = params[:, 1]
        tau_raw = params[:, 2]
        d_raw = params[:, 3]

        # ε∞ typically in [1.5, 5.0]
        eps_inf = 1.5 + 3.5 * torch.sigmoid(eps_inf_raw)

        # εs > ε∞, typically [5, 100]
        eps_s = eps_inf + 95 * torch.sigmoid(eps_s_raw)

        # τ (relaxation time) typically [0.1ps, 100ps] for THz
        tau = 0.1e-12 + 100e-12 * torch.sigmoid(tau_raw)

        # Thickness
        d = Config.D_MIN + (Config.D_MAX - Config.D_MIN) * torch.sigmoid(d_raw)

        # Compute n(ω), κ(ω) from Debye model
        n_freq, kappa_freq = self.debye_to_nk(eps_inf, eps_s, tau, omega)

        return n_freq, kappa_freq, d

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function
def get_dispersive_network(name, **kwargs):
    """
    Get a dispersive network by name.

    Args:
        name: 'polynomial', 'debye', etc.
        **kwargs: Additional arguments for network

    Returns:
        network instance
    """
    networks = {
        'polynomial': THz_Dispersive_Polynomial,
        'debye': THz_Dispersive_Debye,
    }

    if name not in networks:
        raise ValueError(f"Unknown network: {name}. Available: {list(networks.keys())}")

    return networks[name](**kwargs)


# Example usage
if __name__ == '__main__':
    print("Testing dispersive networks...")

    # Create dummy input
    B = 4
    n_freq = 8193
    x = torch.randn(B, 2, n_freq)
    frequencies = torch.linspace(0, 5e12, n_freq)

    # Test polynomial network
    print("\n1. Polynomial Network:")
    net_poly = THz_Dispersive_Polynomial(n_terms_n=3, n_terms_kappa=2)
    print(f"   Parameters: {net_poly.count_parameters():,}")

    n_freq_out, kappa_freq_out, d_out = net_poly(x, frequencies)
    print(f"   Output shapes:")
    print(f"     n(ω): {n_freq_out.shape}")
    print(f"     κ(ω): {kappa_freq_out.shape}")
    print(f"     d:    {d_out.shape}")
    print(f"   n(ω) range: [{n_freq_out.min():.3f}, {n_freq_out.max():.3f}]")
    print(f"   κ(ω) range: [{kappa_freq_out.min():.5f}, {kappa_freq_out.max():.5f}]")

    # Test Debye network
    print("\n2. Debye Network:")
    net_debye = THz_Dispersive_Debye()
    print(f"   Parameters: {net_debye.count_parameters():,}")

    n_freq_out, kappa_freq_out, d_out = net_debye(x, frequencies)
    print(f"   Output shapes:")
    print(f"     n(ω): {n_freq_out.shape}")
    print(f"     κ(ω): {kappa_freq_out.shape}")
    print(f"     d:    {d_out.shape}")
    print(f"   n(ω) range: [{n_freq_out.min():.3f}, {n_freq_out.max():.3f}]")
    print(f"   κ(ω) range: [{kappa_freq_out.min():.5f}, {kappa_freq_out.max():.5f}]")

    print("\n✓ All tests passed!")
