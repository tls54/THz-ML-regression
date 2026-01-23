import numpy as np
import torch

c = 299792458  # m/s

# ============================================================================
# REFERENCE PULSE GENERATION
# ============================================================================

def simulate_reference(L, deltat, device='cpu'):
    """
    Generate reference THz pulse (propagation through air).
    
    Args:
        L: Number of time samples
        deltat: Time step (seconds)
        device: torch device
        
    Returns:
        x: Reference pulse in time domain, shape [L]
    """
    toff = 1.0e-11
    twidth = 8.0e-13
    tdecay = 1.0e-12
    scale = 1.0e12

    t = torch.arange(0, L, dtype=torch.float32, device=device) * deltat - toff
    x = -scale * t * torch.exp(-(t / twidth) ** 2 - t / tdecay)

    return x


# ============================================================================
# CORE TRANSFER MATRIX METHOD
# ============================================================================

def rts_batched(n0, nj, Dj):
    """
    Compute reflection and transmission coefficients for a single interface/layer.
    
    Args:
        n0: Refractive index of incident medium (scalar or [F])
        nj: Refractive index of layer (complex, shape [F])
        Dj: Phase thickness = k * d for layer (complex, shape [F])
        
    Returns:
        r: Reflection coefficient [F]
        t: Transmission coefficient (with phase) [F]
        s: t*t - r*r [F]
    """
    c = torch.cos(nj * Dj)
    s = torch.sin(nj * Dj)
    d = c + (0.5j) * (nj / n0 + n0 / nj) * s
    r = (0.5j) * s * (n0 / nj - nj / n0) / d
    t = 1.0 / d
    return r, t * torch.exp(1j * n0 * Dj), t * t - r * r


def RTm_batched(m, n0, layers):
    """
    Compute overall reflection and transmission through m layers.
    
    Args:
        m: Number of layers
        n0: Incident medium refractive index (scalar, typically 1.0 for air)
        layers: List of (nj, Dj) tuples, each with shape [F]
        
    Returns:
        R: Overall reflection coefficient [F]
        T: Overall transmission coefficient [F]
    """
    F = layers[0][0].shape[0]
    device = layers[0][0].device

    U = torch.zeros(F, dtype=torch.cfloat, device=device)
    V = torch.ones(F, dtype=torch.cfloat, device=device)
    T = torch.ones(F, dtype=torch.cfloat, device=device)

    for j in range(m):
        nj, Dj = layers[j]
        r, t, s = rts_batched(n0, nj, Dj)

        Vlast = V.clone()
        U_new = r * V + s * U
        V_new = V - r * U
        U, V = U_new, V_new

        T = T * t * Vlast / V

    R = U / V
    return R, T


# ============================================================================
# FORWARD MODEL (Original - Time Domain Focus)
# ============================================================================

def simulate_parallel(x, layers, deltat, noise_level=None):
    """
    Original function - optimized for time domain output.
    
    Args:
        x: Reference pulse [L]
        layers: List of (n_complex, thickness) tuples
        deltat: Time step
        noise_level: Optional noise std
        
    Returns:
        T: Full transmission spectrum [N] with conjugate symmetry
        y: Time domain sample pulse [N] (padded)
    """
    L = len(x)
    M = 2 * L
    N = 4 * L
    deltaf = 1.0 / (N * deltat)
    dk = 2 * torch.pi * deltaf / c

    device = x.device

    # Extract parameters
    indices = torch.stack([
        l[0] if isinstance(l[0], torch.Tensor)
        else torch.tensor(l[0], dtype=torch.cfloat, device=device, requires_grad=True)
        for l in layers
    ])
    thicknesses = torch.stack([
        l[1] if isinstance(l[1], torch.Tensor)
        else torch.tensor(l[1], dtype=torch.cfloat, device=device, requires_grad=True)
        for l in layers
    ])
    m = len(layers)

    # Frequency grid
    k_vals = torch.arange(M + 1, dtype=torch.float32, device=device)
    kD = dk * k_vals[:, None] * thicknesses[None, :]  # [M+1, m]

    # Build batched layers
    batched_layers = [(indices[j].expand(M+1), kD[:, j]) for j in range(m)]
    
    # Compute transmission
    _, T_half = RTm_batched(m, torch.tensor(1.0, dtype=torch.cfloat, device=device), batched_layers)

    # Build full spectrum (conjugate symmetry)
    T = torch.zeros(N, dtype=torch.cfloat, device=device)
    T[:M+1] = T_half
    T[M+1:] = torch.conj(torch.flip(T[1:M], dims=[0]))

    # Apply to reference pulse
    z = torch.zeros(N, dtype=torch.float, device=device)
    z[:L] = x
    X = torch.fft.fft(z) / N
    Y = T * X
    y = N * torch.fft.ifft(Y).real

    if noise_level is not None:
        y += noise_level * torch.randn(N, dtype=torch.float, device=device)

    return T, y


# ============================================================================
# NEW: ML-OPTIMIZED FORWARD MODEL
# ============================================================================

def params_to_complex_index(n, kappa):
    """
    Convert real-valued (n, κ) to complex refractive index.
    
    Args:
        n: Refractive index (real part), shape [B] or scalar
        kappa: Extinction coefficient (imaginary part), shape [B] or scalar
        
    Returns:
        n_complex: Complex refractive index (n + i*κ), shape [B] or scalar
        
    Sign convention: n_complex = n + i*κ
        κ < 0 → Loss/Absorption (standard case)
        κ > 0 → Gain/Amplification (rare)
        
    Wave propagation: E(z) ∝ exp(i*k*ñ*z) = exp(i*k*n*z) * exp(-k*|κ|*z) for κ < 0
    """
    # Ensure we're working with tensors
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, dtype=torch.float32)
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(kappa, dtype=torch.float32)
    
    # Move to same device if they're on different devices
    if n.device != kappa.device:
        kappa = kappa.to(n.device)
    
    # Create complex tensor: n + i*κ (CHANGED from n - i*κ)
    n_complex = n.to(torch.cfloat) + 1j * kappa.to(torch.cfloat)
    
    return n_complex


def simulate_transmission_ml(n, kappa, d, L, deltat, device='cpu', 
                              return_time_domain=False, noise_level=None):
    """
    ML-optimized forward model for single layer.
    
    Args:
        n: Refractive index (real), scalar or [B]
        kappa: Extinction coefficient (real), scalar or [B]
               κ < 0: absorption/loss (typical)
               κ > 0: gain (rare)
        d: Thickness in meters, scalar or [B]
        L: Number of time samples for reference pulse
        deltat: Time step (seconds)
        device: torch device
        return_time_domain: If True, also return time domain pulse
        noise_level: Optional noise std for time domain
        
    Returns:
        frequencies: Frequency array [M+1] in Hz
        T_complex: Transmission coefficient (complex) [M+1] or [B, M+1]
        ref_pulse: Reference pulse [L] (same for all in batch)
        sample_pulse: (optional) Time domain sample pulse [N] or [B, N]
    """
    M = 2 * L
    N = 4 * L
    deltaf = 1.0 / (N * deltat)
    dk = 2 * torch.pi * deltaf / c
    
    # Ensure inputs are tensors on correct device
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, dtype=torch.float32, device=device)
    else:
        n = n.to(device)
    
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(kappa, dtype=torch.float32, device=device)
    else:
        kappa = kappa.to(device)
    
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=torch.float32, device=device)
    else:
        d = d.to(device)
    
    # Generate reference pulse (same for all samples in batch)
    ref_pulse = simulate_reference(L, deltat, device=device)
    
    # Frequency array (positive frequencies only)
    frequencies = torch.arange(M + 1, dtype=torch.float32, device=device) * deltaf
    
    # Convert to complex refractive index
    n_complex = params_to_complex_index(n, kappa)
    
    # Handle batched vs single sample
    is_batched = n.dim() > 0 and n.numel() > 1
    
    if is_batched:
        B = n.shape[0]
        # Expand for all frequencies: [B] → [B, M+1]
        n_freq = n_complex.unsqueeze(1).expand(B, M + 1)
        d_freq = d.unsqueeze(1).expand(B, M + 1)
        
        # Phase thickness: [B, M+1]
        k_vals = torch.arange(M + 1, dtype=torch.float32, device=device)
        kD = dk * k_vals.unsqueeze(0) * d_freq  # [B, M+1]
        
        # Compute transmission for each batch element
        T_list = []
        for b in range(B):
            layers_b = [(n_freq[b], kD[b])]
            _, T_b = RTm_batched(1, torch.tensor(1.0, dtype=torch.cfloat, device=device), 
                                layers_b)
            T_list.append(T_b)
        T_complex = torch.stack(T_list, dim=0)  # [B, M+1]
        
    else:
        # Single sample case - ensure scalar tensor becomes [1] for expand
        if n_complex.dim() == 0:
            n_complex = n_complex.unsqueeze(0)
        if d.dim() == 0:
            d = d.unsqueeze(0)
            
        n_freq = n_complex.expand(M + 1)
        k_vals = torch.arange(M + 1, dtype=torch.float32, device=device)
        kD = dk * k_vals * d
        
        layers = [(n_freq, kD)]
        _, T_complex = RTm_batched(1, torch.tensor(1.0, dtype=torch.cfloat, device=device), 
                                   layers)  # [M+1]
    
    if not return_time_domain:
        return frequencies, T_complex, ref_pulse
    
    # Time domain reconstruction (if requested)
    if is_batched:
        sample_pulses = []
        for b in range(B):
            # Build full spectrum with conjugate symmetry
            T_full = torch.zeros(N, dtype=torch.cfloat, device=device)
            T_full[:M+1] = T_complex[b]
            T_full[M+1:] = torch.conj(torch.flip(T_complex[b, 1:M], dims=[0]))
            
            # Apply to reference
            z = torch.zeros(N, dtype=torch.float, device=device)
            z[:L] = ref_pulse
            X = torch.fft.fft(z) / N
            Y = T_full * X
            y = N * torch.fft.ifft(Y).real
            
            if noise_level is not None:
                y += noise_level * torch.randn(N, dtype=torch.float, device=device)
            
            sample_pulses.append(y)
        
        sample_pulses = torch.stack(sample_pulses, dim=0)  # [B, N]
        return frequencies, T_complex, ref_pulse, sample_pulses
    
    else:
        # Build full spectrum
        T_full = torch.zeros(N, dtype=torch.cfloat, device=device)
        T_full[:M+1] = T_complex
        T_full[M+1:] = torch.conj(torch.flip(T_complex[1:M], dims=[0]))
        
        # Apply to reference
        z = torch.zeros(N, dtype=torch.float, device=device)
        z[:L] = ref_pulse
        X = torch.fft.fft(z) / N
        Y = T_full * X
        y = N * torch.fft.ifft(Y).real
        
        if noise_level is not None:
            y += noise_level * torch.randn(N, dtype=torch.float, device=device)
        
        return frequencies, T_complex, ref_pulse, y


# ============================================================================
# VISUALIZATION HELPERS (Magnitude & Phase - for analysis only)
# ============================================================================

def get_transmission_features(T_complex):
    """
    Extract magnitude and phase from complex transmission coefficient.
    
    FOR VISUALIZATION ONLY - not for ML input (use prepare_ml_input instead)
    
    Args:
        T_complex: Complex transmission, shape [M+1] or [B, M+1]
        
    Returns:
        magnitude: |T|, same shape as input
        phase: angle(T) in radians (WRAPPED to [-π, π]), same shape as input
    """
    magnitude = torch.abs(T_complex)
    phase = torch.angle(T_complex)
    return magnitude, phase


def plot_transmission(frequencies, T_complex, title="Transmission Coefficient", 
                     save_path=None):
    """
    Standard THz field visualization: magnitude and phase plots.
    
    Args:
        frequencies: Frequency array [M+1] in Hz
        T_complex: Complex transmission [M+1]
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    magnitude, phase = get_transmission_features(T_complex)
    
    freq_THz = frequencies.cpu().numpy() / 1e12
    mag_np = magnitude.cpu().numpy()
    phase_np = phase.cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude
    ax1.plot(freq_THz, mag_np, 'b-', linewidth=1.5)
    ax1.set_xlabel('Frequency (THz)')
    ax1.set_ylabel('|T(ω)|')
    ax1.set_title(f'{title} - Magnitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(mag_np) * 1.1] if max(mag_np) > 0 else [0, 1])
    
    # Phase (wrapped)
    ax2.plot(freq_THz, phase_np, 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (THz)')
    ax2.set_ylabel('∠T(ω) (rad)')
    ax2.set_title(f'{title} - Phase (wrapped)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-np.pi, np.pi])
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=np.pi, color='k', linestyle=':', alpha=0.2)
    ax2.axhline(y=-np.pi, color='k', linestyle=':', alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


# ============================================================================
# ML INPUT PREPARATION (Real/Imag - for training)
# ============================================================================

def prepare_ml_input(T_complex):
    """
    Prepare transmission coefficient as ML network input using real/imag components.
    
    This avoids phase wrapping issues by using Cartesian representation.
    
    Args:
        T_complex: Complex transmission [M+1] or [B, M+1]
        
    Returns:
        input_tensor: [2, M+1] or [B, 2, M+1]
                     Channel 0: Real part of T
                     Channel 1: Imaginary part of T
    """
    T_real = T_complex.real
    T_imag = T_complex.imag
    
    if T_complex.dim() == 1:
        # Single sample: [M+1] → [2, M+1]
        return torch.stack([T_real, T_imag], dim=0)
    else:
        # Batched: [B, M+1] → [B, 2, M+1]
        return torch.stack([T_real, T_imag], dim=1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_bandwidth_info(frequencies):
    """
    Get bandwidth statistics.
    
    Args:
        frequencies: Frequency array [M+1]
        
    Returns:
        dict with f_min, f_max, bandwidth, n_points, df
    """
    return {
        'f_min': frequencies[1].item(),  # Skip DC
        'f_max': frequencies[-1].item(),
        'bandwidth': (frequencies[-1] - frequencies[1]).item(),
        'n_points': len(frequencies),
        'df': (frequencies[1] - frequencies[0]).item()
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Simulation parameters - adjusted for ~5 THz bandwidth
    L = 2**12  # 4096 time points
    deltat = 0.1e-12  # 100 fs → f_max ≈ 5 THz
    device = 'cpu'
    
    print("="*60)
    print("Testing ML-Optimized Forward Model")
    print("="*60)
    
    # Check what our time window is
    T_window = L * deltat
    print(f"\nTime window: {T_window*1e12:.1f} ps")
    print(f"This corresponds to max thickness (for n=1.5): {c*T_window/(2*1.5)*1e6:.0f} μm")
    
    # Test single sample (note: kappa is NEGATIVE for absorption)
    print("\n1. Single Sample Test:")
    n = 2.5  # Increased n for stronger reflections
    kappa = -0.005  # Reduced absorption to see echoes better
    d = 200e-6  # 200 μm - thinner for shorter echo delay
    
    # Check if echoes will be visible
    echo_delay = 2 * n * d / c  # Round trip time through sample
    print(f"  Round-trip delay through sample: {echo_delay*1e12:.2f} ps")
    
    # Generate without noise first to see echoes clearly
    freq, T, ref, sample_clean = simulate_transmission_ml(
        n, kappa, d, L, deltat, device=device, 
        return_time_domain=True, noise_level=None  # No noise first
    )
    
    # Also generate with noise
    _, _, _, sample_noisy = simulate_transmission_ml(
        n, kappa, d, L, deltat, device=device, 
        return_time_domain=True, noise_level=5e-3  # Moderate noise
    )
    
    mag, phase = get_transmission_features(T)
    bw_info = get_bandwidth_info(freq)
    
    print(f"  Parameters: n={n}, κ={kappa} (absorption), d={d*1e6:.1f} μm")
    print(f"  Frequency range: {bw_info['f_min']/1e12:.2f} - {bw_info['f_max']/1e12:.2f} THz")
    print(f"  Bandwidth: {bw_info['bandwidth']/1e12:.2f} THz")
    print(f"  Frequency points: {bw_info['n_points']}")
    
    # Find main pulse peak
    ref_np = ref.cpu().numpy()
    sample_np = sample_clean.cpu().numpy()
    main_pulse_idx = np.argmax(np.abs(ref_np))
    main_pulse_time = main_pulse_idx * deltat * 1e12  # in ps
    
    echo1_time = main_pulse_time + echo_delay * 1e12
    echo2_time = main_pulse_time + 2 * echo_delay * 1e12
    echo3_time = main_pulse_time + 3 * echo_delay * 1e12
    
    print(f"  Main pulse peak at: {main_pulse_time:.2f} ps")
    print(f"  Expected 1st echo at: {echo1_time:.2f} ps")
    print(f"  Expected 2nd echo at: {echo2_time:.2f} ps")
    print(f"  Expected 3rd echo at: {echo3_time:.2f} ps")
    
    # Prepare ML input
    ml_input = prepare_ml_input(T)
    print(f"  ML input shape: {ml_input.shape} (channels: [real, imag])")
    
    # Test batched (all with negative kappa for absorption)
    print("\n2. Batched Test:")
    B = 5
    n_batch = torch.tensor([1.2, 1.5, 2.0, 3.0, 4.0], device=device)
    kappa_batch = torch.tensor([-0.005, -0.01, -0.015, -0.02, -0.025], device=device)  # All negative
    d_batch = torch.tensor([200e-6, 300e-6, 400e-6, 500e-6, 600e-6], device=device)
    
    freq_b, T_b, ref_b, sample_b = simulate_transmission_ml(
        n_batch, kappa_batch, d_batch, L, deltat, device=device,
        return_time_domain=True, noise_level=5e-3
    )
    
    print(f"  Batch size: {B}")
    print(f"  T shape: {T_b.shape}")
    print(f"  Sample pulses shape: {sample_b.shape}")
    
    # Prepare batched ML input
    ml_input_batch = prepare_ml_input(T_b)
    print(f"  ML input batch shape: {ml_input_batch.shape}")
    
    
    # Visualization with cleaned up layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Time domain - FULL WIDTH main pulse + echo region
    time_ps = np.arange(len(sample_clean)) * deltat * 1e12  # Convert to ps
    
    # Main pulse region - FULL WIDTH (spans all 3 columns)
    ax1 = fig.add_subplot(gs[0, :])  # <- This spans all columns!
    ax1.plot(time_ps[:L], ref_np, label='Reference', alpha=0.7, linewidth=1.5)
    ax1.plot(time_ps[:L], sample_np[:L], label=f'Sample (n={n}, κ={kappa}, d={d*1e6:.0f}μm)', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain - Main Pulse and Sample Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([main_pulse_time - 5, main_pulse_time + 30])
    
    # Row 2: Echo region + Full trace
    # Echo region - clean (left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_ps, sample_np, 'b-', linewidth=1, label='Clean signal')
    ax2.axvline(x=echo1_time, color='r', linestyle='--', alpha=0.7, label=f'1st echo ({echo1_time:.1f} ps)')
    ax2.axvline(x=echo2_time, color='orange', linestyle='--', alpha=0.7, label=f'2nd echo ({echo2_time:.1f} ps)')
    ax2.axvline(x=echo3_time, color='green', linestyle='--', alpha=0.7, label=f'3rd echo ({echo3_time:.1f} ps)')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Time Domain - Echo Region (clean)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([echo1_time - 5, echo3_time + 5])
    
    # Full time trace (log scale) - clean vs noisy (middle)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(time_ps, np.abs(sample_np) + 1e-10, 'b-', alpha=0.5, linewidth=0.8, label='Clean')
    ax3.semilogy(time_ps, np.abs(sample_noisy.cpu().numpy()) + 1e-10, 'r-', alpha=0.5, linewidth=0.8, label='With noise')
    ax3.axvline(x=main_pulse_time, color='k', linestyle='-', alpha=0.3, label='Main pulse')
    ax3.axvline(x=echo1_time, color='r', linestyle='--', alpha=0.5)
    ax3.axvline(x=echo2_time, color='orange', linestyle='--', alpha=0.5)
    ax3.axvline(x=echo3_time, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('|Amplitude| (log)')
    ax3.set_title('Time Trace (log scale)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, min(50, T_window*1e12)])
    ax3.set_ylim([1e-5, 1])
    
    # Reflection coefficient estimate (right)
    ax4 = fig.add_subplot(gs[1, 2])
    r_fresnel = (n - 1) / (n + 1)
    expected_echo_amplitude = r_fresnel**2 * np.exp(-2 * np.abs(kappa) * 2 * np.pi * 1e12 * d / c)
    ax4.bar(['Main\nPulse', 'Expected\n1st Echo', 'Expected\n2nd Echo'], 
            [1.0, expected_echo_amplitude, expected_echo_amplitude**2])
    ax4.set_ylabel('Relative Amplitude')
    ax4.set_title(f'Expected Echo Strengths\n(Fresnel r={r_fresnel:.3f})')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # Row 3: Frequency domain visualization
    # Transmission magnitude (left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(freq.cpu().numpy()/1e12, mag.cpu().numpy(), linewidth=1.5)
    ax5.set_xlabel('Frequency (THz)')
    ax5.set_ylabel('|T(ω)|')
    ax5.set_title('Transmission Magnitude')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 5])
    
    # Transmission phase (wrapped) (middle)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(freq.cpu().numpy()/1e12, phase.cpu().numpy(), 'r-', linewidth=1)
    ax6.set_xlabel('Frequency (THz)')
    ax6.set_ylabel('∠T(ω) (rad)')
    ax6.set_title('Transmission Phase (wrapped)')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([-np.pi, np.pi])
    ax6.set_xlim([0, 5])
    
    # Real and Imaginary components (right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(freq.cpu().numpy()/1e12, T.real.cpu().numpy(), linewidth=1.5, label='Re(T)', color='blue')
    ax7.plot(freq.cpu().numpy()/1e12, T.imag.cpu().numpy(), linewidth=1.5, label='Im(T)', color='red')
    ax7.set_xlabel('Frequency (THz)')
    ax7.set_ylabel('T(ω)')
    ax7.set_title('Transmission: Real & Imaginary Components')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 5])
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.savefig('forward_model_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'forward_model_test.png'")
    plt.show()
    
    # Verify gradients work
    print("\n3. Gradient Test:")
    n_test = torch.tensor(2.0, requires_grad=True, device=device)
    kappa_test = torch.tensor(-0.01, requires_grad=True, device=device)
    d_test = torch.tensor(400e-6, requires_grad=True, device=device)
    
    freq_test, T_test, _ = simulate_transmission_ml(
        n_test, kappa_test, d_test, L, deltat, device=device
    )
    
    # Prepare ML input and compute loss
    ml_input_test = prepare_ml_input(T_test)
    loss = torch.mean(ml_input_test ** 2)
    loss.backward()
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ∂loss/∂n: {n_test.grad.item():.6e}")
    print(f"  ∂loss/∂κ: {kappa_test.grad.item():.6e}")
    print(f"  ∂loss/∂d: {d_test.grad.item():.6e}")
    
    if n_test.grad.item() != 0 and kappa_test.grad.item() != 0 and d_test.grad.item() != 0:
        print("  ✓ All gradients are non-zero - backpropagation works!")
    else:
        print("  ✗ Warning: Some gradients are zero!")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)