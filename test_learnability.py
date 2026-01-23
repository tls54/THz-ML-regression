"""
Test if THz parameter extraction is fundamentally learnable.

This script performs multiple diagnostic tests to determine if the inverse problem
(transmission spectrum → material parameters) has a unique solution.

Tests:
1. Parameter Sensitivity Analysis - How much does T(ω) change with each parameter?
2. Parameter Correlation Analysis - Are different parameters correlated in their effects?
3. Uniqueness Test - Can different parameter sets produce similar spectra?
4. Noise Sensitivity - How does noise affect parameter extraction?
5. Information Content - Is there enough information in the spectrum?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
if './src' not in sys.path:
    sys.path.insert(0, './src')

from simulate import simulate_transmission_ml, prepare_ml_input
from training_config import Config

config = Config()

# Simulation parameters
L = 4096
deltat = 0.1e-12
device = 'cpu'


def test_parameter_sensitivity(n_base=2.0, kappa_base=-0.01, d_base=500e-6):
    """
    Test 1: Parameter Sensitivity Analysis

    How much does the transmission spectrum change when we perturb each parameter?
    If sensitivity is very low, the parameter is hard to extract.
    """
    print("\n" + "="*70)
    print("TEST 1: Parameter Sensitivity Analysis")
    print("="*70)
    print("Testing how transmission changes with small parameter perturbations...")
    print(f"Base parameters: n={n_base}, κ={kappa_base}, d={d_base*1e6:.1f} μm\n")

    # Generate base spectrum
    freq, T_base, _ = simulate_transmission_ml(
        n_base, kappa_base, d_base, L, deltat, device=device
    )

    # Test perturbations (±1%, ±5%)
    perturbations = [0.01, 0.05]  # 1% and 5%

    results = {}

    for param_name, param_value, param_idx in [
        ('n', n_base, 0),
        ('κ', kappa_base, 1),
        ('d', d_base, 2)
    ]:
        results[param_name] = {}

        for p in perturbations:
            # Perturb parameter
            n_pert = n_base * (1 + p) if param_idx == 0 else n_base
            k_pert = kappa_base * (1 + p) if param_idx == 1 else kappa_base
            d_pert = d_base * (1 + p) if param_idx == 2 else d_base

            # Generate perturbed spectrum
            _, T_pert, _ = simulate_transmission_ml(
                n_pert, k_pert, d_pert, L, deltat, device=device
            )

            # Compute difference metrics
            diff = T_pert - T_base

            # L2 norm of difference (in complex space)
            l2_diff = torch.sqrt(torch.sum(torch.abs(diff)**2)).item()

            # Max absolute difference
            max_diff = torch.max(torch.abs(diff)).item()

            # Relative change
            base_norm = torch.sqrt(torch.sum(torch.abs(T_base)**2)).item()
            rel_change = l2_diff / base_norm

            results[param_name][f'{p*100:.0f}%'] = {
                'l2_diff': l2_diff,
                'max_diff': max_diff,
                'rel_change': rel_change
            }

            print(f"{param_name} +{p*100:.0f}% perturbation:")
            print(f"  L2 difference:      {l2_diff:.6f}")
            print(f"  Max |ΔT|:           {max_diff:.6f}")
            print(f"  Relative change:    {rel_change:.4%}")

    # Sensitivity ranking
    print("\nSensitivity Ranking (5% perturbation, by relative change):")
    sensitivities = {
        name: results[name]['5%']['rel_change']
        for name in results.keys()
    }
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    for i, (name, sens) in enumerate(sorted_sens, 1):
        print(f"  {i}. {name}: {sens:.4%} relative change")

    print("\nInterpretation:")
    min_sens = sorted_sens[-1][1]
    if min_sens < 0.001:  # <0.1% change for 5% parameter change
        print(f"  ⚠️  WARNING: {sorted_sens[-1][0]} has very low sensitivity ({min_sens:.4%})")
        print(f"     This parameter may be difficult to extract accurately.")
    elif min_sens < 0.01:  # <1% change
        print(f"  ⚠️  CAUTION: {sorted_sens[-1][0]} has low sensitivity ({min_sens:.4%})")
        print(f"     May require low noise and large dataset.")
    else:
        print(f"  ✓ All parameters show reasonable sensitivity (>1% for 5% perturbation)")

    return results, sensitivities


def test_parameter_correlation(n_range=(1.5, 3.0), kappa_range=(-0.05, -0.001),
                                d_range=(200e-6, 800e-6), n_samples=100):
    """
    Test 2: Parameter Correlation Analysis

    If changing parameter A in one direction can be compensated by changing
    parameter B, they are correlated and the inverse problem is ill-posed.
    """
    print("\n" + "="*70)
    print("TEST 2: Parameter Correlation Analysis")
    print("="*70)
    print("Sampling parameter space to check for correlations...")
    print(f"Testing {n_samples} random parameter combinations\n")

    # Random sampling
    torch.manual_seed(42)
    n_samples_t = torch.rand(n_samples) * (n_range[1] - n_range[0]) + n_range[0]
    k_samples_t = torch.rand(n_samples) * (kappa_range[1] - kappa_range[0]) + kappa_range[0]
    d_samples_t = torch.rand(n_samples) * (d_range[1] - d_range[0]) + d_range[0]

    # Generate spectra
    spectra = []
    for i in range(n_samples):
        _, T, _ = simulate_transmission_ml(
            n_samples_t[i].item(), k_samples_t[i].item(),
            d_samples_t[i].item(), L, deltat, device=device
        )
        # Flatten to real vector
        ml_input = prepare_ml_input(T)  # [2, M+1]
        spectra.append(ml_input.flatten())

    spectra = torch.stack(spectra)  # [n_samples, 2*(M+1)]

    # Convert to numpy for correlation analysis
    n_np = n_samples_t.numpy()
    k_np = k_samples_t.numpy()
    d_np = d_samples_t.numpy()
    spectra_np = spectra.numpy()

    # Compute correlation matrix between parameters
    params_matrix = np.column_stack([n_np, k_np, d_np * 1e6])  # d in μm for scaling

    param_corr = np.corrcoef(params_matrix.T)

    print("Parameter-Parameter Correlations:")
    print("(Values close to ±1 indicate strong correlation)")
    print(f"\n  n vs κ:  {param_corr[0, 1]:+.3f}")
    print(f"  n vs d:  {param_corr[0, 2]:+.3f}")
    print(f"  κ vs d:  {param_corr[1, 2]:+.3f}")

    # Compute correlation between each parameter and spectrum features
    # Use PCA to get dominant spectrum components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    spectra_pca = pca.fit_transform(spectra_np)

    print(f"\nSpectrum variance explained by first 10 components:")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  First 3 components: {pca.explained_variance_ratio_[:3].sum():.2%}")

    # Correlation between parameters and spectral components
    print("\nParameter vs Spectral Component Correlations:")
    print("(High values mean parameter affects this spectral mode)")

    for i in range(3):  # First 3 PCA components
        n_corr = np.corrcoef(n_np, spectra_pca[:, i])[0, 1]
        k_corr = np.corrcoef(k_np, spectra_pca[:, i])[0, 1]
        d_corr = np.corrcoef(d_np, spectra_pca[:, i])[0, 1]

        print(f"\n  PC{i+1} (explains {pca.explained_variance_ratio_[i]:.1%} variance):")
        print(f"    n: {n_corr:+.3f}")
        print(f"    κ: {k_corr:+.3f}")
        print(f"    d: {d_corr:+.3f}")

    print("\nInterpretation:")
    max_param_corr = np.max(np.abs(param_corr[np.triu_indices_from(param_corr, k=1)]))
    if max_param_corr > 0.9:
        print(f"  ⚠️  WARNING: Strong parameter correlation detected ({max_param_corr:.3f})")
        print("     The inverse problem may be ill-posed.")
    elif max_param_corr > 0.7:
        print(f"  ⚠️  CAUTION: Moderate parameter correlation ({max_param_corr:.3f})")
        print("     Regularization may be needed.")
    else:
        print(f"  ✓ Parameters are relatively independent (max corr: {max_param_corr:.3f})")

    return param_corr, pca


def test_uniqueness(n_ref=2.0, kappa_ref=-0.01, d_ref=500e-6,
                    noise_level=0.01, n_trials=1000):
    """
    Test 3: Uniqueness Test

    Generate a reference spectrum, then search for other parameter sets
    that produce similar spectra. If we find many, the problem is not unique.
    """
    print("\n" + "="*70)
    print("TEST 3: Uniqueness Test")
    print("="*70)
    print("Searching for different parameters that produce similar spectra...")
    print(f"Reference: n={n_ref}, κ={kappa_ref}, d={d_ref*1e6:.1f} μm")
    print(f"Noise level: {noise_level*100:.1f}%\n")

    # Generate reference spectrum (with noise)
    freq, T_ref, _ = simulate_transmission_ml(
        n_ref, kappa_ref, d_ref, L, deltat, device=device
    )

    # Add noise
    noise_real = noise_level * torch.randn_like(T_ref.real)
    noise_imag = noise_level * torch.randn_like(T_ref.imag)
    T_ref_noisy = T_ref + noise_real + 1j * noise_imag

    ml_ref = prepare_ml_input(T_ref_noisy).flatten()
    ref_norm = torch.norm(ml_ref).item()

    # Random search for similar spectra
    torch.manual_seed(123)

    similar_params = []
    distances = []

    for _ in range(n_trials):
        # Random parameters
        n_test = torch.rand(1).item() * (Config.N_MAX - Config.N_MIN) + Config.N_MIN
        k_test = torch.rand(1).item() * (Config.KAPPA_MAX - Config.KAPPA_MIN) + Config.KAPPA_MIN
        d_test = torch.rand(1).item() * (Config.D_MAX - Config.D_MIN) + Config.D_MIN

        # Generate spectrum
        _, T_test, _ = simulate_transmission_ml(
            n_test, k_test, d_test, L, deltat, device=device
        )

        ml_test = prepare_ml_input(T_test).flatten()

        # Compute normalized distance
        dist = torch.norm(ml_test - ml_ref).item() / ref_norm
        distances.append(dist)

        # If very similar (within noise level), record it
        if dist < noise_level * 2:  # Within 2x noise level
            param_dist = np.sqrt(
                ((n_test - n_ref) / n_ref)**2 +
                ((k_test - kappa_ref) / kappa_ref)**2 +
                ((d_test - d_ref) / d_ref)**2
            )
            similar_params.append({
                'n': n_test, 'κ': k_test, 'd': d_test,
                'spectrum_dist': dist,
                'param_dist': param_dist
            })

    distances = np.array(distances)

    print(f"Searched {n_trials} random parameter sets")
    print(f"\nSpectrum distance statistics:")
    print(f"  Min:    {distances.min():.6f}")
    print(f"  Mean:   {distances.mean():.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Max:    {distances.max():.6f}")

    print(f"\nParameter sets producing similar spectra (dist < {noise_level*2:.3f}):")
    print(f"  Found: {len(similar_params)}")

    if len(similar_params) > 0:
        print("\n  Top 5 closest matches:")
        sorted_similar = sorted(similar_params, key=lambda x: x['spectrum_dist'])[:5]
        for i, p in enumerate(sorted_similar, 1):
            print(f"    {i}. n={p['n']:.3f}, κ={p['κ']:.5f}, d={p['d']*1e6:.1f} μm")
            print(f"       Spectrum dist: {p['spectrum_dist']:.6f}, Param dist: {p['param_dist']:.3f}")

    print("\nInterpretation:")
    if len(similar_params) > n_trials * 0.01:  # More than 1%
        print(f"  ⚠️  WARNING: {len(similar_params)} similar spectra found ({len(similar_params)/n_trials*100:.1f}%)")
        print("     The inverse problem may not be unique.")
    elif len(similar_params) > 1:
        print(f"  ⚠️  CAUTION: {len(similar_params)} similar spectra found")
        print("     Some parameter combinations may be ambiguous.")
    else:
        print(f"  ✓ No similar spectra found - solution appears unique")

    return similar_params, distances


def test_noise_sensitivity(n=2.0, kappa=-0.01, d=500e-6,
                           noise_levels=[0.001, 0.005, 0.01, 0.05, 0.1]):
    """
    Test 4: Noise Sensitivity

    How much does added noise affect the spectrum? This sets a lower bound
    on achievable parameter extraction accuracy.
    """
    print("\n" + "="*70)
    print("TEST 4: Noise Sensitivity Analysis")
    print("="*70)
    print("Testing how noise affects spectral measurements...")
    print(f"Parameters: n={n}, κ={kappa}, d={d*1e6:.1f} μm\n")

    # Generate clean spectrum
    freq, T_clean, _ = simulate_transmission_ml(
        n, kappa, d, L, deltat, device=device
    )
    ml_clean = prepare_ml_input(T_clean).flatten()

    results = []

    for noise_lvl in noise_levels:
        # Generate multiple noisy versions
        n_samples = 100
        snr_db_list = []

        for _ in range(n_samples):
            noise_real = noise_lvl * torch.randn_like(T_clean.real)
            noise_imag = noise_lvl * torch.randn_like(T_clean.imag)
            T_noisy = T_clean + noise_real + 1j * noise_imag

            ml_noisy = prepare_ml_input(T_noisy).flatten()

            # Compute SNR
            signal_power = torch.mean(ml_clean**2).item()
            noise_power = torch.mean((ml_noisy - ml_clean)**2).item()
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr_db_list.append(snr_db)

        avg_snr = np.mean(snr_db_list)
        std_snr = np.std(snr_db_list)

        results.append({
            'noise_level': noise_lvl,
            'snr_db': avg_snr,
            'snr_std': std_snr
        })

        print(f"Noise level {noise_lvl*100:5.1f}%: SNR = {avg_snr:6.2f} ± {std_snr:.2f} dB")

    print("\nInterpretation:")
    print("  SNR > 40 dB: Excellent - high precision possible")
    print("  SNR 20-40 dB: Good - moderate precision expected")
    print("  SNR < 20 dB: Poor - limited precision, need large dataset")

    # Check typical experimental noise level
    typical_noise = 0.01  # 1% typical
    typical_result = [r for r in results if abs(r['noise_level'] - typical_noise) < 0.001]
    if typical_result:
        snr = typical_result[0]['snr_db']
        print(f"\nFor typical experimental noise (~1%):")
        print(f"  Expected SNR: {snr:.1f} dB")
        if snr > 40:
            print("  ✓ Excellent signal quality - high precision achievable")
        elif snr > 20:
            print("  ✓ Good signal quality - moderate precision achievable")
        else:
            print("  ⚠️  Poor signal quality - may limit extraction accuracy")

    return results


def test_information_content(n=2.0, kappa=-0.01, d=500e-6):
    """
    Test 5: Information Content

    How many effective degrees of freedom are in the spectrum?
    We need at least 3 (for n, κ, d).
    """
    print("\n" + "="*70)
    print("TEST 5: Information Content Analysis")
    print("="*70)
    print("Analyzing information content via singular value decomposition...\n")

    # Generate diverse spectra
    n_samples = 500
    torch.manual_seed(456)

    spectra_list = []
    for _ in range(n_samples):
        n_rand = torch.rand(1).item() * (Config.N_MAX - Config.N_MIN) + Config.N_MIN
        k_rand = torch.rand(1).item() * (Config.KAPPA_MAX - Config.KAPPA_MIN) + Config.KAPPA_MIN
        d_rand = torch.rand(1).item() * (Config.D_MAX - Config.D_MIN) + Config.D_MIN

        _, T, _ = simulate_transmission_ml(
            n_rand, k_rand, d_rand, L, deltat, device=device
        )
        ml_input = prepare_ml_input(T).flatten()
        spectra_list.append(ml_input.numpy())

    spectra_matrix = np.array(spectra_list)  # [n_samples, n_features]

    # Mean center
    spectra_centered = spectra_matrix - spectra_matrix.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(spectra_centered, full_matrices=False)

    # Compute variance explained
    variance_explained = (S**2) / np.sum(S**2)
    cumulative_variance = np.cumsum(variance_explained)

    print("Singular value decomposition results:")
    print(f"  Total features: {spectra_matrix.shape[1]}")
    print(f"  Total samples: {n_samples}")

    print("\nVariance explained by top singular values:")
    for i in range(min(10, len(S))):
        print(f"  SV {i+1}: {variance_explained[i]*100:6.2f}%  (cumulative: {cumulative_variance[i]*100:6.2f}%)")

    # Find effective dimensionality
    n_dims_90 = np.argmax(cumulative_variance > 0.90) + 1
    n_dims_95 = np.argmax(cumulative_variance > 0.95) + 1
    n_dims_99 = np.argmax(cumulative_variance > 0.99) + 1

    print(f"\nEffective dimensionality:")
    print(f"  90% variance: {n_dims_90} dimensions")
    print(f"  95% variance: {n_dims_95} dimensions")
    print(f"  99% variance: {n_dims_99} dimensions")

    print("\nInterpretation:")
    if n_dims_95 < 3:
        print(f"  ⚠️  WARNING: Only {n_dims_95} effective dimensions for 3 parameters")
        print("     The problem is underdetermined!")
    elif n_dims_95 == 3:
        print(f"  ⚠️  CAUTION: Exactly 3 dimensions for 3 parameters")
        print("     The problem is barely determined - any noise will cause issues")
    elif n_dims_95 < 10:
        print(f"  ⚠️  Low effective dimensionality ({n_dims_95})")
        print("     Limited information - may need physics constraints")
    else:
        print(f"  ✓ Good effective dimensionality ({n_dims_95} >> 3)")
        print("     Sufficient information to extract 3 parameters")

    return S, variance_explained, cumulative_variance


def generate_learnability_report():
    """Run all tests and generate comprehensive report."""

    print("\n" + "#"*70)
    print("# THz PARAMETER EXTRACTION - LEARNABILITY ANALYSIS")
    print("#"*70)
    print("\nThis analysis tests whether the inverse problem is well-posed")
    print("and if neural networks can theoretically solve it.\n")

    # Run all tests
    sensitivity_results, sensitivities = test_parameter_sensitivity()
    corr_matrix, pca = test_parameter_correlation()
    similar_params, distances = test_uniqueness()
    noise_results = test_noise_sensitivity()
    svd_S, var_explained, cum_var = test_information_content()

    # Generate summary
    print("\n" + "#"*70)
    print("# SUMMARY - IS THE PROBLEM LEARNABLE?")
    print("#"*70)

    flags = []

    # Check sensitivity
    min_sens = min(sensitivities.values())
    if min_sens < 0.001:
        flags.append(("❌", "Very low parameter sensitivity detected"))
    elif min_sens < 0.01:
        flags.append(("⚠️ ", "Low parameter sensitivity"))
    else:
        flags.append(("✅", "Good parameter sensitivity"))

    # Check correlations
    max_corr = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    if max_corr > 0.9:
        flags.append(("❌", "Strong parameter correlations"))
    elif max_corr > 0.7:
        flags.append(("⚠️ ", "Moderate parameter correlations"))
    else:
        flags.append(("✅", "Low parameter correlations"))

    # Check uniqueness
    if len(similar_params) > 10:
        flags.append(("❌", "Multiple solutions found - non-unique"))
    elif len(similar_params) > 1:
        flags.append(("⚠️ ", "Some ambiguous cases found"))
    else:
        flags.append(("✅", "Solution appears unique"))

    # Check information content
    n_dims = np.argmax(cum_var > 0.95) + 1
    if n_dims < 3:
        flags.append(("❌", "Insufficient information content"))
    elif n_dims < 10:
        flags.append(("⚠️ ", "Limited information content"))
    else:
        flags.append(("✅", "Rich information content"))

    print("\nLearnability Checklist:")
    for symbol, message in flags:
        print(f"  {symbol}  {message}")

    # Overall verdict
    n_good = sum(1 for s, _ in flags if s == "✅")
    n_bad = sum(1 for s, _ in flags if s == "❌")

    print("\nOverall Verdict:")
    if n_bad > 0:
        print("  ⚠️  CONCERNING: The problem has fundamental limitations")
        print("     Consider:")
        print("       - Adding physics-informed constraints")
        print("       - Using multi-scale measurements")
        print("       - Restricting parameter ranges")
        print("       - Bayesian approaches with strong priors")
    elif n_good == len(flags):
        print("  ✅ LEARNABLE: The problem appears well-posed!")
        print("     Neural networks should work well with:")
        print("       - Sufficient training data (>10,000 samples recommended)")
        print("       - Appropriate noise modeling")
        print("       - Standard architectures (CNN, ResNet)")
    else:
        print("  ⚠️  CHALLENGING but LEARNABLE")
        print("     Recommendations:")
        print("       - Use physics-informed loss (hybrid)")
        print("       - Large dataset (>50,000 samples)")
        print("       - Data augmentation")
        print("       - Regularization")

    print("\n" + "#"*70)
    print("Analysis complete!")
    print("#"*70 + "\n")


if __name__ == '__main__':
    generate_learnability_report()
