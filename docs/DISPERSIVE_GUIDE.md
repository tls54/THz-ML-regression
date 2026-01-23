# Learning Frequency-Dependent n(ω) and κ(ω)

Guide for extracting dispersive material parameters from THz spectroscopy.

## Problem Overview

**Standard (Current) Problem:**
- Extract constant n, κ, d
- 3 parameters total
- Assumes non-dispersive material

**Dispersive Problem:**
- Extract n(ω), κ(ω), d
- Potentially 1000s of parameters
- Much more challenging!

---

## Why This is Hard

### 1. Underdetermined Problem

With ~8000 frequency points:
- You'd need to extract ~16,000 values (n and κ at each frequency)
- But you only have ~8000 measurements (real + imag at each frequency)
- **Problem: More unknowns than measurements!**

With noise, this gets worse:
- Effective information content drops
- Need strong regularization

### 2. Non-Uniqueness

Multiple n(ω), κ(ω) curves can produce similar transmission spectra, especially:
- At low SNR
- In frequency regions with low transmission
- At band edges

### 3. Physics Constraints Required

Unlike constant n, κ - you MUST enforce:
- **Kramers-Kronig relations** (causality)
- **Smoothness** (materials don't have infinitely sharp features at THz frequencies)
- **Physical bounds** (n ≥ 1, κ ≤ 0 for passive materials)

---

## Recommended Approaches

### Approach 1: Parametric Dispersion Models ⭐ (Recommended)

**Idea:** Instead of predicting raw n(ω), predict parameters of a known dispersion model.

**Common THz Dispersion Models:**

#### Debye Model (polar materials, dielectrics)
```
ε(ω) = ε∞ + (εs - ε∞)/(1 - iωτ)
n̂(ω) = √ε(ω)
```
**Parameters to predict:** [ε∞, εs, τ, d]  (4 parameters)

#### Drude Model (metals, semiconductors)
```
ε(ω) = ε∞ - ωp²/(ω² + iγω)
```
**Parameters to predict:** [ε∞, ωp, γ, d]  (4 parameters)

#### Lorentz Oscillator (resonances)
```
ε(ω) = ε∞ + Σⱼ fⱼ/(ω₀ⱼ² - ω² - iγⱼω)
```
**Parameters to predict:** [ε∞, ω₀₁, γ₁, f₁, ω₀₂, ..., d]  (3N+2 for N oscillators)

#### Polynomial Expansion (generic smooth dispersion)
```
n(ω) = n₀ + n₁·ω + n₂·ω² + ...
κ(ω) = κ₀ + κ₁·ω + κ₂·ω² + ...
```
**Parameters to predict:** [n₀, n₁, n₂, ..., κ₀, κ₁, κ₂, ..., d]  (2M+1 for M terms)

**Advantages:**
- ✅ Low-dimensional (3-20 parameters instead of 1000s)
- ✅ Physics-motivated
- ✅ Smooth by construction
- ✅ Interpretable results

**Disadvantages:**
- ⚠️ Must choose correct model for your material
- ⚠️ Can't represent arbitrary dispersion

**Use when:** You know the material class (dielectric, semiconductor, etc.)

---

### Approach 2: Basis Function Expansion

**Idea:** Represent n(ω) and κ(ω) as linear combinations of smooth basis functions.

**Example: Chebyshev Polynomials**
```python
n(ω) = Σᵢ₌₀ᴺ aᵢ·Tᵢ(ω)  # Chebyshev polynomials
κ(ω) = Σᵢ₌₀ᴺ bᵢ·Tᵢ(ω)
```

**Network predicts:** [a₀, a₁, ..., aₙ, b₀, b₁, ..., bₙ, d]  (2N+2 parameters)

**Advantages:**
- ✅ More flexible than parametric models
- ✅ Smooth by choosing N small (e.g., N=5-10)
- ✅ Can approximate any smooth function

**Disadvantages:**
- ⚠️ Less interpretable
- ⚠️ Doesn't automatically satisfy Kramers-Kronig

**Use when:** Unknown material type, but expect smooth dispersion

---

### Approach 3: Direct Prediction with Regularization

**Idea:** Predict n(ω) and κ(ω) directly at each frequency, but heavily regularize.

**Network Architecture:**
```
Input: T(ω) [2, 8193]
    ↓
CNN Encoder
    ↓
Bottleneck [256]  ← Forces compressed representation
    ↓
CNN Decoder
    ↓
Output: [n(ω), κ(ω), d]  [2, 8193] + [1]
```

**Loss Function:**
```python
loss = MSE_loss + λ₁·smoothness_loss + λ₂·KK_loss + λ₃·physics_loss

# Smoothness: penalize oscillations
smoothness_loss = ||∇²n(ω)||² + ||∇²κ(ω)||²

# Kramers-Kronig consistency
KK_loss = ||n(ω) - KK_transform(κ(ω))||²

# Physics: reconstruction via forward model
physics_loss = ||T_pred(n(ω), κ(ω), d) - T_true||²
```

**Advantages:**
- ✅ Most flexible - no model assumptions
- ✅ Can represent any dispersion

**Disadvantages:**
- ⚠️ High-dimensional (hard to train)
- ⚠️ Requires careful regularization tuning
- ⚠️ Need large dataset (100k+ samples)
- ⚠️ Kramers-Kronig enforcement is complex

**Use when:** Maximum flexibility needed, have lots of data

---

## Implementation Strategy

### Step 1: Choose Your Model

**Decision tree:**

```
Do you know the material class?
├─ Yes → Use parametric model (Debye, Drude, Lorentz)
└─ No
   └─ Is dispersion expected to be smooth and weak?
      ├─ Yes → Polynomial expansion (3-5 terms)
      └─ No → Basis functions or direct prediction
```

### Step 2: Modify Forward Model

Your current `simulate_transmission_ml()` assumes constant n, κ.

**Need new function:**
```python
def simulate_transmission_dispersive(n_freq, kappa_freq, d, L, deltat, device='cpu'):
    """
    Forward model with frequency-dependent n(ω) and κ(ω).

    Args:
        n_freq: [B, M+1] - refractive index at each frequency
        kappa_freq: [B, M+1] - extinction coefficient at each frequency
        d: [B] - thickness
        L: time samples
        deltat: time step

    Returns:
        T_complex: [B, M+1] - transmission coefficient
    """
    # n and κ are already at each frequency!
    # Just build complex index and compute transfer matrix
    ...
```

### Step 3: Modify Network Architecture

**For parametric models:**
```python
class THz_Dispersive_Parametric(nn.Module):
    def __init__(self, model_type='debye'):
        super().__init__()

        # Standard CNN encoder
        self.encoder = THz_Encoder_CNN()

        # Output depends on model
        if model_type == 'debye':
            n_params = 4  # [ε∞, εs, τ, d]
        elif model_type == 'drude':
            n_params = 4  # [ε∞, ωp, γ, d]
        elif model_type == 'lorentz':
            n_params = 3 * n_oscillators + 2

        self.fc_params = nn.Linear(encoder_dim, n_params)

    def forward(self, x, frequencies):
        # Encode spectrum
        features = self.encoder(x)

        # Predict model parameters
        params = self.fc_params(features)

        # Compute n(ω), κ(ω) from parameters
        n_freq, kappa_freq = self.dispersion_model(params, frequencies)

        # Extract thickness
        d = params[:, -1]

        return n_freq, kappa_freq, d
```

**For basis functions:**
```python
class THz_Dispersive_Basis(nn.Module):
    def __init__(self, n_basis=10):
        super().__init__()
        self.n_basis = n_basis
        self.encoder = THz_Encoder_CNN()
        self.fc_coeffs = nn.Linear(encoder_dim, 2*n_basis + 1)  # n, κ coeffs + d

    def forward(self, x, frequencies):
        features = self.encoder(x)
        coeffs = self.fc_coeffs(features)

        # Split coefficients
        n_coeffs = coeffs[:, :self.n_basis]
        kappa_coeffs = coeffs[:, self.n_basis:2*self.n_basis]
        d = coeffs[:, -1]

        # Evaluate basis functions
        n_freq = self.evaluate_basis(n_coeffs, frequencies)
        kappa_freq = self.evaluate_basis(kappa_coeffs, frequencies)

        return n_freq, kappa_freq, d
```

### Step 4: Modified Loss Function

**Physics-informed loss is ESSENTIAL:**

```python
class DispersiveLoss(nn.Module):
    def __init__(self, alpha_smooth=0.1, alpha_kk=0.1, alpha_physics=0.5):
        super().__init__()
        self.alpha_smooth = alpha_smooth
        self.alpha_kk = alpha_kk
        self.alpha_physics = alpha_physics

    def forward(self, n_pred, kappa_pred, d_pred, T_true, frequencies):
        # Smoothness regularization
        dn = torch.diff(n_pred, dim=1)  # First derivative
        ddn = torch.diff(dn, dim=1)     # Second derivative
        dk = torch.diff(kappa_pred, dim=1)
        ddk = torch.diff(dk, dim=1)

        smoothness_loss = torch.mean(ddn**2) + torch.mean(ddk**2)

        # Kramers-Kronig consistency (optional but recommended)
        n_from_kappa = kramers_kronig_transform(kappa_pred, frequencies)
        kk_loss = torch.mean((n_pred - n_from_kappa)**2)

        # Physics: forward model reconstruction
        T_pred = simulate_transmission_dispersive(
            n_pred, kappa_pred, d_pred, L, deltat, device
        )
        physics_loss = torch.mean(torch.abs(T_pred - T_true)**2)

        total_loss = (
            self.alpha_physics * physics_loss +
            self.alpha_smooth * smoothness_loss +
            self.alpha_kk * kk_loss
        )

        return total_loss
```

---

## Practical Considerations

### Dataset Generation

**For parametric models:**
```python
# Sample model parameters instead of constant n, κ
eps_inf = np.random.uniform(2.0, 5.0)
eps_s = np.random.uniform(10.0, 100.0)
tau = np.random.uniform(0.1e-12, 10e-12)  # relaxation time
d = np.random.uniform(100e-6, 1000e-6)

# Compute n(ω), κ(ω) from model
n_freq, kappa_freq = debye_model(eps_inf, eps_s, tau, frequencies)

# Simulate spectrum
T = simulate_transmission_dispersive(n_freq, kappa_freq, d, ...)
```

**For general dispersion:**
```python
# Generate random smooth dispersion
n_base = 2.0
n_coeffs = np.random.randn(5) * 0.1  # Small perturbations
n_freq = n_base + polynomial(frequencies, n_coeffs)

kappa_base = -0.01
kappa_coeffs = np.random.randn(5) * 0.005
kappa_freq = kappa_base + polynomial(frequencies, kappa_coeffs)
```

### Validation

**Test if your extraction works:**

1. **Synthetic data with known dispersion**
   - Generate data from Debye model
   - Extract parameters
   - Compare to ground truth

2. **Consistency checks**
   - Do extracted n(ω), κ(ω) satisfy Kramers-Kronig?
   - Are they smooth?
   - Forward simulation: does T_pred match T_true?

3. **Experimental validation**
   - Compare to literature values for known materials
   - Check against ellipsometry or other techniques

---

## Expected Challenges

### 1. Low Sensitivity Regions

Some frequency ranges have low transmission → hard to extract n(ω), κ(ω) there.

**Solution:**
- Weight loss by transmission magnitude
- Use multi-thickness samples
- Combine with reflection measurements

### 2. High-Frequency Noise

At high frequencies, SNR drops → noisy n(ω), κ(ω) predictions.

**Solution:**
- Increase smoothness regularization at high freq
- Use physics priors (many materials have n→1 at high freq)

### 3. Thickness-Dispersion Coupling

Changes in d and n(ω) can partially compensate each other.

**Solution:**
- Fix d if known
- Use multi-angle measurements
- Strong physics-informed loss

---

## Simplified Starting Point

**Easiest approach to try first:**

### 3-Parameter Polynomial Model

```python
n(ω) = n₀ + n₁·ω + n₂·ω²
κ(ω) = κ₀ + κ₁·ω  # Usually less dispersive
d = d
```

**Total parameters:** 6 (n₀, n₁, n₂, κ₀, κ₁, d)

This is:
- ✅ Simple to implement
- ✅ Only 2x more parameters than constant case
- ✅ Captures weak dispersion
- ✅ Good starting point for testing

---

## When is Dispersion Extraction Worthwhile?

**You should extract n(ω), κ(ω) if:**

✅ Material is known to be dispersive (e.g., near resonances)
✅ You have high-SNR data (>40 dB)
✅ You need full spectroscopic information
✅ Constant n, κ model gives poor fits

**Stick with constant n, κ if:**

✅ Material is weakly dispersive in your frequency range
✅ You only need average properties
✅ Data is noisy
✅ Constant model works well (R² > 0.9)

---

## Summary

**To learn frequency-dependent n(ω) and κ(ω):**

1. **Choose parameterization** (parametric > basis > direct)
2. **Modify forward model** to accept n(ω), κ(ω) arrays
3. **Add strong regularization** (smoothness, Kramers-Kronig)
4. **Use physics-informed loss** (forward model reconstruction)
5. **Start simple** (polynomial with 3-5 terms)
6. **Validate carefully** (synthetic data first)

The key insight: **You're not predicting 8000 parameters, you're predicting 5-20 parameters of a smooth function.**
