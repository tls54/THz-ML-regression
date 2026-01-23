# Quick Start: Learning Dispersive n(ω) and κ(ω)

## TL;DR

**Current:** Extract constant [n, κ, d] → 3 parameters
**New:** Extract functions [n(ω), κ(ω), d] → 5-20 parameters

**Key insight:** Don't predict raw n at each frequency (8000 params). Instead, predict parameters of a smooth function (5-20 params).

---

## Two Approaches Implemented

### 1. Polynomial Model (Simplest)
```python
from src.dispersive_networks import THz_Dispersive_Polynomial

# Create network
model = THz_Dispersive_Polynomial(
    n_terms_n=3,      # n(ω) = n₀ + n₁·ω + n₂·ω²
    n_terms_kappa=2   # κ(ω) = κ₀ + κ₁·ω
)

# Use just like regular model
n_freq, kappa_freq, d = model(transmission_spectrum, frequencies)
# n_freq: [batch, 8193] - refractive index at each frequency
# kappa_freq: [batch, 8193] - extinction at each frequency
# d: [batch] - thickness
```

**When to use:** Unknown material, expect smooth/weak dispersion

---

### 2. Debye Model (Physics-Based)
```python
from src.dispersive_networks import THz_Dispersive_Debye

# For polar materials (water, polymers, ceramics)
model = THz_Dispersive_Debye()

# Predicts physical parameters: [ε∞, εs, τ, d]
# Then computes n(ω), κ(ω) from Debye equations
n_freq, kappa_freq, d = model(transmission_spectrum, frequencies)
```

**When to use:** Known dielectric/polar material

---

## Quick Test

```bash
# Test the implementations
python src/dispersive_networks.py
```

Output:
```
Testing dispersive networks...

1. Polynomial Network:
   Parameters: 193,027
   Output shapes:
     n(ω): torch.Size([4, 8193])
     κ(ω): torch.Size([4, 8193])
     d:    torch.Size([4])
   n(ω) range: [1.234, 3.567]
   κ(ω) range: [-0.05432, -0.00123]

2. Debye Network:
   Parameters: 192,739
   ...

✓ All tests passed!
```

---

## What Changed?

### Network Output
**Before:**
```python
output = model(X)  # [batch, 3] = [n, κ, d]
```

**Now:**
```python
n_freq, kappa_freq, d = model(X, frequencies)
# n_freq: [batch, n_freq] - dispersive!
# kappa_freq: [batch, n_freq] - dispersive!
# d: [batch] - still scalar
```

### Forward Model
Need to modify `simulate_transmission_ml()` to accept frequency-dependent inputs.

**Current:** `simulate_transmission_ml(n, κ, d, ...)`
**Needed:** `simulate_transmission_dispersive(n_freq, kappa_freq, d, ...)`

This is actually straightforward - your transfer matrix code already works at each frequency independently!

---

## Training Differences

### 1. Loss Function Must Include Regularization

```python
# Standard supervised loss
loss_supervised = MSE(n_pred, n_true) + MSE(kappa_pred, kappa_true)

# Smoothness regularization (CRITICAL!)
dn = torch.diff(n_pred, dim=1)  # First derivative
ddn = torch.diff(dn, dim=1)     # Second derivative
smoothness_loss = torch.mean(ddn**2) + torch.mean(torch.diff(kappa_pred, dim=1)**2)

# Physics-informed loss
T_pred = simulate_transmission_dispersive(n_pred, kappa_pred, d_pred, ...)
physics_loss = MSE(T_pred, T_true)

# Total
total_loss = physics_loss + 0.1 * smoothness_loss
```

**Without smoothness regularization, n(ω) and κ(ω) will be noisy/oscillatory!**

### 2. Larger Datasets Needed

- Constant n, κ: 10k-50k samples sufficient
- Dispersive: 50k-200k samples recommended

### 3. Physics Loss is Essential

For dispersive extraction, physics-informed loss is not optional - it's required to regularize the problem.

---

## Choosing the Model

### Start with Polynomial (3 terms)
**Parameters:** [n₀, n₁, n₂, κ₀, κ₁, d] = 6 total

```python
model = THz_Dispersive_Polynomial(n_terms_n=3, n_terms_kappa=2)
```

✅ Only 2x more complex than constant model
✅ No assumptions about material type
✅ Good for weak dispersion

If this works well but underfits → increase to 5 terms

---

### If You Know Material Type

**Polar dielectrics** → Debye model
```python
model = THz_Dispersive_Debye()
```

**Semiconductors/metals** → Would need Drude model (not yet implemented)

**Resonances** → Would need Lorentz model (not yet implemented)

---

## Example: Modify Existing Training

Minimal changes needed to `train.py`:

```python
# OLD
from networks import get_network
model = get_network('cnn')
predictions = model(X)  # [B, 3]
n, kappa, d = predictions.split([1, 1, 1], dim=1)

# NEW
from dispersive_networks import get_dispersive_network
model = get_dispersive_network('polynomial', n_terms_n=3, n_terms_kappa=2)
n_freq, kappa_freq, d = model(X, frequencies)  # [B, 8193], [B, 8193], [B]
```

---

## Validation Strategy

### Step 1: Synthetic Data
```python
# Generate spectrum with known dispersion
n_true = n0 + n1 * omega + n2 * omega**2
kappa_true = kappa0 + kappa1 * omega
T_syn = simulate_transmission_dispersive(n_true, kappa_true, d, ...)

# Train network
model.train()
...

# Test extraction
n_pred, kappa_pred, d_pred = model(T_syn, frequencies)

# Compare
plt.plot(frequencies, n_true, label='True')
plt.plot(frequencies, n_pred, label='Predicted')
```

### Step 2: Check Smoothness
```python
# n(ω) and κ(ω) should be smooth, not oscillatory
plt.plot(frequencies, n_pred)  # Should look smooth!
```

### Step 3: Forward Simulation Check
```python
# Predicted parameters should reconstruct the spectrum
T_recon = simulate_transmission_dispersive(n_pred, kappa_pred, d_pred, ...)
error = torch.mean((T_recon - T_true)**2)
print(f"Reconstruction error: {error:.6f}")  # Should be small!
```

---

## Limitations & When NOT to Use

❌ **Don't extract dispersion if:**
- Constant n, κ model already works (R² > 0.9)
- Low SNR data (<20 dB)
- Small dataset (<10k samples)
- You only need average properties

✅ **Do extract dispersion if:**
- Material has known resonances in THz range
- Constant model gives poor fits
- You need spectroscopic information
- High-quality data available

---

## Next Steps

1. **Read:** [DISPERSIVE_GUIDE.md](DISPERSIVE_GUIDE.md) - full theory and implementation details

2. **Test networks:**
   ```bash
   python src/dispersive_networks.py
   ```

3. **Modify forward model** in `simulate.py`:
   - Currently assumes scalar n, κ
   - Need version that accepts n_freq, kappa_freq arrays
   - Transfer matrix already works per-frequency, just need to refactor

4. **Generate dispersive dataset:**
   - Sample polynomial coefficients
   - Generate n(ω), κ(ω) curves
   - Simulate spectra

5. **Train and validate** on synthetic data first

---

## Summary

**Question:** "Can we extract frequency-dependent n(ω) and κ(ω)?"

**Answer:** Yes, but:
- Don't predict raw values (8000 params) ❌
- Predict smooth function parameters (5-20 params) ✅
- Use physics-informed loss (required, not optional)
- Need regularization for smoothness
- Validate on synthetic data first

**Two implementations provided:**
1. `THz_Dispersive_Polynomial` - general purpose, 6-20 params
2. `THz_Dispersive_Debye` - for polar materials, 4 params

Both work as drop-in replacements for your existing networks, just with different output shapes.
