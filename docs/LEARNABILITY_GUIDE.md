# How to Test if THz Parameter Extraction is Learnable

## Quick Answer

Run the notebook: **[test_learnability.ipynb](test_learnability.ipynb)**

This tests whether your inverse problem (spectrum → parameters) has a unique, stable solution.

---

## The 5 Key Tests

### 1. **Parameter Sensitivity** ✓
**Question:** Do parameter changes produce detectable spectrum changes?

**What to check:**
- Perturb each parameter by 5%
- Measure how much the spectrum changes

**Interpretation:**
- ✅ **Good:** >5% spectrum change for 5% parameter change
- ⚠️  **Concerning:** 1-5% spectrum change
- ❌ **Bad:** <1% spectrum change (parameter is unlearnable)

**Why it matters:** If a parameter barely affects the spectrum, it's impossible to extract it accurately from noisy data.

---

### 2. **Uniqueness** ✓
**Question:** Can different parameter sets produce identical spectra?

**What to check:**
- Generate a reference spectrum
- Search for other parameters producing the same spectrum
- Count how many "matches" you find

**Interpretation:**
- ✅ **Good:** 0-1 similar spectra (unique solution)
- ⚠️  **Concerning:** 2-10 similar spectra (mostly unique)
- ❌ **Bad:** >10 similar spectra (non-unique, ill-posed)

**Why it matters:** Non-unique problems have multiple correct answers. Neural networks will average over them and give poor predictions.

---

### 3. **Information Content (PCA)** ✓
**Question:** How many degrees of freedom are in the spectrum?

**What to check:**
- Generate 500+ random spectra
- Perform PCA (Principal Component Analysis)
- Count dimensions needed for 95% variance

**Interpretation:**
- ✅ **Good:** >10 effective dimensions (rich information)
- ⚠️  **Concerning:** 3-10 dimensions (barely sufficient)
- ❌ **Bad:** <3 dimensions (underdetermined problem)

**Why it matters:** You need at least as many independent measurements as parameters (3). More is better for robustness.

**Rule of thumb:** Want 3-10x more information than parameters (so 9-30 effective dimensions for 3 parameters).

---

### 4. **Parameter Correlation** ✓
**Question:** Are the parameters independent, or do they compensate each other?

**What to check:**
- Compute correlation matrix between n, κ, d
- Look for high correlations (>0.7)

**Interpretation:**
- ✅ **Good:** All correlations <0.5 (independent)
- ⚠️  **Concerning:** Some correlations 0.5-0.8
- ❌ **Bad:** Any correlation >0.9 (highly correlated)

**Why it matters:** If increasing `n` can be compensated by decreasing `κ`, the neural network sees them as the same thing.

**Example of bad correlation:**
- If n↑ and d↓ both increase phase accumulation similarly → they're correlated
- Network can't tell which changed

---

### 5. **Noise Sensitivity** ✓
**Question:** How does realistic noise affect the spectrum?

**What to check:**
- Add typical experimental noise (1-5%)
- Measure Signal-to-Noise Ratio (SNR)

**Interpretation:**
- ✅ **Good:** SNR >40 dB (excellent)
- ⚠️  **Concerning:** SNR 20-40 dB (moderate)
- ❌ **Bad:** SNR <20 dB (poor, limits accuracy)

**Why it matters:** Noise sets a fundamental limit on parameter extraction accuracy. No amount of training data can overcome this.

---

## Interpreting Results

### ✅ **All Tests Pass → LEARNABLE**

The problem is well-posed and neural networks should work well!

**Recommendations:**
- Use standard CNN or ResNet architecture
- Train on 10k-100k samples
- Hybrid loss (supervised + physics) works well
- Expect high accuracy (R² > 0.95)

**Example training:**
```python
python src/train.py --network cnn --dataset production_v1 --epochs 100
```

---

### ⚠️ **Some Tests Fail → CHALLENGING BUT POSSIBLE**

The problem has some difficulties but is solvable with care.

**Recommendations:**
- Use physics-informed loss (or hybrid with high physics weight)
- Generate >100k training samples
- Add strong regularization (dropout, weight decay)
- May need deeper network (ResNet, MultiScale)
- Expect moderate accuracy (R² > 0.8)

**Example training:**
```python
python src/train.py --network resnet --loss physics --epochs 200
```

**Additional strategies:**
- Restrict parameter ranges (narrow down the search space)
- Add data augmentation
- Use ensemble methods
- Physics-guided architecture design

---

### ❌ **Many Tests Fail → FUNDAMENTAL ISSUES**

The inverse problem is ill-posed and may not be solvable by pure ML.

**Possible solutions:**

1. **Restrict Parameter Ranges**
   - If you know n ∈ [1.8, 2.2] instead of [1.1, 7.0], learning is easier
   - Reduce search space to improve uniqueness

2. **Bayesian Approaches**
   - Use strong prior distributions
   - Probabilistic predictions with uncertainty
   - Helps when multiple solutions exist

3. **Multi-Measurement**
   - Combine THz-TDS with other techniques
   - Multiple angles, polarizations, temperatures
   - More information → better determined problem

4. **Physics Regularization**
   - Enforce known physics constraints
   - Limit parameter combinations to physically realistic ones

5. **Traditional Methods**
   - Consider optimization-based approaches
   - Gradient descent in parameter space
   - May work better than neural networks for ill-posed problems

---

## Quick Diagnostic Checklist

Run [test_learnability.ipynb](test_learnability.ipynb) and check:

- [ ] All parameters show >5% sensitivity?
- [ ] <5 similar spectra found in uniqueness test?
- [ ] >10 effective dimensions in PCA?
- [ ] All parameter correlations <0.7?
- [ ] SNR >20 dB for typical noise?

**Score:**
- **5/5:** Excellent, proceed with confidence ✅
- **3-4/5:** Good, proceed with care ⚠️
- **1-2/5:** Concerning, consider alternatives ❌

---

## What Makes a Problem Learnable?

### The Physics Perspective

For the inverse problem `T(ω) → (n, κ, d)` to be learnable:

1. **Injectivity:** Different parameters must produce different spectra
   - One-to-one mapping
   - Tested by uniqueness test

2. **Sensitivity:** Small parameter changes must produce detectable spectrum changes
   - Information is present in the signal
   - Tested by sensitivity analysis

3. **Stability:** Small noise shouldn't completely change the answer
   - Continuous inverse mapping
   - Tested by noise sensitivity

4. **Sufficiency:** The spectrum must contain enough information
   - Dimensionality of spectrum ≥ dimensionality of parameters
   - Tested by PCA

### The ML Perspective

Neural networks can learn the inverse mapping if:

1. **Deterministic:** Same input always gives same output
   - No randomness in the forward model ✓

2. **Continuous:** Nearby spectra come from nearby parameters
   - Smooth parameter space ✓

3. **Sufficient Data:** Enough examples to cover the space
   - Need 10k+ samples for well-posed problems
   - Need 100k+ for ill-posed problems

4. **Generalizable:** Patterns exist that transfer to unseen data
   - Not just memorization

---

## Real-World Considerations

### Experimental Noise Types

1. **Measurement Noise:** Random detector noise
   - Gaussian, ~1-5% typical
   - Addressed by SNR test

2. **Systematic Errors:** Water vapor absorption, alignment drift
   - Can bias results
   - Need careful calibration

3. **Model Mismatch:** Real samples aren't perfect single layers
   - Surface roughness, inhomogeneity
   - May need modified physics model

### Domain Knowledge

Even if tests show "learnable," consider:

1. **Physical Constraints:**
   - Is n > 1? (must be for passive materials)
   - Is κ < 0? (absorption, not gain)
   - Reasonable thickness range?

2. **Prior Knowledge:**
   - Known material type?
   - Expected parameter ranges?
   - Can dramatically improve results

3. **Measurement Conditions:**
   - Frequency range matters
   - Thicker samples → more oscillations → easier to extract d
   - Higher absorption → weaker signal → harder to extract κ

---

## Examples from Literature

### Well-Posed THz Problems ✅
- Thin polymer films (d = 10-100 μm, low absorption)
- Crystalline semiconductors (known dispersion relations)
- Liquids in calibrated cells (known thickness)

### Challenging THz Problems ⚠️
- Thick, highly absorbing materials (d > 1mm, high κ)
- Multi-layer structures (3+ layers)
- Anisotropic materials (tensor n)

### Ill-Posed THz Problems ❌
- Very thin films (d < 10 μm) - low sensitivity
- Highly dispersive materials - parameter correlation
- Materials with unknown structure - underdetermined

---

## Next Steps After Testing

### If Learnable ✅

1. Generate large dataset
   ```bash
   python src/generate_dataset.py --samples 50000
   ```

2. Train baseline model
   ```bash
   python src/train.py --network cnn --epochs 100
   ```

3. Evaluate and iterate
   ```bash
   python src/evaluate.py path/to/best_model.pt
   ```

### If Challenging ⚠️

1. Try physics-informed loss first
2. Analyze failure modes
3. Consider restricted parameter ranges
4. Add domain-specific constraints

### If Ill-Posed ❌

1. Consult the learnability tests to identify which specific issue(s)
2. Consider problem reformulation
3. Combine with complementary measurements
4. Use traditional optimization methods instead

---

## Additional Resources

- **Theory:** Hadamard's well-posedness conditions
- **Practice:** Inverse problems textbooks (e.g., Vogel, Tarantola)
- **THz-specific:** Jepsen et al., "Terahertz spectroscopy and imaging" (2011)

---

## Summary

**The learnability analysis answers:**
> "Can neural networks theoretically solve this problem, or are there fundamental mathematical/physical barriers?"

**It does NOT answer:**
- What architecture is best? (need experiments)
- How much data is needed? (rule of thumb: 10k-100k)
- What hyperparameters to use? (need tuning)

**Run the tests BEFORE spending weeks generating data and training models!**

5 tests × 2 minutes each = 10 minutes to save potentially weeks of wasted effort.
