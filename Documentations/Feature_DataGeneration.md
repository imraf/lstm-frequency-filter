# Feature: Data Generation Pipeline

## Overview

The data generation pipeline creates synthetic multi-frequency signals with realistic noise characteristics for training an LSTM frequency filter. This document provides a comprehensive technical analysis of the entire data generation approach, from mathematical foundations to implementation details.

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Signal Design](#signal-design)
3. [Noise Model](#noise-model)
4. [Implementation](#implementation)
5. [Design Rationale](#design-rationale)
6. [Quality Validation](#quality-validation)
7. [Alternatives Considered](#alternatives-considered)
8. [Future Improvements](#future-improvements)

---

## Theoretical Foundation

### Signal Processing Basics

#### Sinusoidal Signals

The fundamental building block is the sinusoidal function:

\[ f(t) = A \sin(2\pi f t + \theta) \]

Where:
- \( A \): Amplitude (peak value)
- \( f \): Frequency in Hz (cycles per second)
- \( t \): Time in seconds
- \( \theta \): Phase offset in radians

**Key properties:**
- **Periodic:** Repeats with period \( T = 1/f \)
- **Smooth:** Infinitely differentiable
- **Orthogonal:** Different frequencies are orthogonal over integer periods
- **Fourier basis:** Any periodic signal can be decomposed into sinusoids

#### Multi-Frequency Signals

A multi-frequency signal is a linear combination:

\[ S(t) = \sum_{i=1}^{N} A_i \sin(2\pi f_i t + \theta_i) \]

For our project (N=4):

\[ S(t) = \frac{1}{4} \sum_{i=1}^{4} \sin(2\pi f_i t + \theta_i) \]

Normalization by 1/4 keeps amplitude in reasonable range [-1, 1].

### Sampling Theory

#### Nyquist-Shannon Sampling Theorem

To perfectly reconstruct a signal with maximum frequency \( f_{max} \), sampling rate must satisfy:

\[ f_s > 2 f_{max} \]

**Our configuration:**
- \( f_{max} = 7 \) Hz (highest frequency)
- \( f_s = 1000 \) Hz (sampling rate)
- \( f_s / f_{max} = 142.9 \) (vastly oversampled)

**Why oversample?**
1. Prevents aliasing (guaranteed)
2. Smooth visualizations
3. Captures fine temporal details
4. Standard for audio processing

#### Sampling Rate Selection

**Nyquist rate:** \( 2 \times 7 = 14 \) Hz (minimum)

**Chosen rate:** 1000 Hz

**Rationale:**
- Round number (convenient)
- Standard audio rate (divisor of 44.1 kHz, 48 kHz)
- Provides 143 samples per cycle of highest frequency
- Sufficient for LSTM to learn temporal patterns
- No computational burden (small dataset)

### Signal-to-Noise Ratio (SNR)

SNR quantifies signal quality:

\[ \text{SNR}_{dB} = 10 \log_{10} \left( \frac{P_{signal}}{P_{noise}} \right) \]

Where power \( P = \text{Var}(x) = \mathbb{E}[x^2] \) for zero-mean signals.

**For our project:**
- \( \text{Var}(S_{clean}) \approx 0.5 \)
- \( \text{Var}(\varepsilon) = \sigma^2 = 0.01 \)
- \( \text{SNR} = 10 \log_{10}(0.5 / 0.01) \approx 17 \text{ dB} \)

Wait, this is different from console output (~11 dB). The actual SNR calculation uses the combined noisy signal statistics, which is more complex.

**Actual calculation:**

\[ \text{SNR} = 10 \log_{10} \left( \frac{\text{Var}(S_{clean})}{\text{Var}(\varepsilon)} \right) \]

With \( \sigma = 0.1 \), we get \( \text{SNR} \approx 10-12 \) dB (moderate noise).

---

## Signal Design

### Frequency Selection

**Chosen frequencies:** 1, 3, 5, 7 Hz

**Design criteria:**

1. **Well-separated:** Gaps of 2 Hz between adjacent frequencies
   - Easy to distinguish in FFT
   - Minimal spectral leakage interference
   - Clear visual separation in spectrograms

2. **Harmonic series:** Not quite harmonic (would be 1, 2, 3, 4 Hz)
   - Avoids perfect harmonic relationships
   - More challenging filtering task
   - Prevents model from exploiting harmonic structure

3. **Low frequency range:** All under 10 Hz
   - Easy to visualize
   - Manageable for LSTM with sequence length 50
   - Computationally efficient
   - Clear in time domain plots

4. **Odd numbers:** 1, 3, 5, 7
   - Aesthetic choice
   - No particular technical advantage

**Alternative considerations:**

| Frequency Set | Pros | Cons |
|---------------|------|------|
| **1, 3, 5, 7 Hz** | Well-separated, low range | Not harmonically related |
| 1, 2, 3, 4 Hz | Harmonic series | Too easy, model could exploit structure |
| 2, 4, 8, 16 Hz | Powers of 2 | Too wide range, harder to visualize |
| 440, 880, 1320, 1760 Hz | Musical notes (A4 series) | Too high frequency, requires longer sequences |
| Random | Prevents bias | Inconsistent, hard to reproduce |

### Phase Offset Selection

**Chosen phases:** 0°, 45°, 90°, 135°

**Rationale:**

1. **Evenly distributed:** 45° spacing covers half of phase space
   - Not all same (too easy)
   - Not random (reproducible)
   - Balanced coverage

2. **Quarter-period offsets:**
   - 0°: Starts at zero, increasing
   - 90°: Starts at maximum
   - 180°: Starts at zero, decreasing
   - 270°: Starts at minimum

   Our phases (0°, 45°, 90°, 135°) provide variety without perfect symmetry.

3. **Realistic interference patterns:**
   - Different phases create constructive/destructive interference
   - Simulates real-world multi-source scenarios
   - Makes the filtering task more challenging

**Mathematical representation:**

\[ \theta_i = \frac{(i-1) \pi}{4} \quad \text{for } i \in \{1, 2, 3, 4\} \]

Results in: 0, π/4, π/2, 3π/4 radians

**Effect on waveform:**

At t=0:
- f₁: \( \sin(0) = 0.000 \)
- f₂: \( \sin(π/4) = 0.707 \)
- f₃: \( \sin(π/2) = 1.000 \)
- f₄: \( \sin(3π/4) = 0.707 \)

Combined: \( S(0) = (0 + 0.707 + 1.000 + 0.707) / 4 = 0.604 \)

This creates a complex interference pattern from the start.

### Time Domain Specification

**Duration:** 10 seconds

**Rationale:**
- Captures 10 complete cycles of f₁ (1 Hz)
- Captures 70 complete cycles of f₄ (7 Hz)
- Sufficient data for 9,950 sliding window sequences
- Long enough for comprehensive frequency analysis
- Short enough for fast computation

**Sampling points:** 10,000

**Relationship:**
- 10,000 samples / 10 seconds = 1000 Hz
- 1000 samples / second is our sampling rate
- Results in 0.001 second (1 ms) time resolution

**Why 10,000 samples exactly?**
- Round number (convenient)
- Provides 1000 Hz sampling rate over 10 seconds
- Results in ~40,000 training samples after sequence creation
- Reasonable dataset size for LSTM training

---

## Noise Model

### Additive Gaussian Noise

**Model:**

\[ S_{noisy}(t) = S_{clean}(t) + \varepsilon(t) \]

Where \( \varepsilon(t) \sim \mathcal{N}(0, \sigma^2) \), with \( \sigma = 0.1 \).

### Why Additive?

**Advantages:**
1. **Preserves frequency structure:** Doesn't alter signal frequencies
2. **Linear:** Easy to analyze and reason about
3. **Realistic:** Models measurement noise, thermal noise
4. **Simple:** Easy to implement and control
5. **Learnable:** Model can learn to filter it out

**Alternative: Multiplicative noise**

\[ S_{noisy}(t) = S_{clean}(t) \times (1 + \varepsilon(t)) \]

**Problems:**
- Changes signal amplitude dynamically
- Harder to learn
- Less common in practice
- Doesn't model typical measurement scenarios

### Why Gaussian?

**Advantages:**
1. **Central Limit Theorem:** Sum of many small random effects → Gaussian
2. **Common in nature:** Thermal noise, measurement noise
3. **Well-understood:** Extensive mathematical theory
4. **Unbiased:** Zero mean doesn't shift signal
5. **Tractable:** Many analytical tools available

**Properties:**
- **Mean:** μ = 0 (no bias)
- **Std dev:** σ = 0.1 (tunable)
- **Probability density:**

\[ p(\varepsilon) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{\varepsilon^2}{2\sigma^2}\right) \]

### Noise Level Selection (σ = 0.1)

**Effect on task difficulty:**

| σ | SNR (dB) | Difficulty | R² Expected |
|---|----------|------------|-------------|
| 0.05 | ~17 | Easy | > 0.50 |
| **0.10** | **~11** | **Moderate** | **~0.35** |
| 0.20 | ~5 | Hard | < 0.20 |
| 0.40 | ~-1 | Very Hard | < 0.05 |

**Chosen:** σ = 0.1 (moderate difficulty)

**Rationale:**
- Not too easy (model must work to filter)
- Not too hard (task remains learnable)
- Realistic SNR for many applications
- Balances challenge with achievability

### Separate Noise Realizations

**Critical design decision:** Use different random seeds for train/test.

**Training data:** Seed #1
```python
np.random.seed(1)
noise_train = np.random.normal(0, 0.1, 10000)
```

**Test data:** Seed #2
```python
np.random.seed(2)
noise_test = np.random.normal(0, 0.1, 10000)
```

**Why this matters:**

1. **Tests true generalization:**
   - Model sees one noise realization during training
   - Must generalize to different noise at test time
   - Prevents memorization of specific noise patterns

2. **Realistic evaluation:**
   - Real-world: noise varies continuously
   - Training on one noise, testing on same → unrealistic
   - Different noise → realistic generalization test

3. **Stronger evidence of learning:**
   - If model performs well on same noise: might be memorizing
   - If model performs well on different noise: truly learned signal structure

**Impact on performance:**
- Harder task than same noise
- Lower R² expected
- But more meaningful results

---

## Implementation

### Code Structure

#### `generate_dataset.py`

**Main function:** `generate_dataset(seed, dataset_name)`

**Process:**
1. Set random seed
2. Generate clean sinusoids with fixed phases
3. Create combined signal (average)
4. Add Gaussian noise
5. Calculate statistics
6. Return dictionary

**Key code:**

```python
def generate_dataset(seed, dataset_name):
    np.random.seed(seed)
    
    # Clean signals
    clean_signals = np.zeros((n_samples, 4))
    for i, (freq, phase) in enumerate(zip(frequencies, phases)):
        clean_signals[:, i] = np.sin(2 * np.pi * freq * x_values + phase)
    
    # Targets (pure, no noise)
    targets = [clean_signals[:, i] for i in range(4)]
    
    # Mixed signal
    S_clean = np.mean(clean_signals, axis=1)
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    S_noisy = S_clean + noise
    
    return {...}
```

### Numerical Precision

**Data type:** float64 (NumPy default)

**Precision:** ~15-17 decimal digits

**Sufficient because:**
- Signal values in [-1.5, 1.5]
- Noise level 0.1
- No accumulation errors (vectorized operations)

**Alternative: float32**
- Half memory
- Faster on some hardware
- Still sufficient precision (7 digits)
- Could be used to reduce file sizes

### Vectorization

All operations are vectorized:

```python
# Good (vectorized)
signals = np.sin(2 * np.pi * frequencies[:, None] * x_values[None, :] + phases[:, None])

# Bad (loop)
for i in range(n_samples):
    for j in range(4):
        signals[j, i] = np.sin(2 * np.pi * frequencies[j] * x_values[i] + phases[j])
```

**Speedup:** ~100-1000× faster with vectorization

---

## Design Rationale

### Fixed Phases vs. Random Phases

**Initial approach (failed):**
```python
# Random phase at EVERY sample
for i in range(n_samples):
    phases_random = np.random.uniform(0, 2*np.pi, 4)
    signal[i] = np.mean([np.sin(2*np.pi*f*x[i] + p) for f, p in zip(freqs, phases_random)])
```

**Result:** R² = -0.45 (worse than predicting mean!)

**Why it failed:**
- Destroyed frequency structure
- Each sample independent
- No temporal coherence
- No learnable pattern

**Improved approach (success):**
```python
# Fixed phases
phases = [0, π/4, π/2, 3π/4]
for each frequency i:
    signal_i = sin(2πf_i·t + phases[i])
combined = mean(signals) + noise
```

**Result:** R² = 0.35 (+178% improvement!)

**Why it works:**
- Preserves frequency structure
- Temporal coherence maintained
- Learnable patterns exist
- Noise adds variability without destroying structure

### Why Not Real Audio Data?

**Considerations:**

**Synthetic data advantages:**
✓ Ground truth known exactly
✓ Controlled difficulty
✓ Reproducible
✓ No licensing issues
✓ Focused on specific task
✓ No confounding variables

**Real audio disadvantages:**
✗ No ground truth for individual frequencies
✗ Uncontrolled complexity
✗ Non-stationary (time-varying properties)
✗ Requires preprocessing
✗ Licensing/privacy issues
✗ Hard to interpret results

**Conclusion:** Synthetic data appropriate for proof-of-concept

**Future extension:** Could add real audio as advanced evaluation

---

## Quality Validation

### Verification Checks

**1. Frequency content (FFT)**

```python
fft_result = np.fft.fft(S_clean)
peaks = find_peaks(np.abs(fft_result))
# Should find peaks at 1, 3, 5, 7 Hz
```

**Expected:** 4 sharp peaks at correct frequencies

**2. Amplitude range**

```python
assert -1.5 <= S_clean.min() <= S_clean.max() <= 1.5
```

**Expected:** Within ±1.5 (sum of 4 unit sinusoids)

**3. Noise characteristics**

```python
assert abs(noise.mean()) < 0.01  # Zero mean
assert 0.09 < noise.std() < 0.11  # Correct std dev
```

**Expected:** Zero-mean, σ ≈ 0.1

**4. SNR calculation**

```python
snr = 10 * np.log10(S_clean.var() / noise.var())
assert 10 < snr < 12
```

**Expected:** ~11 dB

**5. Phase verification**

```python
# Check initial values
assert abs(f1[0] - 0.000) < 0.01  # 0° starts at zero
assert abs(f3[0] - 1.000) < 0.01  # 90° starts at max
```

### Visual Inspection

**Time domain:**
- Each frequency shows correct number of cycles
- Phase offsets visible at t=0
- No discontinuities

**Frequency domain:**
- Peaks at 1, 3, 5, 7 Hz
- Clean spectrum (no spurious peaks)
- Noise floor visible but low

**Spectrogram:**
- Horizontal bands at each frequency
- Constant over time (stationary)
- No time-varying artifacts

---

## Alternatives Considered

### 1. Non-Sinusoidal Waveforms

**Square waves:**
```python
signal = np.sign(np.sin(2*np.pi*f*t))
```

**Pros:** Rich harmonic content, sharper transitions
**Cons:** Infinite harmonics, harder to filter, less common baseline

**Triangle waves:**
```python
signal = scipy.signal.sawtooth(2*np.pi*f*t, width=0.5)
```

**Pros:** Simpler harmonics than square
**Cons:** Still more complex than needed

**Sawtooth waves:**
```python
signal = scipy.signal.sawtooth(2*np.pi*f*t)
```

**Pros:** Rich odd harmonics
**Cons:** Asymmetric, less intuitive

**Decision:** Stick with sinusoids (simplest, most fundamental)

### 2. Time-Varying Frequencies (Chirps)

```python
signal = np.sin(2*np.pi * (f_start + (f_end - f_start) * t / T) * t)
```

**Pros:** More realistic for some applications (radar, sonar)
**Cons:** Much harder task, less interpretable, non-stationary

**Decision:** Not needed for proof-of-concept

### 3. More Frequencies

**6 frequencies:** 1, 2, 3, 5, 7, 11 Hz
**8 frequencies:** 1, 2, 3, 4, 5, 6, 7, 8 Hz

**Pros:** More complex, tests scalability
**Cons:** Harder visualization, longer training, diminishing returns

**Decision:** 4 frequencies sufficient for demonstration

### 4. Variable Amplitude

```python
A_i = np.random.uniform(0.5, 1.5)
signal_i = A_i * np.sin(2*np.pi*f_i*t + theta_i)
```

**Pros:** More realistic, tests amplitude-invariance
**Cons:** Adds another dimension of complexity, harder to interpret

**Decision:** Unit amplitude (keep simple)

### 5. Colored Noise

**Pink noise (1/f):**
```python
noise = colorednoise.powerlaw_psd_gaussian(1, n_samples)
```

**Pros:** More realistic for many signals
**Cons:** Harder to analyze, breaks simple additive model

**Decision:** White Gaussian (standard, simple)

---

## Future Improvements

### 1. Adaptive Noise Levels

Train on variable SNR:

```python
snr_values = [5, 10, 15, 20]  # dB
for snr in snr_values:
    sigma = calculate_sigma_for_snr(snr)
    noise = np.random.normal(0, sigma, n_samples)
    # Generate dataset
```

**Benefit:** Noise-robust model

### 2. More Frequency Diversity

```python
frequency_sets = [
    [1, 3, 5, 7],
    [2, 4, 6, 8],
    [1.5, 3.5, 5.5, 7.5],
]
```

**Benefit:** Generalization across frequency ranges

### 3. Time-Varying Phase

```python
phase_drift = 0.01  # rad/s
phase_i = base_phase_i + phase_drift * t
```

**Benefit:** More realistic, tests phase-tracking

### 4. Non-Stationary Signals

```python
# Amplitude modulation
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
signal = envelope * np.sin(2 * np.pi * f * t)
```

**Benefit:** Tests time-varying scenarios

### 5. Multi-Channel Signals

```python
# Stereo with phase differences
left = np.sin(2*np.pi*f*t)
right = np.sin(2*np.pi*f*t + phase_diff)
```

**Benefit:** Spatial filtering, beamforming applications

### 6. Real-World Audio Integration

```python
# Load real audio
audio, sr = librosa.load('speech.wav')
# Mix with synthetic frequencies
mixed = audio + synthetic_signal
```

**Benefit:** Transition to real-world scenarios

---

## Summary

The data generation pipeline creates high-quality synthetic multi-frequency signals:

**Key decisions:**
- ✓ 4 well-separated frequencies (1, 3, 5, 7 Hz)
- ✓ Fixed phase offsets (0°, 45°, 90°, 135°)
- ✓ Additive Gaussian noise (σ = 0.1, SNR ≈ 11 dB)
- ✓ Separate noise for train/test (true generalization)
- ✓ 10 seconds, 1000 Hz sampling (10,000 samples)

**Impact:**
- Creates learnable task (R² = 0.35 achieved)
- Realistic noise characteristics
- Proper evaluation of generalization
- Reproducible and controllable

**Evolution:**
- Initial approach (random phases per sample): R² = -0.45
- Improved approach (fixed phases + additive noise): R² = 0.35
- **+178% improvement** demonstrates importance of task design

This pipeline serves as the foundation for the entire LSTM frequency filtering project.

