# Script 01: Generate Dataset

**File:** `generate_dataset.py`

## Quick Reference

| Attribute | Value |
|-----------|-------|
| **Purpose** | Generate synthetic frequency dataset with realistic noise |
| **Input** | None (parameters hardcoded) |
| **Output** | 4 files in `data/` directory |
| **Runtime** | ~5 seconds |
| **Dependencies** | numpy, pandas, pathlib |

## Usage

```bash
python generate_dataset.py
```

No command-line arguments required. All parameters are configured in the script.

## What This Script Does

### Overview

Generates synthetic signal data with 4 phase-shifted sinusoidal frequencies, mixed together with additive Gaussian noise. Creates two separate datasets:
- **Training set** (Seed #1): One noise realization
- **Test set** (Seed #2): Completely different noise realization

This separation ensures the model truly learns to extract frequencies rather than memorizing noise patterns.

### Step-by-Step Process

#### 1. **Configuration** (Lines 20-36)

Defines the 4 frequencies and their fixed phase offsets:

```python
freq1 = 1.0   # 1 Hz, phase = 0° (0 rad)
freq2 = 3.0   # 3 Hz, phase = 45° (π/4 rad)
freq3 = 5.0   # 5 Hz, phase = 90° (π/2 rad)
freq4 = 7.0   # 7 Hz, phase = 135° (3π/4 rad)

noise_std = 0.1  # Gaussian noise standard deviation
```

**Why these values?**
- **Frequencies** well-separated in frequency domain (easy to visualize in FFT)
- **Phase offsets** create realistic multi-frequency signal interference
- **Noise level** (σ=0.1) provides moderate challenge (~11 dB SNR)

#### 2. **Time Domain Setup** (Lines 47-51)

```python
n_samples = 10000
x_start = 0.0
x_end = 10.0  # 10 seconds
sampling_rate = 1000  # 1000 Hz
x_values = np.linspace(x_start, x_end, n_samples)
```

Creates time axis:
- **10,000 samples** over **10 seconds**
- **Sampling rate**: 1000 Hz (sufficient for 7 Hz max frequency)
- Nyquist frequency = 500 Hz >> 7 Hz ✓

#### 3. **Dataset Generation Function** (Lines 58-120)

The `generate_dataset(seed, dataset_name)` function:

**a) Sets random seed for reproducibility**
```python
np.random.seed(seed)
```

**b) Generates clean sinusoids with FIXED phases**
```python
for i, (freq, phase) in enumerate(zip(frequencies, phases)):
    clean_signals[:, i] = np.sin(2 * np.pi * freq * x_values + phase)
```

Mathematical form:
\[ f_i(t) = \sin(2\pi \cdot f_i \cdot t + \theta_i) \]

**c) Creates pure target signals**
```python
target1 = clean_signals[:, 0]
target2 = clean_signals[:, 1]
target3 = clean_signals[:, 2]
target4 = clean_signals[:, 3]
```

These are the **ground truth** the model will try to predict (no noise added).

**d) Combines signals (averaged)**
```python
S_clean = np.mean(clean_signals, axis=1)  # Equivalent to (1/4) * Σ
```

\[ S_{clean}(t) = \frac{1}{4} \sum_{i=1}^{4} f_i(t) \]

**e) Adds Gaussian noise**
```python
noise = np.random.normal(0, noise_std, n_samples)
S_noisy = S_clean + noise
```

\[ S_{noisy}(t) = S_{clean}(t) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2) \]

where σ = 0.1

**f) Calculates SNR**
```python
SNR = 10 * np.log10(S_clean.var() / noise.var())
```

Signal-to-Noise Ratio in decibels: ~11 dB

#### 4. **Generate Both Datasets** (Lines 122-126)

```python
train_data = generate_dataset(seed=1, dataset_name="TRAINING SET")
test_data = generate_dataset(seed=2, dataset_name="TEST SET")
```

**Critical:** Same frequencies and phases, but **different noise realizations**. This tests true generalization.

#### 5. **Save Files** (Lines 128-201)

Saves 4 files:

**a) Training dataset (NPZ)**
```python
np.savez('data/frequency_data_train.npz',
         x=train_data['x'],
         S=train_data['S_noisy'],
         # ... and more arrays
         seed=1)
```

**b) Test dataset (NPZ)**
```python
np.savez('data/frequency_data_test.npz',
         # ... similar structure
         seed=2)
```

**c) Training CSV (human-readable)**
```python
dataset_train.to_csv('data/frequency_dataset_train.csv', index=False)
```

**d) Compatibility file (for visualization script)**
```python
np.savez('data/frequency_data.npz',
         x=train_data['x'],
         f1=train_data['target1'],
         # ... backward compatibility
```

#### 6. **Print Statistics** (Lines 203-232)

Displays comprehensive statistics about generated datasets.

## Parameters & Configuration

### Modifiable Parameters

All parameters are hardcoded in the script. To modify:

#### Frequencies

```python
# Lines 20-24
freq1 = 1.0   # Change to your desired frequency
freq2 = 3.0   # Must be positive real numbers
freq3 = 5.0   # Recommend < Nyquist/2 = 250 Hz
freq4 = 7.0   # For current sampling rate
```

**Constraint:** All frequencies must be less than Nyquist frequency (sampling_rate / 2 = 500 Hz)

#### Phase Offsets

```python
# Lines 27-32
phase1 = 0.0           # 0° - Change in radians
phase2 = np.pi / 4     # 45°
phase3 = np.pi / 2     # 90°
phase4 = 3 * np.pi / 4 # 135°
```

**Range:** [0, 2π) radians or [0°, 360°)

#### Noise Level

```python
# Line 35
noise_std = 0.1  # Standard deviation of Gaussian noise
```

**Effect:**
- Lower (0.05): Easier task, higher SNR (~17 dB)
- Current (0.1): Moderate, ~11 dB SNR
- Higher (0.2): Harder task, lower SNR (~5 dB)

#### Time Domain

```python
# Lines 47-50
n_samples = 10000    # Number of data points
x_start = 0.0        # Start time (seconds)
x_end = 10.0         # End time (seconds)
sampling_rate = 1000 # Samples per second
```

**Relationships:**
- `sampling_rate = n_samples / (x_end - x_start)`
- Nyquist frequency = `sampling_rate / 2`
- Time resolution = `1 / sampling_rate`

### Non-Modifiable Design Choices

**Q: Why fixed phases instead of random phases per sample?**

A: Initial approach used random phases at every sample, which destroyed frequency structure:
- Random phases → R² = -0.45 (worse than predicting mean!)
- Fixed phases + additive noise → R² = 0.35 (+178% improvement!)

Fixed phases preserve the learnable frequency structure while noise adds realistic variability.

**Q: Why additive Gaussian noise?**

A: 
- Preserves frequency content (just adds random component)
- Models realistic measurement noise
- Allows model to learn signal structure despite noise
- Different noise realizations test generalization

**Q: Why two datasets with different seeds?**

A:
- Prevents model from memorizing specific noise patterns
- Tests true generalization to unseen noise
- More realistic evaluation (real-world noise varies)

## Output Files

### 1. `data/frequency_data_train.npz`

**Format:** NumPy compressed archive

**Size:** ~800 KB

**Contents:**
- `x`: Time values (10,000 samples)
- `S`: Noisy mixed signal
- `S_clean`: Clean mixed signal (no noise)
- `noise`: The noise that was added
- `target1`, `target2`, `target3`, `target4`: Pure frequency components
- `clean1`, `clean2`, `clean3`, `clean4`: Same as targets
- `frequencies`: Array [1.0, 3.0, 5.0, 7.0]
- `phases`: Array of phase offsets in radians
- `noise_std`: 0.1
- `sampling_rate`: 1000
- `seed`: 1

**Usage:**
```python
data = np.load('data/frequency_data_train.npz')
x = data['x']
S = data['S']
target1 = data['target1']
```

### 2. `data/frequency_data_test.npz`

**Format:** NumPy compressed archive

**Size:** ~800 KB

**Contents:** Identical structure to training file, but:
- Different noise realization (seed=2)
- Same frequencies, phases, time axis
- Used for final evaluation

### 3. `data/frequency_dataset_train.csv`

**Format:** CSV (comma-separated values)

**Size:** ~1.5 MB

**Structure:**
```csv
Sample,Time(s),S_noisy(t),Target1(t),Target2(t),Target3(t),Target4(t)
0,0.0000,0.6035,0.0000,0.7071,1.0000,0.7071
1,0.0010,0.5987,0.0063,0.7088,0.9999,0.7054
...
```

**Columns:**
- `Sample`: Row index (0-9999)
- `Time(s)`: Time in seconds
- `S_noisy(t)`: Mixed signal with noise
- `Target1(t)` through `Target4(t)`: Individual pure frequencies

**Usage:** Human-readable, can open in Excel/spreadsheet

### 4. `data/frequency_data.npz`

**Format:** NumPy compressed archive

**Size:** ~400 KB

**Purpose:** Compatibility with visualization script

**Contents:**
- `x`: Time values
- `f1`, `f2`, `f3`, `f4`: Frequency components (from training data)
- `S`: Noisy mixed signal (from training data)
- `frequencies`: [1.0, 3.0, 5.0, 7.0]
- `phases`: [0.0, 0.0, 0.0, 0.0] (backward compatibility, not actual phases)

## Technical Details

### Mathematical Foundation

#### Signal Generation

Each frequency component is generated as:

\[ f_i(t) = \sin(2\pi f_i t + \theta_i) \]

Where:
- \( f_i \): Frequency in Hz (1, 3, 5, or 7)
- \( t \): Time in seconds [0, 10]
- \( \theta_i \): Phase offset in radians (0, π/4, π/2, 3π/4)

#### Mixed Signal (Clean)

\[ S_{clean}(t) = \frac{1}{4} \sum_{i=1}^{4} f_i(t) = \frac{1}{4} \sum_{i=1}^{4} \sin(2\pi f_i t + \theta_i) \]

Averaging (dividing by 4) keeps amplitude in reasonable range.

#### Noisy Mixed Signal

\[ S_{noisy}(t) = S_{clean}(t) + \varepsilon_t \]

where \( \varepsilon_t \sim \mathcal{N}(0, \sigma^2) \), σ = 0.1

#### Signal-to-Noise Ratio

\[ \text{SNR} = 10 \log_{10} \left( \frac{\text{Var}(S_{clean})}{\text{Var}(\varepsilon)} \right) \text{ dB} \]

With current parameters: SNR ≈ 11 dB

### Implementation Details

#### Why linspace?

```python
x_values = np.linspace(x_start, x_end, n_samples)
```

`linspace` creates evenly spaced points including both endpoints:
- More intuitive than `arange` for time series
- Exactly `n_samples` points
- No floating-point accumulation errors

#### Why np.mean vs explicit division?

```python
S_clean = np.mean(clean_signals, axis=1)  # vs. np.sum(...) / 4
```

- More readable and intention-clear
- Handles variable number of frequencies if modified
- Numerically equivalent for this case

#### Noise Generation

```python
noise = np.random.normal(0, noise_std, n_samples)
```

- `loc=0`: Mean of 0 (zero-mean noise)
- `scale=noise_std`: Standard deviation
- `size=n_samples`: One noise value per time point

### Memory Considerations

**Arrays created:**
- `x_values`: 10,000 × 8 bytes = 78 KB
- `clean_signals`: 10,000 × 4 × 8 bytes = 312 KB
- Total working memory: ~1-2 MB

**Output files:**
- NPZ compression reduces size by ~50%
- CSV is larger (text format) but human-readable

### Computational Complexity

- **Time complexity:** O(n × m) where n=samples, m=frequencies
  - For 10,000 samples, 4 frequencies: 40,000 sine evaluations
- **Space complexity:** O(n × m)
- **Runtime:** ~5 seconds (dominated by I/O, not computation)

## Code Walkthrough

### Key Functions

#### `generate_dataset(seed, dataset_name)`

**Purpose:** Generate one complete dataset (train or test)

**Parameters:**
- `seed` (int): Random seed for noise generation
- `dataset_name` (str): Display name for console output

**Returns:** Dictionary with keys:
- `x`: Time axis
- `S_noisy`: Mixed signal with noise
- `S_clean`: Mixed signal without noise
- `noise`: The noise array
- `target1` through `target4`: Pure frequency targets
- `clean1` through `clean4`: Same as targets

**Process:**
1. Set random seed
2. Generate clean sinusoids with fixed phases
3. Create targets (no noise)
4. Mix signals (average)
5. Add Gaussian noise
6. Calculate statistics
7. Return dictionary

### Important Variables

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `frequencies` | ndarray | (4,) | Hz values [1, 3, 5, 7] |
| `phases` | ndarray | (4,) | Phase offsets in radians |
| `x_values` | ndarray | (10000,) | Time axis in seconds |
| `clean_signals` | ndarray | (10000, 4) | All 4 clean frequencies |
| `S_clean` | ndarray | (10000,) | Clean mixed signal |
| `S_noisy` | ndarray | (10000,) | Noisy mixed signal |
| `noise` | ndarray | (10000,) | Gaussian noise added |
| `train_data` | dict | - | Complete training dataset |
| `test_data` | dict | - | Complete test dataset |

## Troubleshooting

### Issue: "MemoryError"

**Cause:** Not enough RAM for arrays

**Solutions:**
1. Reduce `n_samples` to 5000
2. Close other applications
3. Use a machine with more RAM

### Issue: "Import Error: No module named 'numpy'"

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install numpy pandas matplotlib
```

### Issue: "Permission denied: data/"

**Cause:** No write access to directory

**Solutions:**
1. Run from project root directory
2. Check folder permissions
3. Create `data/` folder manually: `mkdir data`

### Issue: Different SNR values

**Cause:** Random noise varies

**Expected:** SNR should be around 10-12 dB
- Training set (Seed #1): ~10.9 dB
- Test set (Seed #2): ~10.9 dB (similar, not identical)

**Action:** Values within ±1 dB are normal

### Issue: Files not created

**Cause:** Script error before save

**Solutions:**
1. Check console for error messages
2. Verify `data/` folder exists
3. Check disk space: `df -h`
4. Run script with Python directly: `python generate_dataset.py`

### Issue: Want to modify frequencies

**Process:**
1. Edit lines 20-24 (frequency values)
2. Optionally edit lines 27-32 (phase offsets)
3. Rerun script: `python generate_dataset.py`
4. **Important:** Rerun ALL subsequent scripts (pipeline depends on this data)

## Performance Considerations

### Optimization Opportunities

**Current implementation is already efficient:**
- Vectorized NumPy operations
- No unnecessary loops
- Efficient file I/O

**If you needed to scale up:**

1. **More samples:** Increase `n_samples`
   - Linear time/space increase
   - 100,000 samples → ~50 seconds

2. **More frequencies:** Add more frequency components
   - Requires code changes (not just parameter tweaks)
   - Linear increase in computation

3. **Parallel generation:** Not needed (already fast)

### Memory Usage

**Peak memory:** ~10 MB (negligible)

**Output files:** ~3 MB total

**Bottleneck:** Disk I/O, not computation

## Extending the Script

### Add More Frequencies

```python
# Add frequency 5
freq5 = 10.0  # 10 Hz
frequencies = np.array([freq1, freq2, freq3, freq4, freq5])

# Add phase
phase5 = np.pi  # 180°
phases = np.array([phase1, phase2, phase3, phase4, phase5])

# Update clean_signals array size
clean_signals = np.zeros((n_samples, 5))  # Changed from 4 to 5
```

**Also update:**
- Target generation (add `target5`)
- NPZ save statements
- CSV column names

### Change Noise Type

**Current:** Additive Gaussian

**Alternative:** Multiplicative noise
```python
# Instead of: S_noisy = S_clean + noise
S_noisy = S_clean * (1 + noise)  # Multiplicative
```

**Alternative:** Uniform noise
```python
noise = np.random.uniform(-0.1, 0.1, n_samples)  # Uniform instead of Gaussian
```

### Variable Phase per Frequency

**Current:** Fixed phases

**Alternative:** Random phases (but not per-sample!)
```python
np.random.seed(seed)
phases = np.random.uniform(0, 2*np.pi, 4)  # Random but fixed per dataset
```

## Related Documentation

- **Next step:** `Script_02_VisualizeData.md` - Visualize generated data
- **Theory:** `MathematicalFoundations.md` - Signal processing background
- **Design:** `ArchitectureDecisions.md` - Why this approach
- **Feature:** `Feature_DataGeneration.md` - Complete data generation pipeline

## Summary

This script generates the foundation of the entire project:
- ✓ 10,000 samples of 4 phase-shifted frequencies
- ✓ Moderate Gaussian noise (SNR ≈ 11 dB)
- ✓ Separate train/test sets with different noise
- ✓ Multiple file formats (NPZ, CSV)
- ✓ Well-documented, reproducible process

**Key achievement:** Creates learnable task where LSTM can extract frequencies despite overlapping signals and noise.

