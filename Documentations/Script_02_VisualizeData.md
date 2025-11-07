# Script 02: Visualize Data

**File:** `visualize_data.py`

## Quick Reference

| Attribute | Value |
|-----------|-------|
| **Purpose** | Create visualizations of generated frequency data |
| **Input** | `data/frequency_data.npz` |
| **Output** | 4 PNG visualizations |
| **Runtime** | ~10 seconds |
| **Dependencies** | numpy, matplotlib, scipy |

## Usage

```bash
python visualize_data.py
```

**Prerequisites:** Must run `generate_dataset.py` first to create input data.

## What This Script Does

### Overview

Creates 4 fundamental visualizations to understand the frequency data:
1. **Time domain signals** - How each frequency looks over time
2. **Frequency domain (FFT)** - Spectral analysis showing frequency peaks
3. **Spectrogram** - Time-frequency representation
4. **Signal overlay** - All frequencies combined

These visualizations help verify data quality and understand signal characteristics before training.

### Step-by-Step Process

#### 1. **Load Data** (Lines 10-18)

```python
data = np.load('data/frequency_data.npz')
x = data['x']           # Time axis
f1 = data['f1']         # 1 Hz signal
f2 = data['f2']         # 3 Hz signal
f3 = data['f3']         # 5 Hz signal
f4 = data['f4']         # 7 Hz signal
S = data['S']           # Combined signal
frequencies = data['frequencies']  # [1, 3, 5, 7]
phases = data['phases']  # Phase offsets
```

#### 2. **Visualization 1: Time Domain Signals** (Lines 28-73)

Creates 5 subplots showing individual frequencies and combined signal.

**What it shows:**
- Each frequency's oscillation pattern
- Phase offsets visible at t=0
- Combined signal showing interference patterns

**Code structure:**
```python
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
plot_range = (x >= 0) & (x <= 2)  # Plot first 2 seconds only

# Plot each frequency
axes[0].plot(x_plot, f1[plot_range], 'b-', label='f1...')
axes[1].plot(x_plot, f2[plot_range], 'g-', label='f2...')
# ... etc
```

**Why first 2 seconds only?**
- Full 10 seconds too crowded to see detail
- 2 seconds shows: 2 cycles of f₁, 6 of f₂, 10 of f₃, 14 of f₄
- Enough to see frequency differences

**Output:** `visualizations/01_time_domain_signals.png`

#### 3. **Visualization 2: Frequency Domain (FFT)** (Lines 75-138)

Performs Fast Fourier Transform on each signal to show frequency content.

**FFT computation:**
```python
sampling_rate = len(x) / (x[-1] - x[0])  # 1000 Hz
fft_S = np.fft.fft(S)
freqs = np.fft.fftfreq(len(x), 1/sampling_rate)
positive_freqs = freqs[:len(freqs)//2]  # Only positive frequencies
```

**What it shows:**
- Sharp peaks at 1, 3, 5, 7 Hz
- Combined signal shows all 4 peaks
- Confirms frequency separation

**Mathematical background:**

FFT transforms time-domain signal to frequency domain:

\[ F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt \]

For discrete signals (DFT):

\[ X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi k n / N} \]

**Output:** `visualizations/02_frequency_domain_fft.png`

#### 4. **Visualization 3: Spectrogram** (Lines 140-161)

Creates time-frequency heatmap showing how frequency content evolves.

**Spectrogram computation:**
```python
from scipy import signal as scipy_signal
f_spec, t_spec, Sxx = scipy_signal.spectrogram(
    S, 
    sampling_rate, 
    nperseg=1024  # Window size
)
```

**What it shows:**
- Horizontal bands at each frequency (1, 3, 5, 7 Hz)
- Constant over time (stationary signal)
- Intensity shows power at each frequency

**Parameters:**
- `nperseg=1024`: Window size (affects frequency resolution)
- Larger → better frequency resolution, worse time resolution
- Current value balances both

**Output:** `visualizations/03_spectrogram.png`

#### 5. **Visualization 4: Signal Overlay** (Lines 163-184)

Plots all signals on same axes for direct comparison.

**What it shows:**
- How individual frequencies combine
- Phase relationships visible
- Constructive/destructive interference

**Colors:**
- Blue: f₁ (1 Hz)
- Green: f₂ (3 Hz)
- Red: f₃ (5 Hz)
- Magenta: f₄ (7 Hz)
- Black: Combined signal S(x)

**Output:** `visualizations/04_overlay_signals.png`

## Output Files

### 1. `visualizations/01_time_domain_signals.png`

**Size:** ~400 KB

**Dimensions:** 14" × 12" (4200 × 3600 pixels at 300 DPI)

**Structure:**
- 5 subplots (stacked vertically)
- Each shows first 2 seconds
- X-axis: Time (seconds)
- Y-axis: Amplitude

**What to look for:**
✓ f₁ shows 2 complete cycles
✓ f₂ shows 6 complete cycles
✓ f₃ shows 10 complete cycles
✓ f₄ shows 14 complete cycles
✓ S shows complex interference pattern

### 2. `visualizations/02_frequency_domain_fft.png`

**Size:** ~400 KB

**Structure:**
- 5 subplots (stacked vertically)
- X-axis: Frequency (Hz), limited to 0-10 Hz
- Y-axis: Magnitude

**What to look for:**
✓ Sharp peaks at exact frequencies (1, 3, 5, 7 Hz)
✓ Minimal noise floor
✓ Combined signal shows all 4 peaks
✓ Peak heights proportional to signal strength

**Reading the plot:**
- Peak height = amplitude of that frequency
- Width = frequency resolution
- Location = actual frequency value

### 3. `visualizations/03_spectrogram.png`

**Size:** ~500 KB

**Structure:**
- Single plot with color heatmap
- X-axis: Time (seconds, 0-10)
- Y-axis: Frequency (Hz, 0-10)
- Color: Power in dB (decibels)

**What to look for:**
✓ Four horizontal bands at 1, 3, 5, 7 Hz
✓ Bands are consistent over time
✓ No frequency drift or modulation
✓ Red dashed lines mark expected frequencies

**Color scale:**
- Brighter/yellower: Higher power
- Darker/purple: Lower power

### 4. `visualizations/04_overlay_signals.png`

**Size:** ~400 KB

**Structure:**
- Single plot with all signals
- X-axis: Time (seconds, 0-2)
- Y-axis: Amplitude

**What to look for:**
✓ All frequencies visible simultaneously
✓ Phase offsets visible at t=0
✓ Combined signal (black) shows sum
✓ Constructive interference where signals align
✓ Destructive interference where signals cancel

## Technical Details

### FFT Analysis

#### Why FFT?

**Advantages:**
- Fast: O(N log N) vs O(N²) for DFT
- Reveals frequency content
- Standard tool in signal processing

**Implementation:**
```python
fft_S = np.fft.fft(S)  # Complex values
magnitude = np.abs(fft_S)  # Get magnitude
phase = np.angle(fft_S)  # Get phase (not plotted here)
```

#### Frequency Resolution

\[ \Delta f = \frac{f_s}{N} = \frac{1000}{10000} = 0.1 \text{ Hz} \]

Where:
- \( f_s \) = sampling rate (1000 Hz)
- \( N \) = number of samples (10,000)

**What this means:**
- Can distinguish frequencies 0.1 Hz apart
- Plenty of resolution for our frequencies (1, 3, 5, 7 Hz)

#### Nyquist Frequency

\[ f_{Nyquist} = \frac{f_s}{2} = \frac{1000}{2} = 500 \text{ Hz} \]

Maximum frequency that can be represented. Our highest frequency (7 Hz) << 500 Hz, so no aliasing.

### Spectrogram Details

#### What is a Spectrogram?

A spectrogram is a visual representation of the spectrum of frequencies in a signal as they vary with time.

**Computation:**
1. Divide signal into overlapping windows
2. Compute FFT for each window
3. Stack results to show time evolution

**Parameters:**
- `nperseg=1024`: Window size (1024 samples ≈ 1 second)
- `overlap`: Default 50% (512 samples)
- `window`: Default Hann window (reduces spectral leakage)

#### Reading a Spectrogram

**Horizontal bands:** Sustained frequencies
**Vertical lines:** Transient events
**Intensity:** Power at that time-frequency point

For our signals:
- All are sustained → horizontal bands
- Constant amplitude → uniform intensity
- No transients → no vertical lines

### Matplotlib Configuration

#### Figure Sizes

```python
figsize=(14, 12)  # Width × Height in inches
dpi=300  # Resolution (dots per inch)
```

**Resulting dimensions:**
- Width: 14 × 300 = 4200 pixels
- Height: 12 × 300 = 3600 pixels

High resolution for publication-quality figures.

#### Tight Layout

```python
plt.tight_layout()
```

Automatically adjusts subplot spacing to prevent overlap.

#### Saving

```python
plt.savefig('path.png', dpi=300, bbox_inches='tight')
```

- `dpi=300`: High resolution
- `bbox_inches='tight'`: Crop whitespace

## Parameters & Configuration

### Modifiable Visualization Parameters

#### Time Range for Plots

```python
# Line 35
plot_range = (x >= 0) & (x <= 2)  # Plot first 2 seconds
```

**Change to:**
- `x <= 5`: Show 5 seconds (more cycles)
- `x <= 1`: Show 1 second (less crowded)
- `x <= 10`: Show all data (very crowded)

#### Figure Size

```python
# Line 31
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
```

**Change to:**
- `figsize=(10, 8)`: Smaller file size
- `figsize=(20, 16)`: Larger, more detail

#### DPI (Resolution)

```python
# Line 71
plt.savefig('...', dpi=300, bbox_inches='tight')
```

**Options:**
- `dpi=150`: Lower resolution, smaller files
- `dpi=300`: Publication quality (current)
- `dpi=600`: Very high resolution (larger files)

#### Frequency Range for FFT

```python
# Lines 98, 106, 114, 122, 128
ax.set_xlim([0, 10])
```

**Change to:**
- `[0, 20]`: Show more frequency range
- `[0, 5]`: Focus on lower frequencies

#### Spectrogram Window Size

```python
# Line 144
f_spec, t_spec, Sxx = scipy_signal.spectrogram(S, sampling_rate, nperseg=1024)
```

**Effect of changing `nperseg`:**
- Smaller (512): Better time resolution, worse frequency resolution
- Larger (2048): Better frequency resolution, worse time resolution

### Style Customization

#### Colors

```python
# Lines 39-58
'b-'  # Blue solid line
'g-'  # Green solid line
'r-'  # Red solid line
'm-'  # Magenta solid line
'k-'  # Black solid line
```

**Matplotlib color codes:**
- `'b'`: blue, `'g'`: green, `'r'`: red
- `'c'`: cyan, `'m'`: magenta, `'y'`: yellow, `'k'`: black
- `'-'`: solid, `'--'`: dashed, `':'`: dotted

#### Line Width

```python
linewidth=1.5  # Thickness of lines
```

**Options:**
- `0.5`: Thin lines
- `1.5`: Medium (current)
- `3.0`: Thick lines

## Code Walkthrough

### Important Functions

This script doesn't define custom functions; it uses numpy and matplotlib built-ins:

**From NumPy:**
- `np.load()`: Load data from NPZ file
- `np.fft.fft()`: Fast Fourier Transform
- `np.fft.fftfreq()`: Frequency axis for FFT

**From Matplotlib:**
- `plt.subplots()`: Create figure and axes
- `ax.plot()`: Plot line
- `plt.savefig()`: Save figure to file

**From SciPy:**
- `scipy_signal.spectrogram()`: Compute spectrogram

### Code Structure

```
Load Data
├─ Read frequency_data.npz
└─ Extract arrays

For each visualization:
├─ Create figure and axes
├─ Plot data
├─ Set labels and titles
├─ Add legend and grid
├─ Save to file
└─ Close figure
```

### Key Variables

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `x` | ndarray | (10000,) | Time axis in seconds |
| `f1`-`f4` | ndarray | (10000,) | Individual frequencies |
| `S` | ndarray | (10000,) | Combined signal |
| `frequencies` | ndarray | (4,) | [1.0, 3.0, 5.0, 7.0] Hz |
| `phases` | ndarray | (4,) | Phase offsets |
| `sampling_rate` | float | scalar | 1000 Hz |
| `fft_S` | ndarray (complex) | (10000,) | FFT of combined signal |
| `freqs` | ndarray | (10000,) | Frequency axis for FFT |

## Troubleshooting

### Issue: "FileNotFoundError: data/frequency_data.npz"

**Cause:** Input file doesn't exist

**Solution:**
```bash
python generate_dataset.py  # Run step 1 first
```

### Issue: "No module named 'scipy'"

**Cause:** SciPy not installed

**Solution:**
```bash
pip install scipy
```

### Issue: Visualizations are blank

**Cause:** Matplotlib backend issue

**Solution:**
```python
# Add at top of script
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Issue: "Permission denied" when saving

**Cause:** Can't write to visualizations folder

**Solution:**
```bash
mkdir visualizations
chmod 755 visualizations
```

### Issue: Plots look different than expected

**Possible causes:**
1. **Different data:** Regenerate with `generate_dataset.py`
2. **Modified parameters:** Check frequency values match
3. **Random seed:** Should be consistent if dataset unchanged

### Issue: FFT peaks not at expected frequencies

**Diagnostic:**
```python
# Add after line 83
print(f"Sampling rate: {sampling_rate}")
print(f"Frequency resolution: {sampling_rate / len(x)}")
print(f"Expected peaks at: {frequencies}")
```

**Common cause:** Incorrect sampling rate calculation

### Issue: Spectrogram shows vertical stripes

**Cause:** Window size too large or too small

**Solution:** Adjust `nperseg` parameter (line 144):
```python
nperseg=512   # Smaller windows
# or
nperseg=2048  # Larger windows
```

## Performance Considerations

### Runtime Breakdown

| Operation | Time | Percentage |
|-----------|------|------------|
| Load data | ~0.1s | 1% |
| Time domain plot | ~2s | 20% |
| FFT computation | ~0.5s | 5% |
| FFT plot | ~2s | 20% |
| Spectrogram computation | ~1s | 10% |
| Spectrogram plot | ~3s | 30% |
| Overlay plot | ~1.5s | 15% |
| **Total** | **~10s** | **100%** |

**Bottleneck:** Matplotlib rendering, not computation

### Memory Usage

**Peak memory:** ~50 MB

**Arrays:**
- Loaded data: ~0.4 MB
- FFT results: ~0.4 MB (complex, double size)
- Spectrogram: ~2 MB (time-frequency grid)
- Matplotlib figures: ~40 MB (temporary, released after save)

### Optimization Opportunities

**If visualizations are too slow:**

1. **Reduce resolution:**
```python
dpi=150  # Instead of 300
```

2. **Reduce data points:**
```python
# Downsample for plotting
plot_every = 10
x_plot = x[::plot_every]
f1_plot = f1[::plot_every]
```

3. **Skip spectrogram:**
Comment out lines 140-161 (most computationally expensive)

## Extending the Script

### Add New Visualization: Phase Plot

```python
# After line 184
fig, ax = plt.subplots(figsize=(14, 6))

# Compute instantaneous phase
phase1 = np.angle(scipy_signal.hilbert(f1))
phase2 = np.angle(scipy_signal.hilbert(f2))
phase3 = np.angle(scipy_signal.hilbert(f3))
phase4 = np.angle(scipy_signal.hilbert(f4))

# Plot
ax.plot(x_plot, phase1[plot_range], 'b-', label='f1 phase')
ax.plot(x_plot, phase2[plot_range], 'g-', label='f2 phase')
ax.plot(x_plot, phase3[plot_range], 'r-', label='f3 phase')
ax.plot(x_plot, phase4[plot_range], 'm-', label='f4 phase')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Phase (radians)')
ax.set_title('Instantaneous Phase')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('visualizations/05_phase.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Add Waterfall Plot

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create 3D plot of FFT over time windows
# ... implementation details

plt.savefig('visualizations/waterfall.png', dpi=300)
plt.close()
```

### Save Interactive HTML

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_plot, y=f1[plot_range], name='f1'))
# ... add more traces

fig.write_html('visualizations/interactive.html')
```

## Related Documentation

- **Previous step:** `Script_01_GenerateDataset.md` - Data generation
- **Next step:** `Script_03_PrepareTraining.md` - Prepare sequences
- **Theory:** `MathematicalFoundations.md` - FFT and signal processing
- **Results:** `ResultsInterpretation.md` - How to read visualizations

## Summary

This script creates essential visualizations to:
- ✓ Verify data quality (correct frequencies, amplitudes)
- ✓ Understand signal characteristics (time/frequency domain)
- ✓ Visualize the challenge (overlapping frequencies)
- ✓ Prepare for model training (baseline understanding)

**Key outputs:** 4 publication-quality visualizations showing the frequency data from multiple perspectives.

