# Script 05: Evaluate Model

**File:** `evaluate_model.py`

## Quick Reference

| Attribute | Value |
|-----------|-------|
| **Purpose** | Evaluate trained LSTM model and create comprehensive visualizations |
| **Input** | `models/best_model.pth`, `data/training_data.npz` |
| **Output** | 6 visualizations + evaluation results file |
| **Runtime** | ~20-30 seconds |
| **Dependencies** | numpy, torch, matplotlib, sklearn, scipy |

## Usage

```bash
python evaluate_model.py
```

**Prerequisites:** Must run `train_model.py` first to create the trained model.

## What This Script Does

### Overview

Comprehensive model evaluation on the test set:

1. **Loads trained model** and test data
2. **Makes predictions** on test set (Seed #2, unseen noise)
3. **Calculates metrics** (MSE, RMSE, MAE, R², correlation)
4. **Per-frequency analysis** (individual performance for each frequency)
5. **Creates 6 visualizations** showing prediction quality and errors

This script provides the complete picture of model performance and generalization.

### Step-by-Step Process

#### 1. **Load Model Architecture** (Lines 18-41)

Defines the same model architecture used in training:

```python
class FrequencyFilterLSTM(nn.Module):
    # Same architecture as train_model.py
    # Must match exactly to load weights correctly
```

#### 2. **Load Data and Model** (Lines 45-64)

```python
# Load test data
data = np.load('data/training_data.npz')
X_test = data['X_test']    # (39804, 50, 5)
y_test = data['y_test']    # (39804, 50, 1)
selectors = data['selectors']  # (4, 4) one-hot matrix

# Load trained model
model = FrequencyFilterLSTM(5, 128, 2, 1, 0.2).to(device)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()  # Set to evaluation mode
```

**Crucial:** `model.eval()` disables dropout and batch normalization updates.

#### 3. **Make Predictions** (Lines 70-74)

```python
X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():  # No gradient computation (faster, less memory)
    y_pred = model(X_test_tensor).cpu().numpy()
```

**Result:** `y_pred` shape (39804, 50, 1) - predictions for all test sequences

#### 4. **Calculate Overall Metrics** (Lines 84-102)

Flatten arrays for sklearn metrics:

```python
y_test_flat = y_test.flatten()  # (1,990,200,) all values
y_pred_flat = y_pred.flatten()

# Calculate metrics
mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)
correlation, p_value = pearsonr(y_test_flat, y_pred_flat)
```

**Metrics:**
- **MSE:** Average squared error
- **RMSE:** Root MSE (same units as signal)
- **MAE:** Mean absolute error (robust to outliers)
- **R²:** Coefficient of determination (% variance explained)
- **Correlation:** Pearson correlation coefficient

#### 5. **Per-Frequency Metrics** (Lines 105-122)

Calculate metrics separately for each frequency:

```python
for i in range(4):
    # Find samples for this frequency
    selector_mask = np.all(X_test[:, 0, 1:] == selectors[i], axis=1)
    y_test_freq = y_test[selector_mask].flatten()
    y_pred_freq = y_pred[selector_mask].flatten()
    
    # Calculate metrics
    mse_freq = mean_squared_error(y_test_freq, y_pred_freq)
    r2_freq = r2_score(y_test_freq, y_pred_freq)
    # ... etc
```

**Result:** Individual performance for f₁, f₂, f₃, f₄

#### 6. **Create Visualizations**

**Visualization 1: Predictions vs Actual** (Lines 141-175)

4×2 grid showing sample predictions for each frequency:
- Row i: Frequency fᵢ
- Column 1: First sample
- Column 2: Second sample
- Blue line: Actual (ground truth)
- Red dashed: Predicted

**What to look for:**
✓ Predicted closely follows actual
✓ Phase alignment correct
✓ Amplitude matches
✓ No systematic lag or lead

**Visualization 2: Error Distribution** (Lines 180-204)

Two subplots:
1. **Histogram:** Distribution of prediction errors
2. **Q-Q Plot:** Normality check

**What Q-Q plot shows:**
- Points on diagonal → errors are normally distributed
- Points below diagonal → lighter tails than normal
- Points above diagonal → heavier tails than normal

**Why this matters:**
Normal error distribution suggests:
✓ Unbiased predictions
✓ No systematic errors
✓ Appropriate model complexity

**Visualization 3: Scatter Plot** (Lines 209-234)

Predicted vs. actual values scatter plot:
- X-axis: Actual values
- Y-axis: Predicted values
- Red dashed line: Perfect prediction (y=x)

**Interpretation:**
- Points close to line: Good predictions
- Spread around line: Error magnitude
- Systematic deviation: Bias

**Visualization 4: Frequency Spectrum Comparison** (Lines 239-284)

FFT of predictions vs. actual for each frequency:

**Process:**
1. Take first sample for each frequency
2. Compute FFT of actual signal
3. Compute FFT of predicted signal
4. Plot magnitude spectra

**What to look for:**
✓ Peak at correct frequency
✓ Predicted and actual peaks align
✓ Similar magnitude
✓ No spurious peaks

**Visualization 5: Long Sequence Predictions** (Lines 289-325)

Extended time series (20 concatenated sequences):

**Shows:**
- Model maintains accuracy over long durations
- No drift or degradation
- Temporal coherence across sequences

**Visualization 6: Per-Frequency Metrics** (Lines 330-376)

Bar charts comparing performance across frequencies:
- MSE
- RMSE  
- MAE
- R² Score

**Shows which frequencies:**
- Are easiest/hardest to filter
- Have best/worst performance
- Need improvement

#### 7. **Save Results** (Lines 380-387)

```python
np.savez('models/evaluation_results.npz',
         y_test=y_test,
         y_pred=y_pred,
         mse=mse,
         rmse=rmse,
         mae=mae,
         r2=r2,
         correlation=correlation)
```

## Output Files

### 1. `models/evaluation_results.npz`

**Size:** ~160 MB

**Contents:**
- `y_test`: Ground truth test sequences (39804, 50, 1)
- `y_pred`: Model predictions (39804, 50, 1)
- `mse`, `rmse`, `mae`, `r2`, `correlation`: Scalar metrics

**Usage:**
```python
results = np.load('models/evaluation_results.npz')
predictions = results['y_pred']
r2_score = results['r2']
```

### 2. `visualizations/08_predictions_vs_actual.png`

**Shows:** Sample predictions for each frequency (4×2 grid)

**Size:** ~600 KB

**What to check:**
- Predicted matches actual waveform
- Correct frequency visible
- Minimal lag or lead
- Amplitude preserved

### 3. `visualizations/09_error_distribution.png`

**Shows:** Error histogram and Q-Q plot

**What to check:**
- Histogram centered at zero (unbiased)
- Q-Q plot follows diagonal (normally distributed)
- Standard deviation reasonable

**Ideal characteristics:**
✓ Mean error ≈ 0
✓ Symmetric distribution
✓ Q-Q points on line

### 4. `visualizations/10_scatter_pred_vs_actual.png`

**Shows:** Correlation plot

**What to check:**
- Points cluster around diagonal
- Tight clustering = low error
- R² value shown in title

**Good performance:**
- R² > 0.3 (moderate)
- No systematic bias (symmetric around line)

### 5. `visualizations/11_frequency_spectrum_comparison.png`

**Shows:** FFT comparison (2×2 grid, one per frequency)

**What to check:**
- Peak at correct frequency
- Predicted and actual peaks align
- Similar magnitudes

### 6. `visualizations/12_long_sequence_predictions.png`

**Shows:** Extended time series (4 rows, one per frequency)

**What to check:**
- Consistency over time
- No drift or degradation
- Maintains phase lock

### 7. `visualizations/13_per_frequency_metrics.png`

**Shows:** Bar charts (2×2 grid: MSE, RMSE, MAE, R²)

**What to check:**
- Which frequencies perform best
- Performance variation across frequencies
- All metrics reasonable

## Understanding the Metrics

### Overall Performance

**Expected values:**

| Metric | Expected | Good | Excellent |
|--------|----------|------|-----------|
| R² | 0.30-0.40 | > 0.35 | > 0.50 |
| Correlation | 0.60-0.70 | > 0.65 | > 0.80 |
| RMSE | 0.50-0.60 | < 0.55 | < 0.40 |
| MAE | 0.35-0.45 | < 0.40 | < 0.30 |

**Why these values?**

This is a **challenging task:**
- 4 overlapping frequencies
- Moderate noise (SNR ≈ 11 dB)
- Test set has different noise than training
- Time-domain filtering (no explicit FFT)

R² = 0.35 means model explains 35% of variance - **reasonable for this difficulty!**

### R² Score Interpretation

**Formula:**

\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \]

**Interpretation:**
- **R² = 1.0:** Perfect predictions
- **R² = 0.35:** Model explains 35% of variance
- **R² = 0.0:** Model performs as well as predicting mean
- **R² < 0:** Model worse than predicting mean

**Context:**
- vs. Random baseline: Model is **54% better** (MAE comparison)
- vs. Mean baseline: Model is **41% better**

### Correlation Coefficient

**Formula:**

\[ r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}} \]

**Interpretation:**
- **r = 1.0:** Perfect positive correlation
- **r = 0.63:** Strong positive correlation
- **r = 0.0:** No correlation
- **r = -1.0:** Perfect negative correlation

**Significance:**
p-value ≈ 0 means correlation is statistically significant

### Per-Frequency Performance

**Typical pattern:**

| Frequency | R² | Difficulty |
|-----------|-----|------------|
| f₁ (1 Hz) | 0.64 | Easiest |
| f₂ (3 Hz) | 0.18 | Harder |
| f₃ (5 Hz) | 0.17 | Harder |
| f₄ (7 Hz) | 0.40 | Moderate |

**Why is f₁ easiest?**
- Lower frequency → longer wavelength
- More temporal context in 50-sample window
- Less interference from higher frequencies

**Why are f₂ and f₃ harder?**
- Higher frequencies → shorter wavelengths
- Less complete cycles in window
- More susceptible to noise

## Technical Details

### Evaluation Mode

```python
model.eval()
```

**What it does:**
- Sets dropout layers to identity (no random drops)
- Sets batch normalization to use running statistics
- Essential for reproducible predictions

**Without eval():**
- Dropout still active → random predictions
- Batch norm uses batch statistics → inconsistent

### No Gradient Computation

```python
with torch.no_grad():
    predictions = model(input)
```

**Benefits:**
- **Faster:** No gradient computation or storage
- **Less memory:** No need to store intermediate values
- **Same results:** Forward pass identical

**Memory savings:** ~50% for inference

### Metric Calculation

**Why flatten arrays?**

sklearn metrics expect 1D arrays:

```python
y_test_flat = y_test.flatten()
# (39804, 50, 1) → (1,990,200,)
```

**Alternative:** Calculate per-sequence metrics and average:

```python
r2_per_sequence = [r2_score(y_test[i], y_pred[i]) for i in range(len(y_test))]
average_r2 = np.mean(r2_per_sequence)
```

### Frequency Domain Analysis

**Why FFT?**
Verify model preserves frequency content:

```python
fft_test = np.fft.fft(y_test_signal)
fft_pred = np.fft.fft(y_pred_signal)
```

**Good model:**
- FFT peaks at same frequencies
- Similar magnitudes
- No spurious frequencies

**Poor model:**
- Peaks at wrong frequencies
- Phase noise
- Missing frequency components

## Troubleshooting

### Issue: "FileNotFoundError: models/best_model.pth"

**Cause:** Model not trained yet

**Solution:**
```bash
python train_model.py  # Train model first
```

### Issue: Metrics are very poor (R² < 0)

**Possible causes:**
1. Wrong model architecture
2. Model didn't train properly
3. Data mismatch

**Diagnostic:**
```python
# Check model loaded correctly
print(model)

# Check prediction range
print(f"Predictions range: [{y_pred.min()}, {y_pred.max()}]")
print(f"Actual range: [{y_test.min()}, {y_test.max()}]")

# Should be similar (around [-1.5, 1.5])
```

**Solutions:**
1. Verify model architecture matches training
2. Check training completed successfully
3. Regenerate data if needed

### Issue: Predictions are all the same value

**Cause:** Model collapsed to predicting mean

**Diagnostic:**
```python
print(f"Unique predictions: {len(np.unique(y_pred.round(3)))}")
# Should be >> 100
```

**Solution:** Retrain with different hyperparameters

### Issue: "CUDA out of memory"

**Cause:** Batch too large for GPU memory

**Solution:** Process in smaller batches:

```python
# Instead of processing all at once
batch_size = 1000
predictions = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_tensor = torch.FloatTensor(batch).to(device)
    with torch.no_grad():
        pred = model(batch_tensor).cpu().numpy()
    predictions.append(pred)
y_pred = np.concatenate(predictions, axis=0)
```

### Issue: Different results each run

**Cause:** Model not in eval mode or random seed issue

**Solution:**
```python
# Ensure eval mode
model.eval()

# Set seeds if needed
torch.manual_seed(42)
```

### Issue: Visualizations look wrong

**Possible causes:**
1. Matplotlib backend
2. Data range issues
3. Wrong axis limits

**Solutions:**
1. Try different backend: `matplotlib.use('Agg')`
2. Check data ranges before plotting
3. Adjust xlim/ylim in plot code

## Performance Considerations

### Runtime Breakdown

| Operation | Time | Percentage |
|-----------|------|------------|
| Load model | ~1s | 5% |
| Make predictions | ~5s | 20% |
| Calculate metrics | ~2s | 10% |
| Create visualizations | ~15s | 65% |
| **Total** | **~23s** | **100%** |

**Bottleneck:** Matplotlib rendering

### Memory Usage

**Peak memory:** ~200 MB

**Breakdown:**
- Model: 0.8 MB
- Test data: 160 MB
- Predictions: 160 MB
- Intermediate arrays: ~10 MB

**If memory is limited:**
Process test set in batches (see troubleshooting)

### Optimization

**Speed up visualizations:**

```python
# Reduce DPI
plt.savefig(..., dpi=150)  # Instead of 300

# Downsample data for plotting
plot_every = 10
y_plot = y_test[::plot_every]
```

**Speed up predictions:**

```python
# Increase batch size if memory allows
batch_size = 2000  # Instead of 1000
```

## Extending the Script

### Add Confidence Intervals

```python
# Monte Carlo Dropout for uncertainty estimation
model.train()  # Enable dropout
n_samples = 100
predictions = []

for _ in range(n_samples):
    with torch.no_grad():
        pred = model(X_test_tensor).cpu().numpy()
    predictions.append(pred)

predictions = np.array(predictions)
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)

# Plot with confidence intervals
plt.plot(mean_pred[0, :, 0], label='Mean')
plt.fill_between(range(50), 
                 mean_pred[0, :, 0] - 2*std_pred[0, :, 0],
                 mean_pred[0, :, 0] + 2*std_pred[0, :, 0],
                 alpha=0.3, label='95% CI')
```

### Add Confusion Matrix for Phase

```python
# Discretize phase into bins
def get_phase(signal):
    return np.angle(scipy.signal.hilbert(signal))

phase_true = np.array([get_phase(y_test[i, :, 0]) for i in range(100)])
phase_pred = np.array([get_phase(y_pred[i, :, 0]) for i in range(100)])

# Bin into 8 phase categories
phase_bins = np.linspace(-np.pi, np.pi, 9)
phase_true_binned = np.digitize(phase_true, phase_bins)
phase_pred_binned = np.digitize(phase_pred, phase_bins)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(phase_true_binned.flatten(), phase_pred_binned.flatten())
```

### Add Statistical Tests

```python
from scipy import stats

# Test if errors are normally distributed
stat, p_value = stats.normaltest(errors)
print(f"Normality test: statistic={stat}, p-value={p_value}")

# Test if errors have zero mean
t_stat, p_value = stats.ttest_1samp(errors, 0)
print(f"Zero mean test: t={t_stat}, p-value={p_value}")

# Test for autocorrelation in errors
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_stat, lb_pvalue = acorr_ljungbox(errors[:1000], lags=10)
print(f"Autocorrelation test: {lb_pvalue}")
```

### Add Baseline Comparisons

```python
# Random baseline
y_random = np.random.normal(0, np.std(y_test), y_test.shape)
mae_random = mean_absolute_error(y_test.flatten(), y_random.flatten())

# Mean baseline
y_mean = np.full_like(y_test, np.mean(y_test))
mae_mean = mean_absolute_error(y_test.flatten(), y_mean.flatten())

# Compare
print(f"Model MAE: {mae}")
print(f"Random MAE: {mae_random} (model is {(mae_random-mae)/mae_random*100:.1f}% better)")
print(f"Mean MAE: {mae_mean} (model is {(mae_mean-mae)/mae_mean*100:.1f}% better)")
```

## Related Documentation

- **Previous step:** `Script_04_TrainModel.md` - Model training
- **Next step:** `Script_06_Summary.md` - Results summary
- **Metrics:** `Feature_EvaluationMetrics.md` - Detailed metrics explanation
- **Results:** `ResultsInterpretation.md` - How to interpret visualizations
- **Improvements:** `HyperparameterGuide.md` - How to improve performance

## Summary

This script provides comprehensive model evaluation:
- ✓ Tests on unseen noise (different seed)
- ✓ Calculates 5 key metrics (MSE, RMSE, MAE, R², correlation)
- ✓ Per-frequency analysis (individual performance)
- ✓ 6 detailed visualizations
- ✓ R² = 0.35 (moderate performance for challenging task)
- ✓ 54% better than random baseline
- ✓ 41% better than mean baseline

**Key achievement:** Model successfully generalizes to test set with different noise, demonstrating true learning of frequency patterns.

