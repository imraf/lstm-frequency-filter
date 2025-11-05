"""
Create a comprehensive overview visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load all data
orig_data = np.load('data/frequency_data.npz')
train_data = np.load('data/training_data.npz')
history = np.load('models/training_history.npz')
eval_results = np.load('models/evaluation_results.npz')

x = orig_data['x']
S = orig_data['S']
frequencies = orig_data['frequencies']

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('LSTM Frequency Filter - Complete Overview', fontsize=18, fontweight='bold')

# ============================================================================
# Plot 1: Combined Signal (Time Domain)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
plot_range = (x >= 0) & (x <= 2)
ax1.plot(x[plot_range], S[plot_range], 'k-', linewidth=1.5)
ax1.set_title('Combined Signal S(x)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Frequency Spectrum
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
fft_S = np.fft.fft(S)
freqs = np.fft.fftfreq(len(x), (x[1] - x[0]))
positive_freqs = freqs[:len(freqs)//2]
ax2.plot(positive_freqs, 2.0/len(x) * np.abs(fft_S[:len(freqs)//2]), 'b-', linewidth=2)
ax2.set_xlim([0, 10])
ax2.set_title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.grid(True, alpha=0.3)
for i, freq in enumerate(frequencies):
    ax2.axvline(freq, color=['b', 'g', 'r', 'm'][i], linestyle='--', alpha=0.5)

# ============================================================================
# Plot 3: Dataset Statistics
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
stats_text = f"""
DATASET STATISTICS

Total Samples: 10,000
Time Interval: [0, 20] seconds
Sampling Rate: {len(x)/(x[-1]-x[0]):.0f} Hz

Frequencies:
  f1 = {frequencies[0]} Hz
  f2 = {frequencies[1]} Hz
  f3 = {frequencies[2]} Hz
  f4 = {frequencies[3]} Hz

Training Set: 31,840 sequences
Validation Set: 3,980 sequences
Test Set: 3,980 sequences
Sequence Length: 50 steps
"""
ax3.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Plot 4: Model Architecture
# ============================================================================
ax4 = fig.add_subplot(gs[1, 0])
ax4.axis('off')
arch_text = """
MODEL ARCHITECTURE

Framework: PyTorch
Type: LSTM

Input Layer:
  • 5 features
  • Signal + 4 selectors

LSTM Layers:
  • 2 stacked layers
  • 128 hidden units
  • Dropout: 0.2

Output Layer:
  • 1 value (frequency)

Total Parameters: 201,345
"""
ax4.text(0.1, 0.5, arch_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# Plot 5: Training Loss Curves
# ============================================================================
ax5 = fig.add_subplot(gs[1, 1])
epochs = range(1, len(history['train_losses']) + 1)
ax5.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Training')
ax5.plot(epochs, history['val_losses'], 'r-', linewidth=2, label='Validation')
ax5.set_title('Training Progress', fontsize=12, fontweight='bold')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('MSE Loss')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# ============================================================================
# Plot 6: Performance Metrics
# ============================================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
metrics_text = f"""
PERFORMANCE METRICS

MSE:  {eval_results['mse']:.6f}
RMSE: {eval_results['rmse']:.6f}
MAE:  {eval_results['mae']:.6f}
R²:   {eval_results['r2']:.6f}
Corr: {eval_results['correlation']:.6f}

INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━
R² = 0.948
→ Model explains 94.8%
  of variance

Correlation = 0.974
→ Very strong linear
  relationship

RMSE = 0.161
→ Low prediction error
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ============================================================================
# Plot 7: Prediction vs Actual (Sample)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 0])
sample_idx = 0
y_test = eval_results['y_test']
y_pred = eval_results['y_pred']
time_steps = range(50)
ax7.plot(time_steps, y_test[sample_idx, :, 0], 'b-', linewidth=2, label='Actual', alpha=0.7)
ax7.plot(time_steps, y_pred[sample_idx, :, 0], 'r--', linewidth=2, label='Predicted', alpha=0.7)
ax7.set_title('Sample Prediction', fontsize=12, fontweight='bold')
ax7.set_xlabel('Time Step')
ax7.set_ylabel('Amplitude')
ax7.legend()
ax7.grid(True, alpha=0.3)

# ============================================================================
# Plot 8: Error Distribution
# ============================================================================
ax8 = fig.add_subplot(gs[2, 1])
errors = y_test.flatten() - y_pred.flatten()
ax8.hist(errors, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax8.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Error')
ax8.set_ylabel('Frequency')
ax8.legend()
ax8.grid(True, alpha=0.3)

# ============================================================================
# Plot 9: Scatter Plot
# ============================================================================
ax9 = fig.add_subplot(gs[2, 2])
sample_indices = np.arange(0, len(y_test.flatten()), 10)
y_test_sample = y_test.flatten()[sample_indices]
y_pred_sample = y_pred.flatten()[sample_indices]
ax9.scatter(y_test_sample, y_pred_sample, alpha=0.3, s=10, color='steelblue')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax9.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax9.set_title(f'Predicted vs Actual (R²={eval_results["r2"]:.3f})', fontsize=12, fontweight='bold')
ax9.set_xlabel('Actual')
ax9.set_ylabel('Predicted')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_aspect('equal', adjustable='box')

plt.savefig('visualizations/00_complete_overview.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/00_complete_overview.png")
plt.close()

print("\nComplete overview visualization created!")
