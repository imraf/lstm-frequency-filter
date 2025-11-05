"""
Step 2: Prepare training data with one-hot selectors
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = np.load('data/frequency_data.npz')
x = data['x']
f1 = data['f1']
f2 = data['f2']
f3 = data['f3']
f4 = data['f4']
S = data['S']
frequencies = data['frequencies']

n_samples = len(x)
print(f"Loaded data with {n_samples} samples")
print(f"Frequencies: {frequencies}")

# ============================================================================
# Create Training Samples
# ============================================================================
# For LSTM, we need to create sequences from the time series data
# We'll use a sliding window approach

# Hyperparameters for sequence creation
sequence_length = 50  # Use 50 time steps as input
step_size = 1  # Stride between sequences

print(f"\nCreating sequences with:")
print(f"  Sequence length: {sequence_length}")
print(f"  Step size: {step_size}")

# Calculate number of sequences
n_sequences = (n_samples - sequence_length) // step_size + 1
print(f"  Total sequences: {n_sequences}")

# One-hot selectors for the 4 frequencies
# c1 = [1, 0, 0, 0] -> select f1
# c2 = [0, 1, 0, 0] -> select f2
# c3 = [0, 0, 1, 0] -> select f3
# c4 = [0, 0, 0, 1] -> select f4
selectors = np.array([
    [1, 0, 0, 0],  # c1
    [0, 1, 0, 0],  # c2
    [0, 0, 1, 0],  # c3
    [0, 0, 0, 1]   # c4
])

# Stack all target frequencies
all_frequencies = np.stack([f1, f2, f3, f4], axis=1)  # Shape: (n_samples, 4)

# Create sequences
# For each sequence, we'll create 4 training samples (one for each selector)
# Input: S(x) sequence + selector (one-hot vector)
# Output: fi(x) sequence (the selected frequency)

X_sequences = []  # Input signal sequences
X_selectors = []  # Input selectors
y_sequences = []  # Target frequency sequences

for i in range(0, n_samples - sequence_length, step_size):
    # Extract sequence of S(x)
    S_seq = S[i:i+sequence_length]
    
    # Extract target frequency sequences
    f_seqs = all_frequencies[i:i+sequence_length, :]  # Shape: (seq_len, 4)
    
    # Create 4 training samples for this sequence (one per frequency)
    for j in range(4):
        X_sequences.append(S_seq)
        X_selectors.append(selectors[j])
        y_sequences.append(f_seqs[:, j])

# Convert to numpy arrays
X_sequences = np.array(X_sequences)  # Shape: (n_train, sequence_length)
X_selectors = np.array(X_selectors)  # Shape: (n_train, 4)
y_sequences = np.array(y_sequences)  # Shape: (n_train, sequence_length)

print(f"\nTraining data shapes:")
print(f"  X_sequences (signal): {X_sequences.shape}")
print(f"  X_selectors (one-hot): {X_selectors.shape}")
print(f"  y_sequences (target): {y_sequences.shape}")

# ============================================================================
# The LSTM input will combine S(x) sequence with the selector
# Two approaches:
# 1. Concatenate selector to each time step
# 2. Use selector as additional input
# We'll use approach 1: broadcast selector across all time steps
# ============================================================================

# Expand selector to match sequence length
# Shape: (n_train, sequence_length, 4)
X_selectors_expanded = np.repeat(X_selectors[:, np.newaxis, :], sequence_length, axis=1)

# Expand signal sequence to have feature dimension
# Shape: (n_train, sequence_length, 1)
X_sequences_expanded = X_sequences[:, :, np.newaxis]

# Concatenate signal and selector
# Final input shape: (n_train, sequence_length, 5)
# 5 features = 1 signal value + 4 selector values
X_combined = np.concatenate([X_sequences_expanded, X_selectors_expanded], axis=2)

print(f"\nCombined input shape: {X_combined.shape}")
print(f"  (samples, sequence_length, features)")
print(f"  Features: 1 signal value + 4 selector values = 5 total")

# Expand target to have feature dimension
# Shape: (n_train, sequence_length, 1)
y_sequences_expanded = y_sequences[:, :, np.newaxis]

print(f"\nTarget output shape: {y_sequences_expanded.shape}")

# ============================================================================
# Split into train/validation/test sets
# ============================================================================
from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp (which will be split into val and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_combined, y_sequences_expanded, test_size=0.2, random_state=42
)

# Second split: Split temp into 50% validation, 50% test (10% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\nData split:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_combined)*100:.1f}%)")
print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X_combined)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_combined)*100:.1f}%)")

# Save the processed data
np.savez('data/training_data.npz',
         X_train=X_train,
         X_val=X_val,
         X_test=X_test,
         y_train=y_train,
         y_val=y_val,
         y_test=y_test,
         sequence_length=sequence_length,
         selectors=selectors)

print(f"\nTraining data saved to data/training_data.npz")

# ============================================================================
# Visualize sample training data
# ============================================================================
Path("visualizations").mkdir(exist_ok=True)

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Sample Training Data: Input Signal & Target Frequencies', fontsize=16)

# Show one example for each frequency selector
for i in range(4):
    # Get a sample for this selector
    selector_mask = np.all(X_train[:, 0, 1:] == selectors[i], axis=1)
    sample_idx = np.where(selector_mask)[0][0]
    
    sample_input = X_train[sample_idx, :, 0]  # Signal
    sample_selector = X_train[sample_idx, 0, 1:]  # Selector (same for all time steps)
    sample_target = y_train[sample_idx, :, 0]  # Target
    
    # Plot input signal
    axes[i, 0].plot(sample_input, 'k-', linewidth=1.5, label='Input S(x)')
    axes[i, 0].set_ylabel(f'Amplitude', fontsize=10)
    axes[i, 0].set_title(f'Input Signal with Selector {sample_selector}', fontsize=11)
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend(loc='upper right')
    
    # Plot target frequency
    axes[i, 1].plot(sample_target, ['b-', 'g-', 'r-', 'm-'][i], linewidth=1.5, 
                    label=f'Target f{i+1} ({frequencies[i]} Hz)')
    axes[i, 1].set_ylabel(f'Amplitude', fontsize=10)
    axes[i, 1].set_title(f'Target Frequency f{i+1}', fontsize=11)
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend(loc='upper right')
    
    if i == 3:
        axes[i, 0].set_xlabel('Time Step', fontsize=11)
        axes[i, 1].set_xlabel('Time Step', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/05_training_samples.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/05_training_samples.png")
plt.close()

# ============================================================================
# Visualize the model input/output structure
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Model Input/Output Structure', fontsize=16)

# Take one sample
sample_idx = 0
sample_input = X_train[sample_idx]
sample_output = y_train[sample_idx]

# Plot input features over time
axes[0].plot(sample_input[:, 0], 'k-', linewidth=2, label='S(x) - Signal')
axes[0].plot(sample_input[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='Selector[0]')
axes[0].plot(sample_input[:, 2], 'g--', linewidth=1.5, alpha=0.7, label='Selector[1]')
axes[0].plot(sample_input[:, 3], 'r--', linewidth=1.5, alpha=0.7, label='Selector[2]')
axes[0].plot(sample_input[:, 4], 'm--', linewidth=1.5, alpha=0.7, label='Selector[3]')
axes[0].set_ylabel('Value', fontsize=11)
axes[0].set_title('Input: Combined Signal + One-Hot Selector (5 features)', fontsize=12)
axes[0].legend(loc='upper right', ncol=5)
axes[0].grid(True, alpha=0.3)

# Plot output
axes[1].plot(sample_output[:, 0], 'b-', linewidth=2, label='Target Frequency')
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].set_xlabel('Time Step', fontsize=11)
axes[1].set_title('Output: Selected Frequency Signal (1 feature)', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_model_io_structure.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/06_model_io_structure.png")
plt.close()

print("\n" + "="*70)
print("LOSS FUNCTION RECOMMENDATION")
print("="*70)
print("""
For this frequency filtering task, the recommended loss function is:

** Mean Squared Error (MSE) **

Rationale:
1. Regression Problem: We're predicting continuous signal values (not classification)
2. Signal Matching: MSE penalizes deviations in amplitude proportionally to their square
3. Smooth Gradients: Provides smooth gradients for LSTM training
4. Standard Choice: MSE is the standard loss for time series regression

Alternative loss functions to consider:
- Mean Absolute Error (MAE): More robust to outliers, but less common for signals
- Huber Loss: Combination of MSE and MAE, good for noisy data
- Custom Signal-to-Noise Ratio (SNR) loss: Could maximize SNR of filtered signal

We'll use MSE as the primary loss function.
""")

print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)
print("""
Beyond the loss function, we'll track these metrics:

1. **MSE**: Primary metric for training
2. **RMSE**: Root Mean Squared Error (more interpretable, same units as signal)
3. **MAE**: Mean Absolute Error (robust metric)
4. **R² Score**: Coefficient of determination (how well we explain variance)
5. **Correlation**: Pearson correlation between predicted and actual signals
6. **Frequency Domain Error**: MSE in frequency domain (FFT comparison)

These will be computed during evaluation to assess model performance.
""")

print("\nTraining data preparation complete!")
