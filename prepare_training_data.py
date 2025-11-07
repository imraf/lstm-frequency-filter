"""
Step 2: Prepare training data with one-hot selectors
Now processing separate train/test datasets (different noise seeds)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("PREPARING TRAINING DATA")
print("="*70)

# Load TRAINING data (Seed #1)
print("\nLoading training dataset (Seed #1)...")
train_data = np.load('data/frequency_data_train.npz')
x_train = train_data['x']
S_train = train_data['S']
target1_train = train_data['target1']
target2_train = train_data['target2']
target3_train = train_data['target3']
target4_train = train_data['target4']
frequencies = train_data['frequencies']

n_samples_train = len(x_train)
print(f"  Training samples: {n_samples_train}")
print(f"  Frequencies: {frequencies}")

# Load TEST data (Seed #2)
print("\nLoading test dataset (Seed #2)...")
test_data = np.load('data/frequency_data_test.npz')
x_test_raw = test_data['x']
S_test = test_data['S']
target1_test = test_data['target1']
target2_test = test_data['target2']
target3_test = test_data['target3']
target4_test = test_data['target4']

n_samples_test = len(x_test_raw)
print(f"  Test samples: {n_samples_test}")
print(f"  NOTE: Test set uses completely different noise (Seed #2)")

# ============================================================================
# Create Sequences from Both Datasets
# ============================================================================
# For LSTM, we need to create sequences from the time series data
# We'll use a sliding window approach

# Hyperparameters for sequence creation
sequence_length = 50  # Use 50 time steps as input
step_size = 1  # Stride between sequences

print(f"\n" + "="*70)
print("SEQUENCE CREATION PARAMETERS")
print("="*70)
print(f"  Sequence length: {sequence_length}")
print(f"  Step size: {step_size}")

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

def create_sequences(S, targets, sequence_length, step_size, dataset_name):
    """
    Create sequences from signal and targets
    
    For each sequence, we create 4 training samples (one for each selector)
    Input: S(x) sequence + selector (one-hot vector)
    Output: fi(x) sequence (the selected frequency)
    """
    print(f"\nCreating sequences for {dataset_name}...")
    
    # Stack all target frequencies
    all_frequencies = np.stack(targets, axis=1)  # Shape: (n_samples, 4)
    
    X_sequences = []  # Input signal sequences
    X_selectors = []  # Input selectors
    y_sequences = []  # Target frequency sequences
    
    n_samples = len(S)
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
    
    print(f"  Sequences created: {len(X_sequences)}")
    print(f"  X_sequences shape: {X_sequences.shape}")
    print(f"  X_selectors shape: {X_selectors.shape}")
    print(f"  y_sequences shape: {y_sequences.shape}")
    
    return X_sequences, X_selectors, y_sequences

# Create sequences for TRAINING data (Seed #1)
X_train_seq, X_train_sel, y_train_seq = create_sequences(
    S_train,
    [target1_train, target2_train, target3_train, target4_train],
    sequence_length,
    step_size,
    "TRAINING SET (Seed #1)"
)

# Create sequences for TEST data (Seed #2)
X_test_seq, X_test_sel, y_test_seq = create_sequences(
    S_test,
    [target1_test, target2_test, target3_test, target4_test],
    sequence_length,
    step_size,
    "TEST SET (Seed #2)"
)

# ============================================================================
# Combine Features (Signal + Selector)
# ============================================================================
# The LSTM input will combine S(x) sequence with the selector
# We broadcast selector across all time steps
# ============================================================================

def combine_features(X_seq, X_sel, y_seq, dataset_name):
    """
    Combine signal sequences with selector one-hot encoding
    
    Final input shape: (n_samples, sequence_length, 5)
    5 features = 1 signal value + 4 selector values
    """
    print(f"\nCombining features for {dataset_name}...")
    
    # Expand selector to match sequence length
    # Shape: (n_samples, sequence_length, 4)
    X_sel_expanded = np.repeat(X_sel[:, np.newaxis, :], sequence_length, axis=1)
    
    # Expand signal sequence to have feature dimension
    # Shape: (n_samples, sequence_length, 1)
    X_seq_expanded = X_seq[:, :, np.newaxis]
    
    # Concatenate signal and selector
    # Final input shape: (n_samples, sequence_length, 5)
    X_combined = np.concatenate([X_seq_expanded, X_sel_expanded], axis=2)
    
    # Expand target to have feature dimension
    # Shape: (n_samples, sequence_length, 1)
    y_expanded = y_seq[:, :, np.newaxis]
    
    print(f"  Combined input shape: {X_combined.shape}")
    print(f"  Features: 1 signal + 4 selectors = 5 total")
    print(f"  Target shape: {y_expanded.shape}")
    
    return X_combined, y_expanded

# Combine features for training data (Seed #1)
X_train_combined, y_train_combined = combine_features(
    X_train_seq, X_train_sel, y_train_seq, "TRAINING SET"
)

# Combine features for test data (Seed #2)
X_test_combined, y_test_combined = combine_features(
    X_test_seq, X_test_sel, y_test_seq, "TEST SET"
)

# ============================================================================
# Split Training Data into Train/Validation
# Test data remains separate (Seed #2)
# ============================================================================
from sklearn.model_selection import train_test_split

print(f"\n" + "="*70)
print("DATA SPLITTING")
print("="*70)
print(f"Training data (Seed #1) will be split into train/val")
print(f"Test data (Seed #2) remains separate for final evaluation")

# Split training data: 90% train, 10% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_combined, y_train_combined, test_size=0.1, random_state=42
)

# Test data is already separate (different noise seed)
X_test = X_test_combined
y_test = y_test_combined

print(f"\nFinal data split:")
print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_val.shape[0])*100:.1f}% of Seed #1)")
print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/(X_train.shape[0]+X_val.shape[0])*100:.1f}% of Seed #1)")
print(f"  Test set:       {X_test.shape[0]} samples (100% from Seed #2)")
print(f"\n  CRITICAL: Test set uses COMPLETELY DIFFERENT NOISE from training!")

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

print(f"\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Sample Training Data: Noisy Input Signal & Pure Target Frequencies', fontsize=16)

# Show one example for each frequency selector
for i in range(4):
    # Get a sample for this selector
    selector_mask = np.all(X_train[:, 0, 1:] == selectors[i], axis=1)
    sample_idx = np.where(selector_mask)[0][0]
    
    sample_input = X_train[sample_idx, :, 0]  # Noisy signal S
    sample_selector = X_train[sample_idx, 0, 1:]  # Selector (same for all time steps)
    sample_target = y_train[sample_idx, :, 0]  # Pure target
    
    # Plot input signal (noisy)
    axes[i, 0].plot(sample_input, 'k-', linewidth=1.5, label='Input S(t) [Noisy]', alpha=0.7)
    axes[i, 0].set_ylabel(f'Amplitude', fontsize=10)
    axes[i, 0].set_title(f'Input Signal with Selector {sample_selector}', fontsize=11)
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend(loc='upper right')
    
    # Plot target frequency (pure, no noise)
    axes[i, 1].plot(sample_target, ['b-', 'g-', 'r-', 'm-'][i], linewidth=1.5, 
                    label=f'Target f{i+1} ({frequencies[i]} Hz) [Pure]')
    axes[i, 1].set_ylabel(f'Amplitude', fontsize=10)
    axes[i, 1].set_title(f'Target Frequency f{i+1} (No Noise)', fontsize=11)
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend(loc='upper right')
    
    if i == 3:
        axes[i, 0].set_xlabel('Time Step', fontsize=11)
        axes[i, 1].set_xlabel('Time Step', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/05_training_samples.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/05_training_samples.png")
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
print("✓ Saved: visualizations/06_model_io_structure.png")
plt.close()

print("\n" + "="*70)
print("SEQUENCE LENGTH JUSTIFICATION (L=50)")
print("="*70)
print("""
This implementation uses L=50 (sequence length of 50 timesteps) instead of L=1.

PEDAGOGICAL JUSTIFICATION:

1. TEMPORAL ADVANTAGE:
   - At 1000 Hz sampling, L=50 covers 50ms of signal
   - For f1 (1 Hz): Captures 5% of one cycle
   - For f4 (7 Hz): Captures 35% of one cycle
   - LSTM can learn phase and frequency patterns across multiple timesteps
   - Hidden states accumulate information about oscillation patterns

2. COMPARISON TO L=1:
   - L=1: LSTM must rely ONLY on hidden state continuity between samples
   - L=50: LSTM sees context window + maintains hidden state
   - Result: Stronger temporal modeling with both local context and memory

3. OUTPUT HANDLING:
   - Model outputs sequence-to-sequence (50 inputs → 50 outputs)
   - Each timestep prediction uses context from previous steps in sequence
   - Maintains temporal coherence within sequences

4. PRACTICAL BENEFITS:
   - More efficient training (processes 50 samples at once)
   - Better gradient flow (BPTT over 50 steps, not just 1)
   - Natural sliding window inference for real-time applications

If L=1 were used, we would need explicit manual state management between
samples, which is pedagogically interesting but computationally inefficient.
""")

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

These will be computed during evaluation to assess model performance.
""")
