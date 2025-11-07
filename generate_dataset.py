"""
Step 1: Generate dataset with 4 frequencies and combined signal
IMPROVED VERSION - Learnable task with realistic noise:
- Time domain: 0-10 seconds
- Sampling rate: 1000 Hz (10,000 samples)
- Frequencies: f1=1Hz, f2=3Hz, f3=5Hz, f4=7Hz with FIXED phase offsets
- Phase offsets: 0°, 45°, 90°, 135° (creates realistic multi-frequency signal)
- Additive Gaussian noise for robustness: S(t) = (1/4) * Σ sin_i(t) + ε, ε ~ N(0, 0.1²)
- Two separate datasets with different noise realizations (Seed #1 for train, Seed #2 for test)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directories
Path("data").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

# Define the 4 frequencies (in Hz) with fixed phase offsets
freq1 = 1.0   # 1 Hz
freq2 = 3.0   # 3 Hz
freq3 = 5.0   # 5 Hz
freq4 = 7.0   # 7 Hz
frequencies = np.array([freq1, freq2, freq3, freq4])

# Fixed phase offsets (in radians) - creates realistic multi-frequency signal
# 0°, 45°, 90°, 135° = 0, π/4, π/2, 3π/4
phase1 = 0.0           # 0°
phase2 = np.pi / 4     # 45°
phase3 = np.pi / 2     # 90°
phase4 = 3 * np.pi / 4 # 135°
phases = np.array([phase1, phase2, phase3, phase4])

# Noise parameters
noise_std = 0.1  # Standard deviation for Gaussian noise (10% of signal range)

print("="*70)
print("IMPROVED DATASET GENERATION - Learnable Task")
print("="*70)
print(f"\nFrequencies (Hz): f1={freq1}, f2={freq2}, f3={freq3}, f4={freq4}")
print(f"Phase offsets (rad): {phases[0]:.4f}, {phases[1]:.4f}, {phases[2]:.4f}, {phases[3]:.4f}")
print(f"Phase offsets (deg): {np.degrees(phases[0]):.1f}°, {np.degrees(phases[1]):.1f}°, {np.degrees(phases[2]):.1f}°, {np.degrees(phases[3]):.1f}°")
print(f"Noise level (σ): {noise_std}")

# Time domain parameters
n_samples = 10000
x_start = 0.0
x_end = 10.0  # CORRECTED: 0-10 seconds (not 20)
sampling_rate = 1000  # CORRECTED: 1000 Hz
x_values = np.linspace(x_start, x_end, n_samples)

print(f"\nTime Domain:")
print(f"  Interval: [{x_start}, {x_end}] seconds")
print(f"  Samples: {n_samples}")
print(f"  Sampling Rate (Fs): {sampling_rate} Hz")

def generate_dataset(seed, dataset_name):
    """
    Generate a dataset with fixed phase offsets and additive Gaussian noise
    
    IMPROVED APPROACH:
    - Fixed phases: θ = [0°, 45°, 90°, 135°] - preserves frequency structure
    - Clean sinusoids: sin_i(t) = sin(2π·fi·t + θi)
    - Additive Gaussian noise: S(t) = (1/4) * Σ sin_i(t) + ε, where ε ~ N(0, σ²)
    - Different noise realization per seed for train/test generalization
    """
    np.random.seed(seed)
    
    print(f"\n{'-'*70}")
    print(f"Generating {dataset_name} (Seed #{seed})")
    print(f"{'-'*70}")
    
    # Initialize arrays for clean signals with fixed phases
    clean_signals = np.zeros((n_samples, 4))
    
    # Generate clean signals with FIXED phase offsets
    print("\nGenerating clean frequency components with fixed phases:")
    for i, (freq, phase) in enumerate(zip(frequencies, phases)):
        clean_signals[:, i] = np.sin(2 * np.pi * freq * x_values + phase)
        print(f"  f{i+1}(t) = sin(2π·{freq}·t + {np.degrees(phase):.1f}°)")
    
    # Generate PURE target signals (same as clean, no noise) - Ground Truth
    target1 = clean_signals[:, 0]
    target2 = clean_signals[:, 1]
    target3 = clean_signals[:, 2]
    target4 = clean_signals[:, 3]
    
    print("\nGround Truth Targets (identical to clean signals):")
    print(f"  Target_i(t) = sin(2π·fi·t + θi)")
    
    # Create clean mixed signal (average of all frequencies)
    S_clean = np.mean(clean_signals, axis=1)  # Equivalent to (1/4) * Σ
    
    # Add Gaussian noise to mixed signal
    noise = np.random.normal(0, noise_std, n_samples)
    S_noisy = S_clean + noise
    
    print(f"\nMixed Signal with Gaussian noise:")
    print(f"  S_clean(t) = (1/4) * Σ sin_i(t)")
    print(f"  S_noisy(t) = S_clean(t) + ε, where ε ~ N(0, {noise_std}²)")
    print(f"  S_clean range: [{S_clean.min():.4f}, {S_clean.max():.4f}]")
    print(f"  S_noisy range: [{S_noisy.min():.4f}, {S_noisy.max():.4f}]")
    print(f"  Noise std: {noise.std():.4f}")
    print(f"  SNR (Signal-to-Noise Ratio): {10 * np.log10(S_clean.var() / noise.var()):.2f} dB")
    
    return {
        'x': x_values,
        'S_noisy': S_noisy,
        'S_clean': S_clean,
        'noise': noise,
        'target1': target1,
        'target2': target2,
        'target3': target3,
        'target4': target4,
        'clean1': clean_signals[:, 0],
        'clean2': clean_signals[:, 1],
        'clean3': clean_signals[:, 2],
        'clean4': clean_signals[:, 3]
    }

# Generate TRAINING dataset (Seed #1)
train_data = generate_dataset(seed=1, dataset_name="TRAINING SET")

# Generate TEST dataset (Seed #2) - Same frequencies, COMPLETELY DIFFERENT NOISE
test_data = generate_dataset(seed=2, dataset_name="TEST SET")

# Save training dataset
print("\n" + "="*70)
print("SAVING DATASETS")
print("="*70)

np.savez('data/frequency_data_train.npz',
         x=train_data['x'],
         S=train_data['S_noisy'],
         S_clean=train_data['S_clean'],
         noise=train_data['noise'],
         target1=train_data['target1'],
         target2=train_data['target2'],
         target3=train_data['target3'],
         target4=train_data['target4'],
         clean1=train_data['clean1'],
         clean2=train_data['clean2'],
         clean3=train_data['clean3'],
         clean4=train_data['clean4'],
         frequencies=frequencies,
         phases=phases,
         noise_std=noise_std,
         sampling_rate=sampling_rate,
         seed=1)

print("\n✓ Training dataset saved: data/frequency_data_train.npz (Seed #1)")

# Save test dataset
np.savez('data/frequency_data_test.npz',
         x=test_data['x'],
         S=test_data['S_noisy'],
         S_clean=test_data['S_clean'],
         noise=test_data['noise'],
         target1=test_data['target1'],
         target2=test_data['target2'],
         target3=test_data['target3'],
         target4=test_data['target4'],
         clean1=test_data['clean1'],
         clean2=test_data['clean2'],
         clean3=test_data['clean3'],
         clean4=test_data['clean4'],
         frequencies=frequencies,
         phases=phases,
         noise_std=noise_std,
         sampling_rate=sampling_rate,
         seed=2)

print("✓ Test dataset saved: data/frequency_data_test.npz (Seed #2)")

# Create CSV for reference (using training data)
dataset_train = pd.DataFrame({
    'Sample': np.arange(n_samples),
    'Time(s)': train_data['x'],
    'S_noisy(t)': train_data['S_noisy'],
    'Target1(t)': train_data['target1'],
    'Target2(t)': train_data['target2'],
    'Target3(t)': train_data['target3'],
    'Target4(t)': train_data['target4']
})

dataset_train.to_csv('data/frequency_dataset_train.csv', index=False)
print("✓ Training CSV saved: data/frequency_dataset_train.csv")

# Also keep the old filename for compatibility with visualization
np.savez('data/frequency_data.npz',
         x=train_data['x'],
         f1=train_data['target1'],
         f2=train_data['target2'],
         f3=train_data['target3'],
         f4=train_data['target4'],
         S=train_data['S_noisy'],
         frequencies=frequencies,
         phases=np.array([0.0, 0.0, 0.0, 0.0]))  # No fixed phases anymore

print("✓ Compatibility file saved: data/frequency_data.npz")

# Statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)
print("\nTraining Set (Seed #1):")
print(f"  Samples: {len(train_data['x'])}")
print(f"  Time range: [{train_data['x'][0]:.3f}, {train_data['x'][-1]:.3f}] seconds")
print(f"  S_clean range: [{train_data['S_clean'].min():.4f}, {train_data['S_clean'].max():.4f}]")
print(f"  S_noisy range: [{train_data['S_noisy'].min():.4f}, {train_data['S_noisy'].max():.4f}]")
print(f"  Noise std: {train_data['noise'].std():.4f}")

print("\nTest Set (Seed #2):")
print(f"  Samples: {len(test_data['x'])}")
print(f"  S_clean range: [{test_data['S_clean'].min():.4f}, {test_data['S_clean'].max():.4f}]")
print(f"  S_noisy range: [{test_data['S_noisy'].min():.4f}, {test_data['S_noisy'].max():.4f}]")
print(f"  Noise std: {test_data['noise'].std():.4f}")

print("\nTarget Signals (same for both sets):")
for i in range(1, 5):
    print(f"  Target{i} = sin(2π·{frequencies[i-1]}·t + {np.degrees(phases[i-1]):.1f}°)")

print("\n" + "="*70)
print("IMPROVEMENTS IMPLEMENTED:")
print("="*70)
print("✓ Fixed phase offsets: 0°, 45°, 90°, 135° (realistic multi-frequency signal)")
print("✓ Additive Gaussian noise: σ = 0.1 (preserves frequency structure)")
print("✓ Learnable task: Model can now extract frequencies from noisy input")
print("✓ Separate noise realizations: Seed #1 (train), Seed #2 (test)")
print("✓ Maintains challenge: Noise + overlapping frequencies + selector mechanism")
print("="*70)

print("\n✅ Dataset generation complete - Improved learnable version!")
