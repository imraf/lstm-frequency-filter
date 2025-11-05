"""
Step 1: Generate dataset with 4 frequencies and combined signal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directories
Path("data").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define the 4 frequencies (in Hz)
# Using distinct frequencies to make them easier to separate
freq1 = 1.0   # 1 Hz
freq2 = 3.0   # 3 Hz
freq3 = 5.0   # 5 Hz
freq4 = 7.0   # 7 Hz

print(f"Frequencies:")
print(f"  f1 = {freq1} Hz")
print(f"  f2 = {freq2} Hz")
print(f"  f3 = {freq3} Hz")
print(f"  f4 = {freq4} Hz")

# Sample 10,000 values over an interval
# Using interval [0, 20] seconds to capture multiple periods of all frequencies
n_samples = 10000
x_start = 0.0
x_end = 20.0
x_values = np.linspace(x_start, x_end, n_samples)

print(f"\nSampling {n_samples} points over interval [{x_start}, {x_end}]")

# Calculate individual frequency signals
# f_i(x) = sin(2*pi*freq_i*x)
f1_x = np.sin(2 * np.pi * freq1 * x_values)
f2_x = np.sin(2 * np.pi * freq2 * x_values)
f3_x = np.sin(2 * np.pi * freq3 * x_values)
f4_x = np.sin(2 * np.pi * freq4 * x_values)

# Calculate combined signal S(x)
S_x = f1_x + f2_x + f3_x + f4_x

# Create dataset table
dataset = pd.DataFrame({
    'Sample': np.arange(n_samples),
    'X': x_values,
    'f1(x)': f1_x,
    'f2(x)': f2_x,
    'f3(x)': f3_x,
    'f4(x)': f4_x,
    'S(x)': S_x
})

# Save dataset
dataset.to_csv('data/frequency_dataset.csv', index=False)
print(f"\nDataset saved to data/frequency_dataset.csv")
print(f"Dataset shape: {dataset.shape}")
print(f"\nFirst few rows:")
print(dataset.head(10))
print(f"\nDataset statistics:")
print(dataset.describe())

# Save numpy arrays for easier loading during training
np.savez('data/frequency_data.npz', 
         x=x_values,
         f1=f1_x,
         f2=f2_x,
         f3=f3_x,
         f4=f4_x,
         S=S_x,
         frequencies=np.array([freq1, freq2, freq3, freq4]))

print(f"\nNumpy arrays saved to data/frequency_data.npz")
