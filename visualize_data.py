"""
Step 2: Create visualizations for frequencies and signals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
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
phases = data['phases']

print(f"Loaded data with {len(x)} samples")
print(f"Frequencies: {frequencies}")
print(f"Phases (radians): {phases}")
print(f"Phases (degrees): {np.degrees(phases)}")

# Create visualization directory
Path("visualizations").mkdir(exist_ok=True)

# ============================================================================
# Visualization 1: Individual Frequencies in Time Domain
# ============================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
fig.suptitle('Time Domain - Phase-Shifted Frequencies and Combined Signal', fontsize=16)

# Plot only first 2 seconds for clarity
plot_range = (x >= 0) & (x <= 2)
x_plot = x[plot_range]

# Individual frequencies with phase information
axes[0].plot(x_plot, f1[plot_range], 'b-', linewidth=1.5, 
            label=f'f1(x) = sin(2π·{frequencies[0]}·x + {phases[0]:.3f}), θ={np.degrees(phases[0]):.0f}°')
axes[0].set_ylabel('f1(x)', fontsize=10)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_plot, f2[plot_range], 'g-', linewidth=1.5, 
            label=f'f2(x) = sin(2π·{frequencies[1]}·x + {phases[1]:.3f}), θ={np.degrees(phases[1]):.0f}°')
axes[1].set_ylabel('f2(x)', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_plot, f3[plot_range], 'r-', linewidth=1.5, 
            label=f'f3(x) = sin(2π·{frequencies[2]}·x + {phases[2]:.3f}), θ={np.degrees(phases[2]):.0f}°')
axes[2].set_ylabel('f3(x)', fontsize=10)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

axes[3].plot(x_plot, f4[plot_range], 'm-', linewidth=1.5, 
            label=f'f4(x) = sin(2π·{frequencies[3]}·x + {phases[3]:.3f}), θ={np.degrees(phases[3]):.0f}°')
axes[3].set_ylabel('f4(x)', fontsize=10)
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

# Combined signal
axes[4].plot(x_plot, S[plot_range], 'k-', linewidth=1.5, label='S(x) = f1 + f2 + f3 + f4')
axes[4].set_ylabel('S(x)', fontsize=10)
axes[4].set_xlabel('Time (seconds)', fontsize=12)
axes[4].legend(loc='upper right')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/01_time_domain_signals.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/01_time_domain_signals.png")
plt.close()

# ============================================================================
# Visualization 2: Frequency Domain (FFT Analysis)
# ============================================================================
# Compute FFT for each signal
sampling_rate = len(x) / (x[-1] - x[0])  # samples per second
fft_S = np.fft.fft(S)
fft_f1 = np.fft.fft(f1)
fft_f2 = np.fft.fft(f2)
fft_f3 = np.fft.fft(f3)
fft_f4 = np.fft.fft(f4)

# Frequency axis
freqs = np.fft.fftfreq(len(x), 1/sampling_rate)

# Only plot positive frequencies
positive_freqs = freqs[:len(freqs)//2]

fig, axes = plt.subplots(5, 1, figsize=(14, 12))
fig.suptitle('Frequency Domain (FFT Analysis)', fontsize=16)

# Individual frequencies
axes[0].plot(positive_freqs, 2.0/len(x) * np.abs(fft_f1[:len(freqs)//2]), 'b-', linewidth=1.5)
axes[0].set_ylabel('f1 Magnitude', fontsize=10)
axes[0].set_xlim([0, 10])
axes[0].grid(True, alpha=0.3)
axes[0].axvline(frequencies[0], color='b', linestyle='--', alpha=0.5, label=f'{frequencies[0]} Hz')
axes[0].legend()

axes[1].plot(positive_freqs, 2.0/len(x) * np.abs(fft_f2[:len(freqs)//2]), 'g-', linewidth=1.5)
axes[1].set_ylabel('f2 Magnitude', fontsize=10)
axes[1].set_xlim([0, 10])
axes[1].grid(True, alpha=0.3)
axes[1].axvline(frequencies[1], color='g', linestyle='--', alpha=0.5, label=f'{frequencies[1]} Hz')
axes[1].legend()

axes[2].plot(positive_freqs, 2.0/len(x) * np.abs(fft_f3[:len(freqs)//2]), 'r-', linewidth=1.5)
axes[2].set_ylabel('f3 Magnitude', fontsize=10)
axes[2].set_xlim([0, 10])
axes[2].grid(True, alpha=0.3)
axes[2].axvline(frequencies[2], color='r', linestyle='--', alpha=0.5, label=f'{frequencies[2]} Hz')
axes[2].legend()

axes[3].plot(positive_freqs, 2.0/len(x) * np.abs(fft_f4[:len(freqs)//2]), 'm-', linewidth=1.5)
axes[3].set_ylabel('f4 Magnitude', fontsize=10)
axes[3].set_xlim([0, 10])
axes[3].grid(True, alpha=0.3)
axes[3].axvline(frequencies[3], color='m', linestyle='--', alpha=0.5, label=f'{frequencies[3]} Hz')
axes[3].legend()

# Combined signal - shows all 4 peaks
axes[4].plot(positive_freqs, 2.0/len(x) * np.abs(fft_S[:len(freqs)//2]), 'k-', linewidth=1.5)
axes[4].set_ylabel('S Magnitude', fontsize=10)
axes[4].set_xlabel('Frequency (Hz)', fontsize=12)
axes[4].set_xlim([0, 10])
axes[4].grid(True, alpha=0.3)
for i, freq in enumerate(frequencies):
    axes[4].axvline(freq, color=['b', 'g', 'r', 'm'][i], linestyle='--', alpha=0.5)
axes[4].legend(['Combined Signal S(x)', f'{frequencies[0]} Hz', f'{frequencies[1]} Hz', 
                f'{frequencies[2]} Hz', f'{frequencies[3]} Hz'])

plt.tight_layout()
plt.savefig('visualizations/02_frequency_domain_fft.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/02_frequency_domain_fft.png")
plt.close()

# ============================================================================
# Visualization 3: Spectrogram of Combined Signal
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
f_spec, t_spec, Sxx = scipy_signal.spectrogram(S, sampling_rate, nperseg=1024)
im = ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
ax.set_ylabel('Frequency (Hz)', fontsize=12)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_title('Spectrogram of Combined Signal S(x)', fontsize=14)
ax.set_ylim([0, 10])

# Mark the frequency components
for i, freq in enumerate(frequencies):
    ax.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'f{i+1} = {freq} Hz' if i == 0 else f'{freq} Hz')

ax.legend(loc='upper right')
plt.colorbar(im, ax=ax, label='Power (dB)')
plt.tight_layout()
plt.savefig('visualizations/03_spectrogram.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/03_spectrogram.png")
plt.close()

# ============================================================================
# Visualization 4: All Frequencies Overlaid
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(x_plot, f1[plot_range], 'b-', linewidth=1.5, alpha=0.7, 
        label=f'f1 ({frequencies[0]} Hz, θ={np.degrees(phases[0]):.0f}°)')
ax.plot(x_plot, f2[plot_range], 'g-', linewidth=1.5, alpha=0.7, 
        label=f'f2 ({frequencies[1]} Hz, θ={np.degrees(phases[1]):.0f}°)')
ax.plot(x_plot, f3[plot_range], 'r-', linewidth=1.5, alpha=0.7, 
        label=f'f3 ({frequencies[2]} Hz, θ={np.degrees(phases[2]):.0f}°)')
ax.plot(x_plot, f4[plot_range], 'm-', linewidth=1.5, alpha=0.7, 
        label=f'f4 ({frequencies[3]} Hz, θ={np.degrees(phases[3]):.0f}°)')
ax.plot(x_plot, S[plot_range], 'k-', linewidth=2, alpha=0.8, label='S(x) (Combined)')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Overlay of Phase-Shifted Frequencies and Combined Signal', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/04_overlay_signals.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/04_overlay_signals.png")
plt.close()

print("\nAll visualizations created successfully!")
print(f"\nPhase shifts used:")
for i, (freq, phase) in enumerate(zip(frequencies, phases)):
    print(f"  f{i+1} ({freq} Hz): θ = {phase:.4f} rad ({np.degrees(phase):.1f}°)")
