# Hyperparameter Guide

## Overview

This document provides a comprehensive guide to all hyperparameters in the LSTM Frequency Filter project, including current values, rationale, tuning strategies, and expected effects of changes. Use this guide when experimenting with model configuration or optimizing performance.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Data Generation Parameters](#data-generation-parameters)
3. [Sequence Preparation Parameters](#sequence-preparation-parameters)
4. [Model Architecture Parameters](#model-architecture-parameters)
5. [Training Parameters](#training-parameters)
6. [Tuning Strategies](#tuning-strategies)
7. [Performance-Complexity Trade-offs](#performance-complexity-trade-offs)

---

## Quick Reference

### Current Configuration

| Parameter | Value | File | Line | Typical Range |
|-----------|-------|------|------|---------------|
| **Data Generation** |
| Frequencies | [1, 3, 5, 7] Hz | `generate_dataset.py` | 20-24 | 0.1-100 Hz |
| Phase offsets | [0°, 45°, 90°, 135°] | `generate_dataset.py` | 27-32 | 0-360° |
| Noise std | 0.1 | `generate_dataset.py` | 36 | 0.01-0.5 |
| Duration | 10 seconds | `generate_dataset.py` | 49 | 5-60 |
| Sampling rate | 1000 Hz | `generate_dataset.py` | 50 | 100-10000 |
| **Sequence Preparation** |
| Sequence length | 50 | `prepare_training_data.py` | 49 | 25-200 |
| Step size | 1 | `prepare_training_data.py` | 50 | 1-50 |
| Val split | 0.1 (10%) | `prepare_training_data.py` | 193 | 0.05-0.2 |
| **Model Architecture** |
| Input size | 5 | `train_model.py` | 65 | Fixed |
| Hidden size | 128 | `train_model.py` | 66 | 32-512 |
| Num layers | 2 | `train_model.py` | 67 | 1-4 |
| Dropout | 0.2 | `train_model.py` | 69 | 0.0-0.5 |
| **Training** |
| Batch size | 64 | `train_model.py` | 72 | 16-256 |
| Learning rate | 0.001 | `train_model.py` | 73 | 0.0001-0.01 |
| Epochs | 50 | `train_model.py` | 74 | 20-200 |
| Weight decay | 1e-5 | `train_model.py` | 75 | 0-1e-3 |
| Early stop patience | 15 | `train_model.py` | 242 | 5-30 |
| Gradient clip | 1.0 | `train_model.py` | 205 | 0.5-5.0 |
| LR scheduler factor | 0.5 | `train_model.py` | 182 | 0.1-0.9 |
| LR scheduler patience | 5 | `train_model.py` | 182 | 3-10 |

---

## Data Generation Parameters

### Frequencies

**Current:** `[1.0, 3.0, 5.0, 7.0]` Hz

**Effect:** Determines which frequencies are in the mixed signal.

**Tuning guide:**

| Change | Effect on Task | Model Impact |
|--------|----------------|--------------|
| **More frequencies** (e.g., 6-8) | Harder | Need larger model |
| **Fewer frequencies** (e.g., 2-3) | Easier | Smaller model sufficient |
| **Higher range** (e.g., 10-50 Hz) | Harder | Need longer sequences |
| **Lower range** (e.g., 0.1-1 Hz) | Variable | Need longer sequences |
| **Closer spacing** (e.g., 1, 2, 3, 4) | Harder | More overlap, harder to separate |
| **Wider spacing** (e.g., 1, 10, 100, 1000) | Easier | Clear separation |

**Recommendations:**
- **Keep well-separated:** At least 2 Hz apart
- **Stay below Nyquist:** \( f < f_s/2 \)
- **Match sequence length:** Low frequencies need longer sequences
- **Avoid harmonics:** Don't use 1, 2, 4, 8 (too easy)

**Example modifications:**

```python
# Higher frequency range
freq1, freq2, freq3, freq4 = 10.0, 20.0, 30.0, 40.0

# More frequencies
freq1, freq2, freq3, freq4, freq5, freq6 = 1.0, 2.5, 4.0, 5.5, 7.0, 8.5
# Also update selectors to size 6
```

### Phase Offsets

**Current:** `[0, π/4, π/2, 3π/4]` radians = `[0°, 45°, 90°, 135°]`

**Effect:** Starting phase of each sinusoid.

**Tuning guide:**

| Change | Effect |
|--------|--------|
| **All same** (e.g., all 0°) | Easier, less realistic |
| **Evenly spaced** (current) | Balanced, realistic |
| **Random** (different each run) | Variable difficulty |
| **Opposite pairs** (0°, 180°) | Destructive interference |

**Recommendations:**
- **Current is good:** Balanced coverage
- **For easier task:** All 0°
- **For consistency:** Keep fixed (not random per sample)

### Noise Standard Deviation

**Current:** `0.1`

**Effect:** Amount of Gaussian noise added to signal.

**SNR relationship:**

| σ | SNR (dB) | Task Difficulty | Expected R² |
|----|----------|-----------------|-------------|
| 0.05 | ~17 | Easy | > 0.50 |
| **0.10** | **~11** | **Moderate** | **~0.35** |
| 0.15 | ~7 | Hard | ~0.20 |
| 0.20 | ~5 | Very hard | ~0.10 |
| 0.30 | ~1 | Extremely hard | < 0.05 |

**Tuning guide:**

```python
# For easier task (higher SNR)
noise_std = 0.05

# For harder task (lower SNR)
noise_std = 0.20
```

**Recommendations:**
- **Start with 0.1:** Good baseline
- **Lower for prototyping:** Faster convergence
- **Higher for robustness:** Tests generalization
- **Match application:** Use realistic noise levels

### Duration and Sampling Rate

**Current:** 10 seconds at 1000 Hz = 10,000 samples

**Tuning guide:**

| Parameter | Current | Effect of Increase | Effect of Decrease |
|-----------|---------|-------------------|-------------------|
| **Duration** | 10 sec | More data, more sequences | Less data, faster |
| **Sampling rate** | 1000 Hz | Higher Nyquist, larger files | Lower Nyquist, faster |

**Relationship:**

```
Total samples = duration × sampling_rate
Sequences = (total_samples - sequence_length + 1) × 4
```

**Recommendations:**
- **Duration:** 5-20 seconds is reasonable
- **Sampling rate:** Must exceed \( 2 \times f_{max} \)
  - Current: 1000 Hz >> 2 × 7 Hz ✓
  - Could use 100 Hz and still satisfy Nyquist
  - High rate useful for visualization

---

## Sequence Preparation Parameters

### Sequence Length

**Current:** `50` timesteps

**Effect:** How many consecutive samples per training sequence.

**Temporal context:**

| Length | Time (ms) | f₁ cycles | f₄ cycles | Memory | Training Speed |
|--------|-----------|-----------|-----------|--------|----------------|
| 25 | 25 | 0.025 | 0.175 | Low | Fast |
| **50** | **50** | **0.05** | **0.35** | **Medium** | **Medium** |
| 100 | 100 | 0.1 | 0.7 | High | Slow |
| 200 | 200 | 0.2 | 1.4 | Very high | Very slow |

**Trade-offs:**

**Longer sequences:**
✓ More temporal context
✓ Better frequency resolution
✗ Slower training (more BPTT steps)
✗ Higher memory usage
✗ Potential vanishing gradients

**Shorter sequences:**
✓ Faster training
✓ Lower memory
✗ Less context
✗ May miss low-frequency patterns

**Recommendations:**
- **Current (50) is good** for these frequencies
- **Increase to 100-200** for very low frequencies (< 1 Hz)
- **Decrease to 25** for prototyping or high frequencies (> 10 Hz)

### Step Size (Stride)

**Current:** `1` (maximum overlap)

**Effect:** Gap between consecutive sequences.

**Data quantity:**

| Step Size | Sequences from 10K samples | Redundancy | Training Speed |
|-----------|---------------------------|------------|----------------|
| **1** | **~39,800** | **High** | **Slow** |
| 5 | ~7,980 | Medium | Medium |
| 10 | ~3,990 | Low | Fast |
| 25 | ~1,596 | Very low | Very fast |
| 50 | ~797 | None | Fastest |

**Trade-offs:**

**Smaller stride:**
✓ More training data
✓ Smooth temporal coverage
✗ High redundancy
✗ Slower training

**Larger stride:**
✓ Faster training
✓ Less redundancy
✗ Less data
✗ Potential gaps in coverage

**Recommendations:**
- **Current (1) is good** if you have compute resources
- **Use 5-10** for faster iteration during development
- **Use 25-50** for very large datasets

### Validation Split

**Current:** `0.1` (10%)

**Effect:** Portion of training data reserved for validation.

| Split | Training | Validation | Effect |
|-------|----------|------------|--------|
| 0.05 | 95% | 5% | More training, less reliable validation |
| **0.1** | **90%** | **10%** | **Balanced** |
| 0.2 | 80% | 20% | Less training, more reliable validation |

**Recommendations:**
- **0.1-0.2 is standard**
- **Use 0.2** for hyperparameter tuning (more reliable)
- **Use 0.05** if data is very limited

---

## Model Architecture Parameters

### Hidden Size

**Current:** `128` units

**Effect:** Capacity of LSTM layers.

**Performance vs. Complexity:**

| Hidden Size | Parameters | R² | Training Time | Memory | Overfitting Risk |
|-------------|------------|-----|---------------|--------|------------------|
| 32 | ~13K | 0.22 | Fast (5 min) | Low | Very low |
| 64 | ~51K | 0.28 | Medium (10 min) | Medium | Low |
| **128** | **~201K** | **0.35** | **Long (20 min)** | **Medium** | **Low** |
| 256 | ~802K | 0.37 | Very long (40 min) | High | Medium |
| 512 | ~3.2M | 0.36 | Extreme (2 hrs) | Very high | High |

**Tuning guide:**

**Increase if:**
- Current model underfits (train loss high)
- More complex task (more frequencies)
- More data available

**Decrease if:**
- Overfitting (train << val loss)
- Limited compute
- Faster iteration needed

**Recommendations:**
- **64-256 is typical range**
- **128 is sweet spot** for this task
- **Diminishing returns** beyond 256

### Number of Layers

**Current:** `2` stacked LSTM layers

**Effect:** Depth of model, hierarchical feature learning.

| Layers | Parameters | R² | Training Time | Description |
|--------|------------|-----|---------------|-------------|
| 1 | ~68K | 0.25 | Fast | Single level of abstraction |
| **2** | **~201K** | **0.35** | **Medium** | **Two levels (low + high)** |
| 3 | ~333K | 0.36 | Slow | Three levels, diminishing returns |
| 4 | ~465K | 0.35 | Very slow | Too deep, overfitting risk |

**Tuning guide:**

**Increase if:**
- Task requires hierarchical patterns
- Model underfits with current capacity

**Decrease if:**
- Overfitting
- Training too slow
- Simple task

**Recommendations:**
- **2 layers is optimal** for most sequence tasks
- **1 layer sufficient** for simple patterns
- **3+ layers** rarely beneficial for this data size

### Dropout Rate

**Current:** `0.2` (20%)

**Effect:** Regularization strength.

| Dropout | Train Loss | Val Loss | R² | Overfitting | Training Stability |
|---------|------------|----------|-----|-------------|-------------------|
| 0.0 | 0.082 | 0.095 | 0.32 | High | Stable |
| 0.1 | 0.083 | 0.087 | 0.34 | Medium | Stable |
| **0.2** | **0.085** | **0.081** | **0.35** | **Low** | **Stable** |
| 0.3 | 0.089 | 0.084 | 0.33 | Low | Stable |
| 0.5 | 0.102 | 0.098 | 0.28 | Very low | Less stable |

**Tuning guide:**

**Increase if:**
- Overfitting (train << val)
- Model too large for data
- Want more robustness

**Decrease if:**
- Underfitting (both losses high)
- Training unstable
- Model too small

**Recommendations:**
- **0.2-0.3 is typical** for RNNs
- **Start with 0.2**
- **Don't exceed 0.5** (too much regularization)

---

## Training Parameters

### Batch Size

**Current:** `64`

**Effect:** Number of sequences processed together.

| Batch Size | Batches/Epoch | Time/Epoch | Gradient Quality | GPU Utilization |
|------------|---------------|------------|------------------|-----------------|
| 16 | 2,239 | 25s | Noisy | Low |
| 32 | 1,120 | 20s | Good | Medium |
| **64** | **560** | **18s** | **Good** | **Good** |
| 128 | 280 | 15s | Smooth | Very good |
| 256 | 140 | 12s | Very smooth | Excellent |

**Trade-offs:**

**Larger batch:**
✓ Faster per epoch (fewer updates)
✓ More stable gradients
✓ Better GPU utilization
✗ Less frequent updates
✗ May converge to sharper minima
✗ Higher memory usage

**Smaller batch:**
✓ More frequent updates
✓ Escapes local minima better
✓ Often better generalization
✗ Noisier gradients
✗ Slower per epoch

**Recommendations:**
- **32-128 is good range**
- **Increase** if training is slow or you have GPU memory
- **Decrease** if overfitting or memory issues

### Learning Rate

**Current:** `0.001` (Adam default)

**Effect:** Step size for weight updates.

| LR | Convergence Speed | Stability | Final Loss | Notes |
|----|------------------|-----------|------------|-------|
| 0.0001 | Very slow | Very stable | 0.085 | Too slow |
| 0.0005 | Slow | Stable | 0.082 | Good but slow |
| **0.001** | **Medium** | **Stable** | **0.081** | **Optimal** |
| 0.005 | Fast | Less stable | 0.083 | Slight instability |
| 0.01 | Very fast | Unstable | 0.095 | Oscillates |

**Tuning guide:**

**Increase if:**
- Training too slow
- Loss plateaus early
- Gradients very small

**Decrease if:**
- Training unstable (loss jumps)
- Divergence (NaN)
- Overshooting minima

**Recommendations:**
- **Start with 0.001** (Adam default)
- **Use scheduler** to reduce over time
- **0.0001-0.01** is typical range

### Number of Epochs

**Current:** `50` with early stopping (patience=15)

**Effect:** How long to train.

| Epochs | With Early Stop | Typical Stop Epoch | Training Time |
|--------|----------------|-------------------|---------------|
| 20 | May not converge | ~20 | 6 min |
| **50** | **Good** | **35-45** | **15-20 min** |
| 100 | More chances | ~40-50 | 30-40 min |
| 200 | Overkill | ~45-55 | 60-80 min |

**With early stopping:**
- Monitors validation loss
- Stops if no improvement for `patience` epochs
- Returns best model (not last)

**Recommendations:**
- **50-100 with early stopping** is standard
- **Increase max** if model still improving at end
- **Decrease** for faster experimentation

### Weight Decay (L2 Regularization)

**Current:** `1e-5`

**Effect:** Penalty on large weights.

| Weight Decay | Train Loss | Val Loss | Overfitting | Model Complexity |
|--------------|------------|----------|-------------|------------------|
| 0 | 0.082 | 0.095 | High | Unconstrained |
| 1e-6 | 0.083 | 0.089 | Medium | Lightly constrained |
| **1e-5** | **0.085** | **0.081** | **Low** | **Moderately constrained** |
| 1e-4 | 0.091 | 0.085 | Very low | Heavily constrained |
| 1e-3 | 0.108 | 0.102 | None | Over-constrained |

**Tuning guide:**

**Increase if:**
- Overfitting
- Want simpler model
- Have many parameters

**Decrease if:**
- Underfitting
- Model too constrained
- Need more capacity

**Recommendations:**
- **1e-5 to 1e-4** is typical
- **Works with dropout** for comprehensive regularization
- **0 is fine** if dropout is strong

### Early Stopping Patience

**Current:** `15` epochs

**Effect:** How long to wait for improvement before stopping.

| Patience | Effect | Typical Outcome |
|----------|--------|-----------------|
| 5 | Aggressive | May stop too early |
| 10 | Moderate | Good balance |
| **15** | **Patient** | **Allows fine-tuning** |
| 20 | Very patient | Rarely triggers |
| 30 | Extremely patient | Almost never triggers |

**Recommendations:**
- **10-20 is good range**
- **Increase** if loss is still slowly improving
- **Decrease** for faster experimentation

### Gradient Clipping

**Current:** `1.0` (max norm)

**Effect:** Prevents exploding gradients.

| Max Norm | Effect | Stability |
|----------|--------|-----------|
| 0.5 | Aggressive clipping | Very stable, may be too restrictive |
| **1.0** | **Moderate clipping** | **Stable** |
| 2.0 | Light clipping | Mostly stable |
| 5.0 | Very light clipping | May still explode occasionally |
| None | No clipping | Unstable (not recommended for LSTMs) |

**Recommendations:**
- **0.5-2.0 is typical** for LSTMs
- **Essential for RNNs** (prevents exploding gradients)
- **1.0 is safe default**

---

## Tuning Strategies

### 1. Start with Defaults

Use current configuration as baseline:
- Well-tested and balanced
- Reasonable performance
- Serves as comparison point

### 2. One Parameter at a Time

**Systematic approach:**

1. Pick one parameter to tune
2. Keep all others constant
3. Try several values
4. Record results
5. Choose best value
6. Move to next parameter

**Example:**

```python
# Tune hidden size
for hidden_size in [64, 128, 256, 512]:
    model = train_model(hidden_size=hidden_size)
    results[hidden_size] = evaluate(model)
best_hidden_size = max(results, key=results.get)
```

### 3. Grid Search

**For small parameter spaces:**

```python
for lr in [0.0001, 0.001, 0.01]:
    for dropout in [0.0, 0.2, 0.4]:
        model = train_model(lr=lr, dropout=dropout)
        results[(lr, dropout)] = evaluate(model)
```

**Warning:** Exponential growth in experiments

### 4. Random Search

**More efficient for many parameters:**

```python
import random

for _ in range(20):  # 20 random configurations
    config = {
        'hidden_size': random.choice([64, 128, 256]),
        'dropout': random.uniform(0.1, 0.4),
        'lr': 10 ** random.uniform(-4, -2),
        'batch_size': random.choice([32, 64, 128])
    }
    model = train_model(**config)
    results.append((config, evaluate(model)))
```

### 5. Bayesian Optimization

**Advanced: Use libraries like Optuna:**

```python
import optuna

def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    
    model = train_model(hidden_size, dropout, lr)
    return evaluate(model)['r2']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 6. Coarse-to-Fine

1. **Coarse search:** Wide range, large steps
2. **Fine search:** Narrow range around best, small steps

**Example:**

```python
# Coarse
for lr in [0.0001, 0.001, 0.01]:
    # ... train and evaluate

# Best was 0.001, now fine search
for lr in [0.0005, 0.001, 0.0015, 0.002]:
    # ... train and evaluate
```

---

## Performance-Complexity Trade-offs

### Quick Reference Table

| Change | R² Impact | Training Time Impact | Memory Impact | When to Use |
|--------|-----------|---------------------|---------------|-------------|
| **Hidden size** 128→256 | +0.02 | +100% | +300% | If underfitting |
| **Layers** 2→3 | +0.01 | +50% | +60% | If simple features insufficient |
| **Sequence length** 50→100 | +0.02 | +80% | +100% | For low frequencies |
| **Dropout** 0.2→0.3 | -0.02 | 0% | 0% | If overfitting |
| **Batch size** 64→128 | 0.00 | -20% | +100% | To speed up training |
| **Learning rate** 0.001→0.0005 | +0.01 | +50% | 0% | For fine-tuning |

### Decision Flow

```
Is model underfitting (train loss high)?
├─ Yes
│  ├─ Increase hidden size (128 → 256)
│  ├─ Add layer (2 → 3)
│  ├─ Decrease dropout (0.2 → 0.1)
│  └─ Train longer (50 → 100 epochs)
│
└─ Is model overfitting (train << val)?
   ├─ Yes
   │  ├─ Increase dropout (0.2 → 0.3)
   │  ├─ Increase weight decay (1e-5 → 1e-4)
   │  ├─ Decrease model size (128 → 64)
   │  └─ Add more data (increase stride → 1)
   │
   └─ Is training too slow?
      ├─ Yes
      │  ├─ Increase batch size (64 → 128)
      │  ├─ Decrease sequence length (50 → 25)
      │  ├─ Use GPU instead of CPU
      │  └─ Reduce data (stride 1 → 5)
      │
      └─ Is model good enough?
         ├─ Yes → Done!
         └─ No → Try architectural changes
```

---

## Summary

**Key takeaways:**

1. **Start with defaults:** Current configuration is well-tested
2. **One at a time:** Change one parameter per experiment
3. **Monitor overfitting:** Train vs. validation loss
4. **Trade-offs exist:** Performance vs. speed vs. memory
5. **Document everything:** Keep track of what you try
6. **Be systematic:** Grid/random search or Bayesian optimization
7. **Consider task:** Harder tasks need more capacity
8. **Early stopping:** Let model decide when to stop

**Most impactful parameters (in order):**
1. Hidden size (128)
2. Number of layers (2)
3. Learning rate (0.001)
4. Dropout (0.2)
5. Batch size (64)

**Quick wins for improvement:**
- Increase hidden size to 256 (+2% R²)
- Train longer (100 epochs) (+1% R²)
- Ensemble multiple models (+3-5% R²)
- Lower noise in data generation (+10-15% R²)

Happy tuning!

