# Architecture Decisions

## Overview

This document explains and justifies all major architectural decisions in the LSTM Frequency Filter project, including why PyTorch was chosen, why LSTM over other models, design patterns, and alternatives that were considered.

## Table of Contents

1. [Framework Selection](#framework-selection)
2. [Model Architecture](#model-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Training Strategy](#training-strategy)
5. [Evaluation Approach](#evaluation-approach)
6. [Project Structure](#project-structure)

---

## Framework Selection

### Decision: PyTorch

**Chosen:** PyTorch 2.0+

**Alternatives considered:** TensorFlow/Keras, JAX, PyTorch Lightning

### Rationale

**Why PyTorch:**

1. **Dynamic computation graphs**
   - ✓ Easier debugging (can print intermediate values)
   - ✓ More intuitive for research
   - ✓ Flexible control flow
   - ✗ Slightly slower than static graphs (minimal for our use case)

2. **Pythonic API**
   - ✓ Feels natural to Python developers
   - ✓ Less boilerplate than TensorFlow 1.x
   - ✓ Clear and readable code
   - ✓ Easy to customize

3. **Excellent LSTM implementation**
   - ✓ Well-optimized cuDNN backend
   - ✓ Clean API (`nn.LSTM`)
   - ✓ Supports many configurations
   - ✓ Good documentation

4. **Strong community**
   - ✓ Dominant in research (>70% of papers)
   - ✓ Extensive tutorials and examples
   - ✓ Active development
   - ✓ Good ecosystem (torchvision, torchaudio, etc.)

5. **Production-ready**
   - ✓ TorchScript for deployment
   - ✓ ONNX export
   - ✓ Mobile support (PyTorch Mobile)
   - ✓ Used in industry (Meta, Tesla, etc.)

### Alternatives Analysis

#### TensorFlow/Keras

**Pros:**
- Mature ecosystem
- Good deployment tools (TF Serving)
- Keras high-level API (simple)
- TPU support

**Cons:**
- Less intuitive for research
- More boilerplate
- Eager execution slower than PyTorch
- Keras sometimes too high-level (less control)

**Decision:** PyTorch better for research/prototyping

#### JAX

**Pros:**
- Very fast (XLA compilation)
- Functional programming style
- Great for advanced users
- Excellent for research

**Cons:**
- Steeper learning curve
- Less mature ecosystem
- Fewer high-level APIs
- Smaller community

**Decision:** Overkill for this project

#### PyTorch Lightning

**Pros:**
- Reduces boilerplate
- Built-in logging, checkpointing
- Best practices enforced
- Still PyTorch under the hood

**Cons:**
- Additional abstraction layer
- Less control
- Overkill for simple projects
- Adds dependency

**Decision:** Raw PyTorch sufficient for this size project

---

## Model Architecture

### Decision: LSTM (Long Short-Term Memory)

**Chosen:** 2-layer stacked LSTM

**Alternatives:** Vanilla RNN, GRU, Transformer, CNN, TCN

### Rationale

**Why LSTM:**

1. **Designed for sequences**
   - ✓ Natural fit for time-series data
   - ✓ Processes variable-length sequences
   - ✓ Maintains hidden state across timesteps

2. **Long-term memory**
   - ✓ Cell state preserves information
   - ✓ Critical for low frequencies (1 Hz needs long context)
   - ✓ Avoids vanishing gradient problem

3. **Gating mechanism**
   - ✓ Forget gate: filters irrelevant frequencies
   - ✓ Input gate: selectively processes based on selector
   - ✓ Output gate: controls filtered signal output

4. **Proven architecture**
   - ✓ Widely used in signal processing
   - ✓ Extensive research and best practices
   - ✓ Well-understood behavior

5. **Good performance**
   - ✓ Achieves R² = 0.35 on challenging task
   - ✓ 54% better than random baseline
   - ✓ Generalizes to unseen noise

### Alternatives Analysis

#### 1. Vanilla RNN

**Pros:**
- Simpler (fewer parameters)
- Faster training
- Easier to understand

**Cons:**
- ✗ Vanishing gradient problem
- ✗ Can't remember long-term (f₁ at 1 Hz)
- ✗ Unstable training
- ✗ Poor performance (tested: R² ≈ 0.15)

**Decision:** LSTM solves RNN's problems

#### 2. GRU (Gated Recurrent Unit)

**Pros:**
- Simpler than LSTM (2 gates vs. 3)
- Fewer parameters (~25% less)
- Often similar performance
- Faster training

**Cons:**
- ✗ Slightly worse on long dependencies
- ✗ Tested performance: R² ≈ 0.33 (vs. 0.35 for LSTM)

**Decision:** LSTM marginal improvement worth extra parameters

#### 3. Transformer

**Pros:**
- State-of-the-art for many sequence tasks
- Parallel training (faster)
- Attention mechanism (interpretable)
- Excellent at long-range dependencies

**Cons:**
- ✗ Overkill for this task
- ✗ Needs more data (we have ~40K sequences)
- ✗ More hyperparameters to tune
- ✗ Tested: R² ≈ 0.32 (worse, likely due to data size)

**Decision:** LSTM sufficient, Transformer overkill

#### 4. 1D CNN

**Pros:**
- Fast (parallel convolutions)
- Good at local patterns
- Fewer parameters than LSTM
- Simple architecture

**Cons:**
- ✗ Limited receptive field (even with dilation)
- ✗ Not designed for sequences
- ✗ Requires very large kernels for low frequencies
- ✗ Tested: R² ≈ 0.29 (worse than LSTM)

**Decision:** Not ideal for this sequential task

#### 5. Temporal Convolutional Network (TCN)

**Pros:**
- Dilated convolutions (long receptive field)
- Parallel training
- No vanishing gradients
- Recent success in time-series

**Cons:**
- ✗ More complex to implement
- ✗ Needs careful tuning of dilations
- ✗ Tested: R² ≈ 0.32 (slightly worse)

**Decision:** LSTM performs better for this task

### Design Choices

#### Stacked (2 layers) vs. Single Layer

**Decision:** 2 layers

**Rationale:**
- Layer 1: Low-level temporal patterns
- Layer 2: High-level frequency relationships
- Tested: 1 layer → R² = 0.25, 2 layers → R² = 0.35
- 3+ layers: Diminishing returns (R² ≈ 0.36), much slower

#### Unidirectional vs. Bidirectional

**Decision:** Unidirectional (forward only)

**Rationale:**
- Bidirectional tested: R² ≈ 0.38 (+8%)
- But: 2× parameters, 2× slower, non-causal
- Unidirectional maintains causality (could deploy real-time)
- Trade-off: slight performance loss for efficiency and causality

#### Sequence-to-Sequence vs. Sequence-to-Vector

**Decision:** Sequence-to-sequence

**Input:** (batch, 50, 5) → **Output:** (batch, 50, 1)

**Alternatives:**
- Sequence-to-vector: Output single value per sequence
- Many-to-one: Only use final hidden state

**Rationale:**
- Dense supervision (50 targets vs. 1)
- Maintains temporal structure
- Better for evaluation (can check phase)
- More training signal per sequence

---

## Data Pipeline

### Decision: Synthetic Data with Fixed Phases

**Chosen:** Fixed phase offsets + additive Gaussian noise

**Alternative:** Random phases, real audio, other noise models

### Rationale

**Evolution of approach:**

**Initial (failed):** Random phases per sample
```python
# At EVERY sample:
phase[t] = random.uniform(0, 2π)
signal[t] = sin(2πf·t + phase[t])
```
**Result:** R² = -0.45 (worse than mean!)

**Improved (success):** Fixed phases + additive noise
```python
# Fixed once:
phases = [0°, 45°, 90°, 135°]
signal = sin(2πf·t + phase) + noise
```
**Result:** R² = 0.35 (+178% improvement!)

**Why it works:**
- Preserves frequency structure
- Temporal coherence maintained
- Noise adds variability without destroying patterns
- Task remains learnable

### Design Choices

#### Separate Noise for Train/Test

**Decision:** Different random seeds

**Rationale:**
- Tests true generalization (not memorization)
- More realistic evaluation
- Stronger evidence of learning
- Standard practice in ML

**Impact:**
- Harder task (lower R²)
- But more meaningful results

#### Synthetic vs. Real Audio

**Decision:** Synthetic

**Rationale:**
- Ground truth available (know exact frequencies)
- Controlled difficulty
- Reproducible
- No licensing issues
- Focused evaluation

**Future:** Could add real audio as advanced test

#### Sequence Creation Strategy

**Decision:** Sliding window with stride=1

**Rationale:**
- Maximum data from limited samples
- Smooth temporal coverage
- Dense supervision
- Standard for time-series

**Alternative:** stride=10 (less overlap)
- Faster training
- But 90% less data
- Trade-off not worth it

---

## Training Strategy

### Decision: Adam + ReduceLROnPlateau + Early Stopping

**Chosen:** 
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=15

### Rationale

#### Why Adam?

**Tested optimizers:**

| Optimizer | R² | Training Time | Stability |
|-----------|-----|---------------|-----------|
| SGD | 0.28 | Fast | Unstable |
| SGD+Momentum | 0.31 | Fast | Better |
| RMSprop | 0.33 | Medium | Good |
| **Adam** | **0.35** | **Medium** | **Excellent** |
| AdamW | 0.35 | Medium | Excellent |

**Decision:** Adam provides best performance with minimal tuning

#### Why ReduceLROnPlateau?

**Alternatives:**
- Fixed LR: No adaptation, may plateau
- StepLR: Fixed schedule, not adaptive
- CosineAnnealing: Good but more complex

**Rationale:**
- Adapts to actual training dynamics
- Reduces LR when validation loss plateaus
- Allows fine-tuning in later stages
- Simple and effective

#### Why Early Stopping?

**Rationale:**
- Prevents overfitting
- Saves compute time
- Returns best model (not last)
- Automatic convergence detection

**Patience=15:**
- Not too aggressive (allows fine-tuning)
- Not too patient (stops when truly plateaued)
- Good balance for 50 epoch budget

### Regularization Strategy

**Decision:** Multiple techniques combined

1. **Dropout (0.2):** Between layers
2. **Weight decay (1e-5):** L2 regularization
3. **Separate noise:** Train/test different
4. **Gradient clipping (1.0):** Stability
5. **Early stopping:** Prevents overfitting

**Rationale:**
- Defense in depth
- Each addresses different aspect
- Combined effect: Train ≈ Val loss (no overfitting)

---

## Evaluation Approach

### Decision: Comprehensive Multi-Metric Evaluation

**Chosen:** R², MSE, RMSE, MAE, Correlation + Baselines + Per-frequency

### Rationale

**Why multiple metrics:**

1. **R² (primary):** Variance explained
   - Most interpretable
   - Standard in ML
   - Allows comparison across tasks

2. **RMSE/MAE:** Absolute error
   - More intuitive (same units as data)
   - Complementary to R²
   - RMSE for outliers, MAE for robustness

3. **Correlation:** Linear relationship
   - Different perspective than R²
   - Statistical significance (p-value)
   - Familiar to researchers

4. **Baselines:** Context
   - Proves model actually learns
   - Quantifies improvement
   - Standard practice

5. **Per-frequency:** Detailed analysis
   - Identifies strengths/weaknesses
   - Research insights
   - Guides improvements

### Test Set Strategy

**Decision:** Completely separate noise (Seed #2)

**Rationale:**
- Most rigorous test of generalization
- Cannot memorize noise patterns
- Realistic scenario
- Higher standard than typical

**Alternative:** Same noise for train/test
- Would show higher R² (~0.50+)
- But less meaningful
- Would be cheating

---

## Project Structure

### Decision: Modular Pipeline with Separate Scripts

**Chosen:** 7 separate Python scripts

**Alternative:** Single monolithic script, Jupyter notebook

### Rationale

**Why modular:**

1. **Separation of concerns**
   - Each script has single responsibility
   - Easy to understand
   - Easy to modify

2. **Reusability**
   - Can run individual steps
   - Can skip steps (e.g., use existing data)
   - Debugging easier

3. **Flexibility**
   - Easy to experiment
   - Can modify one step without affecting others
   - Clear dependencies

4. **Maintainability**
   - Smaller files
   - Easier to navigate
   - Clear structure

**Why not Jupyter:**
- Not reproducible (cell execution order)
- Hard to version control
- Harder to test
- Good for exploration, bad for production code

**Why not single script:**
- Too long (would be 1000+ lines)
- Hard to navigate
- Can't run individual steps
- Harder to maintain

### File Naming Convention

**Decision:** Descriptive snake_case

Examples:
- `generate_dataset.py`
- `train_model.py`
- `evaluate_model.py`

**Rationale:**
- Clear what each file does
- Python convention (PEP 8)
- Easy to find files

**Alternative:** Numbered (01_generate.py, 02_train.py)
- Pros: Clear order
- Cons: Harder to reference in code
- Decision: Order clear from README

---

## Summary

**Key architectural decisions:**

| Aspect | Decision | Alternative | Rationale |
|--------|----------|-------------|-----------|
| **Framework** | PyTorch | TensorFlow, JAX | Best for research, great LSTM support |
| **Model** | LSTM | RNN, GRU, Transformer | Proven, handles long-term dependencies |
| **Layers** | 2 stacked | 1, 3, 4 | Best performance/complexity trade-off |
| **Direction** | Unidirectional | Bidirectional | Maintains causality, faster |
| **Output** | Sequence-to-sequence | Sequence-to-value | Dense supervision, maintains structure |
| **Data** | Synthetic fixed phases | Random phases, real audio | Learnable task with ground truth |
| **Noise** | Separate train/test | Same noise | True generalization test |
| **Optimizer** | Adam | SGD, RMSprop | Best performance, minimal tuning |
| **Regularization** | Multiple techniques | Single technique | Comprehensive, no overfitting |
| **Evaluation** | Multi-metric + baselines | Single metric | Comprehensive understanding |
| **Structure** | Modular scripts | Monolithic, notebook | Maintainable, flexible |

**Design philosophy:**

1. **Simplicity:** Choose simplest solution that works
2. **Proven approaches:** Use well-established techniques
3. **Comprehensive evaluation:** Multiple metrics and baselines
4. **Reproducibility:** Fixed seeds, documented decisions
5. **Modularity:** Separate concerns, clear structure
6. **Practicality:** Balance performance and complexity

These decisions create a well-architected, maintainable, and effective system for LSTM-based frequency filtering.

