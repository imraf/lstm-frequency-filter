# Script 03: Prepare Training Data

**File:** `prepare_training_data.py`

## Quick Reference

| Attribute | Value |
|-----------|-------|
| **Purpose** | Create sequences and train/val/test splits for LSTM training |
| **Input** | `data/frequency_data_train.npz`, `data/frequency_data_test.npz` |
| **Output** | `data/training_data.npz` + 2 visualizations |
| **Runtime** | ~5-10 seconds |
| **Dependencies** | numpy, matplotlib, sklearn |

## Usage

```bash
python prepare_training_data.py
```

**Prerequisites:** Must run `generate_dataset.py` first.

## What This Script Does

### Overview

Transforms raw time-series data into sequence format suitable for LSTM training:

1. **Creates sequences** using sliding window (length=50, stride=1)
2. **Generates one-hot selectors** for each frequency
3. **Combines features** (signal + selector)
4. **Splits data** into train (90%) / validation (10%)
5. **Keeps test separate** (different noise seed)
6. **Creates visualizations** of training samples

This preparation is crucial for LSTM training, which requires fixed-length sequences.

### Step-by-Step Process

#### 1. **Load Data** (Lines 13-40)

Loads two separate datasets:

**Training dataset (Seed #1):**
```python
train_data = np.load('data/frequency_data_train.npz')
S_train = train_data['S']           # Noisy mixed signal
target1_train = train_data['target1']  # Pure f1
# ... targets 2-4
```

**Test dataset (Seed #2):**
```python
test_data = np.load('data/frequency_data_test.npz')
S_test = test_data['S']           # Different noise!
target1_test = test_data['target1']  # Same frequencies, different noise
```

**Critical:** Test set uses **completely different noise** from training. This ensures the model learns frequency patterns, not noise patterns.

#### 2. **Sequence Creation** (Lines 48-69)

Defines parameters and one-hot selectors:

```python
sequence_length = 50  # 50 timesteps per sequence
step_size = 1        # Stride = 1 (maximum overlap)

selectors = np.array([
    [1, 0, 0, 0],  # Select f1
    [0, 1, 0, 0],  # Select f2
    [0, 0, 1, 0],  # Select f3
    [0, 0, 0, 1]   # Select f4
])
```

**Why sequence_length=50?**

At 1000 Hz sampling rate, 50 samples = 0.05 seconds:
- f₁ (1 Hz): 0.05 cycles (5% of one cycle)
- f₄ (7 Hz): 0.35 cycles (35% of one cycle)

Provides temporal context while remaining computationally tractable.

#### 3. **Create Sequences Function** (Lines 70-111)

The `create_sequences()` function performs sliding window extraction:

**Process:**
1. Stack all target frequencies
2. Slide window across signal
3. For each window, create 4 training samples (one per selector)
4. Return sequences, selectors, and targets

**Mathematical representation:**

For signal S of length N, create sequences:

\[ X_i = [S_{i}, S_{i+1}, ..., S_{i+L-1}] \]

where i ∈ [0, N-L] and L=50 (sequence length).

For each sequence, create 4 training samples:
- Input: (X_i, [1,0,0,0]) → Target: [f1_i, ..., f1_{i+49}]
- Input: (X_i, [0,1,0,0]) → Target: [f2_i, ..., f2_{i+49}]
- Input: (X_i, [0,0,1,0]) → Target: [f3_i, ..., f3_{i+49}]
- Input: (X_i, [0,0,0,1]) → Target: [f4_i, ..., f4_{i+49}]

**Example:**
- Raw data: 10,000 samples
- Possible sequences: 10,000 - 50 = 9,950
- With 4 selectors: 9,950 × 4 = 39,800 training samples

#### 4. **Generate Sequences for Both Datasets** (Lines 113-129)

```python
# Training data (Seed #1)
X_train_seq, X_train_sel, y_train_seq = create_sequences(
    S_train,
    [target1_train, target2_train, target3_train, target4_train],
    sequence_length,
    step_size,
    "TRAINING SET (Seed #1)"
)

# Test data (Seed #2)
X_test_seq, X_test_sel, y_test_seq = create_sequences(
    S_test,
    [target1_test, target2_test, target3_test, target4_test],
    sequence_length,
    step_size,
    "TEST SET (Seed #2)"
)
```

#### 5. **Combine Features** (Lines 131-177)

The `combine_features()` function merges signal and selector:

**Input format before:**
- Signal sequence: (batch, 50)
- Selector: (batch, 4)

**Input format after:**
- Combined: (batch, 50, 5)

**How it works:**
```python
# Expand selector to match sequence length
X_sel_expanded = np.repeat(X_sel[:, np.newaxis, :], sequence_length, axis=1)
# Shape: (batch, 50, 4)

# Expand signal to have feature dimension
X_seq_expanded = X_seq[:, :, np.newaxis]
# Shape: (batch, 50, 1)

# Concatenate
X_combined = np.concatenate([X_seq_expanded, X_sel_expanded], axis=2)
# Shape: (batch, 50, 5)
```

**Resulting features at each timestep:**
- Feature 0: Signal value S(t)
- Feature 1: Selector bit 0 (1 if f1 selected, else 0)
- Feature 2: Selector bit 1 (1 if f2 selected, else 0)
- Feature 3: Selector bit 2 (1 if f3 selected, else 0)
- Feature 4: Selector bit 3 (1 if f4 selected, else 0)

#### 6. **Split Data** (Lines 179-206)

Uses sklearn's train_test_split:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_combined, y_train_combined, 
    test_size=0.1,  # 10% validation
    random_state=42  # Reproducibility
)
```

**Final splits:**
- **Training:** 90% of Seed #1 data (~35,820 sequences)
- **Validation:** 10% of Seed #1 data (~3,980 sequences)
- **Test:** 100% of Seed #2 data (~39,800 sequences)

**Why this split?**
- Train/val from same noise realization (Seed #1): Hyperparameter tuning
- Test from different noise (Seed #2): True generalization test

#### 7. **Save Prepared Data** (Lines 207-217)

```python
np.savez('data/training_data.npz',
         X_train=X_train,
         X_val=X_val,
         X_test=X_test,
         y_train=y_train,
         y_val=y_val,
         y_test=y_test,
         sequence_length=sequence_length,
         selectors=selectors)
```

#### 8. **Create Visualizations** (Lines 219-298)

**Visualization 1: Training Samples** (Lines 228-263)
- Shows 4 examples (one per frequency selector)
- Left column: Noisy input signal
- Right column: Pure target frequency

**Visualization 2: Model I/O Structure** (Lines 268-298)
- Shows input features (5 features) over time
- Shows output (1 feature) over time
- Illustrates the sequence-to-sequence nature

## Output Files

### 1. `data/training_data.npz`

**Size:** ~40 MB

**Format:** NumPy compressed archive

**Contents:**

| Array | Shape | Description |
|-------|-------|-------------|
| `X_train` | (35820, 50, 5) | Training input sequences |
| `X_val` | (3980, 50, 5) | Validation input sequences |
| `X_test` | (39800, 50, 5) | Test input sequences |
| `y_train` | (35820, 50, 1) | Training target sequences |
| `y_val` | (3980, 50, 1) | Validation target sequences |
| `y_test` | (39800, 50, 1) | Test target sequences |
| `sequence_length` | scalar | 50 |
| `selectors` | (4, 4) | One-hot selector matrix |

**Usage:**
```python
data = np.load('data/training_data.npz')
X_train = data['X_train']
y_train = data['y_train']
```

### 2. `visualizations/05_training_samples.png`

**Shows:** Sample input/output pairs for each frequency selector

**Structure:** 4 rows × 2 columns
- Row 1: f₁ selector examples
- Row 2: f₂ selector examples
- Row 3: f₃ selector examples
- Row 4: f₄ selector examples

**What to look for:**
✓ Left column (input) shows noisy mixed signal
✓ Right column (target) shows clean single frequency
✓ Each selector correctly isolates its frequency
✓ Target frequencies match expected patterns

### 3. `visualizations/06_model_io_structure.png`

**Shows:** Detailed view of one sample's structure

**Structure:** 2 subplots
- Top: Input features over 50 timesteps
- Bottom: Output feature over 50 timesteps

**What to look for:**
✓ Signal (feature 0) oscillates
✓ Selector bits (features 1-4) are constant
✓ Exactly one selector bit = 1
✓ Output matches selected frequency pattern

## Technical Details

### Sequence Length Justification (L=50)

**Why not L=1?**

With L=1, each input would be a single timestep:
- Input: [S(t), c₁, c₂, c₃, c₄]
- Output: f_i(t)

**Problems:**
- Minimal temporal context
- LSTM must rely entirely on hidden state
- Harder to learn phase and frequency patterns

**Why not L=1000?**

With L=1000 (1 full second):
- Better frequency resolution
- BUT: Slower training (longer backpropagation)
- Higher memory usage
- Diminishing returns

**Why L=50 is optimal:**

✓ Captures 0.05 seconds of context
✓ Reasonable memory usage
✓ Good gradient flow in BPTT
✓ Efficient batch processing
✓ Balances context vs. computation

### Stride (Step Size)

**Current: stride=1 (maximum overlap)**

```python
step_size = 1
```

**Effect:**
- Creates 9,950 sequences from 10,000 samples
- Adjacent sequences differ by only 1 sample
- Maximum data augmentation
- Smooth temporal coverage

**Alternatives:**

**stride=25 (50% overlap):**
- Creates ~400 sequences
- Less redundancy
- Faster training
- BUT: Less data for model to learn from

**stride=50 (no overlap):**
- Creates ~200 sequences
- Minimum redundancy
- Much faster training
- BUT: Significant data loss

**Why stride=1?**
- We have computational resources
- More data helps generalization
- Smooth temporal coverage benefits LSTM

### One-Hot Selector Encoding

**Why one-hot instead of integer?**

**Integer encoding:**
```python
selector = 0  # f1
selector = 1  # f2
selector = 2  # f3
selector = 3  # f4
```

**Problems:**
- Implies ordering (f1 < f2 < f3 < f4)
- Implies distances (f1 is "closer" to f2 than to f4)
- Model might learn incorrect relationships

**One-hot encoding:**
```python
selector_f1 = [1, 0, 0, 0]
selector_f2 = [0, 1, 0, 0]
selector_f3 = [0, 0, 1, 0]
selector_f4 = [0, 0, 0, 1]
```

**Advantages:**
✓ No implicit ordering
✓ Equal "distance" between all frequencies
✓ Clear binary signal: "select this" vs. "don't select this"
✓ Standard practice for categorical variables

### Memory Considerations

**Array sizes:**
- X_train: 35,820 × 50 × 5 × 8 bytes = ~71 MB
- y_train: 35,820 × 50 × 1 × 8 bytes = ~14 MB
- X_val: ~8 MB
- y_val: ~1.6 MB
- X_test: ~80 MB
- y_test: ~16 MB

**Total:** ~190 MB uncompressed, ~40 MB compressed (NPZ)

**Peak memory during script:** ~300 MB (holding multiple copies during processing)

## Parameters & Configuration

### Sequence Length

```python
# Line 49
sequence_length = 50
```

**Effect of changing:**
- **Smaller (25):** Less context, faster training, lower memory
- **Larger (100):** More context, slower training, higher memory

**Recommendation:** 25-100 is reasonable range

### Step Size (Stride)

```python
# Line 50
step_size = 1
```

**Effect of changing:**
- **Larger (10):** ~10× fewer sequences, much faster, less redundancy
- **Smaller (1):** Current maximum overlap

**Trade-off:** Data quantity vs. computation time

### Validation Split

```python
# Line 192
test_size=0.1  # 10% validation
```

**Effect of changing:**
- **0.2:** More validation data, better hyperparameter tuning, less training data
- **0.05:** More training data, less robust validation

**Recommendation:** 0.1-0.2 is standard

### Random Seed

```python
# Line 193
random_state=42
```

**Purpose:** Ensures same train/val split across runs

**Change if:** You want different splits for cross-validation

## Code Walkthrough

### Key Functions

#### `create_sequences(S, targets, sequence_length, step_size, dataset_name)`

**Purpose:** Convert time series to sequences with selectors

**Parameters:**
- `S`: Mixed signal array (10,000 samples)
- `targets`: List of 4 target frequency arrays
- `sequence_length`: Window size (50)
- `step_size`: Stride (1)
- `dataset_name`: For logging

**Returns:**
- `X_sequences`: Signal sequences (n_sequences, 50)
- `X_selectors`: Selector arrays (n_sequences, 4)
- `y_sequences`: Target sequences (n_sequences, 50)

**Algorithm:**
```
For each starting position i in [0, N-L]:
    Extract signal window: S[i:i+L]
    Extract target windows: f1[i:i+L], f2[i:i+L], ...
    
    For each frequency j in [0, 1, 2, 3]:
        Append signal window to X_sequences
        Append selector[j] to X_selectors
        Append target[j] window to y_sequences
```

#### `combine_features(X_seq, X_sel, y_seq, dataset_name)`

**Purpose:** Combine signal and selector into unified input format

**Parameters:**
- `X_seq`: Signal sequences (n, 50)
- `X_sel`: Selectors (n, 4)
- `y_seq`: Targets (n, 50)
- `dataset_name`: For logging

**Returns:**
- `X_combined`: (n, 50, 5) - signal + selector at each timestep
- `y_expanded`: (n, 50, 1) - targets with feature dimension

**Algorithm:**
```
1. Expand selector from (n, 4) to (n, 50, 4) by repeating
2. Expand signal from (n, 50) to (n, 50, 1) by adding dimension
3. Concatenate along feature axis: (n, 50, 1) + (n, 50, 4) = (n, 50, 5)
4. Expand target from (n, 50) to (n, 50, 1) for consistency
```

## Troubleshooting

### Issue: "FileNotFoundError"

**Cause:** Input files not found

**Solution:**
```bash
python generate_dataset.py  # Create input files
```

### Issue: Out of memory

**Cause:** Arrays too large for available RAM

**Solutions:**
1. Increase step_size to reduce sequences:
```python
step_size = 10  # Instead of 1
```

2. Reduce sequence_length:
```python
sequence_length = 25  # Instead of 50
```

3. Close other applications

### Issue: Shapes don't match

**Diagnostic:**
```python
print(f"X_train shape: {X_train.shape}")  # Should be (n, 50, 5)
print(f"y_train shape: {y_train.shape}")  # Should be (n, 50, 1)
```

**Common cause:** Modified sequence_length without rerunning

**Solution:** Delete `data/training_data.npz` and rerun

### Issue: Selectors incorrect

**Diagnostic:**
```python
# Check one-hot encoding
print(selectors)
# Should be:
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]]
```

**Solution:** Verify selectors array (lines 63-68)

### Issue: Train/test split seems wrong

**Expected:**
- Training: ~35,800 sequences
- Validation: ~3,980 sequences
- Test: ~39,800 sequences

**If different:** Check input file sizes and sequence creation

## Performance Considerations

### Runtime Analysis

| Operation | Time | Bottleneck |
|-----------|------|------------|
| Load data | ~0.5s | Disk I/O |
| Create train sequences | ~2s | Array copying |
| Create test sequences | ~2s | Array copying |
| Combine features | ~1s | Array operations |
| Split data | ~0.5s | Random shuffle |
| Save NPZ | ~3s | Disk I/O, compression |
| Visualizations | ~2s | Matplotlib |
| **Total** | **~11s** | |

### Memory Optimization

**Current peak:** ~300 MB

**If memory is limited:**

1. **Process in batches:**
```python
# Instead of creating all sequences at once
for batch_start in range(0, n_samples-sequence_length, batch_size):
    # Process batch
    # Save to temp file
# Merge temp files
```

2. **Use memory-mapped arrays:**
```python
X_train = np.memmap('temp_X_train.dat', dtype='float32', 
                    mode='w+', shape=(n, 50, 5))
```

## Extending the Script

### Add Multi-Frequency Selection

Currently, selectors are mutually exclusive. To select multiple frequencies:

```python
# Allow multiple selections
selectors = np.array([
    [1, 0, 0, 0],  # f1 only
    [0, 1, 0, 0],  # f2 only
    [1, 1, 0, 0],  # f1 + f2
    [1, 0, 1, 0],  # f1 + f3
    # ... more combinations
])

# Targets would sum selected frequencies
if selector[0] and selector[1]:
    target = f1 + f2
```

### Add Sequence Augmentation

```python
# Add noise augmentation
def augment_sequence(X, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Apply during sequence creation
X_augmented = augment_sequence(X_seq)
```

### Add Validation for Data Quality

```python
# After sequence creation
assert X_train.shape[1] == sequence_length, "Wrong sequence length"
assert X_train.shape[2] == 5, "Wrong number of features"
assert np.all((X_train[:, :, 1:] == 0) | (X_train[:, :, 1:] == 1)), "Selectors not binary"
assert np.sum(X_train[:, 0, 1:], axis=1).all() == 1, "Exactly one selector should be 1"
```

## Related Documentation

- **Previous step:** `Script_02_VisualizeData.md` - Data visualization
- **Next step:** `Script_04_TrainModel.md` - LSTM training
- **Theory:** `MathematicalFoundations.md` - Sequence modeling
- **Architecture:** `Feature_ModelArchitecture.md` - Why sequences

## Summary

This script transforms raw time-series into LSTM-ready sequences:
- ✓ Creates 39,800 training samples from 10,000 raw samples
- ✓ Implements sliding window (L=50, stride=1)
- ✓ Adds one-hot selectors for frequency selection
- ✓ Properly splits train/val/test with separate noise realizations
- ✓ Prepares data in exact format LSTM expects: (batch, time, features)

**Critical achievement:** Separate noise for test set ensures true generalization testing.

