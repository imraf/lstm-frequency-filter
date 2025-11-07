# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, error messages, unexpected behaviors, and their solutions for the LSTM Frequency Filter project. Organized by category and severity.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Data Generation Issues](#data-generation-issues)
4. [Training Issues](#training-issues)
5. [Evaluation Issues](#evaluation-issues)
6. [Performance Issues](#performance-issues)
7. [Environment Issues](#environment-issues)

---

## Quick Diagnostics

### First Steps When Something Goes Wrong

**1. Check error message**
```bash
python script_name.py 2>&1 | tee error.log
# Saves full error output to error.log
```

**2. Verify environment**
```bash
python --version  # Should be 3.8+
pip list | grep -E "numpy|torch|pandas"  # Check versions
```

**3. Check file existence**
```bash
ls -lh data/
ls -lh models/
ls -lh visualizations/
```

**4. Verify disk space**
```bash
df -h .  # Check available space
```

**5. Check GPU availability (if applicable)**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Common Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Import error | `pip install --upgrade -r requirements.txt` |
| File not found | Run previous steps in order |
| Out of memory | Reduce batch size to 32 |
| Training slow | Use GPU or reduce epochs to 25 |
| Different results | Check random seeds are set |

---

## Installation Issues

### Issue: ModuleNotFoundError

**Error message:**
```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'torch'
```

**Cause:** Required packages not installed

**Solution:**
```bash
pip install -r requirements.txt
```

**If still failing:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Or install individually
pip install numpy pandas matplotlib scipy torch scikit-learn
```

### Issue: ImportError: DLL load failed (Windows)

**Error message:**
```
ImportError: DLL load failed while importing _ssl
```

**Cause:** Missing Visual C++ Redistributable

**Solution:**
1. Download Microsoft Visual C++ Redistributable
2. Install both x86 and x64 versions
3. Restart terminal
4. Try again

### Issue: torch.cuda not available

**Symptom:** GPU not detected

**Diagnostic:**
```python
import torch
print(torch.cuda.is_available())  # False
print(torch.version.cuda)  # None
```

**Causes & Solutions:**

**1. No GPU installed**
- Solution: Use CPU (automatic fallback)
- Expected: Slower training (~15-20 min vs. 3-5 min)

**2. Wrong PyTorch version**
- Solution: Install CUDA-enabled PyTorch
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**3. CUDA drivers not installed**
- Solution: Install NVIDIA CUDA Toolkit
- Download from: https://developer.nvidia.com/cuda-downloads

### Issue: Version conflicts

**Error message:**
```
ERROR: package 'X' has requirement 'Y>=1.0', but you have 'Y 0.9'
```

**Solution:**
```bash
# Remove conflicting packages
pip uninstall numpy pandas matplotlib scipy torch scikit-learn

# Reinstall fresh
pip install -r requirements.txt
```

**Nuclear option (if all else fails):**
```bash
# Create new virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
pip install -r requirements.txt
```

---

## Data Generation Issues

### Issue: FileNotFoundError during data generation

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/'
```

**Cause:** Data directory doesn't exist

**Solution:**
```bash
mkdir -p data models visualizations
```

### Issue: Permission denied when saving

**Error message:**
```
PermissionError: [Errno 13] Permission denied: 'data/frequency_data.npz'
```

**Solutions:**

**1. Check directory permissions:**
```bash
ls -ld data/
chmod 755 data/  # Make writable
```

**2. File is open elsewhere:**
- Close any programs using the file
- On Windows: Check Excel, Python notebooks

**3. Run with appropriate permissions:**
```bash
# On Unix/Mac (if needed, use cautiously)
sudo python generate_dataset.py
```

### Issue: Different SNR than expected

**Symptom:** Console shows SNR ~8 dB instead of ~11 dB

**Cause:** Different signal statistics or noise level

**Diagnostics:**
```python
import numpy as np
data = np.load('data/frequency_data_train.npz')
S_clean = data['S_clean'] if 'S_clean' in data else None
noise = data['noise'] if 'noise' in data else None

if S_clean is not None and noise is not None:
    snr = 10 * np.log10(S_clean.var() / noise.var())
    print(f"Calculated SNR: {snr:.2f} dB")
```

**Solution:**

If SNR is wrong:
1. Check `noise_std` parameter (should be 0.1)
2. Verify signal generation (clean signals correct?)
3. Regenerate data: `python generate_dataset.py`

**Note:** Small variations (±1 dB) are normal

### Issue: NaN or Inf in generated data

**Symptom:** Dataset contains NaN or Inf values

**Diagnostic:**
```python
import numpy as np
data = np.load('data/frequency_data_train.npz')
for key in data:
    arr = data[key]
    if np.isnan(arr).any():
        print(f"{key} contains NaN")
    if np.isinf(arr).any():
        print(f"{key} contains Inf")
```

**Causes & Solutions:**

**1. Numerical overflow:**
- Check frequency values aren't too large
- Check amplitude scaling

**2. Division by zero:**
- Check noise_std > 0
- Check sampling rate > 0

**3. Corrupted computation:**
- Regenerate: `python generate_dataset.py`

---

## Training Issues

### Issue: Training loss not decreasing

**Symptom:** Loss stays high (~0.5) or doesn't improve

**Diagnostic checklist:**

```python
# Check data loading
data = np.load('data/training_data.npz')
print(f"X_train shape: {data['X_train'].shape}")
print(f"X_train range: [{data['X_train'].min()}, {data['X_train'].max()}]")
print(f"y_train range: [{data['y_train'].min()}, {data['y_train'].max()}]")

# Check model
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

**Causes & Solutions:**

**1. Learning rate too low:**
```python
# Increase LR
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Was 0.001
```

**2. Model too small:**
```python
# Increase capacity
hidden_size = 256  # Was 128
# or
num_layers = 3  # Was 2
```

**3. Data not loaded correctly:**
- Check shapes match expected
- Verify data contains variation (not all zeros)
- Regenerate data if suspicious

**4. Loss function issue:**
```python
# Verify loss calculation
print(f"Train loss: {train_loss}")
print(f"Val loss: {val_loss}")
# Should be finite numbers, not NaN
```

### Issue: Training loss exploding (NaN or Inf)

**Symptom:** Loss becomes NaN after few batches

**Error message:**
```
RuntimeError: Function 'MseLoss' returned nan values in its 0th output.
```

**Causes & Solutions:**

**1. Learning rate too high:**
```python
# Reduce LR
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Was 0.001
```

**2. Gradients exploding:**
```python
# Strengthen gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Was 1.0
```

**3. NaN in input data:**
```python
# Check data
assert not torch.isnan(X_train).any(), "NaN in training data"
assert not torch.isinf(X_train).any(), "Inf in training data"
```

**4. Numerical instability:**
```python
# Add epsilon to prevent division by zero
loss = criterion(outputs + 1e-8, targets + 1e-8)
```

### Issue: Overfitting (Train << Val loss)

**Symptom:** Training loss much lower than validation loss

**Example:**
```
Epoch 30: Train Loss: 0.05 | Val Loss: 0.20
```

**Solutions:**

**1. Increase dropout:**
```python
dropout = 0.4  # Was 0.2
```

**2. Increase weight decay:**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Was 1e-5
```

**3. Reduce model capacity:**
```python
hidden_size = 64  # Was 128
```

**4. More data:**
```python
# In prepare_training_data.py
step_size = 1  # Already at minimum
# Consider data augmentation or collecting more data
```

**5. Early stopping:**
- Already implemented (patience=15)
- May need to reduce patience to 10

### Issue: Slow convergence

**Symptom:** Loss decreasing very slowly

**Solutions:**

**1. Increase learning rate:**
```python
learning_rate = 0.005  # Was 0.001
```

**2. Change optimizer:**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**3. Better initialization:**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

model.apply(init_weights)
```

**4. Batch normalization:**
```python
# Add between LSTM and linear layer
self.batch_norm = nn.BatchNorm1d(sequence_length)
```

### Issue: CUDA out of memory

**Error message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
```

**Solutions (in order of preference):**

**1. Reduce batch size:**
```python
batch_size = 32  # Was 64
# or
batch_size = 16
```

**2. Reduce sequence length:**
```python
# In prepare_training_data.py
sequence_length = 25  # Was 50
# NOTE: Must regenerate training data
```

**3. Reduce model size:**
```python
hidden_size = 64  # Was 128
```

**4. Clear cache:**
```python
# Add to training loop
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()
```

**5. Use CPU:**
```python
device = torch.device('cpu')
# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Training stuck at local minimum

**Symptom:** Loss plateaus early and doesn't improve

**Solutions:**

**1. Restart with different initialization:**
```python
# Change random seed
torch.manual_seed(123)  # Was 42
```

**2. Use learning rate scheduler:**
```python
# Already implemented, but can adjust
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.3  # Was patience=5, factor=0.5
)
```

**3. Add noise to gradients:**
```python
# In training loop, after backward()
for param in model.parameters():
    if param.grad is not None:
        param.grad += torch.randn_like(param.grad) * 0.01
```

---

## Evaluation Issues

### Issue: FileNotFoundError during evaluation

**Error message:**
```
FileNotFoundError: models/best_model.pth
```

**Cause:** Model not trained or training incomplete

**Solution:**
```bash
# Run training
python train_model.py

# Verify model saved
ls -lh models/best_model.pth
```

### Issue: Evaluation metrics very poor (R² < 0)

**Symptom:** R² negative, correlation near 0

**Diagnostic:**
```python
# Check predictions
y_pred = model(X_test)
print(f"Predictions range: [{y_pred.min()}, {y_pred.max()}]")
print(f"Actual range: [{y_test.min()}, {y_test.max()}]")
print(f"Predictions unique values: {len(np.unique(y_pred.round(3)))}")
```

**Causes & Solutions:**

**1. Model collapsed to constant:**
- Predictions all same value
- Solution: Retrain with different hyperparameters
- Increase learning rate or model capacity

**2. Wrong model architecture:**
- Model doesn't match saved weights
- Solution: Verify architecture in `evaluate_model.py` matches `train_model.py`

**3. Data mismatch:**
- Test data different format than training
- Solution: Regenerate all data with `generate_dataset.py` then `prepare_training_data.py`

**4. Model not loaded correctly:**
```python
# Verify loading
model = FrequencyFilterLSTM(5, 128, 2, 1, 0.2)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()  # IMPORTANT!
```

### Issue: Predictions are all zero or constant

**Diagnostic:**
```python
y_pred = model(X_test_tensor)
print(f"Unique predictions: {torch.unique(y_pred)}")
```

**Causes:**

**1. Forgot model.eval():**
```python
model.eval()  # Required before inference
```

**2. Model didn't train:**
- Check training history
- Verify losses decreased

**3. Dead neurons:**
- All activations became zero
- Solution: Retrain with lower learning rate

---

## Performance Issues

### Issue: Training too slow

**Expected:** 15-20 minutes on CPU

**If much slower (>1 hour):**

**Solutions:**

**1. Use GPU:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**2. Reduce data:**
```python
# In prepare_training_data.py
step_size = 10  # Was 1 (90% less data)
```

**3. Reduce epochs:**
```python
num_epochs = 25  # Was 50
```

**4. Increase batch size:**
```python
batch_size = 128  # Was 64 (if memory allows)
```

**5. Profile code:**
```python
import time
start = time.time()
for batch in train_loader:
    # ... training code
    pass
print(f"Time per batch: {(time.time() - start) / len(train_loader)}")
```

### Issue: High memory usage

**Diagnostic:**
```bash
# On Linux/Mac
top -p $(pgrep python)

# In Python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

**Solutions:**

**1. Reduce batch size:**
```python
batch_size = 32  # Was 64
```

**2. Free memory periodically:**
```python
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

**3. Use smaller data types:**
```python
X_train = X_train.astype(np.float32)  # Instead of float64
```

**4. Process data in chunks:**
```python
# For evaluation
chunk_size = 1000
for i in range(0, len(X_test), chunk_size):
    X_chunk = X_test[i:i+chunk_size]
    y_pred_chunk = model(X_chunk)
    # Process chunk
```

### Issue: Low GPU utilization

**Symptom:** GPU usage < 50%

**Diagnostic:**
```bash
nvidia-smi  # Check GPU usage
```

**Solutions:**

**1. Increase batch size:**
```python
batch_size = 256  # Was 64
```

**2. Reduce data loading time:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # Add parallel data loading
    pin_memory=True  # Faster CPU-GPU transfer
)
```

**3. Profile bottleneck:**
```python
# Check if CPU or GPU bound
import torch.cuda
torch.cuda.synchronize()  # Wait for GPU
# Time operations to find bottleneck
```

---

## Environment Issues

### Issue: Different results on different machines

**Symptom:** R² = 0.35 on one machine, 0.32 on another

**Causes:**

**1. Different random seeds:**
- Solution: Verify seeds are set in all scripts

**2. Different library versions:**
```bash
pip list | grep -E "numpy|torch|pandas"
# Compare versions
```

**3. Different hardware (CPU vs GPU):**
- Can cause small differences (~1-2%)
- Both are correct

**4. Floating-point precision:**
- Different hardware = slight differences
- Normal and expected

**Solution for reproducibility:**
```python
# Add to all scripts
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Issue: Matplotlib plots don't show

**Symptom:** Scripts complete but no plot windows appear

**Solutions:**

**1. Check backend:**
```python
import matplotlib
print(matplotlib.get_backend())  # Should not be 'agg'
```

**2. Set interactive backend:**
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

**3. Add plt.show():**
```python
plt.savefig('plot.png')
plt.show()  # Add this
```

**4. Running remotely:**
- Use non-interactive backend
```python
matplotlib.use('Agg')
```

### Issue: Scripts fail on Windows

**Common issues:**

**1. Path separators:**
```python
# Wrong
path = 'data/file.npz'  # Works on Unix, may fail on Windows

# Better
import os
path = os.path.join('data', 'file.npz')

# Or
from pathlib import Path
path = Path('data') / 'file.npz'
```

**2. Line endings:**
- Git may convert LF to CRLF
- Solution: Configure Git
```bash
git config --global core.autocrlf false
```

**3. Permission issues:**
- Run terminal as Administrator if needed

---

## Summary

### Quick Troubleshooting Checklist

When things go wrong:

- [ ] Read error message carefully
- [ ] Check file exists (`ls` or `dir`)
- [ ] Verify prerequisites installed (`pip list`)
- [ ] Check disk space (`df -h`)
- [ ] Try running previous step again
- [ ] Check random seeds are set
- [ ] Verify Python version (3.8+)
- [ ] Look for typos in filenames
- [ ] Check for NaN/Inf in data
- [ ] Consult this guide!

### Getting Help

If issue persists:

1. Check documentation in other files
2. Search error message online
3. Check PyTorch forums/GitHub issues
4. Open issue with:
   - Full error message
   - Python version
   - Library versions (`pip list`)
   - Operating system
   - Steps to reproduce
   - What you've tried

### Prevention

**Best practices to avoid issues:**

1. **Use virtual environment**
2. **Install exact versions** from requirements.txt
3. **Run scripts in order**
4. **Check output** after each step
5. **Backup results** before re-running
6. **Document changes** you make
7. **Test on small data** first
8. **Monitor resources** (memory, disk)
9. **Save intermediate results**
10. **Keep this guide handy!**

---

**Most issues can be resolved by following this guide. Good luck!**

