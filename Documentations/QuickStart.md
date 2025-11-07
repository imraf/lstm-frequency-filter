# Quick Start Guide

## Overview

This guide will get you up and running with the LSTM Frequency Filter project in under 10 minutes. For detailed explanations, see the other documentation files.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~500 MB free disk space
- 10-30 minutes for complete pipeline execution

## Installation

### 1. Clone/Download the Repository

```bash
cd lstm-frequency-filter
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- pandas (data manipulation)
- matplotlib (visualization)
- scipy (signal processing)
- torch (deep learning)
- scikit-learn (metrics)

**Installation time:** ~2-5 minutes

## Running the Complete Pipeline

### Option 1: Run Everything at Once (Recommended for First-Time Users)

```bash
chmod +x run_all.sh
./run_all.sh
```

**Total runtime:** ~15-30 minutes on CPU

This will automatically:
1. Generate the dataset (10,000 samples)
2. Create data visualizations (4 plots)
3. Prepare training sequences
4. Train the LSTM model (50 epochs)
5. Evaluate the model (6+ visualizations)

### Option 2: Run Step-by-Step

Execute each script individually to see detailed output:

```bash
# Step 1: Generate dataset (~5 seconds)
python generate_dataset.py

# Step 2: Visualize data (~10 seconds)
python visualize_data.py

# Step 3: Prepare training data (~5 seconds)
python prepare_training_data.py

# Step 4: Train model (~15-25 minutes)
python train_model.py

# Step 5: Evaluate model (~20 seconds)
python evaluate_model.py

# Step 6: View summary (instant)
python summary.py

# Step 7: Create overview visualization (optional, ~5 seconds)
python create_overview.py
```

## What Gets Created

After running the pipeline, you'll have:

### Data Files (`data/`)
- `frequency_dataset_train.csv` - Training data in CSV format
- `frequency_data_train.npz` - Training data (NumPy format)
- `frequency_data_test.npz` - Test data (NumPy format)
- `frequency_data.npz` - Compatibility file
- `training_data.npz` - Prepared sequences

### Model Files (`models/`)
- `best_model.pth` - Trained LSTM weights (~0.8 MB)
- `training_history.npz` - Loss curves
- `evaluation_results.npz` - Test metrics

### Visualizations (`visualizations/`)
14 PNG files showing:
- Time/frequency domain analysis
- Training progress
- Prediction quality
- Error analysis
- Performance metrics

## Quick Results Check

After completion, run:

```bash
python summary.py
```

You should see:
- **R² Score:** ~0.35 (model explains 35% of variance in noisy signals)
- **Correlation:** ~0.63 (strong positive correlation)
- **RMSE:** ~0.57 (reasonable prediction error)

## Understanding the Results

### What the Model Does

The model learns to:
1. Take a **mixed signal** containing 4 frequencies (1, 3, 5, 7 Hz)
2. Take a **selector** (one-hot vector: which frequency to extract?)
3. Output the **selected frequency** filtered from the mixed signal

### Key Achievement

The LSTM successfully learns frequency filtering in the **time domain** without explicit Fourier transforms!

### Performance Context

- **R² = 0.35** is moderate but reasonable for this challenging task:
  - 4 overlapping frequencies
  - Moderate noise (SNR ≈ 11 dB)
  - Model trained on different noise than test
- **54% better than random baseline**
- **41% better than mean baseline**

## Quick Visualization Tour

Check these visualizations in order:

1. **`00_complete_overview.png`** - One-page project summary
2. **`01_time_domain_signals.png`** - Individual frequencies
3. **`04_overlay_signals.png`** - How frequencies combine
4. **`07_training_loss.png`** - Model learning progress
5. **`08_predictions_vs_actual.png`** - Sample predictions
6. **`10_scatter_pred_vs_actual.png`** - Overall accuracy

## Common Quick Fixes

### Import Errors
```bash
pip install --upgrade numpy pandas matplotlib scipy torch scikit-learn
```

### Out of Memory
Reduce batch size in `train_model.py`:
```python
batch_size = 32  # Change from 64 to 32
```

### Training Too Slow
Reduce epochs in `train_model.py`:
```python
num_epochs = 25  # Change from 50 to 25
```

## Next Steps

### For Users Who Want to Understand More
- Read `RunningTheProject.md` for detailed explanation of each step
- Check `ResultsInterpretation.md` to understand all visualizations
- Review `Feature_EvaluationMetrics.md` for metrics explanation

### For Developers Who Want to Modify
- Read `Feature_ModelArchitecture.md` for LSTM details
- Check `HyperparameterGuide.md` for tuning options
- Review `ArchitectureDecisions.md` for design rationale

### For Researchers
- Read `MathematicalFoundations.md` for complete theory
- Check `Feature_DataGeneration.md` for signal processing details
- Review PRD.md for complete project requirements

## Quick FAQ

**Q: How long does training take?**
A: ~15-25 minutes on CPU, ~5 minutes on GPU

**Q: Can I use fewer epochs?**
A: Yes, 25 epochs gives ~90% of final performance

**Q: What if I get different results?**
A: Small variations are normal. Fixed seeds ensure reproducibility within ~1%

**Q: Can I modify the frequencies?**
A: Yes, edit the frequency values in `generate_dataset.py`

**Q: Can I add more frequencies?**
A: Yes, but requires code changes (selectors, model input size, etc.)

**Q: Does this work on GPU?**
A: Yes, automatically detected. Much faster training.

## Project Structure Quick Reference

```
lstm-frequency-filter/
├── data/                          # Generated datasets
├── models/                        # Trained models
├── visualizations/                # All plots
├── Documentations/               # This documentation
├── generate_dataset.py           # Step 1: Create data
├── visualize_data.py             # Step 2: Visualize
├── prepare_training_data.py      # Step 3: Prepare sequences
├── train_model.py                # Step 4: Train LSTM
├── evaluate_model.py             # Step 5: Evaluate
├── summary.py                    # Step 6: View results
├── create_overview.py            # Step 7: Overview viz
├── run_all.sh                    # Complete pipeline
├── requirements.txt              # Dependencies
└── README.md                     # Full documentation
```

## Support

For issues, questions, or contributions:
1. Check `Troubleshooting.md` for common problems
2. Review the detailed script documentation (`Script_*.md`)
3. Open an issue on the repository

---

**You're now ready to use the LSTM Frequency Filter!**

Run `./run_all.sh` and explore the results in the `visualizations/` folder.

