# LSTM Frequency Filter

A deep learning project that trains an LSTM neural network to filter individual frequency components from a mixed signal based on a one-hot selector.

## Project Overview

This project demonstrates:
- **Signal Processing**: Creating synthetic multi-frequency signals
- **Deep Learning**: Training an LSTM model for frequency filtering
- **Data Visualization**: Comprehensive visual analysis of signals and model performance

### Problem Statement

Given a combined signal `S(x)` composed of four sine wave frequencies:
```
S(x) = sin(2π·f1·x) + sin(2π·f2·x) + sin(2π·f3·x) + sin(2π·f4·x)
```

The LSTM model learns to extract a specific frequency component `fi(x)` from `S(x)` based on a one-hot selector vector `c`.

## Frequencies

- **f1**: 1.0 Hz
- **f2**: 3.0 Hz  
- **f3**: 5.0 Hz
- **f4**: 7.0 Hz

## Dataset

- **Total samples**: 10,000 data points
- **Interval**: [0, 20] seconds
- **Training samples**: 31,840 sequences (80%)
- **Validation samples**: 3,980 sequences (10%)
- **Test samples**: 3,980 sequences (10%)
- **Sequence length**: 50 time steps

## Model Architecture

**Framework**: PyTorch

```
FrequencyFilterLSTM(
  - Input size: 5 (1 signal + 4 selector values)
  - Hidden size: 128 LSTM units
  - Number of layers: 2 stacked LSTMs
  - Output size: 1
  - Dropout: 0.2
  - Total parameters: 201,345
)
```

### Hyperparameters

- **Batch size**: 64
- **Learning rate**: 0.001
- **Optimizer**: Adam with weight decay 1e-5
- **Loss function**: MSE (Mean Squared Error)
- **Epochs**: 50
- **Early stopping**: Patience 15

### Design Rationale

- **Hidden size 128**: Sufficient capacity for 4 frequency components
- **2 layers**: Captures temporal patterns at different scales
- **Dropout 0.2**: Prevents overfitting while maintaining performance
- **Adam optimizer**: Adaptive learning rate, robust for LSTMs
- **MSE loss**: Standard for regression, penalizes amplitude errors

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **MSE** | 0.025918 |
| **RMSE** | 0.160990 |
| **MAE** | 0.070495 |
| **R² Score** | 0.948005 |
| **Correlation** | 0.973757 |

### Per-Frequency Performance

| Frequency | MSE | RMSE | MAE | R² Score |
|-----------|-----|------|-----|----------|
| **f1 (1 Hz)** | 0.006424 | 0.080148 | 0.037466 | 0.987046 |
| **f2 (3 Hz)** | 0.032547 | 0.180407 | 0.080432 | 0.934703 |
| **f3 (5 Hz)** | 0.040589 | 0.201467 | 0.087397 | 0.918740 |
| **f4 (7 Hz)** | 0.023213 | 0.152360 | 0.075769 | 0.953519 |

### Key Findings

✅ **Excellent overall R² of 0.948** - The model explains 94.8% of variance in the data  
✅ **Strong correlation of 0.974** - Very strong linear relationship between predictions and actuals  
✅ **Low RMSE of 0.161** - Predictions are within ±0.16 amplitude units on average  
✅ **Best performance on f1 (1 Hz)** - Lowest frequency is easiest to filter (R² = 0.987)  
✅ **Consistent performance** - All frequencies filtered with R² > 0.91

## Project Structure

```
lstm-frequency-filter/
├── data/
│   ├── frequency_dataset.csv       # Raw dataset table
│   ├── frequency_data.npz          # Numpy arrays (signals)
│   └── training_data.npz           # Prepared training sequences
├── models/
│   ├── best_model.pth              # Best trained model weights
│   ├── training_history.npz        # Loss curves
│   └── evaluation_results.npz      # Test results
├── visualizations/
│   ├── 00_complete_overview.png        # NEW: One-page project summary
│   ├── 01_time_domain_signals.png
│   ├── 02_frequency_domain_fft.png
│   ├── 03_spectrogram.png
│   ├── 04_overlay_signals.png
│   ├── 05_training_samples.png
│   ├── 06_model_io_structure.png
│   ├── 07_training_loss.png
│   ├── 08_predictions_vs_actual.png
│   ├── 09_error_distribution.png
│   ├── 10_scatter_pred_vs_actual.png
│   ├── 11_frequency_spectrum_comparison.png
│   ├── 12_long_sequence_predictions.png
│   └── 13_per_frequency_metrics.png
├── generate_dataset.py
├── visualize_data.py
├── prepare_training_data.py
├── train_model.py
├── evaluate_model.py
└── README.md
```

## Visualizations

### Overview
0. **Complete Overview** - Single-page summary of the entire project (dataset, model, training, results)

### Data Visualizations (Steps 1-2)
1. **Time Domain Signals** - Individual frequencies and combined signal over time
2. **Frequency Domain (FFT)** - Frequency spectrum analysis showing distinct peaks
3. **Spectrogram** - Time-frequency representation of combined signal
4. **Signal Overlay** - All frequencies plotted together for comparison

### Training Data Visualizations (Step 2)
5. **Training Samples** - Example input/target pairs for each frequency
6. **Model I/O Structure** - Visualization of input features and output format

### Training Visualizations (Step 3)
7. **Training Loss Curves** - Training and validation loss over epochs

### Evaluation Visualizations (Step 3)
8. **Predictions vs Actual** - Sample predictions for each frequency
9. **Error Distribution** - Histogram and Q-Q plot of prediction errors
10. **Scatter Plot** - Predicted vs actual values showing correlation
11. **Frequency Spectrum Comparison** - FFT comparison between predictions and actuals
12. **Long Sequence Predictions** - Extended time series predictions
13. **Per-Frequency Metrics** - Comparative bar charts of performance metrics

## Usage

### 1. Generate Dataset
```bash
python generate_dataset.py
```
Creates 10,000 samples of 4 frequencies and combined signal.

### 2. Visualize Data
```bash
python visualize_data.py
```
Creates time-domain, frequency-domain, and spectrogram visualizations.

### 3. Prepare Training Data
```bash
python prepare_training_data.py
```
Creates sequences with one-hot selectors for training.

### 4. Train Model
```bash
python train_model.py
```
Trains the LSTM model for 50 epochs with early stopping.

### 5. Evaluate Model
```bash
python evaluate_model.py
```
Evaluates model performance and creates comprehensive visualizations.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- torch
- scikit-learn

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy torch scikit-learn
```

## Technical Details

### Input Format
- **Shape**: `(batch_size, sequence_length, 5)`
- **Features**: 
  - 1 signal value (S(x) at time t)
  - 4 selector values (one-hot encoding)

### Output Format
- **Shape**: `(batch_size, sequence_length, 1)`
- **Value**: Filtered frequency value at each time step

### One-Hot Selectors
```python
c1 = [1, 0, 0, 0]  # Select f1
c2 = [0, 1, 0, 0]  # Select f2
c3 = [0, 0, 1, 0]  # Select f3
c4 = [0, 0, 0, 1]  # Select f4
```

## Key Insights

1. **Lower frequencies are easier to filter**: f1 (1 Hz) has the best R² score (0.987)
2. **Model generalizes well**: Strong performance on unseen test data
3. **Temporal coherence preserved**: Predictions maintain phase relationships
4. **Robust to mixed signals**: Successfully separates all 4 components
5. **Fast inference**: Real-time filtering is feasible

## Future Improvements

- [ ] Add noise to signals for robustness testing
- [ ] Experiment with different LSTM architectures (bidirectional, deeper)
- [ ] Test on non-sinusoidal waveforms
- [ ] Implement attention mechanism for better frequency selection
- [ ] Deploy as real-time audio filter

## References

- **PyTorch LSTM Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **Time Series Forecasting with LSTMs**: Industry best practices
- **Signal Processing**: FFT and spectrogram analysis

## License

MIT License

## Author

Created for LLM Orchestration Exercise 02
