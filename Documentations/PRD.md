# Product Requirements Document (PRD)
## LSTM Frequency Filter

---

**Document Version:** 1.0  
**Date:** Q4 2024  
**Status:** Final  
**Owner:** Machine Learning Research Team  
**Stakeholders:** Data Science Team, Signal Processing Engineers, ML Researchers

---

## Executive Summary

### Project Overview
The LSTM Frequency Filter is a machine learning research project aimed at developing a deep learning solution for intelligent frequency component extraction from mixed signals. This project demonstrates the application of Long Short-Term Memory (LSTM) neural networks to signal processing tasks, specifically addressing the challenge of decomposing multi-frequency signals into individual components based on user-specified selection criteria.

### Key Objectives
1. Develop an LSTM-based model capable of extracting individual frequency components from mixed signals
2. Achieve R² score > 0.30 on test data, demonstrating meaningful variance explanation
3. Create a reproducible pipeline from data generation through model evaluation
4. Produce comprehensive visualizations documenting the entire process
5. Demonstrate superiority over baseline methods (random and mean predictions)

### Success Criteria Summary
- **Performance Target**: R² ≥ 0.30, Correlation ≥ 0.60
- **Baseline Improvement**: >40% better MAE than mean baseline
- **Deliverables**: 7 modular Python scripts, 14+ visualizations, complete documentation
- **Timeline**: 5-week development cycle

---

## 1. Product Overview

### 1.1 Purpose
This project aims to train an LSTM neural network to act as an intelligent frequency filter that can:
- Decompose complex multi-frequency signals into individual components
- Select specific frequencies using one-hot encoded selectors
- Process signals in the time domain without explicit Fourier transforms
- Handle phase-shifted sinusoidal components with additive noise

### 1.2 Background & Context

**Signal Processing Challenge**  
Traditional frequency filtering relies on Fourier-based methods (FFT, bandpass filters) that require explicit frequency domain transformations. These methods work well for clean signals but can struggle with noisy, overlapping frequency components.

**Machine Learning Opportunity**  
Recent advances in deep learning, particularly recurrent neural networks (RNNs) and LSTMs, have shown promise in learning complex temporal patterns. LSTMs excel at:
- Capturing long-term dependencies in sequential data
- Learning patterns without explicit feature engineering
- Handling variable noise conditions through training
- Processing time-domain signals naturally

**Research Question**  
Can an LSTM network learn to extract specific frequency components from a mixed signal in the time domain, using only a selector vector to indicate which frequency to isolate?

### 1.3 Problem Statement

**Current State**  
Frequency separation typically requires:
- Explicit frequency domain transformation (FFT)
- Manual design of filter parameters
- Knowledge of exact frequencies in advance
- Clean signals with high SNR

**Desired State**  
A neural network that:
- Operates directly on time-domain signals
- Learns filtering behavior from data
- Handles noisy input signals
- Uses simple one-hot selectors to choose frequencies
- Generalizes to unseen noise realizations

**Gap**  
Need for a proof-of-concept demonstrating that LSTMs can learn frequency filtering as a data-driven task, paving the way for more complex adaptive filtering systems.

### 1.4 Target Audience

**Primary Users:**
- **ML Researchers**: Studying time-series processing with neural networks
- **Signal Processing Engineers**: Exploring ML alternatives to traditional methods
- **Data Scientists**: Learning LSTM applications for regression tasks

**Secondary Users:**
- **Students**: Educational resource for LSTM and signal processing
- **Engineers**: Reference implementation for similar problems

---

## 2. Goals & Success Criteria

### 2.1 Primary Goals

**G1: Model Performance**
- Achieve R² score ≥ 0.30 on held-out test set
- Demonstrate correlation ≥ 0.60 between predictions and ground truth
- Outperform random baseline by >50% on MAE
- Outperform mean baseline by >40% on MAE

**G2: Frequency Separation**
- Successfully extract all 4 frequency components (1, 3, 5, 7 Hz)
- Handle phase offsets (0°, 45°, 90°, 135°) correctly
- Maintain performance with moderate noise (SNR ≈ 11 dB)

**G3: Reproducibility & Documentation**
- Provide complete pipeline from data generation to evaluation
- Ensure reproducibility through fixed random seeds
- Create comprehensive visualizations (≥10) showing all aspects
- Document all design decisions and hyperparameter choices

### 2.2 Success Metrics

| Metric Category | Metric | Target | Measurement Method |
|----------------|--------|--------|-------------------|
| **Primary Performance** | R² Score | ≥ 0.30 | sklearn.metrics.r2_score on test set |
| | Correlation | ≥ 0.60 | Pearson correlation coefficient |
| | RMSE | ≤ 0.60 | Root mean squared error |
| **Baseline Comparison** | MAE vs Random | >50% better | Mean absolute error comparison |
| | MAE vs Mean | >40% better | MAE against always-mean prediction |
| **Per-Frequency** | f₁ (1 Hz) R² | ≥ 0.50 | Individual frequency performance |
| | All frequencies | R² > 0 | Positive variance explanation |
| **Training** | Convergence | < 50 epochs | Early stopping or completion |
| | Training time | < 30 min | CPU training duration |
| **Deliverables** | Visualizations | ≥ 10 | Number of output plots |
| | Scripts | 7 | Modular components |

### 2.3 Non-Goals

This project explicitly does NOT aim to:
- Achieve state-of-the-art performance (research/educational focus)
- Process real-world audio signals (synthetic data only)
- Provide real-time inference capability
- Support variable/unknown frequency detection
- Deploy as production service or API
- Handle non-sinusoidal waveforms
- Support multi-channel or stereo signals

---

## 3. User Stories & Use Cases

### 3.1 User Stories

**US-001: Frequency Extraction**  
*As a* signal processing researcher  
*I want to* extract a specific frequency from a mixed signal  
*So that* I can isolate individual components for analysis  
*Acceptance Criteria:* Model achieves R² > 0.30 on test set

**US-002: Visual Understanding**  
*As a* data scientist  
*I want to* view comprehensive visualizations of data and results  
*So that* I can understand model behavior and performance  
*Acceptance Criteria:* ≥10 visualizations covering all pipeline stages

**US-003: Pipeline Reproduction**  
*As an* ML engineer  
*I want to* run the complete pipeline with a single command  
*So that* I can reproduce results and verify implementation  
*Acceptance Criteria:* `run_all.sh` executes all steps successfully

**US-004: Model Comparison**  
*As a* researcher  
*I want to* compare model performance against baselines  
*So that* I can validate that learning occurred  
*Acceptance Criteria:* >40% improvement over mean baseline

**US-005: Per-Frequency Analysis**  
*As a* signal processing engineer  
*I want to* see performance metrics for each frequency separately  
*So that* I can identify which frequencies are easier/harder to extract  
*Acceptance Criteria:* Individual metrics calculated and visualized

### 3.2 Use Cases

**UC-001: Research Demonstration**  
User runs complete pipeline to demonstrate LSTM capability for frequency filtering in a research presentation or paper.

**UC-002: Educational Tutorial**  
Student follows code to learn about LSTM architecture, time-series processing, and signal processing applications.

**UC-003: Baseline Comparison**  
Researcher uses this implementation as a baseline for comparing more advanced architectures (attention, transformers).

**UC-004: Hyperparameter Exploration**  
Data scientist modifies hyperparameters to study their effect on model performance for similar time-series tasks.

---

## 4. Functional Requirements

### 4.1 Data Generation Requirements

**FR-001: Signal Synthesis**  
*Priority: P0 (Critical)*  
The system shall generate synthetic signals with the following specifications:
- 4 distinct frequencies: f₁=1 Hz, f₂=3 Hz, f₃=5 Hz, f₄=7 Hz
- Time domain: [0, 10] seconds
- Sampling rate: 1000 Hz (10,000 samples total)
- Mathematical form: `sin(2π·fᵢ·t + θᵢ)`

**FR-002: Phase Configuration**  
*Priority: P0 (Critical)*  
Each frequency shall have a fixed phase offset:
- f₁: 0° (0 radians)
- f₂: 45° (π/4 radians)
- f₃: 90° (π/2 radians)
- f₄: 135° (3π/4 radians)

**FR-003: Signal Mixing**  
*Priority: P0 (Critical)*  
The system shall create a combined signal:
- Formula: `S(t) = (1/4) · Σᵢ sin(2π·fᵢ·t + θᵢ)`
- Equal weighting of all four components
- Normalized by factor of 1/4

**FR-004: Noise Addition**  
*Priority: P0 (Critical)*  
The system shall add Gaussian noise to the mixed signal:
- Distribution: Normal(μ=0, σ=0.1)
- Applied additively: `S_noisy(t) = S_clean(t) + ε`
- Target SNR: ≈11 dB
- Different random seeds for train and test sets

**FR-005: Data Export**  
*Priority: P0 (Critical)*  
The system shall save generated data in two formats:
- CSV format: `data/frequency_dataset.csv` (tabular, human-readable)
- NPZ format: `data/frequency_data.npz` (NumPy arrays, efficient)
- Include: x values, f₁(x), f₂(x), f₃(x), f₄(x), S(x)

### 4.2 Data Preparation Requirements

**FR-006: Sequence Creation**  
*Priority: P0 (Critical)*  
The system shall create training sequences using a sliding window:
- Window size: 50 timesteps
- Stride: 1 (maximum overlap)
- Total sequences: 9,951 from 10,000 samples

**FR-007: One-Hot Selector Generation**  
*Priority: P0 (Critical)*  
For each sequence, create 4 training samples with selectors:
- Selector for f₁: [1, 0, 0, 0]
- Selector for f₂: [0, 1, 0, 0]
- Selector for f₃: [0, 0, 1, 0]
- Selector for f₄: [0, 0, 0, 1]
- Total training samples: 39,804 (9,951 × 4)

**FR-008: Input Feature Construction**  
*Priority: P0 (Critical)*  
Each input sample shall have shape (50, 5):
- Dimension 0: Signal value S(t) at each timestep
- Dimensions 1-4: One-hot selector (constant across timesteps)

**FR-009: Target Construction**  
*Priority: P0 (Critical)*  
Each target sample shall have shape (50, 1):
- Contains the selected frequency fᵢ(t) values
- Matches the one-hot selector in corresponding input

**FR-010: Data Splitting**  
*Priority: P0 (Critical)*  
The system shall split data into three sets:
- Training: 80% (31,840 samples)
- Validation: 10% (3,980 samples)
- Test: 10% (3,980 samples)
- Save to: `data/training_data.npz`

### 4.3 Model Architecture Requirements

**FR-011: LSTM Architecture**  
*Priority: P0 (Critical)*  
The model shall implement a stacked LSTM architecture:
- Input size: 5 features (1 signal + 4 selectors)
- Hidden size: 128 units
- Number of layers: 2 (stacked)
- Output size: 1 (filtered signal value)
- Framework: PyTorch

**FR-012: Regularization**  
*Priority: P0 (Critical)*  
The model shall include regularization techniques:
- Dropout: 0.2 between LSTM layers
- Weight decay: 1e-5 (L2 regularization)
- Applied during training only

**FR-013: Model Parameters**  
*Priority: P1 (High)*  
The model shall have approximately 200,000 trainable parameters
- Total parameters: 201,345
- All parameters trainable (no frozen layers)

### 4.4 Training Requirements

**FR-014: Optimization Configuration**  
*Priority: P0 (Critical)*  
Training shall use the following configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Loss function: MSE (Mean Squared Error)
- Max epochs: 50

**FR-015: Training Process**  
*Priority: P0 (Critical)*  
The training process shall:
- Shuffle training data each epoch
- Validate on validation set each epoch
- Track both training and validation loss
- Apply gradient clipping (max norm = 1.0)

**FR-016: Learning Rate Scheduling**  
*Priority: P1 (High)*  
Implement ReduceLROnPlateau scheduler:
- Monitor: validation loss
- Factor: 0.5 (halve learning rate)
- Patience: 5 epochs

**FR-017: Early Stopping**  
*Priority: P1 (High)*  
Implement early stopping mechanism:
- Monitor: validation loss
- Patience: 15 epochs
- Save best model based on validation performance

**FR-018: Model Checkpointing**  
*Priority: P0 (Critical)*  
Save model artifacts:
- Best model weights: `models/best_model.pth`
- Training history: `models/training_history.npz`
- Include: train/val losses per epoch

**FR-019: Reproducibility**  
*Priority: P0 (Critical)*  
Ensure reproducible results:
- Set random seed: 42 (NumPy, PyTorch, CUDA)
- Deterministic operations where possible
- Document all hyperparameters

**FR-020: Training Monitoring**  
*Priority: P1 (High)*  
Display training progress:
- Print epoch number, losses, time per epoch
- Show progress every epoch
- Final training summary

### 4.5 Evaluation Requirements

**FR-021: Metric Calculation**  
*Priority: P0 (Critical)*  
Calculate comprehensive performance metrics on test set:
- R² Score (coefficient of determination)
- Pearson correlation coefficient
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**FR-022: Baseline Comparison**  
*Priority: P0 (Critical)*  
Compare against baseline methods:
- Random baseline: random noise predictions
- Mean baseline: always predict mean value
- Calculate MAE for each baseline

**FR-023: Per-Frequency Metrics**  
*Priority: P1 (High)*  
Calculate metrics separately for each frequency:
- Individual R², RMSE, MAE for f₁, f₂, f₃, f₄
- Identify best/worst performing frequencies
- Analyze performance patterns

**FR-024: Error Analysis**  
*Priority: P1 (High)*  
Analyze prediction errors:
- Error distribution (histogram)
- Q-Q plot for normality
- Identify systematic biases

### 4.6 Visualization Requirements

**FR-025: Complete Overview**  
*Priority: P0 (Critical)*  
Create single-page overview visualization showing:
- Dataset statistics
- Model architecture diagram
- Training curves
- Final metrics

**FR-026: Data Visualizations**  
*Priority: P0 (Critical)*  
Generate visualizations of input data:
- Time domain signals (individual frequencies)
- Frequency domain (FFT) analysis
- Spectrogram (time-frequency representation)
- Signal overlay (all frequencies + combined)
- Training sample structure

**FR-027: Training Visualizations**  
*Priority: P0 (Critical)*  
Visualize training process:
- Training and validation loss curves
- Log scale for better visibility
- Mark best epoch

**FR-028: Prediction Visualizations**  
*Priority: P0 (Critical)*  
Show model predictions:
- Sample predictions vs actual (all 4 frequencies)
- Long sequence predictions (extended time series)
- Scatter plot: predicted vs actual

**FR-029: Performance Visualizations**  
*Priority: P1 (High)*  
Visualize performance metrics:
- Error distribution histogram
- Frequency spectrum comparison (FFT)
- Per-frequency metric comparison (bar charts)

**FR-030: Visualization Export**  
*Priority: P0 (Critical)*  
Save all visualizations:
- Format: PNG (high resolution)
- Location: `visualizations/` directory
- Naming: Numbered with descriptive names
- Minimum 10 visualizations required

### 4.7 Pipeline & Automation Requirements

**FR-031: Modular Scripts**  
*Priority: P0 (Critical)*  
Implement 7 separate Python scripts:
1. `generate_dataset.py` - Create synthetic data
2. `visualize_data.py` - Visualize raw data
3. `prepare_training_data.py` - Create sequences
4. `train_model.py` - Train LSTM model
5. `evaluate_model.py` - Test and evaluate
6. `summary.py` - Display results summary
7. `create_overview.py` - Generate overview viz

**FR-032: Pipeline Automation**  
*Priority: P1 (High)*  
Provide complete pipeline execution:
- Shell script: `run_all.sh`
- Executes all 7 scripts in sequence
- Handles dependencies between steps
- Provides progress feedback

**FR-033: Error Handling**  
*Priority: P1 (High)*  
Each script shall:
- Create required directories if missing
- Validate input data before processing
- Provide informative error messages
- Exit gracefully on errors

---

## 5. Technical Requirements

### 5.1 Technology Stack

**TR-001: Deep Learning Framework**  
*Framework: PyTorch 2.0+*

**Rationale:**
- Excellent LSTM implementation with flexible architecture
- Dynamic computation graphs (easier debugging)
- Strong community support and documentation
- Industry standard for research and production
- Better control over training process vs TensorFlow/Keras
- Native Python integration
- Efficient GPU/CPU computation

**TR-002: Core Dependencies**

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computations, array operations |
| pandas | ≥2.0.0 | Data manipulation, CSV handling |
| matplotlib | ≥3.7.0 | Visualization, plotting |
| scipy | ≥1.10.0 | Signal processing (FFT, spectrograms) |
| torch | ≥2.0.0 | Deep learning framework |
| scikit-learn | ≥1.3.0 | Metrics, data splitting |

**TR-003: Python Version**  
- Minimum: Python 3.8
- Recommended: Python 3.10+
- Reasoning: Modern language features, type hints, performance

### 5.2 Model Architecture Specifications

**TR-004: Input Tensor Format**
```
Shape: (batch_size, sequence_length, features)
       (B, 50, 5)

Feature dimensions:
  [0] - S(t): Combined signal value
  [1] - c₁: Selector for f₁ (binary)
  [2] - c₂: Selector for f₂ (binary)
  [3] - c₃: Selector for f₃ (binary)
  [4] - c₄: Selector for f₄ (binary)
```

**TR-005: LSTM Layer Configuration**
```
Layer 1: LSTM(input_size=5, hidden_size=128, num_layers=1)
         Output: (B, 50, 128)
         
Dropout: 0.2

Layer 2: LSTM(input_size=128, hidden_size=128, num_layers=1)
         Output: (B, 50, 128)
         
Dropout: 0.2

Linear:  Linear(in_features=128, out_features=1)
         Output: (B, 50, 1)
```

**TR-006: Output Tensor Format**
```
Shape: (batch_size, sequence_length, output_dim)
       (B, 50, 1)

Values: Amplitude of selected frequency fᵢ(t)
Range: Approximately [-1.5, +1.5] (sinusoidal)
```

**TR-007: Model Parameter Count**
```
Total parameters: 201,345
Trainable parameters: 201,345
Non-trainable parameters: 0

Memory footprint: ~0.8 MB (FP32)
```

### 5.3 Data Specifications

**TR-008: Dataset Size Requirements**
- Training samples: 31,840
- Validation samples: 3,980
- Test samples: 3,980
- Total: 39,800 sequences

**TR-009: Data Types**
- All numerical data: float32
- Selectors: float32 (0.0 or 1.0)
- Ensures consistency with PyTorch tensors

**TR-010: Storage Requirements**
- Raw dataset (NPZ): ~2 MB
- Training data (NPZ): ~40 MB
- Model checkpoint: ~0.8 MB
- Visualizations: ~5 MB
- Total: ~50 MB

### 5.4 Hardware Requirements

**TR-011: Minimum Hardware**
- CPU: Any modern multi-core processor
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 500 MB free space
- GPU: Not required (CPU training acceptable)

**TR-012: Performance Expectations**
- Training time: 15-30 minutes on CPU
- Inference time: <1 second for 3,980 test samples
- Dataset generation: <10 seconds
- Visualization creation: ~30 seconds

---

## 6. Non-Functional Requirements

### 6.1 Performance Requirements

**NFR-001: Training Efficiency**  
Training shall complete within 30 minutes on standard CPU hardware
- Target: ~18 seconds per epoch × 50 epochs
- Acceptable range: 15-30 minutes total

**NFR-002: Inference Speed**  
Model inference shall be fast enough for batch processing
- Test set prediction: < 5 seconds for ~4,000 samples
- Single sample: < 1 millisecond

**NFR-003: Memory Efficiency**  
Peak memory usage shall not exceed 2 GB RAM
- Enables execution on standard laptops
- Batch size 64 fits comfortably in memory

### 6.2 Reliability Requirements

**NFR-004: Reproducibility**  
Results shall be reproducible across different runs
- Fixed random seeds (42) for all random operations
- Deterministic operations where possible
- Same results on same hardware/software

**NFR-005: Numerical Stability**  
Training shall be numerically stable
- Gradient clipping prevents exploding gradients
- No NaN or Inf values during training
- Loss decreases smoothly

### 6.3 Maintainability Requirements

**NFR-006: Code Modularity**  
Codebase shall be organized into separate, focused scripts
- Each script has single responsibility
- Clear interfaces between components
- Easy to modify individual steps

**NFR-007: Code Readability**  
Code shall be well-documented and readable
- Descriptive variable names
- Comments explaining complex logic
- Docstrings for functions (where beneficial)

**NFR-008: Dependency Management**  
Dependencies shall be clearly specified
- `requirements.txt` with version constraints
- Compatible with pip package manager
- No exotic or hard-to-install packages

### 6.4 Portability Requirements

**NFR-009: Cross-Platform Compatibility**  
Code shall run on Windows, macOS, and Linux
- Python standard library usage
- Cross-platform path handling (pathlib)
- No OS-specific dependencies

**NFR-010: Environment Isolation**  
Project shall support virtual environment setup
- Compatible with venv, virtualenv
- No global package installation required

### 6.5 Documentation Requirements

**NFR-011: README Completeness**  
README shall provide comprehensive documentation including:
- Mathematical background and equations
- Architecture diagrams and explanations
- Usage instructions and examples
- Results and visualizations
- References and citations

**NFR-012: Code Comments**  
Code shall include explanatory comments for:
- Hyperparameter choices and rationale
- Complex mathematical operations
- Non-obvious design decisions
- Important assumptions

### 6.6 Testing & Validation Requirements

**NFR-013: Data Validation**  
Scripts shall validate data integrity:
- Check for NaN or Inf values
- Verify expected shapes and types
- Confirm file existence before loading

**NFR-014: Sanity Checks**  
Include sanity checks during execution:
- Data ranges are reasonable
- Loss values decrease during training
- Metrics are within expected bounds

---

## 7. Project Deliverables

### 7.1 Code Deliverables

| Deliverable | Description | Acceptance Criteria |
|------------|-------------|---------------------|
| `generate_dataset.py` | Dataset creation script | Generates 10,000 samples, saves CSV & NPZ |
| `visualize_data.py` | Data visualization | Creates 4+ visualizations of raw data |
| `prepare_training_data.py` | Training data prep | Creates sequences, splits data 80/10/10 |
| `train_model.py` | Model training | Trains LSTM, saves best model & history |
| `evaluate_model.py` | Model evaluation | Tests model, generates 6+ visualizations |
| `summary.py` | Results summary | Displays complete metrics and achievements |
| `create_overview.py` | Overview generation | Creates single-page summary visualization |
| `run_all.sh` | Pipeline automation | Executes complete pipeline successfully |

### 7.2 Model Artifacts

| Artifact | Description | Acceptance Criteria |
|----------|-------------|---------------------|
| `models/best_model.pth` | Trained model weights | ~201K parameters, loads correctly |
| `models/training_history.npz` | Training metrics | Contains loss curves for all epochs |
| `models/evaluation_results.npz` | Test metrics | All evaluation metrics included |

### 7.3 Data Artifacts

| Artifact | Description | Acceptance Criteria |
|----------|-------------|---------------------|
| `data/frequency_dataset.csv` | Raw data (tabular) | 10,000 rows, 6 columns |
| `data/frequency_data.npz` | Raw data (arrays) | Contains x, f1-f4, S arrays |
| `data/training_data.npz` | Training sequences | 39,800 sequences, split into train/val/test |

### 7.4 Visualization Deliverables

Minimum 14 visualizations covering:

| ID | Visualization | Purpose |
|----|---------------|---------|
| 00 | Complete overview | Single-page project summary |
| 01 | Time domain signals | Individual frequency waveforms |
| 02 | Frequency domain FFT | Fourier analysis |
| 03 | Spectrogram | Time-frequency representation |
| 04 | Signal overlay | All frequencies combined |
| 05 | Training samples | Input/output pairs |
| 06 | Model I/O structure | Architecture diagram |
| 07 | Training loss | Loss curves |
| 08 | Predictions vs actual | Sample predictions |
| 09 | Error distribution | Error analysis |
| 10 | Scatter plot | Correlation visualization |
| 11 | Frequency spectrum | FFT comparison |
| 12 | Long sequences | Extended predictions |
| 13 | Per-frequency metrics | Performance comparison |

### 7.5 Documentation Deliverables

| Deliverable | Description | Acceptance Criteria |
|------------|-------------|---------------------|
| `README.md` | Complete project docs | Includes theory, usage, results, references |
| `requirements.txt` | Dependency specification | All packages with versions |
| `pyproject.toml` | Project metadata | Basic project configuration |

---

## 8. Timeline & Milestones

### 8.1 Development Phases

**Phase 1: Foundation & Data Generation (Week 1)**
- **Goal**: Create synthetic dataset and visualizations
- **Deliverables**:
  - `generate_dataset.py`
  - `visualize_data.py`
  - Initial data visualizations (4+)
- **Success Criteria**: Dataset created with correct specifications
- **Risks**: Ensuring noise level is appropriate for learning

**Phase 2: Data Preparation & Model Design (Week 2)**
- **Goal**: Prepare training data and define model architecture
- **Deliverables**:
  - `prepare_training_data.py`
  - Model architecture implementation in `train_model.py`
  - Training data sequences
- **Success Criteria**: Sequences created correctly, model compiles
- **Risks**: Input/output shape mismatches

**Phase 3: Model Training & Optimization (Week 3)**
- **Goal**: Train model and optimize hyperparameters
- **Deliverables**:
  - Complete `train_model.py`
  - Trained model checkpoint
  - Training history
- **Success Criteria**: Model converges, R² > 0.30
- **Risks**: Poor performance, overfitting, long training time

**Phase 4: Evaluation & Visualization (Week 4)**
- **Goal**: Evaluate model and create comprehensive visualizations
- **Deliverables**:
  - `evaluate_model.py`
  - `create_overview.py`
  - All evaluation visualizations (10+)
- **Success Criteria**: All metrics calculated, visualizations generated
- **Risks**: Unclear visualizations, missing metrics

**Phase 5: Documentation & Polish (Week 5)**
- **Goal**: Complete documentation and pipeline automation
- **Deliverables**:
  - `summary.py`
  - `run_all.sh`
  - Complete `README.md`
- **Success Criteria**: Full pipeline runs successfully, complete docs
- **Risks**: Integration issues between scripts

### 8.2 Key Milestones

| Milestone | Target Date | Exit Criteria |
|-----------|-------------|---------------|
| **M1: Data Ready** | End Week 1 | Dataset generated and visualized |
| **M2: Model Defined** | End Week 2 | Architecture implemented and compiles |
| **M3: Model Trained** | End Week 3 | Model achieves R² > 0.30 |
| **M4: Evaluation Complete** | End Week 4 | All metrics and visualizations ready |
| **M5: Project Complete** | End Week 5 | Full pipeline working, docs complete |

### 8.3 Dependencies & Critical Path

```
Data Generation (1 week)
    ↓
Data Preparation (1 week) ← Blocking dependency
    ↓
Model Training (1 week) ← Blocking dependency
    ↓
Evaluation (1 week) ← Blocking dependency
    ↓
Documentation (1 week) ← Can parallelize with evaluation
```

**Critical Path**: Data Generation → Preparation → Training → Evaluation (4 weeks)  
**Total Timeline**: 5 weeks (includes documentation polish)

---

## 9. Risk Assessment & Mitigation

### 9.1 Technical Risks

**RISK-001: Poor Model Performance**  
*Likelihood: Medium | Impact: High*

**Description**: Model fails to achieve target R² > 0.30

**Root Causes**:
- Task too difficult (excessive noise, overlapping frequencies)
- Insufficient model capacity
- Poor hyperparameter choices
- Training instability

**Mitigation Strategies**:
1. **Task Design**: Use fixed phase offsets + additive Gaussian noise (not random phase per sample)
   - This preserves frequency structure, making task learnable
   - Initial approach with random phases achieved R² = -0.45
   - Fixed phases + Gaussian noise achieves R² = 0.35 (+178% improvement)

2. **Noise Level**: Set σ=0.1 for moderate noise (SNR ≈ 11 dB)
   - Too high: task becomes impossible
   - Too low: model doesn't learn robustness

3. **Model Capacity**: Use 128 hidden units × 2 layers = 201K parameters
   - Sufficient for 4 frequency patterns
   - Not so large that overfitting dominates

4. **Hyperparameter Tuning**: Test different configurations if needed
   - Learning rate: 0.0001 to 0.01
   - Hidden size: 64 to 256
   - Layers: 1 to 3

**Contingency**: If R² < 0.20, reduce noise or increase model size

---

**RISK-002: Overfitting**  
*Likelihood: Medium | Impact: Medium*

**Description**: Model memorizes training data, poor generalization

**Indicators**:
- Training loss << validation loss
- High training R², low test R²
- Model performs well on seen sequences, poorly on unseen

**Mitigation Strategies**:
1. **Regularization**:
   - Dropout: 0.2 between LSTM layers
   - Weight decay: 1e-5 (L2 regularization)

2. **Separate Noise Realizations**:
   - Train/validation: Random seed #1
   - Test: Random seed #2
   - Forces model to learn signal structure, not specific noise

3. **Early Stopping**:
   - Monitor validation loss
   - Patience: 15 epochs
   - Stops before severe overfitting occurs

4. **Data Splitting**:
   - 80/10/10 split provides adequate test set
   - Validation set used for hyperparameter selection

**Contingency**: If overfitting detected, increase dropout or reduce model size

---

**RISK-003: Long Training Time**  
*Likelihood: Low | Impact: Medium*

**Description**: Training exceeds 30-minute target, impacting iteration speed

**Root Causes**:
- Inefficient batch size
- Too many epochs without early stopping
- Large model architecture
- Inefficient data loading

**Mitigation Strategies**:
1. **Optimized Batch Size**: 64 provides good GPU/CPU utilization
2. **Early Stopping**: Prevents unnecessary epochs
3. **Efficient Architecture**: 2 layers (not 3-4) balances performance and speed
4. **CPU Acceptable**: ~15 min on CPU is reasonable for research

**Monitoring**: Track time per epoch, target ~18 seconds

**Contingency**: If training time excessive, reduce epochs or increase batch size

---

**RISK-004: Reproducibility Issues**  
*Likelihood: Low | Impact: High*

**Description**: Results vary between runs, reducing credibility

**Root Causes**:
- Uncontrolled random number generation
- Non-deterministic operations (some CUDA ops)
- Missing seed initialization
- Different library versions

**Mitigation Strategies**:
1. **Fixed Seeds**: Set seed=42 for NumPy, PyTorch, CUDA
2. **Version Pinning**: Specify minimum versions in requirements.txt
3. **Documentation**: Document all random sources
4. **Testing**: Verify reproducibility on multiple machines

**Monitoring**: Run pipeline twice, verify identical results

**Contingency**: Document any sources of non-determinism

---

### 9.2 Project Risks

**RISK-005: Scope Creep**  
*Likelihood: Medium | Impact: Medium*

**Description**: Adding features beyond original scope delays completion

**Examples**:
- Adding more frequencies (8, 16, etc.)
- Implementing attention mechanisms
- Real audio signal processing
- Web interface development

**Mitigation**:
- Clearly define out-of-scope items in PRD
- Mark nice-to-have features as "Future Enhancements"
- Focus on achieving core success criteria first

**Contingency**: Defer enhancements to post-v1.0 releases

---

**RISK-006: Insufficient Documentation**  
*Likelihood: Low | Impact: Medium*

**Description**: Users cannot understand or reproduce results

**Mitigation**:
- Comprehensive README with theory and usage
- Code comments for complex sections
- Visual diagrams of architecture and data flow
- Complete equation documentation
- Example outputs and visualizations

**Monitoring**: Review docs for clarity and completeness

---

### 9.3 Risk Summary Matrix

| Risk | Likelihood | Impact | Priority | Status |
|------|-----------|--------|----------|--------|
| Poor Performance | Medium | High | **P0** | Mitigated (task design) |
| Overfitting | Medium | Medium | **P1** | Mitigated (regularization) |
| Long Training | Low | Medium | P2 | Mitigated (architecture) |
| Reproducibility | Low | High | **P1** | Mitigated (seeds) |
| Scope Creep | Medium | Medium | P2 | Managed (clear scope) |
| Poor Documentation | Low | Medium | P2 | Mitigated (comprehensive docs) |

---

## 10. Assumptions & Constraints

### 10.1 Assumptions

**A-001: Data Assumptions**
- 4 frequencies are sufficient to demonstrate concept
- Sinusoidal waveforms adequately represent signal processing challenge
- 10,000 samples provide enough data for training
- Moderate noise (SNR ≈ 11 dB) creates realistic but learnable task

**A-002: Hardware Assumptions**
- CPU training is acceptable for research/educational project
- Users have standard laptop/desktop (4GB+ RAM)
- 500 MB storage available for project files

**A-003: User Assumptions**
- Users have basic Python knowledge
- Users can install packages via pip
- Users understand basic signal processing concepts (helpful but not required)

**A-004: Training Assumptions**
- 50 epochs sufficient for convergence
- Adam optimizer with default settings works well
- MSE is appropriate loss function for this regression task

**A-005: Performance Assumptions**
- R² > 0.30 indicates meaningful learning for this difficult task
- Comparison to baselines demonstrates genuine pattern learning
- Lower frequencies (f₁) naturally easier to extract than higher (f₄)

### 10.2 Constraints

**C-001: Time Domain Only**
- Model operates purely in time domain
- No explicit Fourier features provided to model
- This is an intentional design choice to test LSTM capability

**C-002: Fixed Frequencies**
- Frequencies are fixed (1, 3, 5, 7 Hz)
- Model cannot adapt to arbitrary frequencies
- Not a generalized frequency detector

**C-003: Synthetic Data Only**
- Uses synthetic sinusoidal data
- Not tested on real-world audio signals
- Real signals would require additional preprocessing

**C-004: Single Channel**
- Mono signal only (single time series)
- No multi-channel or stereo processing
- No spatial information

**C-005: Offline Processing**
- Batch processing only, not real-time
- No streaming capabilities
- No online learning

**C-006: Resource Constraints**
- CPU-only training (no GPU required)
- Limited to ~200K parameters for reasonable training time
- Memory footprint kept under 2 GB

**C-007: Framework Constraint**
- PyTorch-specific implementation
- Not framework-agnostic
- Dependencies on PyTorch ecosystem

### 10.3 Dependencies

**D-001: External Libraries**
- Project depends on 6 core libraries (numpy, pandas, matplotlib, scipy, torch, sklearn)
- Breaking changes in these libraries could affect functionality
- Minimum versions specified to ensure compatibility

**D-002: Python Version**
- Requires Python 3.8+ for language features
- Type hints, pathlib, f-strings used throughout

**D-003: Sequential Pipeline**
- Each step depends on previous step completing successfully
- Cannot run training before data generation
- Cannot evaluate before training

---

## 11. Out of Scope

The following items are explicitly **OUT OF SCOPE** for this project:

### 11.1 Real-Time Processing
- ❌ No streaming data support
- ❌ No online inference API
- ❌ No latency optimization for real-time use
- **Rationale**: Research focus, not production deployment

### 11.2 Variable/Adaptive Frequencies
- ❌ No detection of arbitrary frequencies
- ❌ No adaptation to unknown frequency sets
- ❌ No frequency estimation capability
- **Rationale**: Fixed demonstration scenario sufficient for proof-of-concept

### 11.3 Non-Sinusoidal Waveforms
- ❌ No square waves, triangle waves, sawtooth
- ❌ No complex modulated signals
- ❌ No amplitude modulation (AM) or frequency modulation (FM)
- **Rationale**: Sinusoids are foundational and sufficient

### 11.4 Real-World Audio
- ❌ No music processing
- ❌ No speech processing
- ❌ No environmental sound analysis
- **Rationale**: Synthetic data eliminates confounding variables

### 11.5 Multi-Channel Signals
- ❌ No stereo processing
- ❌ No multi-sensor arrays
- ❌ No spatial filtering
- **Rationale**: Single channel sufficient for concept demonstration

### 11.6 Production Deployment
- ❌ No web interface or REST API
- ❌ No mobile deployment
- ❌ No containerization (Docker)
- ❌ No cloud deployment
- **Rationale**: Educational/research project

### 11.7 Advanced Architectures
- ❌ No attention mechanisms (beyond basic LSTM)
- ❌ No transformer models
- ❌ No CNN-LSTM hybrids
- **Rationale**: Focus on demonstrating basic LSTM capability

### 11.8 Hyperparameter Optimization
- ❌ No automated hyperparameter search
- ❌ No Bayesian optimization
- ❌ No grid search implementation
- **Rationale**: Manual hyperparameter selection sufficient

### 11.9 Advanced Features
- ❌ No model compression or quantization
- ❌ No knowledge distillation
- ❌ No federated learning
- ❌ No adversarial training
- **Rationale**: Beyond scope of basic research project

---

## 12. Future Enhancements

Items that could be added in future versions:

### 12.1 Short-Term Enhancements (v1.1 - v1.3)

**E-001: Extended Training**
- Train for 100-200 epochs instead of 50
- Potentially improve R² from 0.35 to 0.40+
- **Effort**: Low | **Impact**: Medium

**E-002: Larger Model**
- Increase hidden size from 128 to 256 units
- Add third LSTM layer
- **Effort**: Low | **Impact**: Medium

**E-003: Bidirectional LSTM**
- Process sequences in both directions
- May improve accuracy by ~5-10%
- **Effort**: Medium | **Impact**: Medium

**E-004: Lower Noise Level**
- Reduce σ from 0.1 to 0.05 (easier task)
- Study performance vs noise trade-off
- **Effort**: Low | **Impact**: Low

**E-005: More Training Data**
- Generate 50K samples instead of 10K
- Extend time domain to [0, 50] seconds
- **Effort**: Low | **Impact**: Medium

### 12.2 Medium-Term Enhancements (v2.0)

**E-006: Attention Mechanism**
- Add attention layer on top of LSTM
- Let model focus on relevant timesteps
- **Effort**: High | **Impact**: High

**E-007: Multi-Frequency Selection**
- Extract multiple frequencies simultaneously
- Selector: [1, 0, 1, 0] → extract f₁ + f₃
- **Effort**: Medium | **Impact**: High

**E-008: More Frequencies**
- Expand from 4 to 8-16 frequencies
- Test model scalability
- **Effort**: Medium | **Impact**: Medium

**E-009: Variable Noise Levels**
- Train on multiple SNRs
- Create noise-robust model
- **Effort**: Medium | **Impact**: High

### 12.3 Long-Term Research Directions (v3.0+)

**E-010: Transformer Architecture**
- Replace LSTM with Transformer
- Compare performance and efficiency
- **Effort**: High | **Impact**: High

**E-011: Real Audio Signals**
- Apply to music/speech
- Requires significant preprocessing
- **Effort**: Very High | **Impact**: Very High

**E-012: Adaptive Frequency Filtering**
- Learn to filter arbitrary frequencies
- Provide target frequency as input (not just selection)
- **Effort**: Very High | **Impact**: Very High

**E-013: Phase Estimation**
- Extract phase θᵢ in addition to amplitude
- Complex-valued output
- **Effort**: High | **Impact**: High

**E-014: Real-Time Deployment**
- Optimize for streaming inference
- Create web interface
- Deploy as microservice
- **Effort**: Very High | **Impact**: Medium

---

## 13. Appendices

### Appendix A: Mathematical Formulations

**Signal Generation**

Individual frequency components:
```
fᵢ(t) = sin(2π·fᵢ·t + θᵢ)

where:
  f₁ = 1 Hz, θ₁ = 0°
  f₂ = 3 Hz, θ₂ = 45°
  f₃ = 5 Hz, θ₃ = 90°
  f₄ = 7 Hz, θ₄ = 135°
  t ∈ [0, 10] seconds
```

Combined signal (clean):
```
S_clean(t) = (1/4) · Σᵢ₌₁⁴ fᵢ(t)
```

Combined signal (noisy):
```
S_noisy(t) = S_clean(t) + ε
where ε ~ N(0, σ²), σ = 0.1
```

**Performance Metrics**

R² Score (Coefficient of Determination):
```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(yᵢ - ŷᵢ)²   (residual sum of squares)
  SS_tot = Σ(yᵢ - ȳ)²    (total sum of squares)
```

Mean Squared Error:
```
MSE = (1/n) · Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

Root Mean Squared Error:
```
RMSE = √MSE
```

Mean Absolute Error:
```
MAE = (1/n) · Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

Pearson Correlation:
```
r = Σ[(yᵢ - ȳ)(ŷᵢ - ŷ̄)] / √[Σ(yᵢ - ȳ)² · Σ(ŷᵢ - ŷ̄)²]
```

### Appendix B: Architecture Diagram

```
Input Shape: (batch=64, seq=50, features=5)
                    ↓
┌────────────────────────────────────────────┐
│         LSTM Layer 1 (128 units)           │
│  - Processes sequential input              │
│  - Captures short-term patterns            │
└────────────────┬───────────────────────────┘
                 ↓ (64, 50, 128)
┌────────────────────────────────────────────┐
│         Dropout (rate=0.2)                 │
└────────────────┬───────────────────────────┘
                 ↓
┌────────────────────────────────────────────┐
│         LSTM Layer 2 (128 units)           │
│  - Captures higher-level patterns          │
│  - Learns frequency-specific features      │
└────────────────┬───────────────────────────┘
                 ↓ (64, 50, 128)
┌────────────────────────────────────────────┐
│         Dropout (rate=0.2)                 │
└────────────────┬───────────────────────────┘
                 ↓
┌────────────────────────────────────────────┐
│      Fully Connected (128 → 1)             │
│  - Maps features to output amplitude       │
└────────────────┬───────────────────────────┘
                 ↓
Output Shape: (batch=64, seq=50, output=1)
```

### Appendix C: Dataset Statistics

**Frequency Specifications**

| Frequency | Hz | Period (s) | Wavelength @ 1000 Hz | Cycles in 10s | Phase (deg) | Phase (rad) |
|-----------|-----|-----------|---------------------|---------------|-------------|-------------|
| f₁ | 1.0 | 1.000 | 1000 samples | 10 | 0° | 0.000 |
| f₂ | 3.0 | 0.333 | 333 samples | 30 | 45° | 0.785 |
| f₃ | 5.0 | 0.200 | 200 samples | 50 | 90° | 1.571 |
| f₄ | 7.0 | 0.143 | 143 samples | 70 | 135° | 2.356 |

**Signal Statistics**

- Mean: ~0 (centered)
- Standard deviation: 1.41 (√2)
- Amplitude range: [-4, +4]
- Sampling rate: 1000 Hz
- Duration: 10 seconds
- Total samples: 10,000

**Noise Statistics**

- Distribution: Gaussian
- Mean: 0
- Standard deviation: 0.1
- Signal-to-Noise Ratio: ~11 dB
- Different realizations for train/test

### Appendix D: Hyperparameter Rationale

**Hidden Size: 128**
- Sufficient capacity for 4 frequency patterns
- Not so large that training is slow or overfitting dominates
- Standard choice for medium-complexity sequence tasks
- Provides 201,345 total parameters

**Number of Layers: 2**
- Single layer insufficient for complex temporal patterns
- Two layers capture both local and longer-term patterns
- Three+ layers show diminishing returns and slower training

**Dropout: 0.2**
- Prevents overfitting without hurting performance
- Applied between LSTM layers and before output
- Standard rate for RNN architectures

**Batch Size: 64**
- Good balance for CPU training efficiency
- Provides stable gradient estimates
- Fits comfortably in memory
- 497 batches per epoch (31,840 / 64)

**Learning Rate: 0.001**
- Adam optimizer default
- Proven effective for LSTMs
- No learning rate tuning needed initially

**Loss Function: MSE**
- Standard for regression tasks
- Penalizes large errors more heavily
- Provides smooth gradients for optimization

**Sequence Length: 50**
- Captures ~7 complete cycles of f₄ (highest frequency)
- Provides sufficient context for f₁ (5 complete cycles)
- Not so long that training is slow
- Balances memory usage and temporal context

### Appendix E: References & Citations

**Foundational Research**

1. **Hochreiter, S., & Schmidhuber, J. (1997)**  
   "Long Short-Term Memory"  
   Neural Computation, 9(8), 1735-1780  
   [https://www.bioinf.jku.at/publications/older/2604.pdf](https://www.bioinf.jku.at/publications/older/2604.pdf)

2. **Greff, K., et al. (2017)**  
   "LSTM: A Search Space Odyssey"  
   IEEE Transactions on Neural Networks and Learning Systems

**Technical Documentation**

3. **PyTorch Documentation**  
   LSTM Layer Implementation  
   [https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

4. **PyTorch Tutorials**  
   Time Series Forecasting with Deep Learning  
   [https://pytorch.org/tutorials/beginner/timeseries_tutorial.html](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)

**Signal Processing Background**

5. **Digital Signal Processing**  
   Wikipedia Article  
   [https://en.wikipedia.org/wiki/Digital_signal_processing](https://en.wikipedia.org/wiki/Digital_signal_processing)

6. **Fourier Transform**  
   Mathematical foundation for frequency analysis  
   [https://en.wikipedia.org/wiki/Fourier_transform](https://en.wikipedia.org/wiki/Fourier_transform)

**Libraries Used**

7. **PyTorch** (v2.0+): Deep learning framework
8. **NumPy** (v1.24+): Numerical computing
9. **Matplotlib** (v3.7+): Visualization
10. **SciPy** (v1.10+): Scientific computing, FFT
11. **scikit-learn** (v1.3+): Machine learning utilities
12. **pandas** (v2.0+): Data manipulation

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Project Owner** | ML Research Team | ___________ | _______ |
| **Technical Lead** | Data Science Team | ___________ | _______ |
| **Stakeholder** | Signal Processing Engineers | ___________ | _______ |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Q4 2024 | ML Research Team | Initial PRD creation |

---

**END OF DOCUMENT**

