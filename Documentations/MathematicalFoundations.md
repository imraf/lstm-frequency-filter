# Mathematical Foundations

## Overview

This document provides comprehensive mathematical background for the LSTM Frequency Filter project, covering signal processing theory, neural network mathematics, optimization theory, and evaluation metrics. All equations are derived and explained in detail.

## Table of Contents

1. [Signal Processing Fundamentals](#signal-processing-fundamentals)
2. [LSTM Mathematics](#lstm-mathematics)
3. [Optimization Theory](#optimization-theory)
4. [Loss Functions](#loss-functions)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Statistical Foundations](#statistical-foundations)

---

## Signal Processing Fundamentals

### Sinusoidal Signals

#### Basic Sinusoid

\[ x(t) = A \sin(2\pi f t + \theta) \]

**Parameters:**
- \( A \): Amplitude (peak value)
- \( f \): Frequency in Hz (cycles per second)
- \( t \): Time in seconds
- \( \theta \): Phase offset in radians

**Period:**

\[ T = \frac{1}{f} \]

**Angular frequency:**

\[ \omega = 2\pi f \text{ (rad/s)} \]

**Alternative form:**

\[ x(t) = A \sin(\omega t + \theta) \]

#### Complex Exponential Representation

Using Euler's formula:

\[ e^{j\theta} = \cos(\theta) + j\sin(\theta) \]

A sinusoid can be written as:

\[ x(t) = A \sin(2\pi f t + \theta) = \text{Im}\{A e^{j(2\pi f t + \theta)}\} \]

Or using real and imaginary parts:

\[ x(t) = \frac{A}{2j} \left( e^{j(2\pi f t + \theta)} - e^{-j(2\pi f t + \theta)} \right) \]

### Multi-Frequency Signals

#### Linear Combination

\[ S(t) = \sum_{i=1}^{N} A_i \sin(2\pi f_i t + \theta_i) \]

**Our specific case (N=4):**

\[ S(t) = \frac{1}{4} \sum_{i=1}^{4} \sin(2\pi f_i t + \theta_i) \]

With:
- \( f_1 = 1 \) Hz, \( \theta_1 = 0 \)
- \( f_2 = 3 \) Hz, \( \theta_2 = \pi/4 \)
- \( f_3 = 5 \) Hz, \( \theta_3 = \pi/2 \)
- \( f_4 = 7 \) Hz, \( \theta_4 = 3\pi/4 \)

#### Orthogonality

Two sinusoids with different frequencies are orthogonal over integer periods:

\[ \int_0^T \sin(2\pi f_1 t) \sin(2\pi f_2 t) \, dt = 0 \quad \text{if } f_1 \neq f_2 \text{ and } T = \frac{k}{f_1} = \frac{m}{f_2} \]

where \( k, m \) are integers.

**Power:**

\[ P = \frac{1}{T} \int_0^T x^2(t) \, dt = \frac{A^2}{2} \]

For unit amplitude (\( A = 1 \)): \( P = 0.5 \)

### Sampling Theory

#### Nyquist-Shannon Sampling Theorem

To avoid aliasing, sampling rate must satisfy:

\[ f_s > 2 f_{max} \]

where \( f_{max} \) is the highest frequency component.

**Nyquist rate:** \( f_N = 2 f_{max} \)

**Our project:**
- \( f_{max} = 7 \) Hz
- \( f_N = 14 \) Hz
- \( f_s = 1000 \) Hz >> 14 Hz ✓

#### Discrete-Time Signal

Sampled signal:

\[ x[n] = x(nT_s) = x\left(\frac{n}{f_s}\right) \]

where \( T_s = 1/f_s \) is the sampling period.

#### Reconstruction

Perfect reconstruction (if \( f_s > 2f_{max} \)):

\[ x(t) = \sum_{n=-\infty}^{\infty} x[n] \, \text{sinc}\left(\frac{t - nT_s}{T_s}\right) \]

where \( \text{sinc}(x) = \frac{\sin(\pi x)}{\pi x} \).

### Noise Model

#### Additive White Gaussian Noise (AWGN)

\[ y(t) = s(t) + n(t) \]

where:
- \( s(t) \): Clean signal
- \( n(t) \): Noise
- \( n(t) \sim \mathcal{N}(0, \sigma^2) \): Gaussian distribution

**Probability density:**

\[ p(n) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{n^2}{2\sigma^2}\right) \]

#### Signal-to-Noise Ratio (SNR)

**Power-based:**

\[ \text{SNR} = \frac{P_signal}{P_noise} = \frac{\mathbb{E}[s^2(t)]}{\mathbb{E}[n^2(t)]} = \frac{\sigma_s^2}{\sigma_n^2} \]

**Decibel scale:**

\[ \text{SNR}_{dB} = 10 \log_{10}\left(\frac{P_signal}{P_noise}\right) = 10 \log_{10}\left(\frac{\sigma_s^2}{\sigma_n^2}\right) \]

**Our project:**
- \( \sigma_s \approx 0.7 \) (clean signal std dev)
- \( \sigma_n = 0.1 \) (noise std dev)
- \( \text{SNR}_{dB} = 10 \log_{10}(0.7^2 / 0.1^2) = 10 \log_{10}(49) \approx 17 \) dB

(Actual measured SNR ≈ 11 dB due to signal combination effects)

### Fourier Analysis

#### Continuous Fourier Transform

\[ X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi f t} \, dt \]

**Inverse:**

\[ x(t) = \int_{-\infty}^{\infty} X(f) e^{j2\pi f t} \, df \]

#### Discrete Fourier Transform (DFT)

For discrete signal \( x[n] \), \( n = 0, 1, ..., N-1 \):

\[ X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N} \]

**Inverse DFT:**

\[ x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j2\pi kn/N} \]

#### Fast Fourier Transform (FFT)

Efficient algorithm for computing DFT:
- Complexity: \( O(N \log N) \) vs. \( O(N^2) \) for naive DFT
- Used in visualization scripts

**Frequency resolution:**

\[ \Delta f = \frac{f_s}{N} \]

**Our project:**
- \( N = 10000 \) samples
- \( f_s = 1000 \) Hz
- \( \Delta f = 0.1 \) Hz (excellent resolution for our frequencies)

#### Power Spectrum

\[ P[k] = |X[k]|^2 = X[k] X^*[k] \]

where \( X^*[k] \) is the complex conjugate.

---

## LSTM Mathematics

### Recurrent Neural Network (RNN) Basics

#### Vanilla RNN

\[ h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \]
\[ y_t = W_{hy} h_t + b_y \]

**Problem:** Vanishing/exploding gradients through time

\[ \frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=1}^{k} \frac{\partial h_{t-i+1}}{\partial h_{t-i}} \]

If all derivatives < 1: gradient vanishes
If all derivatives > 1: gradient explodes

### LSTM Architecture

#### Cell State and Hidden State

LSTM maintains two states:
- **Cell state** \( C_t \): Long-term memory
- **Hidden state** \( h_t \): Short-term memory/output

#### Gates

**1. Forget Gate** (what to forget from cell state):

\[ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \]

**2. Input Gate** (what to add to cell state):

\[ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \]

**3. Candidate Cell State** (new information):

\[ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \]

**4. Cell State Update**:

\[ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \]

**5. Output Gate** (what to output):

\[ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \]

**6. Hidden State Update**:

\[ h_t = o_t \odot \tanh(C_t) \]

#### Activation Functions

**Sigmoid:**

\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

Properties:
- Range: (0, 1)
- Derivative: \( \sigma'(x) = \sigma(x)(1 - \sigma(x)) \)
- Used for gates (0-1 control)

**Hyperbolic Tangent:**

\[ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1} \]

Properties:
- Range: (-1, 1)
- Derivative: \( \tanh'(x) = 1 - \tanh^2(x) \)
- Used for cell state and output (symmetric around 0)

#### Parameter Count

For LSTM with input size \( n \), hidden size \( h \):

**Weight matrices (4 gates):**

\[ W_f, W_i, W_C, W_o \in \mathbb{R}^{h \times (n+h)} \]

Total weights: \( 4h(n+h) \)

**Bias vectors:**

\[ b_f, b_i, b_C, b_o \in \mathbb{R}^h \]

Total biases: \( 4h \)

**Total parameters:**

\[ P = 4h(n+h) + 4h = 4h(n+h+1) \]

**Our project (Layer 1):**
- \( n = 5 \), \( h = 128 \)
- \( P = 4 \times 128 \times (5 + 128 + 1) = 68,608 \)

**Layer 2:**
- \( n = 128 \) (input from Layer 1), \( h = 128 \)
- \( P = 4 \times 128 \times (128 + 128 + 1) = 131,584 \)

### Backpropagation Through Time (BPTT)

#### Forward Pass

For sequence of length \( T \):

\[ h_1, C_1 = \text{LSTM}(x_1, h_0, C_0) \]
\[ h_2, C_2 = \text{LSTM}(x_2, h_1, C_1) \]
\[ \vdots \]
\[ h_T, C_T = \text{LSTM}(x_T, h_{T-1}, C_{T-1}) \]

#### Loss

\[ \mathcal{L} = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t) \]

where \( \hat{y}_t = f(h_t) \) is the prediction.

#### Backward Pass

Gradients flow backward through time:

\[ \frac{\partial \mathcal{L}}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial h_{t-1}} + \frac{\partial \mathcal{L}}{\partial C_t} \frac{\partial C_t}{\partial h_{t-1}} \]

**Key advantage of LSTM:**

Cell state provides additive path:

\[ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \]

\[ \frac{\partial C_t}{\partial C_{t-1}} = f_t \]

Gradient can flow through \( f_t \) without repeated multiplication, avoiding vanishing gradients.

### Gradient Clipping

**Global norm clipping:**

\[ \hat{g} = \begin{cases}
\frac{\text{clip\_norm}}{\|g\|} \cdot g & \text{if } \|g\| > \text{clip\_norm} \\
g & \text{otherwise}
\end{cases} \]

where \( \|g\| = \sqrt{\sum_i g_i^2} \) is the L2 norm.

**Our project:** clip_norm = 1.0

---

## Optimization Theory

### Adam Optimizer

#### Algorithm

**Initialize:**
- \( m_0 = 0 \) (first moment)
- \( v_0 = 0 \) (second moment)
- \( t = 0 \) (timestep)

**For each iteration:**

1. Compute gradient: \( g_t = \nabla_\theta \mathcal{L}(\theta_{t-1}) \)

2. Update biased first moment:
   \[ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \]

3. Update biased second moment:
   \[ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \]

4. Bias correction:
   \[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]
   \[ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \]

5. Update parameters:
   \[ \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \]

**Default hyperparameters:**
- \( \alpha = 0.001 \) (learning rate)
- \( \beta_1 = 0.9 \) (first moment decay)
- \( \beta_2 = 0.999 \) (second moment decay)
- \( \epsilon = 10^{-8} \) (numerical stability)

#### Convergence Analysis

**Regret bound:** Under certain conditions:

\[ \mathcal{R}(T) = \sum_{t=1}^{T} [\mathcal{L}(\theta_t) - \mathcal{L}(\theta^*)] = O(\sqrt{T}) \]

where \( \theta^* \) is the optimal parameter.

**Adaptive learning rate:**

Effective learning rate for parameter \( i \):

\[ \alpha_i^{eff} = \frac{\alpha}{\sqrt{\hat{v}_i} + \epsilon} \]

Adapts to gradient magnitude: large gradients → smaller steps, small gradients → larger steps.

### L2 Regularization (Weight Decay)

**Modified loss:**

\[ \mathcal{L}_{total}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2 \]

**Gradient:**

\[ \nabla_\theta \mathcal{L}_{total} = \nabla_\theta \mathcal{L} + \lambda \theta \]

**Update rule:**

\[ \theta_t = \theta_{t-1} - \alpha (\nabla_\theta \mathcal{L} + \lambda \theta_{t-1}) = (1 - \alpha\lambda) \theta_{t-1} - \alpha \nabla_\theta \mathcal{L} \]

**Effect:** Shrinks weights toward zero, preventing overfitting.

**Our project:** \( \lambda = 10^{-5} \)

### Learning Rate Scheduling

#### ReduceLROnPlateau

**Algorithm:**

```
If val_loss does not improve for `patience` epochs:
    lr ← lr × factor
```

**Our settings:**
- patience = 5
- factor = 0.5

**Schedule example:**
- Epochs 1-15: \( \alpha = 0.001 \)
- No improvement for 5 epochs
- Epochs 16-30: \( \alpha = 0.0005 \)
- Again no improvement
- Epochs 31-50: \( \alpha = 0.00025 \)

---

## Loss Functions

### Mean Squared Error (MSE)

\[ \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \]

**Gradient:**

\[ \frac{\partial \mathcal{L}_{MSE}}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i) \]

**Properties:**
- Convex
- Smooth (differentiable everywhere)
- Penalizes large errors quadratically

### Mean Absolute Error (MAE)

\[ \mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| \]

**Gradient:**

\[ \frac{\partial \mathcal{L}_{MAE}}{\partial \hat{y}_i} = \frac{1}{N} \text{sign}(\hat{y}_i - y_i) \]

**Properties:**
- Convex
- Not smooth at \( \hat{y}_i = y_i \)
- Linear penalty (robust to outliers)

### Huber Loss

\[ \mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases} \]

**Gradient:**

\[ \frac{\partial \mathcal{L}_{\delta}}{\partial \hat{y}} = \begin{cases}
\hat{y} - y & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot \text{sign}(\hat{y} - y) & \text{otherwise}
\end{cases} \]

**Properties:**
- Convex
- Smooth
- Combines MSE (small errors) and MAE (large errors)

---

## Evaluation Metrics

### R² Score (Coefficient of Determination)

\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \]

where:

\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \]

**Alternative formulation:**

\[ R^2 = \frac{SS_{exp}}{SS_{tot}} \]

where:

\[ SS_{exp} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2 \]

**Relationship:**

\[ SS_{tot} = SS_{exp} + SS_{res} \]

**Properties:**
- Range: \( (-\infty, 1] \)
- \( R^2 = 1 \): Perfect predictions
- \( R^2 = 0 \): As good as mean baseline
- \( R^2 < 0 \): Worse than mean baseline

### Pearson Correlation Coefficient

\[ r = \frac{\sum_{i=1}^{n} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{n} (\hat{y}_i - \bar{\hat{y}})^2}} \]

**Alternative formulation:**

\[ r = \frac{\text{Cov}(Y, \hat{Y})}{\sigma_Y \sigma_{\hat{Y}}} \]

**Properties:**
- Range: [-1, 1]
- \( r = 1 \): Perfect positive correlation
- \( r = 0 \): No linear correlation
- \( r = -1 \): Perfect negative correlation

**Relationship to R²:**

For simple linear regression: \( R^2 = r^2 \)

### Statistical Significance

#### Hypothesis Test for Correlation

**Null hypothesis:** \( H_0: \rho = 0 \) (no correlation)

**Test statistic:**

\[ t = r\sqrt{\frac{n-2}{1-r^2}} \]

follows t-distribution with \( n-2 \) degrees of freedom.

**p-value:** Probability of observing \( |t| \) or larger under \( H_0 \).

**Our case:**
- \( r = 0.628 \)
- \( n = 1,990,200 \)
- \( t \approx 1114 \) (enormous!)
- \( p < 10^{-10} \) (highly significant)

---

## Statistical Foundations

### Probability Distributions

#### Gaussian (Normal) Distribution

\[ p(x | \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \]

**Properties:**
- Mean: \( \mu \)
- Variance: \( \sigma^2 \)
- Standard deviation: \( \sigma \)

**Standard normal:** \( \mu = 0, \sigma = 1 \)

\[ p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \]

### Central Limit Theorem

For i.i.d. random variables \( X_1, ..., X_n \) with mean \( \mu \) and variance \( \sigma^2 \):

\[ \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \]

as \( n \to \infty \).

**Implications:**
- Sum of many small random effects → Gaussian
- Justifies Gaussian noise model
- Basis for many statistical tests

### Confidence Intervals

#### For Correlation Coefficient

Using Fisher z-transformation:

\[ z = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right) = \tanh^{-1}(r) \]

\( z \) is approximately normal with:

\[ \mathbb{E}[z] \approx \frac{1}{2} \ln\left(\frac{1+\rho}{1-\rho}\right) \]
\[ \text{Var}(z) \approx \frac{1}{n-3} \]

**95% Confidence interval:**

\[ CI = \tanh\left(z \pm \frac{1.96}{\sqrt{n-3}}\right) \]

**Our case:**
- \( r = 0.628 \)
- \( n = 1,990,200 \)
- \( z = 0.739 \)
- \( CI = [0.626, 0.630] \) (very tight!)

---

## Summary

**Signal Processing:**
- ✓ Sinusoidal signals with frequency, phase, amplitude
- ✓ Nyquist sampling theorem: \( f_s > 2f_{max} \)
- ✓ AWGN model: \( y = s + n \), \( n \sim \mathcal{N}(0, \sigma^2) \)
- ✓ SNR in dB: \( 10\log_{10}(P_s/P_n) \)

**LSTM:**
- ✓ Three gates (forget, input, output) + cell state
- ✓ Solves vanishing gradient problem
- ✓ Parameter count: \( 4h(n+h+1) \)

**Optimization:**
- ✓ Adam: Adaptive learning rates with momentum
- ✓ L2 regularization: \( \mathcal{L} + \lambda\|\theta\|^2 \)
- ✓ Gradient clipping: Prevents exploding gradients

**Metrics:**
- ✓ R²: Variance explained
- ✓ MSE/RMSE/MAE: Prediction error
- ✓ Correlation: Linear relationship strength

This mathematical foundation supports all aspects of the LSTM frequency filtering project.

