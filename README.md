# Chaos Theory in Time Series Analysis

<div align="center">
   
<img src="Lorenz_attractor_yb.svg"  width="55%" />


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/Updated-November%202025-green.svg)]()

</div>

A comprehensive Python toolkit for detecting and analyzing chaotic dynamics in time series data, with particular emphasis on financial market applications. This repository provides implementations of key chaos theory metrics including Lyapunov exponents, Hurst exponent, sample entropy, and fractal dimension analysis.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundations](#theoretical-foundations)
   - [Deterministic Chaos](#deterministic-chaos)
   - [Lyapunov Exponents](#lyapunov-exponents)
   - [Hurst Exponent and Long-Range Dependence](#hurst-exponent-and-long-range-dependence)
   - [Sample Entropy](#sample-entropy)
   - [Fractal Dimension](#fractal-dimension)
   - [BDS Test for Nonlinearity](#bds-test-for-nonlinearity)
3. [Applications in Financial Time Series](#applications-in-financial-time-series)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Mathematical Formulations](#mathematical-formulations)
7. [References](#references)
8. [Citation](#citation)

---

## Introduction

Traditional time series analysis often assumes linearity and stationarity—assumptions that frequently fail when applied to complex real-world systems such as financial markets, climate data, and biological signals. Chaos theory provides an alternative framework for understanding systems that exhibit **sensitive dependence on initial conditions** (SDIC), where small perturbations can lead to dramatically different outcomes over time.

This toolkit implements several established methods from nonlinear dynamics to characterize chaotic behavior in time series:

- **Maximal Lyapunov Exponent (MLE)**: Quantifies the rate of separation of infinitesimally close trajectories
- **Hurst Exponent**: Measures long-range dependence and self-similarity
- **Sample Entropy**: Assesses complexity and regularity
- **Fractal Dimension**: Characterizes the geometric complexity of the attractor
- **BDS Test**: Statistical test for independence and identical distribution

---

## Theoretical Foundations

### Deterministic Chaos

Deterministic chaos refers to complex, aperiodic behavior arising from deterministic nonlinear dynamical systems. A system is considered chaotic if it exhibits three key properties [1, 2]:

1. **Sensitive Dependence on Initial Conditions (SDIC)**: Arbitrarily small differences in initial states grow exponentially over time
2. **Topological Mixing**: The system evolves such that any given region of phase space eventually overlaps with any other region
3. **Dense Periodic Orbits**: Periodic orbits are densely distributed in the phase space

The canonical mathematical example is the logistic map:

$$x_{n+1} = r \cdot x_n (1 - x_n)$$

For certain values of the parameter $r$ (specifically $r > 3.57$), this simple deterministic equation produces chaotic dynamics indistinguishable from random noise by conventional statistical methods [3].

### Lyapunov Exponents

The Lyapunov exponent quantifies the rate at which nearby trajectories in phase space diverge or converge. For a one-dimensional map $x_{n+1} = f(x_n)$, the Lyapunov exponent is defined as [4, 5]:

$$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \ln |f'(x_i)|$$

For continuous-time systems, if the initial separation between two trajectories is $\delta_0$, the separation at time $t$ evolves as:

$$|\delta(t)| \approx e^{\lambda t} |\delta_0|$$

**Interpretation**:
- $\lambda > 0$: Chaotic behavior—nearby trajectories diverge exponentially
- $\lambda = 0$: Marginal stability—characteristic of bifurcation points
- $\lambda < 0$: Stable fixed point or periodic orbit—trajectories converge

The **Lyapunov time** $\tau_\lambda = 1/\lambda$ represents the characteristic timescale over which predictability is lost, providing a fundamental limit on forecasting horizons [6].

#### Algorithms for Estimation

This toolkit implements two algorithms for MLE estimation:

**Rosenstein Algorithm** [7]: Robust method suitable for small datasets that tracks the divergence of nearest neighbors in reconstructed phase space:

$$\lambda_1 = \frac{1}{\Delta t \cdot M} \sum_{i=1}^{M} \ln \frac{d_i(k)}{d_i(0)}$$

**Wolf Algorithm** [8]: Classic algorithm that follows the evolution of a single pair of nearest neighbors, replacing the reference point when trajectories diverge beyond a threshold.

### Hurst Exponent and Long-Range Dependence

The Hurst exponent $H$ originated from Harold Edwin Hurst's hydrological studies of the Nile River [9] and characterizes long-range dependence in time series. It is estimated via Rescaled Range (R/S) analysis:

$$\frac{R(n)}{S(n)} \propto n^H$$

where $R(n)$ is the range of cumulative deviations from the mean and $S(n)$ is the standard deviation over a window of size $n$.

**Interpretation**:
- $H = 0.5$: Brownian motion (random walk)—no long-range correlation
- $0.5 < H < 1$: **Persistent** behavior—positive autocorrelation; trends tend to continue
- $0 < H < 0.5$: **Anti-persistent** behavior—negative autocorrelation; mean-reverting dynamics

The Hurst exponent is intimately connected to the autocorrelation function. For a fractional Brownian motion with Hurst parameter $H$, the autocorrelation at lag $k$ decays as [10]:

$$\rho(k) \sim H(2H-1)k^{2H-2} \quad \text{as } k \to \infty$$

### Sample Entropy

Sample Entropy (SampEn), introduced by Richman and Moorman [11], measures the complexity and regularity of a time series by quantifying the conditional probability that sequences similar for $m$ points remain similar when one additional point is included:

$$\text{SampEn}(m, r, N) = -\ln \frac{A^m(r)}{B^m(r)}$$

where:
- $B^m(r)$ = number of template matches of length $m$ within tolerance $r$
- $A^m(r)$ = number of template matches of length $m+1$ within tolerance $r$
- $N$ = length of the time series

**Interpretation**:
- Lower SampEn → Higher regularity and predictability
- Higher SampEn → Greater complexity and unpredictability

Sample entropy improves upon Approximate Entropy by excluding self-matches, providing a less biased estimate for finite data [12].

### Fractal Dimension

The fractal dimension $D_F$ characterizes the geometric complexity of a time series' phase space trajectory. For self-affine time series, an approximate relationship exists with the Hurst exponent [13]:

$$D_F = 2 - H$$

**Interpretation**:
- $D_F \approx 1.0$: Smooth, regular trajectory
- $D_F \approx 1.5$: Brownian motion
- $D_F \to 2.0$: Space-filling, highly irregular trajectory

More rigorous definitions include the correlation dimension $D_2$ estimated via the Grassberger-Procaccia algorithm [14]:

$$D_2 = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}$$

where $C(r)$ is the correlation integral measuring the fraction of point pairs within distance $r$.

### BDS Test for Nonlinearity

The BDS test [15], developed by Brock, Dechert, and Scheinkman, tests the null hypothesis that a time series consists of independent and identically distributed (i.i.d.) observations. The test statistic is:

$$W_{m,T}(\epsilon) = \sqrt{T} \frac{C_{m,T}(\epsilon) - C_{1,T}(\epsilon)^m}{\sigma_{m,T}(\epsilon)}$$

where $C_{m,T}(\epsilon)$ is the correlation integral at embedding dimension $m$ and distance $\epsilon$.

**Interpretation**:
- Rejection of the null hypothesis indicates the presence of nonlinear structure (deterministic chaos, nonlinear stochastic process, or nonstationarity)
- The BDS test does not distinguish between different types of nonlinearity

---

## Applications in Financial Time Series

### The Efficient Market Hypothesis and Chaos

The Efficient Market Hypothesis (EMH) [16] posits that financial asset prices fully reflect all available information, implying that price changes should follow a random walk. However, extensive empirical evidence has documented departures from the EMH, including:

- **Fat tails**: Return distributions exhibit excess kurtosis [17]
- **Volatility clustering**: Large price changes tend to cluster together [18]
- **Long memory**: Squared and absolute returns show persistent autocorrelation [19]
- **Nonlinear dependence**: Returns may be uncorrelated but not independent [20]

Chaos theory provides tools to investigate whether these anomalies arise from deterministic nonlinear dynamics rather than stochastic processes.

### Practical Applications

#### 1. Market Predictability Assessment

The Lyapunov exponent directly quantifies the horizon over which forecasting is theoretically possible:

$$\text{Forecast Horizon} \approx \frac{1}{\lambda} \ln \frac{\Delta}{\delta_0}$$

where $\delta_0$ is the initial forecast uncertainty and $\Delta$ is the acceptable error threshold.

Studies have estimated positive Lyapunov exponents for various financial time series [21, 22], suggesting chaotic dynamics with limited predictability windows typically ranging from days to weeks.

#### 2. Trend Persistence Detection

The Hurst exponent identifies whether markets exhibit:
- **Trending behavior** ($H > 0.5$): Momentum strategies may be effective
- **Mean reversion** ($H < 0.5$): Contrarian strategies may be appropriate
- **Random walk** ($H \approx 0.5$): Technical analysis is unlikely to add value

Research has found that $H$ varies across different markets, time scales, and regimes [23, 24].

#### 3. Regime Detection

Changes in chaos metrics over time can signal regime shifts:
- Increases in Lyapunov exponent may precede market turbulence
- Changes in Hurst exponent can indicate shifts between trending and mean-reverting regimes
- Sample entropy variations may reflect changing market complexity

#### 4. Risk Management

Understanding the chaotic nature of markets has implications for:
- **Value at Risk (VaR)**: Standard Gaussian assumptions may underestimate tail risk
- **Portfolio optimization**: Correlations may be unstable due to nonlinear dynamics
- **Stress testing**: Scenario analysis should account for SDIC

### Caveats and Limitations

Several important caveats apply when applying chaos theory to financial data [25, 26]:

1. **Data requirements**: Reliable estimation of chaos metrics requires long, stationary time series—conditions rarely met in financial markets
2. **Noise sensitivity**: Real financial data contain substantial noise that can bias Lyapunov exponent estimates
3. **Nonstationarity**: Financial markets are inherently nonstationary; structural breaks invalidate long-run estimates
4. **Finite-sample bias**: Chaos metrics are often poorly estimated from the sample sizes available
5. **Alternative explanations**: Many "chaotic" features can also arise from nonlinear stochastic processes

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install numpy pandas matplotlib scipy statsmodels nolds pynamical tqdm jupyter
```

### Clone Repository

```bash
git clone https://github.com/Javihaus/Chaos-in-Time-Series.git
cd Chaos-in-Time-Series
```

### Recommended: Create Virtual Environment

```bash
python -m venv chaos-env
source chaos-env/bin/activate  # On Windows: chaos-env\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```python
import numpy as np
import nolds
from statsmodels.tsa.stattools import bds, adfuller

# Load your time series data
data = np.random.randn(1000)  # Replace with actual data

# Compute Maximal Lyapunov Exponent (Rosenstein algorithm)
mle = nolds.lyap_r(data)
print(f"Maximal Lyapunov Exponent: {mle:.4f}")

# Compute Hurst Exponent
hurst = nolds.hurst_rs(data)
print(f"Hurst Exponent: {hurst:.4f}")

# Compute Sample Entropy
sampen = nolds.sampen(data)
print(f"Sample Entropy: {sampen:.4f}")

# BDS Test for nonlinearity
bds_stat, pvalue = bds(data)
print(f"BDS Statistic: {bds_stat:.4f}, p-value: {pvalue:.4e}")
```

### Running the Notebook

```bash
jupyter notebook chaos_analysis.ipynb
```

---

## Mathematical Formulations

### Phase Space Reconstruction (Takens' Embedding)

Given a scalar time series $\{x_1, x_2, \ldots, x_N\}$, we reconstruct the phase space using delay coordinates [27]:

$$\mathbf{y}_i = (x_i, x_{i+\tau}, x_{i+2\tau}, \ldots, x_{i+(m-1)\tau})$$

where:
- $m$ = embedding dimension
- $\tau$ = time delay

Takens' theorem guarantees that for sufficiently large $m$ (specifically, $m > 2D$, where $D$ is the dimension of the original attractor), the reconstructed attractor is topologically equivalent to the original.

### Correlation Integral

$$C(r) = \lim_{N \to \infty} \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \Theta(r - \|\mathbf{y}_i - \mathbf{y}_j\|)$$

where $\Theta$ is the Heaviside step function.

### Rescaled Range Analysis

For a time series segment of length $n$:

1. Calculate the mean: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$

2. Calculate cumulative deviations: $Y_t = \sum_{i=1}^{t}(x_i - \bar{x})$

3. Calculate range: $R(n) = \max(Y_1, \ldots, Y_n) - \min(Y_1, \ldots, Y_n)$

4. Calculate standard deviation: $S(n) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$

5. Estimate $H$ from: $\log(R/S) = H \cdot \log(n) + c$

---

## References

[1] Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). Westview Press.

[2] Ott, E. (2002). *Chaos in Dynamical Systems* (2nd ed.). Cambridge University Press.

[3] May, R. M. (1976). Simple mathematical models with very complicated dynamics. *Nature*, 261(5560), 459–467. https://doi.org/10.1038/261459a0

[4] Eckmann, J.-P., & Ruelle, D. (1985). Ergodic theory of chaos and strange attractors. *Reviews of Modern Physics*, 57(3), 617–656. https://doi.org/10.1103/RevModPhys.57.617

[5] Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis* (2nd ed.). Cambridge University Press.

[6] Sprott, J. C. (2003). *Chaos and Time-Series Analysis*. Oxford University Press.

[7] Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D: Nonlinear Phenomena*, 65(1–2), 117–134. https://doi.org/10.1016/0167-2789(93)90009-P

[8] Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985). Determining Lyapunov exponents from a time series. *Physica D: Nonlinear Phenomena*, 16(3), 285–317. https://doi.org/10.1016/0167-2789(85)90011-9

[9] Hurst, H. E. (1951). Long-term storage capacity of reservoirs. *Transactions of the American Society of Civil Engineers*, 116(1), 770–799.

[10] Beran, J. (1994). *Statistics for Long-Memory Processes*. Chapman and Hall/CRC.

[11] Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology-Heart and Circulatory Physiology*, 278(6), H2039–H2049. https://doi.org/10.1152/ajpheart.2000.278.6.H2039

[12] Lake, D. E., Richman, J. S., Griffin, M. P., & Moorman, J. R. (2002). Sample entropy analysis of neonatal heart rate variability. *American Journal of Physiology-Regulatory, Integrative and Comparative Physiology*, 283(3), R789–R797.

[13] Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. *SIAM Review*, 10(4), 422–437.

[14] Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346–349. https://doi.org/10.1103/PhysRevLett.50.346

[15] Brock, W. A., Scheinkman, J. A., Dechert, W. D., & LeBaron, B. (1996). A test for independence based on the correlation dimension. *Econometric Reviews*, 15(3), 197–235. https://doi.org/10.1080/07474939608800353

[16] Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417. https://doi.org/10.2307/2325486

[17] Mandelbrot, B. B. (1963). The variation of certain speculative prices. *The Journal of Business*, 36(4), 394–419.

[18] Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.

[19] Ding, Z., Granger, C. W. J., & Engle, R. F. (1993). A long memory property of stock market returns and a new model. *Journal of Empirical Finance*, 1(1), 83–106.

[20] Hsieh, D. A. (1991). Chaos and nonlinear dynamics: Application to financial markets. *The Journal of Finance*, 46(5), 1839–1877. https://doi.org/10.1111/j.1540-6261.1991.tb04646.x

[21] Peters, E. E. (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*. John Wiley & Sons.

[22] Brock, W. A., Hsieh, D. A., & LeBaron, B. D. (1991). *Nonlinear Dynamics, Chaos, and Instability: Statistical Theory and Economic Evidence*. MIT Press.

[23] Lo, A. W. (1991). Long-term memory in stock market prices. *Econometrica*, 59(5), 1279–1313. https://doi.org/10.2307/2938368

[24] Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236. https://doi.org/10.1080/713665670

[25] Granger, C. W. J. (1994). Is chaotic economic theory relevant for economics? A review essay. *Journal of International and Comparative Economics*, 3, 139–145.

[26] LeBaron, B. (1994). Chaos and nonlinear forecastability in economics and finance. *Philosophical Transactions of the Royal Society of London A*, 348(1688), 397–404.

[27] Takens, F. (1981). Detecting strange attractors in turbulence. In D. Rand & L.-S. Young (Eds.), *Dynamical Systems and Turbulence, Warwick 1980* (pp. 366–381). Springer. https://doi.org/10.1007/BFb0091924

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chaos_timeseries_2025,
  author       = {Javihaus},
  title        = {Chaos-in-Time-Series: A Python Toolkit for Nonlinear Time Series Analysis},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/Javihaus/Chaos-in-Time-Series}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgments

- The [nolds](https://github.com/CSchoel/nolds) library for nonlinear dynamics measures
- The [pynamical](https://github.com/gboeing/pynamical) library for dynamical systems visualization
- The scientific community for developing the theoretical foundations of chaos theory
