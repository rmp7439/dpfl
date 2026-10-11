# Stage 6 Experimental Summary & Reproducibility Table

This document provides a complete, reproducible record of the Stage 6 hyperparameter ablation experiments investigating the interaction between data heterogeneity ($\alpha \in \{0.1, 10.0\}$) and Differential Privacy noise scale ($\sigma \in \{1.0, 2.0\}$) with clipping threshold $C = 1.0$ and target failure probability $\delta = 10^{-5}$.

---

## 1. Authoritative 15-Communication-Round Reference Runs

These runs represent the full 15-round protocol executed on full CIFAR-10 (50,000 train, 10,000 test) across 5 clients.

| $\alpha$ | $\sigma$ | $C$ | Seed | Dataset Mode | Clients | Rounds | Final Acc (%) | Best Acc (%) | Max DP Steps | Sample Rate $q$ | Best Order $\alpha_{\text{RDP}}$ | Epsilon ($\varepsilon$) | Runtime (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 1.0 | 1.0 | 42 | full | 5 | 15 | 24.41 | 27.14 | 3045 | 0.01136 | 6.9 | 2.6974 | 655.4 | Success |
| 0.1 | 2.0 | 1.0 | 42 | full | 5 | 15 | 13.79 | 18.95 | 3045 | 0.01136 | 19.0 | 0.9117 | 614.9 | Success |
| 10.0 | 1.0 | 1.0 | 42 | full | 5 | 15 | 31.51 | 31.51 | 2475 | 0.00719 | 8.3 | 2.0691 | 639.6 | Success |
| 10.0 | 2.0 | 1.0 | 44 | full | 5 | 15 | 15.26 | 15.26 | 2580 | 0.00758 | 23.0 | 0.7271 | 1068.2 | Success |

*Provenance Artifacts*: `results/stage6/multiseed_fixed_20261009_211429/ablation_results.json` and `.csv`.

---

## 2. Multi-Seed Robustness Runs (3 Communication Rounds, Seeds 42, 43, 44)

These runs evaluate multi-seed stability across 3 random seeds ($N=3$) under identical partition hyperparameters on full CIFAR-10 across 5 clients.

### 2.1 Per-Run Experimental Data

| $\alpha$ | $\sigma$ | $C$ | Seed | Clients | Rounds | Final Acc (%) | Best Acc (%) | Max DP Steps | Sample Rate $q$ | Best Order $\alpha_{\text{RDP}}$ | Epsilon ($\varepsilon$) | Runtime (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 1.0 | 1.0 | 42 | 5 | 3 | 17.86 | 17.86 | 561 | 0.01136 | 8.2 | 1.5394 | 488.8 | Success |
| 0.1 | 1.0 | 1.0 | 43 | 5 | 3 | 19.68 | 19.68 | 717 | 0.01075 | 8.3 | 1.4963 | 2126.4 | Success |
| 0.1 | 1.0 | 1.0 | 44 | 5 | 3 | 18.45 | 18.45 | 687 | 0.01099 | 8.2 | 1.5157 | 275.1 | Success |
| 0.1 | 2.0 | 1.0 | 42 | 5 | 3 | 10.71 | 10.71 | 561 | 0.01136 | 34.0 | 0.3367 | 397.6 | Success |
| 0.1 | 2.0 | 1.0 | 43 | 5 | 3 | 12.13 | 12.13 | 717 | 0.01075 | 35.0 | 0.3851 | 306.0 | Success |
| 0.1 | 2.0 | 1.0 | 44 | 5 | 3 | 10.03 | 10.63 | 687 | 0.01099 | 34.0 | 0.3918 | 282.0 | Success |
| 10.0 | 1.0 | 1.0 | 42 | 5 | 3 | 20.00 | 20.78 | 480 | 0.00719 | 9.2 | 1.2595 | 414.8 | Success |
| 10.0 | 1.0 | 1.0 | 43 | 5 | 3 | 15.33 | 15.33 | 498 | 0.00676 | 9.3 | 1.2280 | 282.2 | Success |
| 10.0 | 1.0 | 1.0 | 44 | 5 | 3 | 18.30 | 18.30 | 516 | 0.00662 | 9.4 | 1.2186 | 281.0 | Success |
| 10.0 | 2.0 | 1.0 | 42 | 5 | 3 | 8.98 | 9.57 | 495 | 0.00637 | 39.0 | 0.2952 | 349.3 | Success |
| 10.0 | 2.0 | 1.0 | 43 | 5 | 3 | 10.21 | 11.60 | 498 | 0.00680 | 39.0 | 0.3050 | 288.4 | Success |
| 10.0 | 2.0 | 1.0 | 44 | 5 | 3 | 10.12 | 11.71 | 504 | 0.00758 | 38.0 | 0.3216 | 273.7 | Success |

### 2.2 Three-Seed Statistical Aggregates ($N=3$)

| $\alpha$ | $\sigma$ | $C$ | Seeds | Final Acc Mean (%) | Final Acc Sample Std | Best Acc Mean (%) | Best Acc Sample Std | Mean $\varepsilon$ | Sample Std $\varepsilon$ | Max Worst-Case $\varepsilon$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 1.0 | 1.0 | 42, 43, 44 | 18.66 | $\pm 0.93$ | 18.66 | $\pm 0.93$ | 1.5171 | $\pm 0.0215$ | 1.5394 |
| 0.1 | 2.0 | 1.0 | 42, 43, 44 | 10.96 | $\pm 1.07$ | 11.16 | $\pm 0.84$ | 0.3712 | $\pm 0.0301$ | 0.3918 |
| 10.0 | 1.0 | 1.0 | 42, 43, 44 | 17.88 | $\pm 2.36$ | 18.14 | $\pm 2.73$ | 1.2354 | $\pm 0.0214$ | 1.2595 |
| 10.0 | 2.0 | 1.0 | 42, 43, 44 | 9.77 | $\pm 0.69$ | 10.96 | $\pm 1.21$ | 0.3073 | $\pm 0.0133$ | 0.3216 |

*Statistical Note*: Per-run epsilons represent rigorous upper bounds on the individual sample differential privacy loss. The mean $\varepsilon$ is reported strictly as a descriptive summary of empirical accounting stability across partitions; it does **not** replace the worst-case bound $\max_s \varepsilon_s$, which serves as the formal privacy guarantee.

---

## 3. Analysis & Key Observations

1. **Heterogeneity vs. Privacy Trade-Off**:
   - At $\sigma=1.0$, 15-round near-IID ($\alpha=10.0$) reaches 31.51% test accuracy, outperforming highly skewed non-IID ($\alpha=0.1$) at 24.41% (+7.10% utility gain).
   - Under $\sigma=2.0$, excessive Gaussian perturbation severely constrains representation learning across all settings (13.79% at $\alpha=0.1$ vs. 15.26% at $\alpha=10.0$).
2. **Seed Variation in Homogeneous Settings**:
   - In the multi-seed evaluation, $\alpha=10.0, \sigma=1.0$ exhibited the largest seed sensitivity ($\text{std} = 2.36\%$, range 15.33% to 20.00%), reflecting the stochastic impact of gradient noise injection during early optimization rounds.
3. **Implicit Privacy Cost of Data Skew**:
   - Because Dirichlet partitioning at $\alpha=0.1$ produces uneven local dataset sizes (some clients receive fewer samples), fixed batch sizes ($B=64$) result in higher local sampling rates ($q = B / N_k$).
   - As a consequence, non-IID clients experience faster privacy expenditure, yielding higher $\varepsilon$ guarantees ($\approx 2.6974$ vs. $\approx 2.0691$ at 15 rounds) despite identical communication round counts.
