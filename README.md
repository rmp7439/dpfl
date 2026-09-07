# DP-FL: Differentially Private Federated Learning on CIFAR-10

A research implementation of federated learning with Rényi Differential Privacy (RDP) using:
- **Flower** (`flwr`) for federated simulation
- **Opacus** for DP-SGD with per-sample gradient clipping
- **Dirichlet Non-IID** partitioning (α = 0.1)
- **RDP accounting** with (ε, δ)-DP conversion

---

## Project Goal

Train a CNN on CIFAR-10 across simulated clients under Non-IID data heterogeneity, with and without Differential Privacy, and analyze the privacy–utility tradeoff across 8 (σ, C) configurations.

---

## Stage Overview

| Stage | Description | Script |
|-------|-------------|--------|
| 1 | Dirichlet Non-IID data split, visualization | `data_split.py` |
| 2 | Centralized non-private baseline | `train_baseline.py` |
| 3 | Flower FedAvg without DP (plain SGD) | `federated.py` |
| 4 | Flower FedAvg with Opacus DP-SGD | `federated.py --enable-dp` |
| 5 | RDP accounting, grid search over (σ, C) | `grid_search.py` + `plot_stage5.py` |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Running Stages

### Subset-first (always verify here before full data)

```bash
# Stage 1 – Dirichlet split (subset)
python data_split.py

# Stage 2 – Centralized baseline (subset)
python train_baseline.py

# Stage 3 – Non-private federation (subset)
python federated.py

# Stage 4 – DP-SGD federation (subset)
python federated.py --enable-dp

# Stage 5 – Grid search (subset validation)
python grid_search.py --subset
```

### Full CIFAR-10 (official experiments)

```bash
# Stage 1 – Dirichlet split (full)
python data_split.py --full

# Stage 2 – Centralized baseline (full)
python train_baseline.py --full

# Stage 3 – Non-private federation (full)
python federated.py --full

# Stage 4 – DP-SGD federation (full, default sigma/C from config)
python federated.py --full --enable-dp

# Stage 5 – Official 8-run grid search (full)
python grid_search.py

# Generate plots
python plot_stage5.py
```

---

## Official Full-Data Protocol

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Model | SimpleCNN with GroupNorm (DP-compatible) |
| Dirichlet α | 0.1 |
| Clients | 5 |
| Communication rounds | 3 |
| Local epochs | 1 |
| Batch size | 64 |
| Optimizer (Stage 3) | SGD, lr=0.05 |
| Optimizer (Stage 4/5) | SGD, lr=0.05 |
| σ grid | {0.5, 1.0, 1.5, 2.0} |
| C grid | {0.1, 1.0} |
| δ | 1e-5 |
| Seed | 42 |

> **Note**: GPU is not available in the current environment. All full-data runs execute on CPU.
> Stage 2 baseline uses 5 epochs; federated stages use 3 communication rounds.
> These are deliberately constrained for CPU feasibility while producing genuine experimental evidence.

---

## Privacy Accounting

Privacy is tracked using **Rényi Differential Privacy (RDP)**:

1. Each client uses Opacus `PrivacyEngine` which wraps the model/optimizer/dataloader.
2. The accountant uses `compute_rdp(q, noise_multiplier, steps, orders)` from `opacus.accountants.analysis.rdp`.
3. Sampling rate `q = 1 / ceil(dataset_size / batch_size)` — one step samples approximately one batch.
4. Steps compose additively under RDP: after R rounds × T steps/round = R×T total steps.
5. Optimal Rényi order α is found by minimizing ε(δ) = RDP_α + log(1/δ)/(α-1).
6. Under parallel composition, client privacy losses do **not** sum — global ε is bounded by max client ε.

---

## Results Location

```
results/
  stage1/          Dirichlet split figures and validation JSON
  stage2/          Centralized baseline metrics, plots (subset + full)
  stage3/          Non-private federation metrics, plots (subset + full)
  stage4/          DP-SGD subset validation, grad_sample evidence, runtime
  stage5/
    per_run/       One subdirectory per (σ, C) config: summary.json + rounds.csv
    grid_results.json    All 8 runs compiled
    grid_results.csv     Tabular summary
    accuracy_vs_epsilon.png
    epsilon_vs_round.png
```

---

## Key Distinctions

| Label | Meaning |
|-------|---------|
| `*_subset_*` | 1,000-sample development/validation run |
| `*_full_*` | Full 50,000-sample CIFAR-10 official run |
| `dataset_mode: "subset"` in JSON | Validation / debug run |
| `dataset_mode: "full"` in JSON | Official experiment |

Subset results in `results/stage5/per_run/` from the previous session are *validation runs*, not official results.
Official full-data grid results overwrite these when `grid_search.py` (without `--subset`) completes.

---

## Tests

```bash
python -m unittest discover -s . -p "test_*.py"
```

Tests cover:
- Dirichlet coverage and determinism
- Stage 3 non-DP enforcement (no PrivacyEngine in non-DP path)
- Opacus `grad_sample` presence and shape
- RDP: finiteness, monotonicity, σ ordering, δ correctness, cumulative match
- Grid artifact completeness and internal consistency
