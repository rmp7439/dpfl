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

## Important: Environment Requirements

**Full CIFAR-10 experiments require a GPU environment (Colab or local GPU).**

The federated simulation uses Ray, which has known issues on Windows. Full experiments (Stage 3 full, Stage 4 full, Stage 5 grid) should be run on:
- Google Colab (free tier T4 GPU works)
- Linux/macOS with Ray working
- Windows WSL2 with Ray configured

Subset experiments (1000 samples) work on Windows CPU.

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

> **Note**: Full experiments (Stage 3/4/5 on full CIFAR-10) require Colab/GPU due to Ray compatibility issues on Windows. Stage 2 centralized baseline can run on CPU (30 epochs, ~41 minutes).

---

## Privacy Accounting

Privacy is tracked using **Rényi Differential Privacy (RDP)**:

1. Each client uses Opacus `PrivacyEngine` which wraps the model/optimizer/dataloader.
2. The accountant uses `compute_rdp(q, noise_multiplier, steps, orders)` from `opacus.accountants.analysis.rdp`.
3. Opacus `PrivacyEngine` / `DPDataLoader` performs the actual sampling. The production accounting uses the actual DP loader sample rate.
4. The actual number of DP loader iterations/steps is tracked and RDP composition is performed over those actual steps.
5. Optimal Rényi order α is found by minimizing ε(δ) = RDP_α + log(1/δ)/(α-1).
6. Under parallel composition (clients have disjoint data), global ε is bounded by max client ε — we use the client with the highest sample rate (smallest dataset) as the worst case.
7. `sigma` (noise_multiplier) controls the Gaussian noise level and directly affects epsilon.
8. `C` (max_grad_norm) controls clipping threshold — affects training utility but does NOT appear in RDP formula.

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

### Current Status (as of 2026-09-07)

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1 | ✅ COMPLETE | Full and subset splits validated, all samples assigned exactly once |
| Stage 2 | ✅ COMPLETE | Full baseline: 74.87% accuracy (30 epochs, CPU) |
| Stage 3 | ✅ COMPLETE | Full FedAvg: 33.26% accuracy (3 rounds, 5 clients, non-DP) |
| Stage 4 | ✅ COMPLETE | Full CIFAR-10 run exists, official DP-SGD configuration completed |
| Stage 5 | ✅ COMPLETE | All 8 official full-data configs exist; epsilon/accuracy plots generated |
| Stage 6 | ⏳ PENDING | Ablation studies (varying α and σ) on full CIFAR-10 |

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
