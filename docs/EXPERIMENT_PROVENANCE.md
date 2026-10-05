# EXPERIMENT PROVENANCE

| Experiment | Dataset | Clients | α | σ | C | Rounds | Seed | Optimizer | LR | Accuracy | ε | Evidence | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| Stage 2 Baseline | full | 1 | N/A | N/A | N/A | 30 | 42 | Adam | 0.001 | 74.87% | N/A | `results/stage2/baseline_full_seed42.json` | Fully verified from structured artifact |
| Stage 3 FedAvg | full | 5 | 0.1 | N/A | N/A | 3 | 42 | SGD | 0.05 | 33.26% | N/A | `results/stage3/federation_full_seed42.json` | Fully verified from structured artifact |
| Stage 4 DP-FL | full | 5 | 0.1 | 1.0 | 1.0 | 3 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | 20.14% | 1.5394 | None (Report only) | Not currently verifiable |
| Stage 5 Grid | full | 5 | 0.1 | 2.0 | 0.1 | 3 | NOT VERIFIED | SGD | NOT VERIFIED | 19.54% | 0.3989 | `results/stage5/grid_results.json` | Fully verified from structured artifact |
| Stage 5 Grid | full | 5 | 0.1 | 2.0 | 1.0 | 3 | NOT VERIFIED | SGD | NOT VERIFIED | 12.13% | 0.3989 | `results/stage5/grid_results.json` | Fully verified from structured artifact |
| Stage 5 Grid | full | 5 | 0.1 | 0.5 | 1.0 | 3 | NOT VERIFIED | SGD | NOT VERIFIED | 25.11% | 10.8698 | `results/stage5/grid_results.json` | Fully verified from structured artifact |
| Stage 6 Ablation | full | 5 | 0.1 | 1.0 | 1.0 | 3 | 42 | NOT VERIFIED | NOT VERIFIED | 18.25% | 1.5394 | `results/stage6/ablation_results.json` | Verified from preserved execution log |
| Stage 6 Ablation | full | 5 | 0.1 | 2.0 | 1.0 | 3 | 42 | NOT VERIFIED | NOT VERIFIED | 11.87% | 0.3989 | `results/stage6/ablation_results.json` | Verified from preserved execution log |
| Stage 6 Ablation | full | 5 | 10.0 | 1.0 | 1.0 | 3 | 42 | NOT VERIFIED | NOT VERIFIED | 22.33% | 1.2595 | `results/stage6/ablation_results.json` | Verified from preserved execution log |
| Stage 6 Ablation | full | 5 | 10.0 | 2.0 | 1.0 | 3 | 42 | NOT VERIFIED | NOT VERIFIED | 10.10% | 0.3133 | `results/stage6/ablation_results.json` | Verified from preserved execution log |
