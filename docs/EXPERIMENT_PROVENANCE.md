# EXPERIMENT PROVENANCE

| Experiment | Dataset | Clients | α | σ | C | Rounds | Seed | Optimizer | LR | Accuracy | ε | Evidence | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| Stage 2 Baseline | full | 1 | N/A | N/A | N/A | 30 | 42 | Adam | 0.001 | 74.87% | N/A | `results/stage2/baseline_full_seed42.json` | A. Authoritative current result |
| Stage 3 FedAvg | full | 5 | 0.1 | N/A | N/A | 3 | 42 | SGD | 0.05 | 33.26% | N/A | `results/stage3/federation_full_seed42.json` | A. Authoritative current result |
| Stage 4 DP-FL | full | 5 | 0.1 | 1.0 | 1.0 | 15 | 42 | SGD | 0.05 | 26.06% | 2.6974 | `results/stage4/summary.json` | A. Authoritative current result |
| Stage 4 DP-FL (Old) | full | 5 | 0.1 | 1.0 | 1.0 | 3 | N/A | SGD | N/A | 20.14% | 1.5394 | `results/archive/stage4_legacy/summary.json` | B. Historical/superseded result |
| Stage 5 Grid | full | 5 | 0.1 | 2.0 | 0.1 | 3 | N/A | SGD | N/A | 19.54% | 0.3989 | `results/stage5/full/grid_results.json` | A. Authoritative current result |
| Stage 5 Grid | full | 5 | 0.1 | 2.0 | 1.0 | 3 | N/A | SGD | N/A | 12.13% | 0.3989 | `results/stage5/full/grid_results.json` | A. Authoritative current result |
| Stage 5 Grid | full | 5 | 0.1 | 0.5 | 1.0 | 3 | N/A | SGD | N/A | 25.11% | 10.8698 | `results/stage5/full/grid_results.json` | A. Authoritative current result |
| Stage 6 Ablation | full | 5 | 0.1 | 1.0 | 1.0 | 3 | 42 | N/A | N/A | 18.25% | 1.5394 | `results/stage6/historical_reconstructed/...` | C. Reconstructed result |
| Stage 6 Ablation | full | 5 | 0.1 | 2.0 | 1.0 | 3 | 42 | N/A | N/A | 11.87% | 0.3989 | `results/stage6/historical_reconstructed/...` | C. Reconstructed result |
| Stage 6 Ablation | full | 5 | 10.0 | 1.0 | 1.0 | 3 | 42 | N/A | N/A | 22.33% | 1.2595 | `results/stage6/historical_reconstructed/...` | C. Reconstructed result |
| Stage 6 Ablation | full | 5 | 10.0 | 2.0 | 1.0 | 3 | 42 | N/A | N/A | 10.10% | 0.3133 | `results/stage6/historical_reconstructed/...` | C. Reconstructed result |

Note on Stage 6 Reconstructed Results:
- Original full-data runtime artifacts were lost.
- Values were preserved from validated Colab console output.
- They are reconstructed historical evidence.
- They were not newly rerun in the current pipeline.
- Current subset validation artifacts are separate.
- A new official full-data structured Stage 6 rerun is still required for publication-grade current artifacts.
