# Stage 4 Longer-Round DP-FL Convergence Experiment

## Purpose
This experiment establishes the convergence behavior of Differentially Private Federated Learning (DP-FL) over 15 communication rounds, validating both privacy scaling and utility evolution.

## Authoritative Status
**This directory is the authoritative corrected 15-round artifact.** 
The previous version (`results/stage4_longrun_buggy_prepatch`) suffered from a CSV formatting bug where Round 0 evaluations were shifted into Round 1 training accuracy. That bug has been fixed in the codebase for this run. Round 0/pre-training evaluation is successfully excluded from training-round accuracy logging.

## Exact Command Used
```bash
python scripts/run_federated.py --full --enable-dp --rounds 15 --output-dir results/stage4
```

## Complete Configuration
- Dataset: CIFAR-10 (full dataset, 50,000 train, 10,000 test)
- Clients: 5
- Partitioning: Dirichlet ($\alpha=0.1$)
- Noise Multiplier ($\sigma$): 1.0
- Clipping Norm ($C$): 1.0
- Target Delta: $1e-5$
- Batch Size: 64
- Local Epochs: 1
- Optimizer: SGD
- Learning Rate: 0.05
- Model Architecture: SimpleCNN
- Seed: 42
- Communication Rounds: 15

## Experiment Results
- **Total DP steps**: 3045
- **Final epsilon**: 2.6974
- **Final accuracy**: 26.11%
- **Best accuracy**: 27.29% (Achieved at Round 14)

## Metrics Source
Per-round test accuracy is collected directly from the centralized evaluations performed by the Flower server at the end of each round. Epsilon is continuously monitored and aggregated iteratively per client using the Opacus RDP accountant based on the actual DP steps taken.

## Reproducibility
The parameters are deterministically controlled via the `run_federated.py` CLI script overriding the `FULL_CONFIG` dictionary natively instantiated in `src/config.py`. All local random seeds (`numpy.random`, `torch.manual_seed`) are explicitly seeded at script startup via `--seed 42`.
