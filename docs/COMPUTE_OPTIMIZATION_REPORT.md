# COMPUTE OPTIMIZATION REPORT

## Overview
This report evaluates the performance of the DP-FL pipeline before and after applying optimizations based on the empirical compute audit.

## Environment & Configuration
- **Dataset**: CIFAR-10 subset (1,000 samples)
- **Clients**: 3
- **Communication Rounds**: 2
- **Local Epochs**: 1
- **Batch Size**: 32
- **Seed**: 42
- **Privacy Setup**: $\alpha=0.1, \sigma=1.0, C=1.0$
- **Software**: Windows Local Execution (CPU)

## Implemented Optimizations
1. **Centralized Evaluation Strategy**: Switched the `FedAvg` strategy to evaluate the global model centrally instead of having every client compute redundant metrics on the exact same global test set. This reduces test-set forward passes by $N$ (where $N$ is the number of clients).
2. **Round-Level Checkpointing**: Implemented `CheckpointingFedAvg` which serializes model weights (`.npz`) and client accounting metrics (`.json`) after each successful round. 
   - *Limitation Documented (Scientific Safety)*: Automated resumption via `start_simulation` was specifically omitted. Injecting `initial_parameters` into Flower resets the round counter to 1, which would cause the DP-SGD algorithm to execute additional rounds without accounting for the previous privacy spent, silently breaking the DP composition. Checkpoints are safely preserved on disk for manual inspection, but cannot be automatically resumed without rewriting Flower's internal loop handling.

## Before vs. After Benchmark

| Metric | Before | After | Change |
|---|---:|---:|---:|
| **Total Runtime (Local CPU)** | 51.70 s | 49.35 s | -4.5% (See Note) |
| **Total Evaluation Operations (Per Round)** | 3× (All clients) | 1× (Server only) | -66.7% |
| **DP-SGD Integrity / Final Accuracy** | 12.50% | 13.50% | Preserved |
| **Privacy Accounting Epsilon** | 4.4430 | 4.4430 | Preserved |
| **Artifact Persistence** | Memory Only | Serialized per round | Gained Inspectability |

## Projection to Full Scale
While the local subset runtime is overwhelmingly dominated by Ray's native process initialization cost on Windows (~40 seconds overhead), the real-world Colab execution will see massive benefits:
- **Redundant Compute Savings**: 5 clients on the 10,000-sample full test set would have resulted in 50,000 test inferences *per round*. Centralized evaluation cuts this to 10,000, saving 80% of evaluation time per round.
- **Resilience**: The introduction of checkpoints means 12-hour experiments yield partial results if interrupted, even though automatic resume is intentionally unsupported for scientific/DP accounting safety.
- **Opacus Overhead**: Profiling shows `PrivacyEngine.make_private` initialization takes ~1.75s only on the first setup and drops to 3ms on subsequent clients. Thus, caching Opacus state provides no scientific or time benefit.
