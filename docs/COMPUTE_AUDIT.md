# COMPUTE AUDIT

## 1. Execution Path & Time Breakdown (Measured)
The baseline run on a 1000-sample subset (3 clients, 2 rounds) revealed the following breakdown:
- **Total Runtime**: 51.70s
- **Ray/Flower Initialization Overhead**: ~42s (This is a fixed startup cost heavily penalized on native Windows, but less impactful on full Colab runs).
- **Client Training Time (Opacus DP-SGD)**: ~0.8s - 1.2s per client per round on CPU.
- **Client Setup Time**: ~0.002s (Dataset `Subset` and `SimpleCNN` creation are practically instantaneous).
- **Opacus Setup Time (`make_private`)**: ~1.75s on the very first call (likely JIT/import overhead), and ~0.003s on all subsequent calls.
- **Evaluation Time**: ~0.15s per client per round on a 200-sample test set.

## 2. Identified Bottlenecks

| Component | Time | % Runtime | Priority |
|---|---:|---:|---|
| Redundant Client Evaluation | O(Clients × TestSet) | Medium | HIGH |
| Lack of Checkpointing | Entire Run Lost on Crash | N/A | HIGH |
| DP Local Training | ~1.0s / round | Core Work | MEDIUM |
| Ray Orchestration Startup | ~42s | High (Local) | LOW |
| Opacus/Model Recreation | < 0.005s | Low | LOW |

## 3. Key Findings

1. **Redundant Global Evaluation**: In `run_federated.py`, all clients evaluate the model on the exact same `GLOBAL_TESTSET` during the evaluation phase. For 5 clients on the full 10,000-sample test set, this means the global test set is evaluated 5 times per round, yielding identical results and wasting 80% of the evaluation compute.
2. **Missing Fault Tolerance**: Privacy accounting (`compute_rdp`) and saving results happen strictly *after* `fl.simulation.start_simulation` returns. If a Colab GPU runtime preempts during round 7 of 10, the entire `history` object is lost, destroying all privacy tracking and metrics.
3. **Recreation Overheads are Negligible**: Recreating `SimpleCNN` and `DataLoader` takes just 2 milliseconds. Reusing them via caching is not worth the architectural complexity.
4. **Client Parallelism**: The trace shows clients are executing correctly via Ray, but the `client_fn` deprecation warnings indicate older Flower API usage. 

## 4. Optimization Plan
1. **Implement Centralized Evaluation**: Shift evaluation from clients to the server by providing an `evaluate_fn` to the `FedAvg` strategy and setting `fraction_evaluate=0.0`. This will cut evaluation compute by a factor of the number of clients (e.g., 5x).
2. **Implement Checkpointing Strategy**: Create a custom `CheckpointingFedAvg` strategy that extends `FedAvg`. After every round, it will save the global weights, round metrics, and privacy client stats to disk, allowing safe resumption.
3. **Implement Resume Logic**: Add logic to load the latest checkpoint before `start_simulation`, injecting the resumed weights as `initial_parameters`.
