# Differentially Private Federated Learning (DP-FL) on CIFAR-10

## Project overview
A research implementation of federated learning with Rényi Differential Privacy (RDP) on CIFAR-10. This project evaluates how Differential Privacy (via Opacus) impacts model convergence in a highly non-IID federated environment orchestrated by Flower (`flwr`). 

## Research question
How severely does the strict bound of Differential Privacy degrade the classification accuracy of a CNN under highly heterogeneous (non-IID) federated client distributions? How do variations in noise injection ($\sigma$) and clipping norm ($C$) shift this privacy-utility tradeoff?

## Key contributions
- Dirichlet non-IID client partitioning to simulate realistic data skew.
- Integration of Opacus DP-SGD with Flower federated training.
- Empirical mapping of the privacy-utility tradeoff across a comprehensive grid of hyperparameter configurations.
- Alpha ablation study identifying the empirical utility penalty of data heterogeneity on DP models.
- Parallel composition privacy accounting extracting empirical dataloader sample rates.

## Architecture
- **Model**: SimpleCNN with GroupNorm (3 convolutional layers, MaxPool, Dropout, FC layers). BatchNorm is avoided to ensure per-sample gradient independence.
- **Federated Engine**: Flower (`flwr`) using Ray for virtual client simulation.
- **Privacy Engine**: Opacus for automated per-sample gradient computation, clipping, and noise injection.

## Experimental protocol
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test)
- **Clients**: 5 (3 for quick validation/subset runs)
- **Communication rounds**:
  - Quick validation/subset runs: 2-3 rounds
  - Historical experiments (Stage 6 legacy): 3 rounds
  - Official full experiments (Stages 3, 4, 5): 15 rounds
  - New replication experiments (Stage 6 multi-seed): 15 rounds
- **Local epochs**: 1
- **Batch size**: 64
- **Federated Optimizer**: SGD, learning rate = 0.05
- **Centralized Optimizer**: Adam, learning rate = 0.001
- **Dirichlet $\alpha$**: 0.1 (unless ablated)
- **Seed**: 42 (42, 43, 44 for multi-seed replication)

## Repository structure
```text
DPFL/
├── data/                # Dataset storage
├── figures/             # Generated research figures
├── results/             # Structured experimental results and evidence
├── scripts/             # Experiment and plotting scripts
├── src/                 # Core project modules
├── tests/               # Automated tests
│
├── .gitignore
├── AGENTS.md
├── README.md
├── requirements.txt
└── technical_report.tex
```

## Installation and Reproducibility

### Using Docker (Recommended)
A `Dockerfile` is provided for guaranteed reproducibility without dependency conflicts. It uses Python 3.11 and installs the exact pinned dependencies required for the project.

```bash
docker build -t dpfl .
docker run --rm -v ${PWD}/results:/app/results -v ${PWD}/figures:/app/figures dpfl python scripts/train_baseline.py --full
```

### Local Virtual Environment
If you prefer to run locally, ensure you are using Python 3.11+ and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

## Dataset
CIFAR-10 is automatically downloaded to `./data/`. The non-IID partitioning creates highly specialized subsets. 

## Running the centralized baseline
```bash
python scripts/train_baseline.py --full
```

## Running federated learning
```bash
python scripts/run_federated.py --full
```

## Running DP-FL
```bash
python scripts/run_federated.py --full --enable-dp
```

## Running the Stage 5 grid
```bash
python scripts/run_privacy_grid.py
```

## Running ablations
```bash
python scripts/run_ablation.py
```

## Generating figures
All publication figures are generated through a single canonical pipeline:
```bash
python scripts/generate_figures.py --full
```
For validation runs on small data subsets, append `--subset`.

## Privacy accounting explanation
Privacy is accounted via **Rényi Differential Privacy (RDP)**:
1. The sampling rate used for accounting is derived from the DPDataLoader expected batch size and local dataset size.
2. The number of physical DP steps performed by each client is recorded for each communication round.
3. RDP is composed sequentially across the DP steps performed by each client.
4. Because the Dirichlet partition creates disjoint client datasets, privacy composes in parallel across clients.
5. The global privacy guarantee is defined by the maximum client-level $\varepsilon$, evaluated at $\delta = 1e-5$.

*Note: Clipping norm $C$ strictly bounds per-sample gradient sensitivity, and Opacus scales injected Gaussian noise according to $C$ and $\sigma$.*

## Results summary
- **Stage 2 Centralized**: 74.87% (Note: This empirical baseline fell below the original project target of >85%, thus downstream utility comparisons are made relative to 74.87% rather than the original target).
- **Stage 3 Non-private FL**: 47.38% (15 rounds, $\alpha=0.1$)
- **Stage 4 DP-FL Reference (15 rounds, $\sigma=1.0, C=1.0$)**: 26.11%, $\varepsilon=2.6974$ (best: 27.29% at round 14)
- **Stage 5 Grid (Strongest Privacy)**: 30.57%, $\varepsilon=0.9117$ ($\sigma=2.0, C=0.1$)
- **Stage 5 Grid (Best Accuracy)**: 36.49%, $\varepsilon=19.0778$ ($\sigma=0.5, C=1.0$)
- **Stage 6 Homogeneous ($\alpha=10.0, \sigma=1.0$)**: 31.51%, $\varepsilon=2.0691$
- **Stage 6 Heterogeneous ($\alpha=0.1, \sigma=1.0$)**: 24.41%, $\varepsilon=2.6974$

*All reported values are from the project's official 15-round experimental runs. (Note: A new balanced 15-round multi-seed replication for Stage 6 is pending GPU availability).*

## Reproducibility
The official experiments leverage deterministic seeds (seed=42). However, Ray-based multi-client simulation can introduce execution-level non-determinism. Opacus Secure RNG was disabled for execution speed, therefore the implementation is intended for experimental/research use rather than production privacy deployment.

## Hardware and Runtime Caveats
While experiments were planned for a GPU environment to accelerate Opacus's per-sample gradient hooks, empirical analysis of the execution artifacts confirms that Stages 2, 3, and 4 were executed on a CPU. This is documented transparently in the final report. Stage 5 full grid results and historical Stage 6 ablations reflect CUDA execution where available (e.g., historical Colab T4 runs).

## Stage 6 Robustness Status
The historical Stage 6 ablation results indicate an empirical association between stronger data heterogeneity ($\alpha=0.1$) and lower observed utility under the tested configuration, requiring significantly more DP steps (e.g. 3045 vs 2475) due to specialized datasets shifting the batch sampling rate. A formal multi-seed robustness run with 15 rounds is planned to verify these effects statistically, but is currently **PENDING** due to GPU unavailability in the current local environment. All local runs are constrained to CPU, making a 12-seed 15-round Opacus execution prohibitively slow for immediate replication.

**Google Colab T4 Validation:**
To reproduce the validated GPU stack without `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` errors on Colab T4 instances:
1. Do not use the default cu130 stack which causes mismatch errors.
2. Do not disable cuDNN.
3. Install the verified CUDA 12.6 PyTorch build using the unified requirements file:
   ```bash
   pip install -r requirements.txt
   ```
4. Restart the Colab runtime before importing `torch`.
5. Note: `secure_mode=False` is acceptable for fast experimentation, but final/production privacy runs must use `secure_mode=True`.

## Known Windows/Flower limitations
Ray Virtual Client Engine support on native Windows is highly limited and prone to crashes or timeouts. Researchers on Windows must use WSL2 or execute on a Linux/Colab cloud instance.

## Test suite
The repository includes a focused automated test suite validating:
- Dirichlet data partitioning
- federated training behavior
- privacy mechanisms
- RDP accounting
- experimental result validation

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Report
The formal academic report is located at `technical_report.tex`. Please note that this repository does not include a LaTeX compiler. To compile the PDF locally, you will need a LaTeX distribution (such as TeX Live, MiKTeX, or Overleaf). The `.tex` file is structurally verified to compile with `pdflatex`.

## Citation
If utilizing this repository for further research, please credit this project and the corresponding authors.
