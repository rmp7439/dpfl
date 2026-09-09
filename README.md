# Differentially Private Federated Learning (DP-FL) on CIFAR-10

## Project overview
A research implementation of federated learning with Rényi Differential Privacy (RDP) on CIFAR-10. This project evaluates how Differential Privacy (via Opacus) impacts model convergence in a highly non-IID federated environment orchestrated by Flower (`flwr`). 

## Research question
How severely does the strict bound of Differential Privacy degrade the classification accuracy of a CNN under highly heterogeneous (non-IID) federated client distributions? How do variations in noise injection ($\sigma$) and clipping norm ($C$) shift this privacy-utility tradeoff?

## Key contributions
- Dirichlet non-IID client partitioning to simulate realistic data skew.
- Integration of Opacus DP-SGD with Flower federated training.
- Empirical mapping of the privacy-utility tradeoff across a comprehensive grid of hyperparameter configurations.
- Alpha ablation study identifying the compounding penalty of data heterogeneity on DP models.
- Parallel composition privacy accounting extracting empirical dataloader sample rates.

## Architecture
- **Model**: SimpleCNN with GroupNorm (3 convolutional layers, MaxPool, Dropout, FC layers). BatchNorm is avoided to ensure per-sample gradient independence.
- **Federated Engine**: Flower (`flwr`) using Ray for virtual client simulation.
- **Privacy Engine**: Opacus for automated per-sample gradient computation, clipping, and noise injection.

## Experimental protocol
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test)
- **Clients**: 5
- **Communication rounds**: 3
- **Local epochs**: 1
- **Batch size**: 64
- **Optimizer**: SGD, learning rate = 0.05
- **Dirichlet $\alpha$**: 0.1 (unless ablated)
- **Seed**: 42

## Repository structure
```text
DPFL/
├── archive/             # Archived development and validation material
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

## Installation
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

## Privacy accounting explanation
Privacy is accounted via **Rényi Differential Privacy (RDP)**:
1. The sampling rate used for accounting is derived from the DPDataLoader expected batch size and local dataset size.
2. The number of physical DP steps performed by each client is recorded for each communication round.
3. RDP is composed sequentially across the DP steps performed by each client.
4. Because the Dirichlet partition creates disjoint client datasets, privacy composes in parallel across clients.
5. The global privacy guarantee is defined by the maximum client-level $\varepsilon$, evaluated at $\delta = 1e-5$.

*Note: Clipping norm $C$ strictly bounds per-sample gradient sensitivity, and Opacus scales injected Gaussian noise according to $C$ and $\sigma$.*

## Results summary
- **Stage 2 Centralized**: 74.87%
- **Stage 3 Non-private FL**: 33.26%
- **Stage 4 DP-FL Reference ($\sigma=1.0, C=1.0$)**: 20.14%, $\varepsilon=1.5394$
- **Stage 5 Grid (Strongest Privacy)**: 19.54%, $\varepsilon=0.3989$ ($\sigma=2.0, C=0.1$)
- **Stage 6 Homogeneous ($\alpha=10.0, \sigma=1.0$)**: 22.33%, $\varepsilon=1.2595$
- **Stage 6 Heterogeneous ($\alpha=0.1, \sigma=1.0$)**: 18.25%, $\varepsilon=1.5394$

*All reported values are from the project's official experimental runs. The Stage 6 results were preserved from validated Colab console output following loss of the corresponding runtime artifacts.*

## Reproducibility
The official experiments leverage deterministic seeds (seed=42). However, Ray-based multi-client simulation can introduce execution-level non-determinism. Opacus Secure RNG was disabled for execution speed, therefore the implementation is intended for experimental/research use rather than production privacy deployment.

## GPU requirements
Full-scale DP-FL experiments (Stage 4-6) computationally mandate a GPU (e.g., Tesla T4 on Colab) due to Opacus's per-sample gradient hooks. 

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
The formal academic report is located at `technical_report.tex`.

## Citation
If utilizing this repository for further research, please credit this project and the corresponding authors.
