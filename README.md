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
├── src/dpfl/            # Core library modules (model, config, data)
├── scripts/             # Execution scripts for experiments
├── tests/               # Unit and integration tests
├── results/             # Structured output and archived configurations
├── figures/             # High-resolution (300 DPI) generated plots
└── docs/                # LaTeX technical report and defense notes
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
1. The exact sample rate is queried from Opacus's `DPDataLoader` (expected batch size / local dataset size).
2. The number of physical DP steps per client is recorded per round.
3. RDP is composed sequentially over these steps.
4. Because the Dirichlet split creates disjoint client datasets, privacy composes in parallel across clients. Global $\varepsilon$ is defined by the maximum $\varepsilon$ among all participating clients, evaluated at $\delta=10^{-5}$.
*Note: Clipping norm $C$ strictly bounds sensitivity but does not alter the mathematical formulation of RDP directly (noise is scaled internally as $C \times \sigma$).*

## Results summary
- **Stage 2 Centralized**: 74.87%
- **Stage 3 Non-private FL**: 33.26%
- **Stage 4 DP-FL Reference ($\sigma=1.0, C=1.0$)**: 20.14% ($\varepsilon=1.5394$)
- **Stage 5 Grid (Best Privacy)**: 19.54% at $\varepsilon=0.3989$ ($\sigma=2.0, C=0.1$)
- **Stage 6 Homogeneous ($\alpha=10.0$)**: 22.33% ($\sigma=1.0, \varepsilon=1.2595$)
- **Stage 6 Heterogeneous ($\alpha=0.1$)**: 18.25% ($\sigma=1.0, \varepsilon=1.5394$)

*All reported numbers are strictly from official, reproducible runs. The Stage 6 values are preserved from validated console output after a Colab artifact loss.*

## Reproducibility
The official experiments leverage deterministic seeds. However, the multi-client simulation relies on Ray, which introduces execution non-determinism. Secure RNG for Opacus was disabled for execution speed, restricting this codebase to experimental/research usage rather than production deployment.

## GPU requirements
Full-scale DP-FL experiments (Stage 4-6) computationally mandate a GPU (e.g., Tesla T4 on Colab) due to Opacus's per-sample gradient hooks. 

## Known Windows/Flower limitations
Ray Virtual Client Engine support on native Windows is highly limited and prone to crashes or timeouts. Researchers on Windows must use WSL2 or execute on a Linux/Colab cloud instance.

## Test suite
The repository includes a comprehensive 34-test suite validating Dirichlet distributions, non-private fallbacks, and DP accounting logic.
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Report
The formal academic report is located at `docs/technical_report.tex`.

## Citation
If utilizing this repository for further research, please credit this project and the corresponding authors.
