# Differentially Private Federated Learning (DP-FL) on CIFAR-10

## Project overview
A research implementation of federated learning with Rényi Differential Privacy (RDP) on CIFAR-10. This project evaluates how Differential Privacy (via Opacus) impacts model convergence in a highly non-IID federated environment orchestrated by Flower (`flwr`). 

## Research question
How severely does the strict bound of Differential Privacy degrade the classification accuracy of a CNN under highly heterogeneous (non-IID) federated client distributions? How do variations in noise injection ($\sigma$) and clipping norm ($C$) shift this privacy-utility tradeoff?

## Key contributions
- Dirichlet non-IID client partitioning to simulate realistic data skew.
- Integration of Opacus DP-SGD with Flower federated training.
- Empirical mapping of the privacy-utility tradeoff across a comprehensive grid of hyperparameter configurations.
- Alpha ablation study investigating the interaction between data heterogeneity (Dirichlet $\alpha \in \{0.1, 10.0\}$) and Differential Privacy noise scale on CNN convergence.
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
  - Official full experiments (Stages 3, 4, 5, and Stage 6 reference runs): 15 rounds
  - Multi-seed robustness grid (Stage 6, 12 runs across 3 seeds): 3 rounds
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
├── Dockerfile
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
CIFAR-10 (50,000 training, 10,000 test images across 10 classes) is partitioned across clients using a Dirichlet distribution in `src/data.py`.
- **CLI Configuration**: `src/data.py` accepts `--alpha` (float, default: `0.1`), `--num-clients` (int, default: 5 for full, 3 for subset), `--seed` (default: 42), and `--full` (or subset mode).
- **Partition Artifacts**: Saves parameter-specific validation JSON and histogram plots (`results/stage1/split_validation_{mode}_alpha{alpha}.json` and `results/stage1/client_class_distribution_{mode}_alpha{alpha}.png`).
- **Verified Partition Properties (5 clients, full CIFAR-10, seed 42)**:
  - **$\alpha=0.1$ (High Skew)**: Client sample sizes `[12944, 10514, 11934, 9029, 5579]`; heterogeneity metric (mean class standard deviation across clients) = `1699.28`. Clients hold highly concentrated, disjoint class distributions.
  - **$\alpha=10.0$ (Near-IID)**: Client sample sizes `[8856, 10009, 10553, 10399, 10183]`; heterogeneity metric = `236.88`. Clients hold approximately balanced representations of all 10 classes. 

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

The pipeline enforces strict publication standards:
- **Dimensions**: All figures are sized to single-column standard width of 3.5 inches (canvas preserves exact dimensions without distortion).
- **Formats**: Triple-format export for every figure: standalone PDF (`.pdf`), high-resolution raster PNG (`.png`, verified $\ge 300$ DPI), and native LaTeX PGF (`.pgf`).
- **Engine Verification**: Checks for a working `pdflatex` installation via `check_latex_toolchain()`, outputting `LaTeX engine verified: ...`. In environments without a local TeX binary, `FallbackLatexManager` provides font metric estimation for PGF generation.
- **Atomic Staging**: Figures are rendered in an isolated temporary staging directory before atomic deployment.

## Privacy accounting explanation
Privacy is accounted via **Rényi Differential Privacy (RDP)**:
1. The sampling rate used for accounting is derived from the DPDataLoader (`q = 1 / len(train_loader) = 1 / ceil(n_k / 64)`).
2. The number of physical DP steps performed by each client is recorded for each communication round.
3. RDP is composed sequentially across the DP steps performed by each client.
4. Because the Dirichlet partition creates disjoint client datasets, privacy composes in parallel across clients.
5. The global privacy guarantee is defined by the maximum client-level $\varepsilon$, evaluated at $\delta = 1e-5$.

*Note: Clipping norm $C$ strictly bounds per-sample gradient sensitivity, and Opacus scales injected Gaussian noise according to $C$ and $\sigma$. The privacy analysis uses Opacus RDP accounting with Poisson-sampled batches (`make_private` default) and the loader's sample rate q = 1/len(loader). The reported global bound takes the maximum per-client $\varepsilon$ assuming parallel composition across disjoint partitions and post-processing aggregation; while numerical values match Opacus accountant recomputations, these modeling assumptions have not undergone external human review.*

## Results summary
- **Stage 2 Centralized Baseline**: 74.87% (Adam, lr=0.001; empirical baseline relative to target).
- **Stage 3 Non-private FedAvg**: 47.38% (15 rounds, $\alpha=0.1$, FedAvg control).
- **Stage 4 DP-FL Reference (15 rounds, $\sigma=1.0, C=1.0$)**: 26.11%, $\varepsilon=2.6974$ (best: 27.29% at round 14).
- **Stage 5 Grid (Strongest Privacy)**: 30.57%, $\varepsilon=0.9117$ ($\sigma=2.0, C=0.1$).
- **Stage 5 Grid (Best Accuracy)**: 36.49%, $\varepsilon=19.0778$ ($\sigma=0.5, C=1.0$).
- **Stage 6 15-Round Reference Runs**:
  - $\alpha=10.0, \sigma=1.0$ (seed 42): 31.51%, $\varepsilon=2.0691$
  - $\alpha=0.1, \sigma=1.0$ (seed 42): 24.41%, $\varepsilon=2.6974$
  - $\alpha=10.0, \sigma=2.0$ (seed 44*): 15.26%, $\varepsilon=0.7271$ (*non-like-for-like comparison: seed 44 used due to Ray IPC worker crash on seed 42)
  - $\alpha=0.1, \sigma=2.0$ (seed 42): 13.79% (best: 18.95%), $\varepsilon=0.9117$
- **Stage 6 Multi-Seed Robustness Grid ($N=3$, seeds 42, 43, 44, 3 rounds)**:
  - $\alpha=0.1, \sigma=1.0$: $18.66\% \pm 0.93\%$, $\varepsilon=1.5171 \pm 0.0215$
  - $\alpha=0.1, \sigma=2.0$: $10.96\% \pm 1.07\%$, $\varepsilon=0.3712 \pm 0.0301$
  - $\alpha=10.0, \sigma=1.0$: $17.88\% \pm 2.36\%$, $\varepsilon=1.2354 \pm 0.0214$
  - $\alpha=10.0, \sigma=2.0$: $9.77\% \pm 0.69\%$, $\varepsilon=0.3073 \pm 0.0133$

*(All multi-seed uncertainty values reported in documentation denote sample standard deviation with Bessel's correction, $s$ with $\text{ddof}=1$; the automated storage files `aggregate.csv` and `aggregate.json` record population standard deviation with $\text{ddof}=0$; complete tabular details and side-by-side reconciliation are documented in [`results/stage6/STAGE6_SUMMARY.md`](results/stage6/STAGE6_SUMMARY.md)).*

## Reproducibility architecture and baseline verification

### Random seed derivation and client isolation
In `scripts/run_federated.py`:
- `main()` seeds Python `random`, NumPy, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`) using `args.seed`.
- `derive_seed(base_seed, client_id, server_round)` derives a stable, process-independent integer seed in $[0, 2^{31}-1]$ from `f"{base_seed}:{client_id}:{server_round}"` using SHA-256.
- Each client `fit()` invocation reseeds `random`, `np.random`, and `torch`, creating an isolated `torch.Generator` passed to data loader shuffling and Opacus `make_private(..., noise_generator=gen)`.
- The initial global model is deterministically initialized with `args.seed` and passed as `initial_parameters` (or restored from checkpoint when resuming).
- In `CheckpointingFedAvg.aggregate_fit()`, incoming client results are sorted deterministically by numeric client ID before federated averaging.

### Empirical reproducibility verification
The pipeline's determinism is validated by three integration runs in `results/reproducibility/`:
- **Run A (seed 42) vs. Run B (seed 42)**: Both runs produced identical initial model hash (`00c43041ea88aeae6062c53a51a03a08e6ced7c38624c6cab9f84770086177f3`), identical final model hash (`e23b740fdbc3071a3fced62bab52f30b934d9fd8aab30457f7199e7086d6efce`), array-by-array identical layer weights, identical cumulative epsilon (`4.442962601652252`), identical best alpha (`4.1`), identical total DP steps (`30`), and identical per-round accuracy (`13.5%` round 1, `12.5%` round 2).
- **Run Seed 43 (divergence check)**: Seed 43 produced a different initial model hash (`3f269f49a0401293b0f9c9032383bb2c7de05abc1d8bc05e725d9431efab2fec`), different final model hash (`1f2fa650dd8ca6154a4501f06ff5f87840eec3165eee8c2ea21f698d2a6c1104`), different final accuracy (`7.5%`, best: `7.5%`), different epsilon (`4.696513893070338`), and different best alpha (`4.0`), while maintaining matching structural total DP steps (`30`).

### Centralized baseline reproducibility (Stage 2)
The non-private centralized reference in `results/stage2/` was audited via a complete rerun (`scripts/compare_baselines.py`):
- **Metric Agreement**: Both the original run and rerun achieved **74.87%** final accuracy and **74.87%** best accuracy across 30 epochs (comparison classification: `EXACT_MATCH`, `Trajectory Match: True`).
- **Trajectory Agreement**: All 30 epochs exhibit identical train loss, train accuracy, test loss, and test accuracy (maximum absolute difference = `0.000000`, identical CSV SHA-256: `7e36e5a9...`).
- **Recorded Runtimes**: The reference recorded `2,487.73` seconds while the rerun recorded `887.06` seconds (runtime differed by 1600.67s despite identical training trajectory).
- **Reference Ceiling**: The achieved baseline of 74.87% is below the project plan's original >85% target and serves as the non-private empirical ceiling against which federated and private models are compared.

### Critical historical caveats
- **Chronology of Full-Scale Artifacts**: Comparison of embedded timestamps (ranging from 2026-09-07 to 2026-10-10) and Git commit history (`08ad3f8` on 2026-10-11) confirms that all full-scale Stage 3, 4, 5, and 6 artifacts predate the end-to-end reproducibility patch. Exact bitwise determinism has been demonstrated solely on the two-round subset integration runs; exact full-scale reproducibility is not claimed for historical runs.
- **Secure RNG**: Opacus secure RNG was disabled (`secure_rng: false`) for experimental speed and does not provide production-grade cryptographically secure randomness.

## Hardware and Execution Evidence
All documented experimental stages in the repository (Stage 2 baseline, Stage 3 FedAvg, Stage 4 DP-FL reference, Stage 5 full 15-round grid, and Stage 6 15-round reference runs and multi-seed grid) confirmed CPU execution in their run summary JSON artifacts (`"device": "cpu"`, `"GPU": "None"`). None of the saved authoritative run artifacts indicate CUDA execution. While experiments were initially planned for cloud GPU acceleration, the recorded runtimes reflect CPU-bound execution.

## Stage 6 Multi-Seed Robustness and Heterogeneity Analysis

### Three-Round Multi-Seed Evaluation ($N=3$, seeds 42, 43, 44)
Across the 12 three-round multi-seed runs:
- At $\sigma=1.0$: $\alpha=0.1$ achieved mean accuracy $18.66\% \pm 0.93\%$ ($s$, $\text{ddof}=1$), while $\alpha=10.0$ achieved $17.88\% \pm 2.36\%$.
- At $\sigma=2.0$: $\alpha=0.1$ achieved mean accuracy $10.96\% \pm 1.07\%$, while $\alpha=10.0$ achieved $9.77\% \pm 0.69\%$.
- **Statistical Interpretation**: In these 3-round experiments, the observed accuracy differences between $\alpha=0.1$ and $\alpha=10.0$ are within approximately one standard deviation ($\pm 2.36\%$ and $\pm 1.07\%$) and actually exhibit slightly higher sample means for $\alpha=0.1$. These data do not support a statistically established $\alpha$ effect or heterogeneity penalty at 3 rounds. No claim of statistical significance or statistical confidence is made.

### 15-Round Reference Runs
- **$\sigma=1.0$ Comparison**: At 15 rounds, seed 42 produced $24.41\%$ for $\alpha=0.1$ and $31.51\%$ for $\alpha=10.0$. This is an empirical observation from a single random seed (seed 42), not a general result.
- **$\sigma=2.0$ Comparison**: At 15 rounds, the $\alpha=10.0$ run used seed 44 ($15.26\%$) while the $\alpha=0.1$ run used seed 42 ($13.79\%$). Because the seed 42 run for $\alpha=10.0$ encountered a Ray IPC worker crash and required replacement with seed 44, this represents a non-like-for-like seed comparison. The utility difference cannot be attributed to data heterogeneity alone.
- **Physical DP Step Counts and Parallel Composition**: In our run summary artifacts, the top-level metric `total_dp_steps` records the maximum physical steps executed by any single client ($\max_k T_k$, belonging to the largest local partition), whereas the global privacy guarantee under parallel composition is governed by the worst-case privacy expenditure ($\varepsilon_{\text{global}} = \max_k \varepsilon_k$, belonging to the smallest local partition due to its higher sampling rate $q$). Specifically:
  - For $\alpha=0.1$, the largest client (Client 0, 12,944 samples, $q=0.00493$) executed the maximum $\max_k T_k = 3,045$ steps ($\varepsilon_0 = 1.6808$), while the smallest client (Client 4, 5,579 samples, $q=0.01136$) executed $T_4 = 1,320$ steps but incurred the worst-case privacy bound $\varepsilon_{\text{global}} = \varepsilon_4 = 2.6974$ (minimizing order $\approx 6.9$).
  - For $\alpha=10.0$, the largest client (Client 2, 10,553 samples, $q=0.00606$) executed the maximum $\max_k T_k = 2,475$ steps ($\varepsilon_2 = 1.8797$), while the smallest client (Client 0, 8,856 samples, $q=0.00719$) executed $T_0 = 2,085$ steps but incurred the worst-case privacy bound $\varepsilon_{\text{global}} = \varepsilon_0 = 2.0691$ (minimizing order $\approx 8.3$).

#### Per-Client Accounting Breakdown ($\sigma=1.0, C=1.0, \text{seed}=42, 15\text{ rounds}$)

**Configuration 1: Highly Heterogeneous ($\alpha=0.1$)**

| Client ID | Samples ($n_k$) | Sample Rate ($q_k$) | DP Steps ($T_k$) | Epsilon ($\varepsilon_k$) | Rényi Order ($\alpha_{\text{RDP}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 12,944 | 0.00493 | 3,045 | 1.6808 | 9.4 |
| 1 | 10,514 | 0.00606 | 2,475 | 1.8797 | 8.8 |
| 2 | 11,934 | 0.00535 | 2,805 | 1.7560 | 9.1 |
| 3 | 9,029 | 0.00704 | 2,130 | 2.0442 | 8.4 |
| 4 | 5,579 | 0.01136 | 1,320 | 2.6974 | 6.9 |

*Note: Per-client metrics are recorded directly in the run summary artifact (`client_details`). Global privacy is determined by Client 4 ($\varepsilon_{\text{global}} = 2.6974$), while summary `total_dp_steps` records Client 0 ($T=3,045$).*

**Configuration 2: Near-IID ($\alpha=10.0$)**

| Client ID | Samples ($n_k$) | Sample Rate ($q_k$) | DP Steps ($T_k$) | Epsilon ($\varepsilon_k$) | Rényi Order ($\alpha_{\text{RDP}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 8,856 | 0.00719 | 2,085 | 2.0691 | 8.3 |
| 1 | 10,009 | 0.00637 | 2,355 | 1.9323 | 8.7 |
| 2 | 10,553 | 0.00606 | 2,475 | 1.8797 | 8.8 |
| 3 | 10,399 | 0.00613 | 2,445 | 1.8925 | 8.8 |
| 4 | 10,183 | 0.00625 | 2,400 | 1.9120 | 8.7 |

*Note: Per-client metrics are recorded directly in the run summary artifact (`client_details`). Global privacy is determined by Client 0 ($\varepsilon_{\text{global}} = 2.0691$), while summary `total_dp_steps` records Client 2 ($T=2,475$).*

**Google Colab Environment Notes:**
If executing in cloud GPU environments like Google Colab:
1. Pinned dependencies in `requirements.txt` target CUDA 12.6 PyTorch builds.
2. The runtime must be restarted after dependency installation before importing `torch`.
3. Experiments must enable `secure_mode=True` for final production deployments requiring cryptographic privacy guarantees.

## Known Windows/Flower limitations
Ray Virtual Client Engine support on native Windows is highly limited and prone to crashes or timeouts. Researchers on Windows must use WSL2 or execute on a Linux/Colab cloud instance.

## Test suite
The repository includes an automated test suite validating the entire pipeline:
- **Reproducibility integration** (`tests/test_reproducibility.py`): Validates identical-seed determinism (matching model hashes, layer weights, epsilons, DP steps) and verifies parameter divergence under differing seeds.
- **Figure publication standards** (`tests/test_figures.py`): Verifies single-column 3.5-inch width, $\ge 300$ DPI PNG resolution, PGF export, and smoke-tests genuine `pdflatex` compilation.
- **Baseline output isolation and reproducibility** (`tests/test_baseline.py`): Tests tagged output paths (`--tag rerun`), artifact checksum verification, and exact 30-epoch trajectory matching against reference.
- **Dirichlet partitioning** (`tests/test_data.py`): Validates partition completeness, seeded determinism, alpha heterogeneity ordering ($\text{heterogeneity}(\alpha=10.0) < \text{heterogeneity}(\alpha=0.1)$), schema validation, and CLI argument validation.
- **Privacy accounting and Opacus DP-SGD** (`tests/test_accounting.py`, `tests/test_privacy.py`): Verifies RDP per-step calculation, optimal Rényi order minimization, and privacy conversion.
- **Federated orchestration** (`tests/test_federated.py`): Validates `CheckpointingFedAvg` client participation tracking, deterministic client sorting, and checkpointing.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Report
The formal academic report is located at `technical_report.tex`. Please note that this repository does not include a LaTeX compiler. To compile the PDF locally, you will need a LaTeX distribution (such as TeX Live, MiKTeX, or Overleaf). The `.tex` file is structurally verified to compile with `pdflatex`.

## Citation
If utilizing this repository for further research, please credit this project and the corresponding authors.
