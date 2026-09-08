# Technical Report: Differentially Private Federated Learning (DP-FL) on CIFAR-10

## 1. Abstract
This project implements and evaluates a Differentially Private Federated Learning (DP-FL) system on the CIFAR-10 dataset. By combining Flower for federated orchestration and Opacus for Differential Privacy (DP), we investigate the privacy-utility tradeoff under realistic non-IID data distributions (Dirichlet $\alpha=0.1$). Our comprehensive evaluation across varying noise multipliers ($\sigma$), clipping thresholds ($C$), and data heterogeneity levels demonstrates that DP protects individual data samples at the cost of model utility. The negative impact on convergence is severely compounded by high data heterogeneity.

## 2. Introduction
Federated Learning (FL) allows distributed edge devices to collaboratively train a shared model without transmitting raw data to a central server. However, exchanging model weights or gradients remains vulnerable to inference attacks, where adversaries can reconstruct original training examples. Differential Privacy (DP) mitigates this by injecting calibrated noise and clipping gradients, providing a mathematical guarantee of privacy. In this work, we train a CNN on CIFAR-10 distributed across 5 clients, exploring how data heterogeneity interacts with DP-SGD mechanisms.

## 3. Problem Statement
The objective is to train a convolutional neural network on a heavily skewed, non-IID partition of CIFAR-10 across distributed clients. The system must guarantee $(\varepsilon, \delta)$-Differential Privacy for each client's dataset. We aim to quantify how much accuracy is sacrificed for strong privacy guarantees, particularly when the underlying local data is highly specialized (non-IID).

## 4. Background
### 4.1 Federated Learning
FL operates through synchronized rounds. The server broadcasts a global model, clients train locally, and the server aggregates the updates (e.g., via FedAvg).

### 4.2 Differential Privacy & DP-SGD
DP guarantees that a single data sample's presence or absence does not statistically alter the algorithm's output probability beyond a bound $e^\varepsilon$. DP-SGD achieves this by computing per-sample gradients, clipping them to a maximum $L_2$ norm $C$, and adding Gaussian noise scaled by $C$ and $\sigma$.

### 4.3 Rényi Differential Privacy (RDP)
RDP tracks the privacy loss analytically across multiple training steps using Rényi divergence. RDP provides tighter composition bounds for Gaussian mechanisms compared to standard DP, allowing us to safely convert the cumulative RDP back to $(\varepsilon, \delta)$ metrics after training.

## 5. Threat and Privacy Motivation
A malicious server or a compromised client intercepting updates could execute membership inference or data reconstruction attacks. Our threat model assumes an honest-but-curious server and adversarial participants. DP protects against these threats by ensuring that any output weight matrix is statistically indistinguishable regardless of any single participant's specific training samples.

## 6. Methodology
Our methodology proceeds in sequential stages:
1. Dirichlet data partitioning to simulate non-IID data.
2. Centralized training baseline.
3. Standard Federated Learning (FedAvg).
4. Differentially Private Federated Learning (DP-FedAvg).
5. Grid search over DP hyperparameters ($C, \sigma$).
6. Ablation over data heterogeneity ($\alpha, \sigma$).

## 7. System Architecture
We use the **Flower** (`flwr`) framework to orchestrate virtual clients. Instead of physical network distribution, we leverage the **Ray** Virtual Client Engine to simulate massive concurrency on a single machine or GPU. The local training loop uses standard PyTorch, wrapped entirely by **Opacus**.

## 8. Dataset and Non-IID Partitioning
The CIFAR-10 dataset (50k train, 10k test) is partitioned across 5 clients using a Dirichlet distribution.
- **$\alpha=0.1$**: High heterogeneity (official setup). Clients typically hold only 1-2 classes dominantly.
- **$\alpha=10.0$**: Low heterogeneity (near-IID). Clients have an approximately uniform mix of all 10 classes.

## 9. Model Architecture
We use a SimpleCNN designed explicitly for DP:
- 3 Convolutional layers (3→16→32→64 channels)
- **GroupNorm** instead of BatchNorm (crucial, as BatchNorm breaks per-sample independence)
- MaxPool & ReLU activations
- Fully Connected layer (64*4*4 → 128 → 10)
- Dropout (p=0.5) for regularization

## 10. Federated Training Procedure
- **Clients**: 5
- **Rounds**: 3
- **Local Epochs**: 1
- **Batch Size**: 64
- **Optimizer**: SGD, Learning Rate: 0.05
- **Aggregation**: FedAvg

## 11. DP Mechanism
We implement DP using Opacus. The `PrivacyEngine` wraps the local model, optimizer, and dataloader. The dataloader uses Poisson sampling to create batches, computing true per-sample gradients.
- **Clipping**: Per-sample gradient $L_2$ norms are bounded to $C \in \{0.1, 1.0\}$.
- **Noise**: Gaussian noise scaled by $\sigma \in \{0.5, 1.0, 1.5, 2.0\}$ is added to the batch-averaged gradient.

## 12. Privacy Accounting
Because the dataset is statically partitioned and clients hold disjoint datasets, we utilize **parallel composition**. The global privacy loss is bounded by the privacy loss of the worst-case individual client.
1. We extract the exact sample rate (expected batch size / local dataset size) per client from the Opacus `DPDataLoader`.
2. We track the exact number of DP steps per client per round.
3. RDP is computed for each client over all their individual steps.
4. We convert RDP to $\varepsilon$ using $\delta=10^{-5}$. The global $\varepsilon$ is $\max(\varepsilon_{clients})$.

## 13. Experimental Setup
Experiments were run on a GPU environment (Tesla T4 via Colab) to overcome Ray compatibility issues on Windows and accelerate Opacus's computationally expensive per-sample gradient hooks.

## 14. Centralized Baseline (Stage 2)
Training the model centralized on the full 50,000 CIFAR-10 images for 30 epochs achieves **74.87%** test accuracy. This represents the theoretical upper bound without federation or privacy constraints.

## 15. Non-private FedAvg (Stage 3)
Federating the model across 5 clients with $\alpha=0.1$ (high skew) for 3 rounds severely degrades performance, resulting in **33.26%** accuracy. Non-IID data causes local models to diverge sharply, making standard FedAvg aggregation suboptimal.

## 16. DP-FL (Stage 4)
Adding Opacus DP ($\sigma=1.0, C=1.0$) further impacts the signal-to-noise ratio. The model achieves **20.14%** accuracy at $\varepsilon=1.5394$. The noise makes bridging the non-IID divergence even harder.

## 17. Sigma/C Grid Results (Stage 5)
We mapped the privacy-utility curve. 
- Higher $\sigma$ decreases $\varepsilon$ (stronger privacy) but hurts accuracy.
- Lower $C$ heavily degrades accuracy but does not impact the mathematical $\varepsilon$ directly.
- **Best Privacy**: $\sigma=2.0, C=0.1$ gives $\varepsilon=0.3989$ but accuracy drops to 19.54%.
- **Best Accuracy**: $\sigma=0.5, C=1.0$ gives 25.11% accuracy but weak privacy ($\varepsilon=10.8698$).

## 18. Alpha Ablation (Stage 6)
We varied the data heterogeneity $\alpha \in \{0.1, 10.0\}$ while using DP ($\sigma \in \{1.0, 2.0\}$).
- At $\sigma=1.0$, near-IID data ($\alpha=10.0$) achieves 22.33% accuracy, whereas non-IID ($\alpha=0.1$) achieves only 18.25%.
- This proves that data skew compoundingly penalizes DP-SGD performance.

## 19. Privacy–Utility Tradeoff
Our findings confirm a fundamental tension. To maintain an acceptable privacy guarantee ($\varepsilon < 2.0$), the model accuracy on a complex dataset like CIFAR-10 under non-IID federated conditions collapses to sub-30%. The injected noise overwhelms the already conflicting local gradients.

## 20. Discussion
DP-FL requires significantly more communication rounds to converge because individual updates are noisy and bounded. The interplay between weight divergence (from non-IID partitions) and gradient obfuscation (from DP) is the primary obstacle to production-grade DP-FL.

## 21. Limitations
1. **Rounds**: We were restricted to 3 communication rounds due to computational constraints.
2. **Hyperparameters**: $\sigma$ and $C$ were grid-searched, but LR and momentum were frozen. DP-SGD typically requires specifically tuned learning rates.
3. **Accounting Assumption**: We assumed perfect parallel composition; dynamic client dropout or overlapping data would require more complex accounting.
4. **Secure RNG**: For experimentation, cryptographically secure RNG was disabled to boost speed; production requires it enabled.

## 22. Reproducibility
The official protocol relies on Linux/Colab execution due to Ray compatibility limits on Windows. The repository contains all artifacts, split visualizations, and testing utilities. Due to a Colab runtime termination, the final structured artifacts for Stage 6 were lost, but the raw execution logs were preserved and rigorously archived to maintain experimental integrity without expending redundant GPU compute.

## 23. Conclusion
We successfully implemented a fully accountable DP-FL pipeline using modern frameworks (Flower, Opacus). While mathematical privacy guarantees are achievable, they enforce a severe utility penalty on image classification tasks under realistic non-IID conditions.

## 24. References
1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. Abadi et al. "Deep Learning with Differential Privacy" (DP-SGD)
3. Mironov. "Rényi Differential Privacy"
4. Opacus Documentation & Flower Documentation
