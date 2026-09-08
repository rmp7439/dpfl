# FINAL DEFENSE NOTES: DP-FL on CIFAR-10

## A. 30-second explanation
We successfully simulated a federated learning environment on CIFAR-10 where clients hold non-IID data distributions (simulating real-world heterogeneity). We trained a CNN and applied Differential Privacy (DP-SGD via Opacus) to mathematically guarantee that individual client data points cannot be reverse-engineered from the model updates, and explored the tradeoffs between privacy strength and model accuracy.

## B. 2-minute project pitch
Federated Learning enables collaborative training without sharing raw data, but it remains vulnerable to inference attacks where an adversary can reconstruct training samples from weight updates. To solve this, we implemented Differentially Private Federated Learning (DP-FL). We partitioned the CIFAR-10 dataset across 5 clients using a Dirichlet distribution (alpha=0.1) to create highly skewed, realistic non-IID client datasets. We then replaced standard SGD with DP-SGD using the Opacus library. By clipping per-sample gradients and adding Gaussian noise to the aggregated updates, we bounded the privacy loss (epsilon). Across comprehensive grid searches and ablation studies, we proved that while DP effectively protects privacy, the necessary noise injection and clipping strictly limits model convergence, creating a fundamental accuracy-privacy tradeoff that becomes more pronounced under heterogeneous data distributions.

## C. Architecture explanation
- **Base Model**: A SimpleCNN (3 Conv layers, MaxPool, Dropout, FC layers). We specifically replaced standard BatchNorm with GroupNorm because BatchNorm relies on batch statistics which violate the per-sample independence required for DP.
- **Federated Engine**: Flower (`flwr`) manages the federated rounds, distributing the global model to 5 simulated Ray clients, which train locally and return updated weights.
- **Privacy Engine**: Opacus wraps the local PyTorch optimizers and dataloaders, automatically handling per-sample gradient computation, clipping, and noise injection.

## D. DP explanation
Differential Privacy (DP) guarantees that the output of an algorithm (our trained model) does not change significantly whether any single specific data point is included in the training set or not. This is achieved by:
1. **Clipping**: Limiting the maximum influence any single sample can have on the gradient (by clipping the L2 norm of the per-sample gradient to a constant C).
2. **Noise**: Adding random Gaussian noise to the aggregated gradients to mask individual contributions.

## E. RDP explanation
Rényi Differential Privacy (RDP) is a mathematically convenient generalization of standard (epsilon, delta)-DP. It uses Rényi divergence to track privacy loss across multiple training steps much more tightly than standard composition theorems. After training, we compute the total RDP across all steps and mathematically convert it back to standard (epsilon, delta) guarantees for a chosen delta (1e-5).

## F. Why epsilon changes with sigma
Sigma is the noise multiplier. A larger sigma means more Gaussian noise is added to the gradients relative to the clipping threshold C. More noise means individual samples are more heavily obfuscated, leading to less privacy loss per step. Thus, higher sigma mathematically results in a lower, stronger epsilon.

## G. Why clipping is needed
Without clipping, a single outlier sample could produce an infinitely large gradient, overpowering the injected noise and revealing its presence. By clipping per-sample gradients to a known maximum L2 norm (C), we mathematically bound the sensitivity of the update, allowing us to calibrate exactly how much noise is needed to mask it.

## H. Why DP hurts accuracy
1. **Noise**: The injected Gaussian noise degrades the signal-to-noise ratio of the true gradient, pushing the optimization in random directions.
2. **Clipping**: Clipping alters the true direction of the gradient, slowing down convergence and preventing the model from properly learning from highly informative (large gradient) outlier samples.

## I. Why non-IID matters
In real-world federated learning, clients do not have perfectly randomized representative data (IID). They have biased, specific local data (non-IID). When data is highly skewed (Dirichlet alpha=0.1), clients train models that diverge significantly from one another. Averaging these divergent models is already difficult; adding DP noise and clipping makes resolving these divergent updates even harder, severely penalizing accuracy.

## J. Stage 2 vs Stage 3 vs Stage 4 vs Stage 5 vs Stage 6
- **Stage 2**: Centralized, non-private baseline (one big dataset). Strongest accuracy (74.87%).
- **Stage 3**: Federated, non-private (5 clients, FedAvg). Accuracy drops due to non-IID skew (33.26%).
- **Stage 4**: Federated + DP (Opacus). Proved the pipeline works with default DP settings. Accuracy drops further (20.14%).
- **Stage 5**: Grid Search over noise (sigma) and clipping (C). Mapped the accuracy-vs-privacy curve.
- **Stage 6**: Ablation over data skew (alpha). Proved that higher heterogeneity (alpha=0.1) suffers more under DP than near-IID (alpha=10.0).

## K. Expected professor questions
1. *Why did you use GroupNorm instead of BatchNorm?*
2. *Why is your federated accuracy so much lower than centralized?*
3. *How did you handle privacy composition across the federation?*

## L. Strong answers
1. "BatchNorm normalizes across the batch, meaning one sample's activation depends on another's, breaking the per-sample independence required for Opacus to calculate valid per-sample gradients."
2. "The Dirichlet alpha=0.1 creates extreme non-IID skew. Clients heavily overfit to their local classes. Standard FedAvg struggles to reconcile these vastly diverging weights, and DP noise further destroys the aggregation signal."
3. "Because our clients have disjoint datasets, we applied parallel composition. The global privacy guarantee is bounded by the worst-case individual client. We tracked exact DP dataloader steps per client, calculated RDP for each, and the global epsilon is the maximum of those client epsilons."

## M. Known limitations
- Simulated federation (Ray) instead of real distributed network devices.
- Limited communication rounds (3) due to computational expense of DP-SGD.
- Sub-optimal hyperparameters (lr, momentum) — tuning DP-SGD requires extensive compute which was not available.
- Secure RNG was disabled in Opacus for performance; required for true cryptographically secure production DP.

## N. Exact final numbers
- **Centralized (Stage 2)**: ~74.87%
- **Non-private FL (Stage 3)**: ~33.26%
- **Stage 5 Best Privacy**: epsilon=0.3989 (Accuracy: 19.54% at C=0.1, 12.13% at C=1.0)
- **Stage 6 (alpha=0.1 vs 10.0 at sigma=1.0, C=1.0)**: alpha=0.1 gave 18.25%, alpha=10.0 gave 22.33%. Near-IID mitigates some DP performance loss.
