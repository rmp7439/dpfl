# Stage 4: DP-SGD using Opacus in Federated Learning

## Overview
This stage implements Differentially Private Stochastic Gradient Descent (DP-SGD) using the Opacus library within the Flower federated learning framework. The DP training mechanism protects user privacy by bounding the sensitivity of the model to any single training example.

## Privacy Mechanism Lifecycle
During local client training, a single gradient undergoes the following lifecycle:
1. **Raw Per-Example Gradient:** Opacus hooks into PyTorch's backward pass to compute gradients with respect to *each individual sample* in the batch (`grad_sample`), rather than just the averaged gradient.
2. **Clipping:** Each per-sample gradient is clipped to a maximum L2 norm $C$ (`max_grad_norm`). This bounds the sensitivity of the function; no single example can influence the gradient by more than $C$.
3. **Aggregation:** The clipped per-sample gradients are summed to form a single, bounded aggregate gradient for the batch.
4. **Noise Injection:** Gaussian noise scaled by $\sigma \cdot C$ (where $\sigma$ is `noise_multiplier`) is added to the aggregate gradient. This obfuscates the contribution of any specific individual's clipped gradient.
5. **Optimizer Update:** The noisy, aggregated gradient is normalized by the batch size and used by the optimizer (e.g., Adam or SGD) to update the model parameters.

## Privacy/Utility Tradeoff
- **C (max_grad_norm):** Controls how much the gradient of an outlier example is suppressed. A smaller $C$ guarantees lower sensitivity and requires less absolute noise, but heavily biases the update direction by clipping typical gradients. A larger $C$ preserves the true gradient direction better but requires adding proportionally larger noise.
- **Sigma (noise_multiplier):** Determines the ratio of noise to the clipping bound $C$. Larger $\sigma$ yields stronger privacy (lower epsilon) but degrades model accuracy (utility). Smaller $\sigma$ yields better accuracy but weaker privacy guarantees.

## Results Persistence
Results, validation artifacts, and runtime logs for both subset and full CIFAR-10 executions are stored in this directory (`results/stage4/`).
- `validation.json`: Confirms that Opacus successfully computed `grad_sample` and applied clipping/noise.
- `comparison_full_*.png`: Plots the utility difference between Centralized Baseline (Stage 2), Non-Private Federation (Stage 3), and DP-Federation (Stage 4).
