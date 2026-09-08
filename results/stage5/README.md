# Stage 5: RDP Accounting and Grid Search

This stage implements Rényi Differential Privacy (RDP) accounting to rigorously track privacy loss across multiple communication rounds of federated learning. Furthermore, it executes an 8-configuration grid search over key privacy hyperparameters to empirically visualize the privacy-utility tradeoff.

## What is $(\epsilon, \delta)$-DP?
Differential Privacy (DP) guarantees that the output of a randomized algorithm (our training process) does not change significantly when any single individual's data is added or removed. 
- $\epsilon$ (epsilon) is the privacy budget. A smaller $\epsilon$ represents a stronger privacy guarantee (less information leaked).
- $\delta$ (delta) is the probability that the pure $\epsilon$-DP guarantee fails. We typically set $\delta = 10^{-5}$ which is significantly smaller than $1/N$ for CIFAR-10.

## RDP and Composition
Why do we use Rényi Differential Privacy (RDP) instead of naive $(\epsilon, \delta)$-DP composition?
- **Naive Composition:** If a single optimizer step costs $\epsilon$, $T$ steps cost $T \cdot \epsilon$. This bound grows linearly and becomes extremely loose (pessimistic) for neural networks which require thousands of steps.
- **RDP:** RDP tracks the Rényi divergence between the distributions of the mechanism on adjacent datasets across various "orders" (denoted as $\alpha$). RDP composes linearly, meaning we just add the RDP values for each step at each order $\alpha$. After summing the RDP across all steps, we convert the final RDP bound back to a tight $(\epsilon, \delta)$-DP guarantee by finding the order $\alpha$ that minimizes $\epsilon$.

The order $\alpha$ that yields the tightest (lowest) $\epsilon$ is called the "minimizing Rényi order".

## Grid Search
We sweep over two parameters:
- **$\sigma$ (Noise Multiplier):** $[0.5, 1.0, 1.5, 2.0]$
- **$C$ (Max Grad Norm):** $[0.1, 1.0]$

### Reproducing the Grid Search
To execute the full grid search on CIFAR-10, run from the project root:
```bash
python scripts/run_privacy_grid.py --full
```

To quickly validate the pipeline on a 1,000-sample subset:
```bash
python scripts/run_privacy_grid.py --subset
```

To plot the resulting accuracy vs. cumulative $\epsilon$ curves:
```bash
python plot_stage5.py
```

### Outputs
- `grid_results.csv` and `grid_results.json`: Summary of all configurations.
- `per_run/`: Contains `rounds.csv` for each run, tracing $\epsilon$ iteratively over every round.
- `accuracy_vs_epsilon.png`: The visual representation of the privacy-utility tradeoff.
