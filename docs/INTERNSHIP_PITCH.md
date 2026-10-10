# Two-Minute Internship Pitch: Differentially Private Federated Learning

## Problem
Federated Learning enables model training on decentralized data, but sharing gradients can still leak sensitive client information. Applying Differential Privacy (DP) offers a mathematical guarantee against this, but the strict noise injection and gradient clipping of DP-SGD often degrade model accuracy. The core problem we investigated is how severely this privacy-utility tradeoff degrades under highly heterogeneous (non-IID) client data distributions.

## Approach
To simulate realistic data skew, we partitioned CIFAR-10 across 5 clients using a Dirichlet distribution ($\alpha=0.1$). We built a robust federated framework integrating PyTorch for the CNN model, Flower (`flwr`) via Ray for virtual client simulation, and Opacus for automated per-sample gradient computation and noise injection. Privacy was tracked using Rényi Differential Privacy (RDP) accounting, converting sequential per-client DP steps into a global $(\varepsilon, \delta)$ guarantee using parallel composition.

## Implementation Challenge
The most significant implementation challenge was ensuring deterministic privacy accounting and execution under simulated environments. Because Opacus scales injected Gaussian noise based on the clipping norm $C$, and highly specialized client datasets shift the batch sampling rate, heterogeneous data actually required significantly more DP steps (e.g., 3045 vs 2475 steps) compared to homogeneous splits. Furthermore, integrating Opacus hooks in a Ray virtual client engine required strict environment isolation to prevent memory leaks and CUDA mismatch errors on cloud GPUs.

## Key Result
The results demonstrate the steep penalty of strict privacy: our Stage 4 reference DP-FL model achieved 26.11% accuracy ($\varepsilon=2.6974$) on 15 rounds, compared to the 47.38% non-private federated baseline and a 74.87% centralized baseline (which fell short of the initial >85% target). An initial Stage 6 legacy run suggested that stronger data heterogeneity ($\alpha=0.1$) further degraded utility to 24.41% compared to 31.51% for the homogeneous ($\alpha=10.0$) case. A full 15-round multi-seed replication to statistically verify this robustness effect is fully implemented but pending execution due to GPU availability.

## Limitations
Our evidence maps the privacy-utility tradeoff and confirms DP's penalty, but it does not yet establish statistical significance across multiple random seeds for the heterogeneity ablation. Additionally, the learning rate and momentum were frozen due to compute constraints, meaning the DP configurations are not fully optimized. We also disabled Opacus's secure RNG for speed, making this suitable for experimental validation rather than production deployment.
