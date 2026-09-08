# Stage 6 Archival Results (Reconstructed from Console Output)

**Notice:** These are archived results from the completed 2026-09-08 Colab run. The original `results/stage6/` directory and filesystem artifacts were lost when the Colab runtime disconnected. This file was reconstructed directly from the preserved console logs to prevent unnecessary reruns. No new experiment run was conducted to generate this file.

## Protocol Metadata (Extracted from Logs)
- **Mode:** full (FULL CIFAR-10)
- **Seed:** 42
- **Clients:** 5
- **Communication Rounds:** 3
- **Device:** cuda (Tesla T4)
- **C (Clipping Norm):** 1.0

## The Four Completed Configurations

### 1. alpha=0.1, sigma=1.0, C=1.0
- **Final Accuracy:** 18.25%
- **Final Epsilon:** 1.5394
- **Best RDP alpha:** 8.2
- **Runtime:** 389.50 seconds
- **Round-by-round accuracy:**
  - Round 1: 11.08%
  - Round 2: 15.22%
  - Round 3: 18.25%

### 2. alpha=0.1, sigma=2.0, C=1.0
- **Final Accuracy:** 11.87%
- **Final Epsilon:** 0.3989
- **Best RDP alpha:** 34.0
- **Runtime:** 385.03 seconds
- **Round-by-round accuracy:**
  - Round 1: 10.22%
  - Round 2: 11.08%
  - Round 3: 11.87%

### 3. alpha=10.0, sigma=1.0, C=1.0
- **Final Accuracy:** 22.33%
- **Final Epsilon:** 1.2595
- **Best RDP alpha:** 9.2
- **Runtime:** 389.75 seconds
- **Round-by-round accuracy:**
  - Round 1: 14.23%
  - Round 2: 22.00%
  - Round 3: 22.33%

### 4. alpha=10.0, sigma=2.0, C=1.0
- **Final Accuracy:** 10.10%
- **Final Epsilon:** 0.3133
- **Best RDP alpha:** 38.0
- **Runtime:** 385.49 seconds
- **Round-by-round accuracy:**
  - Round 1: 11.45%
  - Round 2: 10.03%
  - Round 3: 10.10%

---
*Generated directly from preserved `Pasted markdown(7).md`.*
