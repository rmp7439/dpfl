# Google Colab Execution Guide for Stage 6 Replication

This guide provides the exact execution workflow to run the 15-round, 12-seed Stage 6 replication on a Google Colab T4 instance. It includes safe resume behavior so that if the runtime disconnects, previously completed runs are preserved and skipped.

## 1. Setup and Environment

Open a new Google Colab notebook and select a **T4 GPU** runtime.

Execute the following cell to clone the repository and install dependencies securely:

```bash
# Cell 1: Setup
!git clone https://github.com/your-org/dpfl.git
%cd dpfl

# Install pinned dependencies (resolves CuDNN mismatches on Colab T4)
!pip install -r requirements.txt

# IMPORTANT: You MUST restart the Colab runtime after this cell 
# to ensure the correct PyTorch CUDA version is loaded.
# Go to Runtime -> Restart session.
```

## 2. Verify CUDA Availability

After restarting the runtime, verify the environment:

```python
# Cell 2: Verify GPU
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available. Ensure you selected a T4 GPU runtime.")
```

## 3. Mount Google Drive for Durable Storage (Recommended)

To protect against runtime disconnections, mount Google Drive and symlink the results directory:

```python
# Cell 3: Durable Storage
from google.colab import drive
import os

drive.mount('/content/drive')

# Create a persistent directory on Drive
persistent_dir = '/content/drive/MyDrive/dpfl_stage6_replication'
os.makedirs(persistent_dir, exist_ok=True)

# Symlink local results to Drive so outputs are streamed directly to persistent storage
if not os.path.exists('results/stage6/multiseed_replication'):
    os.makedirs('results/stage6', exist_ok=True)
    os.symlink(persistent_dir, 'results/stage6/multiseed_replication')
print("Symlinked results to Google Drive.")
```

## 4. Run Preflight Validation

Run a small, fast subset to verify the pipeline (15 rounds, 3 clients, 1000 samples) before launching the 12-hour job:

```bash
# Cell 4: Preflight
!python scripts/run_ablation.py --subset
```
*Note: This will execute the preflight in `results/stage6/subset_validation_replication`. It should succeed and print `Ablation complete`.*

## 5. Execute Full Multi-Seed Replication

The execution script natively loops through the 12 configurations (alphas 0.1, 10.0; sigmas 1.0, 2.0; seeds 42, 43, 44). It checks for valid `summary.json` files and **safely skips completed runs**, making it safe to re-run this cell if Colab disconnects.

```bash
# Cell 5: Full Replication
!python scripts/run_ablation.py
```

## 6. Aggregate Results and Generate Figures

Once all 12 runs complete successfully, run the aggregation script to compile JSON and CSV summaries containing means and standard deviations, then regenerate the figures.

```bash
# Cell 6: Aggregation & Figures
!python scripts/aggregate_stage6_multiseed.py
!python scripts/generate_figures.py --full
```

## 7. Retrieve Final Artifacts

Your final `.csv`, `.json`, and `figures/` will be safely persisted in your Google Drive folder (`/content/drive/MyDrive/dpfl_stage6_replication`). You can copy the generated figures into the Drive folder for easy access:

```bash
# Cell 7: Backup Figures
!cp -r figures/ /content/drive/MyDrive/dpfl_stage6_replication/
print("All artifacts successfully saved to Google Drive.")
```
