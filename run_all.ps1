Write-Host "Waiting for CIFAR-10 download to complete (~170MB)..."
while ((Get-Item data\cifar-10-python.tar.gz -ErrorAction SilentlyContinue).Length -lt 165000000) {
    Start-Sleep -Seconds 15
}
Start-Sleep -Seconds 20

Write-Host "--- STAGE 1: Data Split ---"
.venv\Scripts\python.exe data_split.py
.venv\Scripts\python.exe data_split.py --full
git add subset_dirichlet_split.png full_dirichlet_split.png
git commit -m "add dirichlet split figures for subset and full dataset"
git push

Write-Host "--- STAGE 2: Centralized Baseline ---"
.venv\Scripts\python.exe train_baseline.py
.venv\Scripts\python.exe train_baseline.py --full

Write-Host "--- STAGE 3: Flower Federation ---"
.venv\Scripts\python.exe federated.py
.venv\Scripts\python.exe federated.py --full

Write-Host "All automated runs completed."
