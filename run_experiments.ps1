Write-Host "Waiting for CIFAR-10 download to complete (needs to reach ~170MB)..."
while ((Get-Item data\cifar-10-python.tar.gz).Length -lt 165000000) {
    Start-Sleep -Seconds 15
}
# Give it a bit of extra time to finish writing the file
Start-Sleep -Seconds 10
Write-Host "Download complete. Extracting and running subset split..."
.venv\Scripts\python.exe data_split.py
Write-Host "Running full dataset split..."
.venv\Scripts\python.exe data_split.py --full
Write-Host "Committing figures to git..."
git add subset_dirichlet_split.png full_dirichlet_split.png
git commit -m "add dirichlet split figures for subset and full dataset"
git push
Write-Host "All experiments and commits done!"
