@echo off
set PYTHONUNBUFFERED=1

echo =========================================
echo RUNNING STAGE 2: CENTRALIZED BASELINE
echo =========================================
.venv\Scripts\python train_baseline.py --full

echo =========================================
echo RUNNING STAGE 3: NON-PRIVATE FEDERATION
echo =========================================
.venv\Scripts\python federated.py --full

echo =========================================
echo RUNNING STAGE 4/5: DP-FL GRID SEARCH
echo =========================================
.venv\Scripts\python grid_search.py --full

echo =========================================
echo GENERATING STAGE 5 FINAL PLOT
echo =========================================
.venv\Scripts\python plot_stage5.py

echo ALL DONE!
