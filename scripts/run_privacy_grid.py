
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
"""
Stage 5 Grid Search: runs 8 configurations of (sigma, C) with DP-SGD Flower federation
on the full CIFAR-10 dataset and compiles results into a unified artifact.

Usage:
    python run_privacy_grid.py            # full CIFAR-10 run
    python run_privacy_grid.py --subset   # 1k-sample subset (for validation only)
"""
import os
import sys
import subprocess
import json
import csv
import argparse
import datetime


def main():
    parser = argparse.ArgumentParser(description="Run Stage 5 Grid Search")
    parser.add_argument("--subset", action="store_true",
                        help="Run on 1k subset (validation only, NOT official results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sigmas = [0.5, 1.0, 1.5, 2.0]
    Cs = [0.1, 1.0]

    mode = "subset" if args.subset else "full"
    base_out_dir = os.path.join("results", "stage5", "per_run")
    grid_dir = os.path.join("results", "stage5")

    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STAGE 5 GRID SEARCH — mode={mode}, seed={args.seed}")
    print(f"{'='*60}")
    print(f"  sigma values : {sigmas}")
    print(f"  C values     : {Cs}")
    print(f"  8 total runs")
    print(f"  Results      : {base_out_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed = []

    # 1. Run all configurations sequentially
    for s in sigmas:
        for c in Cs:
            print(f"\n--- sigma={s}, C={c} ({mode}) ---")
            output_dir = os.path.join(base_out_dir, f"sigma_{s}_C_{c}")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_federated.py"),
                "--enable-dp",
                "--sigma", str(s),
                "--C", str(c),
                "--output-dir", output_dir,
                "--seed", str(args.seed),
            ]
            if not args.subset:
                cmd.append("--full")

            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Run failed for sigma={s}, C={c}: {e}")
                failed.append((s, c))

    # 2. Compile grid results
    print("\nCompiling Grid Results...")
    all_results = []
    for s in sigmas:
        for c in Cs:
            summary_path = os.path.join(base_out_dir, f"sigma_{s}_C_{c}", "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    all_results.append(json.load(f))
            else:
                print(f"WARNING: Missing summary for sigma={s}, C={c}")

    # Save combined JSON
    with open(os.path.join(grid_dir, "grid_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Save combined CSV
    keys = [
        "sigma", "C", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"
    ]
    with open(os.path.join(grid_dir, "grid_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for res in all_results:
            writer.writerow([res.get(k, "") for k in keys])

    print(f"\n{'='*60}")
    print(f"Grid search complete: {len(all_results)} runs compiled")
    if failed:
        print(f"FAILED runs: {failed}")
    print(f"Artifacts: {grid_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
