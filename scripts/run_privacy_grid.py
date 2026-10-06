import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

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

    from config import STAGE5_SIGMAS, STAGE5_C_VALUES
    sigmas = STAGE5_SIGMAS
    Cs = STAGE5_C_VALUES

    mode = "subset_validation" if args.subset else "full"
    grid_dir = os.path.join("results", "stage5", mode)
    base_out_dir = os.path.join(grid_dir, "per_run")

    os.makedirs(base_out_dir, exist_ok=True)

    expected_runs = len(sigmas) * len(Cs)

    print(f"\n{'='*60}")
    print(f"STAGE 5 GRID SEARCH — mode={mode}, seed={args.seed}")
    print(f"{'='*60}")
    print(f"  sigma values : {sigmas}")
    print(f"  C values     : {Cs}")
    print(f"  {expected_runs} total runs expected")
    print(f"  Results      : {base_out_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed = []
    missing = []
    successful = []

    # 1. Run all configurations sequentially
    for s in sigmas:
        for c in Cs:
            print(f"\n--- sigma={s}, C={c} ({mode}) ---")
            output_dir = os.path.join(base_out_dir, f"sigma_{s}_C_{c}")
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "summary.json")
            if os.path.exists(summary_path):
                os.remove(summary_path)

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
            if (s, c) in failed:
                print(f"ERROR: Skipping summary collection for failed run sigma={s}, C={c}")
            elif os.path.exists(summary_path):
                with open(summary_path) as f:
                    all_results.append(json.load(f))
                successful.append((s, c))
            else:
                print(f"ERROR: Missing summary for sigma={s}, C={c}")
                missing.append((s, c))

    # Save combined JSON
    out_json = os.path.join(grid_dir, "grid_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    # Save combined CSV
    keys = [
        "sigma", "C", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"
    ]
    out_csv = os.path.join(grid_dir, "grid_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for res in all_results:
            writer.writerow([res.get(k, "") for k in keys])

    print(f"\n{'='*60}")
    print(f"Grid search complete:")
    print(f"Expected runs  : {expected_runs}")
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs    : {len(failed)}")
    print(f"Missing runs   : {len(missing)}")
    if failed:
        print(f"FAILED configs: {failed}")
    if missing:
        print(f"MISSING configs: {missing}")
    print(f"Artifacts: {grid_dir}")
    print(f"{'='*60}")

    if failed or missing:
        print("ERROR: Not all expected runs completed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
