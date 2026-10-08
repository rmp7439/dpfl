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
    parser = argparse.ArgumentParser(description="Run Stage 6 Ablation")
    parser.add_argument("--subset", action="store_true",
                        help="Run on 1k subset (validation only, NOT official results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for multiseed outputs")
    args = parser.parse_args()

    from config import STAGE6_ALPHAS, STAGE6_SIGMAS, STAGE6_C
    alphas = STAGE6_ALPHAS
    sigmas = STAGE6_SIGMAS
    C_val = STAGE6_C

    mode = "subset_validation" if args.subset else "full"
    
    if args.output_root:
        grid_dir = args.output_root
        base_out_dir = args.output_root
    else:
        grid_dir = os.path.join("results", "stage6", mode)
        base_out_dir = os.path.join(grid_dir, "per_run")

    os.makedirs(base_out_dir, exist_ok=True)

    expected_runs = len(alphas) * len(sigmas)

    print(f"\n{'='*60}")
    print(f"STAGE 6 ABLATION — mode={mode}, seed={args.seed}")
    print(f"{'='*60}")
    print(f"  alpha values : {alphas}")
    print(f"  sigma values : {sigmas}")
    print(f"  C value      : {C_val}")
    print(f"  {expected_runs} total runs expected")
    print(f"  Results      : {base_out_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed = []
    missing = []
    successful = []

    # 1. Run all configurations sequentially
    for a in alphas:
        for s in sigmas:
            print(f"\n--- alpha={a}, sigma={s}, C={C_val} ({mode}) ---")
            if args.output_root:
                dir_name = f"alpha_{a}_sigma_{s}_C_{C_val}_seed_{args.seed}"
            else:
                dir_name = f"alpha_{a}_sigma_{s}_C_{C_val}"
            output_dir = os.path.join(base_out_dir, dir_name)
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "summary.json")
            if os.path.exists(summary_path):
                os.remove(summary_path)

            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_federated.py"),
                "--enable-dp",
                "--alpha", str(a),
                "--sigma", str(s),
                "--C", str(C_val),
                "--output-dir", output_dir,
                "--seed", str(args.seed),
            ]
            if not args.subset:
                cmd.append("--full")

            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Run failed for alpha={a}, sigma={s}: {e}")
                failed.append((a, s))

    # 2. Compile grid results
    print("\nCompiling Ablation Results...")
    all_results = []
    for a in alphas:
        for s in sigmas:
            if args.output_root:
                dir_name = f"alpha_{a}_sigma_{s}_C_{C_val}_seed_{args.seed}"
            else:
                dir_name = f"alpha_{a}_sigma_{s}_C_{C_val}"
            summary_path = os.path.join(base_out_dir, dir_name, "summary.json")
            if (a, s) in failed:
                print(f"ERROR: Skipping summary collection for failed run alpha={a}, sigma={s}")
            elif os.path.exists(summary_path):
                with open(summary_path) as f:
                    all_results.append(json.load(f))
                successful.append((a, s))
            else:
                print(f"ERROR: Missing summary for alpha={a}, sigma={s}")
                missing.append((a, s))

    # Save combined JSON
    json_name = f"ablation_results_seed_{args.seed}.json" if args.output_root else "ablation_results.json"
    out_json = os.path.join(grid_dir, json_name)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    # Save combined CSV
    keys = [
        "alpha", "sigma", "C", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"
    ]
    csv_name = f"ablation_results_seed_{args.seed}.csv" if args.output_root else "ablation_results.csv"
    out_csv = os.path.join(grid_dir, csv_name)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for res in all_results:
            writer.writerow([res.get(k, "") for k in keys])

    print(f"\n{'='*60}")
    print(f"Ablation complete:")
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
