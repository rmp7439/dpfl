
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
"""
Stage 6 Ablation Studies: runs different configurations of (alpha, sigma) 
with DP-SGD Flower federation on the full CIFAR-10 dataset and compiles results.

Usage:
    python stage6_ablation.py            # full CIFAR-10 run
    python stage6_ablation.py --subset   # 1k-sample subset (for validation only)
"""
import os
import sys
import subprocess
import json
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Stage 6 Ablations")
    parser.add_argument("--subset", action="store_true",
                        help="Run on 1k subset (validation only, NOT official results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Stage 6 requires:
    # 1. varying alpha
    # 2. varying noise multiplier sigma
    # 3. alpha=0.1 vs alpha=10 non-IID impact comparison
    alphas = [0.1, 10.0]
    sigmas = [1.0, 2.0]  # A scientifically useful subset of Stage 5 sigmas
    C = 1.0  # Fixed clipping norm for ablations

    mode = "subset" if args.subset else "full"
    base_out_dir = os.path.join("results", "stage6", "per_run")
    stage_dir = os.path.join("results", "stage6")

    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STAGE 6 ABLATION SEARCH — mode={mode}, seed={args.seed}")
    print(f"{'='*60}")
    print(f"  alpha values : {alphas}")
    print(f"  sigma values : {sigmas}")
    print(f"  C            : {C}")
    print(f"  {len(alphas) * len(sigmas)} total runs")
    print(f"  Results      : {base_out_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed = []

    # 1. Run all configurations sequentially
    for a in alphas:
        for s in sigmas:
            print(f"\n--- alpha={a}, sigma={s}, C={C} ({mode}) ---")
            output_dir = os.path.join(base_out_dir, f"alpha_{a}_sigma_{s}_C_{C}")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                sys.executable, "federated.py",
                "--enable-dp",
                "--alpha", str(a),
                "--sigma", str(s),
                "--C", str(C),
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

    # 2. Compile ablation results
    print("\nCompiling Ablation Results...")
    all_results = []
    for a in alphas:
        for s in sigmas:
            summary_path = os.path.join(base_out_dir, f"alpha_{a}_sigma_{s}_C_{C}", "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    all_results.append(json.load(f))
            else:
                print(f"WARNING: Missing summary for alpha={a}, sigma={s}")

    # Save combined JSON
    with open(os.path.join(stage_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Save combined CSV
    keys = [
        "alpha", "sigma", "C", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"
    ]
    with open(os.path.join(stage_dir, "ablation_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for res in all_results:
            writer.writerow([res.get(k, "") for k in keys])

    print(f"\n{'='*60}")
    print(f"Ablation search complete: {len(all_results)} runs compiled")
    if failed:
        print(f"FAILED runs: {failed}")
    print(f"Artifacts: {stage_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
