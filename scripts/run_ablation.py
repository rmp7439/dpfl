import sys
import os
import subprocess
import json
import csv
import argparse
import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

from scripts.validate_artifact import validate_artifact

def main():
    parser = argparse.ArgumentParser(description="Run Stage 6 Ablation (Specific 4 runs)")
    parser.add_argument("--subset", action="store_true",
                        help="Run on subset (validation only, NOT official results)")
    args = parser.parse_args()

    mode = "subset_validation" if args.subset else "multiseed"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = os.path.join("results", "stage6", f"{mode}_fixed_{timestamp}")
    base_out_dir = os.path.join(grid_dir, "per_run")

    os.makedirs(base_out_dir, exist_ok=True)

    # Specific intended reruns for Stage 6
    configs = [
        {"alpha": 0.1, "sigma": 1.0, "C": 1.0, "seed": 42},
        {"alpha": 0.1, "sigma": 2.0, "C": 1.0, "seed": 42},
        {"alpha": 10.0, "sigma": 1.0, "C": 1.0, "seed": 42},
        {"alpha": 10.0, "sigma": 2.0, "C": 1.0, "seed": 44},
    ]

    expected_runs = len(configs)

    print(f"\n{'='*60}")
    print(f"STAGE 6 ABLATION — mode={mode}")
    print(f"{'='*60}")
    print(f"  {expected_runs} total runs expected")
    print(f"  Results      : {base_out_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed = []
    missing = []
    successful = []
    all_results = []

    # 1. Run all configurations sequentially
    for cfg in configs:
        a = cfg["alpha"]
        s = cfg["sigma"]
        c_val = cfg["C"]
        seed = cfg["seed"]

        print(f"\n--- alpha={a}, sigma={s}, C={c_val}, seed={seed} ({mode}) ---")
        dir_name = f"alpha_{a}_sigma_{s}_C_{c_val}_seed_{seed}"
        output_dir = os.path.join(base_out_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "run_federated.py"),
            "--enable-dp",
            "--alpha", str(a),
            "--sigma", str(s),
            "--C", str(c_val),
            "--output-dir", output_dir,
            "--seed", str(seed),
            "--rounds", "15",
        ]
        if not args.subset:
            cmd.append("--full")

        try:
            subprocess.run(cmd, check=True, env=env)
            
            # Immediate strict artifact validation
            summary_path = os.path.join(output_dir, "summary.json")
            if not os.path.exists(summary_path):
                raise RuntimeError(f"Run completed but summary.json is missing at {summary_path}")
                
            validate_artifact(summary_path, {
                "sigma": s,
                "C": c_val,
                "number_of_communication_rounds": 15,
                "num_clients": 5,
                "alpha": a,
                "seed": seed,
                "delta": 1e-5
            })
            
            print(f"--- config alpha={a}, sigma={s}, C={c_val}, seed={seed} verified successfully. ---")
            
        except Exception as e:
            print(f"ERROR: Run or validation failed for config {cfg}: {e}")
            print(f"STOPPING THE ABLATION DUE TO FAILURE.")
            sys.exit(1)

        summary_path = os.path.join(output_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_results.append(json.load(f))
            successful.append(cfg)
        else:
            print(f"ERROR: Missing summary for config {cfg}")
            missing.append(cfg)

    # 2. Save combined results
    out_json = os.path.join(grid_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    keys = [
        "alpha", "sigma", "C", "seed", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"
    ]
    out_csv = os.path.join(grid_dir, "ablation_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for res in all_results:
            writer.writerow([res.get(k, "") for k in keys])

    print(f"\n{'='*60}")
    print(f"Ablation complete:")
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
        sys.exit(1)

if __name__ == "__main__":
    main()
