"""
One-shot script: recompile grid_results.json and grid_results.csv from whatever
per_run/sigma_*_C_*/summary.json files exist. Adds a dataset_mode field to the top-level
so the consistency test can distinguish subset from full runs.
"""
import json, csv, os, glob

base = os.path.join("results", "stage5", "per_run")
out_dir = os.path.join("results", "stage5")

runs = []
for path in sorted(glob.glob(os.path.join(base, "*", "summary.json"))):
    with open(path) as f:
        data = json.load(f)
    runs.append(data)
    print(f"  sigma={data['sigma']}, C={data['C']}, mode={data['dataset_mode']}, eps={data['epsilon']:.4f}")

print(f"\nTotal: {len(runs)} runs")

# Save JSON
with open(os.path.join(out_dir, "grid_results.json"), "w") as f:
    json.dump(runs, f, indent=4)
print(f"Saved grid_results.json ({len(runs)} entries)")

# Save CSV
keys = ["sigma", "C", "dataset_mode", "number_of_communication_rounds",
        "sample_rate", "total_dp_steps", "epsilon", "best_alpha",
        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"]
with open(os.path.join(out_dir, "grid_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(keys)
    for r in runs:
        writer.writerow([r.get(k, "") for k in keys])
print(f"Saved grid_results.csv")
