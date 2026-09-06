import os
import subprocess
import json
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Stage 5 Grid Search")
    parser.add_argument("--subset", action="store_true", help="Run on subset (for fast validation)")
    args = parser.parse_args()
    
    sigmas = [0.5, 1.0, 1.5, 2.0]
    Cs = [0.1, 1.0]
    
    mode = "subset" if args.subset else "full"
    base_out_dir = "results/stage5/per_run"
    
    # 1. Run all configurations
    for s in sigmas:
        for c in Cs:
            print(f"\n=======================================================")
            print(f"Running Configuration: sigma={s}, C={c} (mode={mode})")
            print(f"=======================================================\n")
            
            output_dir = os.path.join(base_out_dir, f"sigma_{s}_C_{c}")
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                ".venv\\Scripts\\python", "federated.py", 
                "--enable-dp", 
                "--sigma", str(s), 
                "--C", str(c), 
                "--output-dir", output_dir
            ]
            if not args.subset:
                cmd.append("--full")
                
            subprocess.run(cmd, check=True)
            
    # 2. Compile global grid results
    print("\nCompiling Grid Results...")
    all_results = []
    for s in sigmas:
        for c in Cs:
            summary_path = os.path.join(base_out_dir, f"sigma_{s}_C_{c}", "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    all_results.append(json.load(f))
            else:
                print(f"WARNING: Missing results for sigma={s}, C={c}")
                
    # Save combined JSON
    os.makedirs("results/stage5", exist_ok=True)
    with open("results/stage5/grid_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Save combined CSV
    if all_results:
        keys_to_save = ["sigma", "C", "dataset_mode", "number_of_communication_rounds", 
                        "sample_rate", "total_dp_steps", "epsilon", "best_alpha", 
                        "final_test_accuracy", "best_test_accuracy", "runtime_seconds", "run_status"]
        
        with open("results/stage5/grid_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys_to_save)
            for res in all_results:
                writer.writerow([res.get(k, "") for k in keys_to_save])
                
    print("Grid search completed successfully!")

if __name__ == "__main__":
    main()
