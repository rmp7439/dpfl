import os
import json
import numpy as np

def main():
    base_dir = os.path.join("results", "stage6", "multiseed")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        exit(1)
        
    all_runs = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            summary_path = os.path.join(item_path, "summary.json")
            if not os.path.exists(summary_path):
                print(f"ERROR: Missing summary.json in {item_path}")
                exit(1)
                
            with open(summary_path, "r") as f:
                data = json.load(f)
                all_runs.append(data)
                
    if not all_runs:
        print("No runs found to aggregate.")
        exit(1)
        
    # Group by (alpha, sigma, C)
    configs = {}
    for run in all_runs:
        key = (run.get("alpha"), run.get("sigma"), run.get("C"))
        if key not in configs:
            configs[key] = []
        configs[key].append(run)
        
    aggregate_results = []
    
    for (alpha, sigma, C), runs in configs.items():
        seeds = sorted(list(set(r.get("seed") for r in runs if r.get("seed") is not None)))
        final_accs = [r.get("final_test_accuracy", 0.0) for r in runs]
        best_accs = [r.get("best_test_accuracy", 0.0) for r in runs]
        
        epsilons = [r.get("epsilon") for r in runs if r.get("epsilon") is not None]
        best_alphas = [r.get("best_alpha") for r in runs if r.get("best_alpha") is not None]
        
        # epsilon and best_alpha handling
        epsilon_val = epsilons[0] if epsilons else None
        if epsilons and len(set(epsilons)) > 1:
            epsilon_val = {
                "mean": float(np.mean(epsilons)),
                "std": float(np.std(epsilons))
            }
            
        best_alpha_val = best_alphas[0] if best_alphas else None
        if best_alphas and len(set(best_alphas)) > 1:
            best_alpha_val = {
                "mean": float(np.mean(best_alphas)),
                "std": float(np.std(best_alphas))
            }
            
        dataset_modes = list(set(r.get("dataset_mode") for r in runs if r.get("dataset_mode") is not None))
        rounds_list = list(set(r.get("number_of_communication_rounds") for r in runs if r.get("number_of_communication_rounds") is not None))
        deltas = list(set(r.get("delta") for r in runs if r.get("delta") is not None))
        
        agg = {
            "alpha": alpha,
            "sigma": sigma,
            "C": C,
            "delta": deltas[0] if len(deltas) == 1 else deltas,
            "seeds": seeds,
            "num_runs": len(runs),
            "final_test_accuracy_mean": float(np.mean(final_accs)),
            "final_test_accuracy_std": float(np.std(final_accs)),
            "best_test_accuracy_mean": float(np.mean(best_accs)),
            "best_test_accuracy_std": float(np.std(best_accs)),
            "epsilon": epsilon_val,
            "best_alpha": best_alpha_val,
            "number_of_communication_rounds": rounds_list[0] if len(rounds_list) == 1 else rounds_list,
            "dataset_mode": dataset_modes[0] if len(dataset_modes) == 1 else dataset_modes
        }
        
        aggregate_results.append(agg)
        
    out_file = os.path.join(base_dir, "aggregate.json")
    with open(out_file, "w") as f:
        json.dump(aggregate_results, f, indent=4)
        
    print(f"Aggregation complete. Processed {len(all_runs)} runs. Results saved to {out_file}")

if __name__ == "__main__":
    main()
