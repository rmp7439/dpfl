import os
import csv
import glob
import matplotlib.pyplot as plt

def main():
    base_dir = "results/stage5/per_run"
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found. Run grid_search.py first.")
        return
        
    runs = glob.glob(os.path.join(base_dir, "*"))
    
    plt.figure(figsize=(12, 8))
    
    # Process DP runs
    for run_dir in sorted(runs):
        run_name = os.path.basename(run_dir)
        rounds_file = os.path.join(run_dir, "rounds.csv")
        
        if not os.path.exists(rounds_file):
            continue
            
        epsilons = []
        accuracies = []
        
        with open(rounds_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epsilons.append(float(row["epsilon"]))
                accuracies.append(float(row["test_acc"]))
                
        if epsilons and accuracies:
            plt.plot(epsilons, accuracies, marker='o', label=f"DP: {run_name}")
            
    # Include Baseline if available (Stage 2)
    # Stage 2 results do not have epsilon, they are purely non-private. 
    # We plot it as a horizontal line representing the accuracy ceiling.
    baseline_files = glob.glob("results/stage2/baseline_full_seed*.json")
    if not baseline_files:
        baseline_files = glob.glob("results/stage2/baseline_subset_seed*.json")
        
    if baseline_files:
        import json
        with open(baseline_files[0], "r") as f:
            baseline_data = json.load(f)
            baseline_acc = baseline_data.get("best_test_accuracy", 0.0)
            if baseline_acc > 0:
                plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f"Centralized Non-Private ({baseline_acc:.2f}%)")
                
    plt.title("Privacy-Utility Tradeoff: Test Accuracy vs Cumulative ε\n(Full CIFAR-10, δ=1e-5, Dirichlet α=0.1)")
    plt.xlabel("Cumulative Privacy Loss (ε)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = "results/stage5/accuracy_vs_epsilon.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
