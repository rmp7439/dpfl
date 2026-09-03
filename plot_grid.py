import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def parse_results(log_file):
    results = []
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return results
        
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("RESULT_CSV:"):
                parts = line.strip().split("RESULT_CSV:")[1].split(",")
                if len(parts) == 6:
                    sigma, c, delta, alpha, epsilon, acc = map(float, parts)
                    results.append((sigma, c, epsilon, acc))
    return results

def plot_results(log_file="grid_search.log"):
    results = parse_results(log_file)
    if not results:
        print("No results to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Group by C
    c_vals = sorted(list(set([r[1] for r in results])))
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    for i, c in enumerate(c_vals):
        subset = [r for r in results if r[1] == c]
        subset.sort(key=lambda x: x[2]) # Sort by epsilon
        
        epsilons = [r[2] for r in subset]
        accuracies = [r[3] for r in subset]
        
        plt.plot(epsilons, accuracies, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'C={c}')
        
        # Annotate with sigma
        for r in subset:
            plt.annotate(f"σ={r[0]}", (r[2], r[3]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Cumulative Epsilon (ε)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Privacy Loss (RDP Accounting)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rdp_grid_search_results.png')
    print("Plot saved to rdp_grid_search_results.png")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "grid_search.log"
    plot_results(log_file)
