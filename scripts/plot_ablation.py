
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
import json
import os
import matplotlib.pyplot as plt

def main():
    stage6_file = "results/stage6/ablation_results.json"
    with open(stage6_file, "r") as f:
        s6_data = json.load(f)
        
    s5_file = "results/stage5/grid_results.json"
    s5_data = []
    if os.path.exists(s5_file):
        with open(s5_file, "r") as f:
            s5_data = json.load(f)
            
    os.makedirs("results/stage6", exist_ok=True)
    
    # Plot 1: Accuracy vs Epsilon (Stage 6)
    plt.figure(figsize=(8,6))
    for d in s6_data:
        label = f"alpha={d['alpha']}, sigma={d['sigma']}"
        marker = 'o' if d['alpha'] == 0.1 else 's'
        color = 'blue' if d['sigma'] == 1.0 else 'red'
        plt.scatter(d['epsilon'], d['final_test_accuracy'], label=label, marker=marker, color=color, s=100)
        plt.annotate(f"  σ={d['sigma']}, α={d['alpha']}", (d['epsilon'], d['final_test_accuracy']))
    plt.title("Stage 6: Final Accuracy vs Epsilon")
    plt.xlabel("Epsilon (Privacy Loss)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/accuracy_vs_epsilon.png', dpi=300)
    plt.savefig('figures/accuracy_vs_epsilon.pdf', dpi=300)
    plt.close()
    
    # Plot 2: Convergence (Stage 6)
    plt.figure(figsize=(8,6))
    for d in s6_data:
        rounds = [1, 2, 3]
        accs = d['round_accuracy']
        label = f"alpha={d['alpha']}, sigma={d['sigma']}"
        marker = 'o' if d['alpha'] == 0.1 else 's'
        plt.plot(rounds, accs, marker=marker, label=label)
    
    # Add Stage 2 baseline
    plt.axhline(y=74.87, color='black', linestyle='--', label='Centralized Baseline (Stage 2)')
    
    plt.title("Stage 6: Convergence (Accuracy vs Round)")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks([1,2,3])
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/convergence.png', dpi=300)
    plt.savefig('figures/convergence.pdf', dpi=300)
    plt.close()
    
    # Plot 3: Stage 5 + Stage 6
    if s5_data:
        plt.figure(figsize=(10,6))
        for d in s5_data:
            if 'final_test_accuracy' in d and 'epsilon' in d:
                plt.scatter(d['epsilon'], d['final_test_accuracy'], color='gray', alpha=0.5, label='Stage 5' if d==s5_data[0] else "")
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            plt.scatter(d['epsilon'], d['final_test_accuracy'], marker=marker, label=f"S6: alpha={d['alpha']}, sigma={d['sigma']}", s=100)
        plt.title("Stage 5 & Stage 6: Accuracy vs Epsilon")
        plt.xlabel("Epsilon (Privacy Loss)")
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True)
        
        # Remove duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.savefig('figures/combined_stage5_stage6.png', dpi=300)
        plt.savefig('figures/combined_stage5_stage6.pdf', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
