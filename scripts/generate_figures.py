import argparse
import os
import sys
import json
import csv
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_run(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    
    rounds_path = summary_path.replace("summary.json", "rounds.csv").replace(".json", ".csv")
    if "federation" in rounds_path:
        rounds_path = rounds_path.replace("federation", "rounds").replace(".csv", ".csv")
        if not os.path.exists(rounds_path):
            rounds_path = os.path.join(os.path.dirname(summary_path), "rounds.csv")
    
    rounds = []
    if os.path.exists(rounds_path):
        with open(rounds_path) as f:
            for row in csv.DictReader(f):
                rounds.append({k: float(v) for k, v in row.items()})
    summary["_rounds"] = rounds
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--subset", action="store_true", help="Use subset artifacts (for validation)")
    parser.add_argument("--full", action="store_true", help="Use full artifacts (for publication)")
    args = parser.parse_args()
    
    if args.subset and args.full:
        print("ERROR: Cannot specify both --subset and --full")
        sys.exit(1)
        
    is_subset = args.subset
    if not is_subset and not args.full:
        print("ERROR: Must specify either --subset or --full.")
        sys.exit(1)
    
    if is_subset:
        print("WARNING: Generating VALIDATION figures using subset data. Not for publication.")
        fig_prefix = "val_"
        title_suffix = " (SUBSET VALIDATION)"
        mode_dir = "subset_validation"
    else:
        print("Generating PUBLICATION figures using full-data artifacts.")
        fig_prefix = ""
        title_suffix = ""
        mode_dir = "full"
        
    os.makedirs("figures", exist_ok=True)
    
    print("Generating Figure 1: Stage 1 Client Label Distribution")
    s1_mode = "full" if not is_subset else "subset"
    s1_file = f"results/stage1/split_validation_{s1_mode}.json"
    if os.path.exists(s1_file):
        with open(s1_file) as f:
            s1_data = json.load(f)
        counts = s1_data.get("class_counts_per_client", {})
        if counts:
            import numpy as np
            n_clients = len(counts)
            n_classes = len(counts["0"])
            x = np.arange(n_classes)
            width = 0.8 / n_clients
            
            plt.figure(figsize=(10, 6))
            for i in range(n_clients):
                plt.bar(x + i*width, counts[str(i)], width, label=f"Client {i}")
            
            plt.title(f"Figure 1: Client Label Distribution under Dirichlet α=0.1{title_suffix}")
            plt.xlabel("CIFAR-10 Class")
            plt.ylabel("Number of Samples")
            plt.xticks(x + width*(n_clients-1)/2, [str(j) for j in range(n_classes)])
            plt.legend()
            plt.grid(axis='y', linestyle=':', alpha=0.6)
            plt.savefig(f"figures/{fig_prefix}fig1_stage1_distribution.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig1_stage1_distribution.pdf", dpi=300)
            plt.close()
    else:
        print(f"WARNING: Stage 1 split validation file not found at {s1_file}, skipping Figure 1.")
        
    print("Generating Figure 2 & 3: Stage 4 DP-FL convergence and privacy accumulation")
    
    # FIGURE 2 & 3: Stage 4
    # Note: stage 4 only has a full run. We use it for plotting if it exists.
    stage4_dir = "results/stage4" if not is_subset else None
    if stage4_dir and not os.path.exists(os.path.join(stage4_dir, "summary.json")):
        print(f"ERROR: Stage 4 full run not found in {stage4_dir}")
        sys.exit(1)
        
    if stage4_dir:
        s4 = load_run(os.path.join(stage4_dir, "summary.json"))
        if s4 and s4["_rounds"]:
            rounds = s4["_rounds"]
            r_nums = [r["round"] for r in rounds]
            accs = [r["test_acc"] for r in rounds]
            eps = [r["global_epsilon"] for r in rounds]
            
            plt.figure(figsize=(8, 6))
            plt.plot(r_nums, accs, marker='o', label="DP-FL (Stage 4)")
            plt.title(f"Figure 2: DP-FL Convergence (Accuracy vs Round){title_suffix}")
            plt.xlabel("Communication Round")
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            plt.savefig(f"figures/{fig_prefix}fig2_stage4_convergence.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig2_stage4_convergence.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(8, 6))
            plt.plot(r_nums, eps, marker='s', color='orange', label="Privacy Loss")
            plt.title(f"Figure 3: Privacy Accumulation (Epsilon vs Round){title_suffix}")
            plt.xlabel("Communication Round")
            plt.ylabel("Cumulative Epsilon")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            plt.savefig(f"figures/{fig_prefix}fig3_stage4_epsilon.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig3_stage4_epsilon.pdf", dpi=300)
            plt.close()

    print("Generating Figure 4: Stage 5 privacy-utility tradeoff")
    # FIGURE 4: Stage 5 tradeoff
    s5_runs = []
    s5_per_run_dir = f"results/stage5/{mode_dir}/per_run"
    if not is_subset and not os.path.exists(s5_per_run_dir):
        print(f"ERROR: Required full artifact {s5_per_run_dir} not found. Failing clearly.")
        sys.exit(1)
        
    if os.path.exists(s5_per_run_dir):
        for d in glob.glob(os.path.join(s5_per_run_dir, "*")):
            r = load_run(os.path.join(d, "summary.json"))
            if r:
                s5_runs.append(r)
        
        if s5_runs:
            plt.figure(figsize=(10, 6))
            for r in s5_runs:
                s = r.get("sigma", 0)
                c = r.get("C", 0)
                if r["_rounds"]:
                    eps = [rd["global_epsilon"] for rd in r["_rounds"]]
                    accs = [rd["test_acc"] for rd in r["_rounds"]]
                    plt.plot(eps, accs, marker='o', label=f"σ={s}, C={c}")
            plt.title(f"Figure 4: Stage 5 Privacy-Utility Tradeoff{title_suffix}")
            plt.xlabel("Cumulative Epsilon")
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"figures/{fig_prefix}fig4_stage5_tradeoff.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig4_stage5_tradeoff.pdf", dpi=300)
            plt.close()

    print("Generating Figure 5 & 6: Stage 6 ablation")
    # FIGURE 5 & 6: Stage 6 ablation
    if not is_subset:
        s6_file = "results/stage6/historical_reconstructed/ablation_results.json"
    else:
        s6_file = "results/stage6/subset_validation/ablation_results.json"
        
    if not os.path.exists(s6_file):
        print(f"ERROR: Required artifact {s6_file} not found. Failing clearly.")
        sys.exit(1)
        
    if os.path.exists(s6_file):
        with open(s6_file) as f:
            s6_data = json.load(f)
            
        plt.figure(figsize=(8, 6))
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            color = 'blue' if d['sigma'] == 1.0 else 'red'
            plt.scatter(d['epsilon'], d['final_test_accuracy'], 
                       label=f"α={d['alpha']}, σ={d['sigma']}", 
                       marker=marker, color=color, s=100)
        plt.title(f"Figure 5: Stage 6 Alpha/Sigma Ablation{title_suffix}")
        plt.xlabel("Epsilon")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.savefig(f"figures/{fig_prefix}fig5_stage6_ablation.png", dpi=300)
        plt.savefig(f"figures/{fig_prefix}fig5_stage6_ablation.pdf", dpi=300)
        plt.close()
        
        # Combined plot
        plt.figure(figsize=(10, 6))
        if s5_runs:
            for i, r in enumerate(s5_runs):
                eps = r.get("epsilon", r["_rounds"][-1]["global_epsilon"] if r["_rounds"] else 0)
                acc = r.get("final_test_accuracy", r["_rounds"][-1]["test_acc"] if r["_rounds"] else 0)
                plt.scatter(eps, acc, color='gray', alpha=0.5, label='Stage 5' if i==0 else "")
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            color = 'blue' if d['sigma'] == 1.0 else 'red'
            plt.scatter(d['epsilon'], d['final_test_accuracy'], 
                       label=f"S6: α={d['alpha']}, σ={d['sigma']}", 
                       marker=marker, color=color, s=100)
        plt.title(f"Figure 6: Combined Privacy-Utility Tradeoff{title_suffix}")
        plt.xlabel("Epsilon")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f"figures/{fig_prefix}fig6_combined.png", dpi=300)
        plt.savefig(f"figures/{fig_prefix}fig6_combined.pdf", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
