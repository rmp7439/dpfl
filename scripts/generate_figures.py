import argparse
import os
import sys
import json
import csv
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# PUBLICATION-QUALITY PLOTTING CONFIGURATION
# ------------------------------------------
# Explicit non-default style configuration avoiding standard matplotlib style.
# Ensures LaTeX compatibility, consistent typography, colors, and line widths.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "axes.prop_cycle": cycler(color=["#E63946", "#457B9D", "#1D3557", "#2A9D8F", "#F4A261", "#E76F51", "#264653"]),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

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
            
            plt.title(f"Client Label Distribution under Dirichlet α=0.1{title_suffix}")
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
            
            # Load centralized baseline for overlay
            central_accs = []
            central_loss = []
            baseline_path = "results/stage2/baseline_full_seed42.csv" if not is_subset else "results/stage2/baseline_subset_seed42.csv"
            if os.path.exists(baseline_path):
                with open(baseline_path) as f:
                    for row in csv.DictReader(f):
                        central_accs.append(float(row["test_acc"]))
                        central_loss.append(float(row["train_loss"]))
            
            # Figure 2a: Test Accuracy Convergence
            plt.figure(figsize=(8, 6))
            plt.plot(r_nums, accs, marker='o', label="DP-FL (Stage 4)")
            if central_accs:
                # We align centralized epochs to communication rounds for visual comparison
                # explicitly labeled to avoid confusion.
                epochs_per_round = len(central_accs) / len(r_nums)
                x_central = [i / epochs_per_round + 1 for i in range(len(central_accs))]
                plt.plot(x_central, central_accs, linestyle='--', color="#E63946", alpha=0.7, 
                         label="Centralized Baseline (Epochs scaled to Rounds)")
            plt.title(f"Stage 4 DP-FL Accuracy Convergence{title_suffix}")
            plt.xlabel("Communication Round")
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"figures/{fig_prefix}fig2a_stage4_accuracy_convergence.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig2a_stage4_accuracy_convergence.pdf", dpi=300)
            plt.close()
            
            # Figure 2b: Training Loss Convergence
            if "train_loss" in rounds[0]:
                dp_loss = [r["train_loss"] for r in rounds]
                plt.figure(figsize=(8, 6))
                plt.plot(r_nums, dp_loss, marker='o', label="DP-FL (Stage 4)")
                if central_loss:
                    plt.plot(x_central, central_loss, linestyle='--', color="#E63946", alpha=0.7, 
                             label="Centralized Baseline (Epochs scaled to Rounds)")
                plt.title(f"Stage 4 Training Loss Convergence{title_suffix}")
                plt.xlabel("Communication Round")
                plt.ylabel("Training Loss (Cross Entropy)")
                plt.grid(True)
                plt.legend()
                plt.savefig(f"figures/{fig_prefix}fig2b_stage4_loss_convergence.png", dpi=300)
                plt.savefig(f"figures/{fig_prefix}fig2b_stage4_loss_convergence.pdf", dpi=300)
                plt.close()
            
            plt.figure(figsize=(8, 6))
            plt.plot(r_nums, eps, marker='s', color='#2A9D8F', label=r"Privacy Loss ($\epsilon$)")
            plt.title(f"Stage 4 Privacy Accumulation{title_suffix}")
            plt.xlabel("Communication Round")
            plt.ylabel(r"Cumulative Epsilon ($\epsilon$)")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"figures/{fig_prefix}fig3_stage4_epsilon.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig3_stage4_epsilon.pdf", dpi=300)
            plt.close()

    print("Generating Figure 4: Stage 5 privacy-utility tradeoff")
    # FIGURE 4: Stage 5 tradeoff
    s5_runs = []
    s5_per_run_dir = f"results/stage5/full_15_rounds_fixed_20261009_191553/per_run"
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
            plt.title(f"Stage 5 Privacy-Utility Tradeoff{title_suffix}")
            plt.xlabel(r"Cumulative Epsilon ($\epsilon$)")
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"figures/{fig_prefix}fig4_stage5_tradeoff.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig4_stage5_tradeoff.pdf", dpi=300)
            plt.close()

    print("Generating Figure 5 & 6: Stage 6 ablation")
    # FIGURE 5 & 6: Stage 6 ablation
    if not is_subset:
        s6_file = "results/stage6/multiseed_fixed_20261009_211429/ablation_results.json"
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
            color = '#1D3557' if d['sigma'] == 1.0 else '#E63946'
            
            # Check if this is the new multiseed format or old subset format
            if 'final_test_accuracy_mean' in d:
                acc_mean = d['final_test_accuracy_mean']
                acc_std = d.get('final_test_accuracy_std', 0.0)
                eps_mean = d.get('epsilon_mean', d.get('epsilon', 0.0))
                eps_std = d.get('epsilon_std', 0.0)
                
                plt.errorbar(eps_mean, acc_mean, xerr=eps_std, yerr=acc_std,
                           label=f"α={d['alpha']}, σ={d['sigma']}", 
                           marker=marker, color=color, markersize=10, capsize=5, linestyle='None')
            else:
                seed_label = f" (seed {d['seed']})" if 'seed' in d else ""
                plt.scatter(d['epsilon'], d['final_test_accuracy'], 
                           label=f"α={d['alpha']}, σ={d['sigma']}{seed_label}", 
                           marker=marker, color=color, s=100)
                
        plt.title(f"Stage 6 Alpha/Sigma Ablation (15 Communication Rounds){title_suffix}")
        plt.xlabel(r"Cumulative Epsilon ($\epsilon$)")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"figures/{fig_prefix}fig5_stage6_ablation.png", dpi=300)
        plt.savefig(f"figures/{fig_prefix}fig5_stage6_ablation.pdf", dpi=300)
        plt.close()

        # FIGURE 5a: Paired Alpha Comparison (alpha=0.1 vs alpha=10.0 under matching sigma & seed)
        print("Generating Figure 5a: Paired Alpha Comparison under matching conditions")
        ms_agg_file = "results/stage6/multiseed/aggregate.json"
        ms_per_run_dir = "results/stage6/multiseed"
        import numpy as np
        if os.path.exists(ms_per_run_dir) and not is_subset:
            # Load individual run summaries for matching seed pairs
            runs_by_cfg = {}
            for d in os.listdir(ms_per_run_dir):
                sp = os.path.join(ms_per_run_dir, d, "summary.json")
                if os.path.exists(sp):
                    with open(sp) as f:
                        run_info = json.load(f)
                    k = (run_info["alpha"], run_info["sigma"], run_info["seed"])
                    runs_by_cfg[k] = run_info["final_test_accuracy"]
            
            if runs_by_cfg:
                plt.figure(figsize=(10, 6))
                seeds = [42, 43, 44]
                x_indices = np.arange(len(seeds))
                bar_width = 0.2
                
                # Plot for sigma=1.0 and sigma=2.0
                plt.bar(x_indices - 1.5*bar_width, [runs_by_cfg.get((0.1, 1.0, s), 0) for s in seeds], bar_width, label="α=0.1, σ=1.0 (Non-IID)", color="#457B9D")
                plt.bar(x_indices - 0.5*bar_width, [runs_by_cfg.get((10.0, 1.0, s), 0) for s in seeds], bar_width, label="α=10.0, σ=1.0 (Near-IID)", color="#1D3557")
                plt.bar(x_indices + 0.5*bar_width, [runs_by_cfg.get((0.1, 2.0, s), 0) for s in seeds], bar_width, label="α=0.1, σ=2.0 (Non-IID)", color="#F4A261")
                plt.bar(x_indices + 1.5*bar_width, [runs_by_cfg.get((10.0, 2.0, s), 0) for s in seeds], bar_width, label="α=10.0, σ=2.0 (Near-IID)", color="#E63946")
                
                plt.title("Stage 6 Paired Heterogeneity Comparison across Matching Seeds (3 Rounds)")
                plt.xlabel("Random Seed")
                plt.ylabel("Final Test Accuracy (%)")
                plt.xticks(x_indices, [f"Seed {s}" for s in seeds])
                plt.grid(axis='y', linestyle=':', alpha=0.6)
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(f"figures/{fig_prefix}fig5a_stage6_alpha_comparison.png", dpi=300)
                plt.savefig(f"figures/{fig_prefix}fig5a_stage6_alpha_comparison.pdf", dpi=300)
                plt.close()

        # FIGURE 5b: Privacy-Utility Summary across Sigma & Alpha Settings with 3-seed error bars
        print("Generating Figure 5b: Stage 6 Multi-Seed Summary (Mean +/- Std)")
        if os.path.exists(ms_agg_file) and not is_subset:
            with open(ms_agg_file) as f:
                ms_agg_data = json.load(f)
            plt.figure(figsize=(9, 6))
            for item in ms_agg_data:
                marker = 'o' if item['alpha'] == 0.1 else 's'
                color = '#1D3557' if item['sigma'] == 1.0 else '#E63946'
                acc_m = item['final_test_accuracy_mean']
                acc_s = item['final_test_accuracy_std']
                eps_m = item['epsilon_mean']
                eps_s = item.get('epsilon_std', 0.0)
                plt.errorbar(eps_m, acc_m, xerr=eps_s, yerr=acc_s,
                             label=f"3-round mean: α={item['alpha']}, σ={item['sigma']}",
                             marker=marker, color=color, markersize=9, capsize=5, linestyle='None')
            
            # Overlay 15-round reference runs
            for d in s6_data:
                marker = '^' if d['alpha'] == 0.1 else 'D'
                color = '#457B9D' if d['sigma'] == 1.0 else '#F4A261'
                plt.scatter(d['epsilon'], d['final_test_accuracy'],
                            label=f"15-round ref: α={d['alpha']}, σ={d['sigma']} (seed {d.get('seed', 42)})",
                            marker=marker, color=color, s=90, edgecolors='black', linewidth=1.2)
                            
            plt.title("Stage 6 Privacy-Utility Summary (3-Seed Mean ± Std vs 15-Round Reference)")
            plt.xlabel(r"Cumulative Epsilon ($\epsilon$)")
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"figures/{fig_prefix}fig5b_stage6_multiseed_summary.png", dpi=300)
            plt.savefig(f"figures/{fig_prefix}fig5b_stage6_multiseed_summary.pdf", dpi=300)
            plt.close()

        # FIGURE 6: Combined plot
        plt.figure(figsize=(10, 6))
        if s5_runs:
            for i, r in enumerate(s5_runs):
                eps = r.get("epsilon", r["_rounds"][-1]["global_epsilon"] if r["_rounds"] else 0)
                acc = r.get("final_test_accuracy", r["_rounds"][-1]["test_acc"] if r["_rounds"] else 0)
                plt.scatter(eps, acc, color='gray', alpha=0.5, label='Stage 5 Grid (15 rounds)' if i==0 else "")
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            color = '#1D3557' if d['sigma'] == 1.0 else '#E63946'
            seed_label = f" (seed {d['seed']})" if 'seed' in d else ""
            plt.scatter(d['epsilon'], d['final_test_accuracy'], 
                        label=f"Stage 6: α={d['alpha']}, σ={d['sigma']}{seed_label}", 
                        marker=marker, color=color, s=110, edgecolors='black', linewidth=1.0)
        plt.title(f"Combined Privacy-Utility Tradeoff across Stages 5 and 6{title_suffix}")
        plt.xlabel(r"Cumulative Epsilon ($\epsilon$)")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"figures/{fig_prefix}fig6_combined.png", dpi=300)
        plt.savefig(f"figures/{fig_prefix}fig6_combined.pdf", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
