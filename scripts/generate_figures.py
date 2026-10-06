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
        # Just fallback to rounds.csv if the specific name isn't found
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
    os.makedirs("figures", exist_ok=True)
    print("Generating Figure 2 & 3: Stage 4 DP-FL convergence and privacy accumulation")
    
    # FIGURE 2 & 3: Stage 4
    stage4_dir = "results/stage4_longrun_fixed"
    s4 = load_run(os.path.join(stage4_dir, "summary.json"))
    if s4 and s4["_rounds"]:
        rounds = s4["_rounds"]
        r_nums = [r["round"] for r in rounds]
        accs = [r["test_acc"] for r in rounds]
        eps = [r["global_epsilon"] for r in rounds]
        
        plt.figure(figsize=(8, 6))
        plt.plot(r_nums, accs, marker='o', label="DP-FL (Stage 4)")
        plt.title("Figure 2: DP-FL Convergence (Accuracy vs Round)")
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.savefig("figures/fig2_stage4_convergence.png", dpi=300)
        plt.savefig("figures/fig2_stage4_convergence.pdf", dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(r_nums, eps, marker='s', color='orange', label="Privacy Loss")
        plt.title("Figure 3: Privacy Accumulation (Epsilon vs Round)")
        plt.xlabel("Communication Round")
        plt.ylabel("Cumulative Epsilon")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.savefig("figures/fig3_stage4_epsilon.png", dpi=300)
        plt.savefig("figures/fig3_stage4_epsilon.pdf", dpi=300)
        plt.close()

    print("Generating Figure 4: Stage 5 privacy-utility tradeoff")
    # FIGURE 4: Stage 5 tradeoff
    s5_runs = []
    for d in glob.glob("results/stage5/per_run/*"):
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
        plt.title("Figure 4: Stage 5 Privacy-Utility Tradeoff")
        plt.xlabel("Cumulative Epsilon")
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("figures/fig4_stage5_tradeoff.png", dpi=300)
        plt.savefig("figures/fig4_stage5_tradeoff.pdf", dpi=300)
        plt.close()

    print("Generating Figure 5 & 6: Stage 6 ablation")
    # FIGURE 5 & 6: Stage 6 ablation
    s6_file = "results/stage6/ablation_results_full.json"
    if not os.path.exists(s6_file):
        s6_file = "results/stage6/ablation_results_subset.json"
        
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
        plt.title("Figure 5: Stage 6 Alpha/Sigma Ablation")
        plt.xlabel("Epsilon")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.savefig("figures/fig5_stage6_ablation.png", dpi=300)
        plt.savefig("figures/fig5_stage6_ablation.pdf", dpi=300)
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
        plt.title("Figure 6: Combined Privacy-Utility Tradeoff")
        plt.xlabel("Epsilon")
        plt.ylabel("Final Accuracy (%)")
        plt.grid(True, linestyle=":", alpha=0.6)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig("figures/fig6_combined.png", dpi=300)
        plt.savefig("figures/fig6_combined.pdf", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
