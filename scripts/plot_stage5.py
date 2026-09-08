
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
"""
Plot Stage 5: Accuracy vs Cumulative Epsilon.
Reads per-run rounds.csv files and generates:
  1. accuracy_vs_epsilon.png — main privacy-utility tradeoff plot
  2. epsilon_vs_round.png    — per-configuration epsilon growth per round

Only plots full-data runs (dataset_mode == "full") from summary.json.
If no full runs exist yet, falls back to whatever is available with a clear label.
"""
import os
import csv
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_run(run_dir):
    summary_path = os.path.join(run_dir, "summary.json")
    rounds_path = os.path.join(run_dir, "rounds.csv")
    if not os.path.exists(summary_path) or not os.path.exists(rounds_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    rounds = []
    with open(rounds_path) as f:
        for row in csv.DictReader(f):
            rounds.append({k: float(v) for k, v in row.items()})
    summary["_rounds"] = rounds
    return summary


def main():
    base_dir = os.path.join("results", "stage5", "per_run")
    out_dir = os.path.join("results", "stage5")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found. Run grid_search.py first.")
        return

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    runs = []
    for d in run_dirs:
        r = load_run(d)
        if r is not None:
            runs.append(r)

    if not runs:
        print("No runs with rounds.csv found.")
        return

    # Prefer full runs; fall back to whatever is present
    full_runs = [r for r in runs if r.get("dataset_mode") == "full"]
    plot_runs = full_runs if full_runs else runs
    mode_label = "Full CIFAR-10" if full_runs else "Subset (validation only)"
    print(f"Plotting {len(plot_runs)} runs ({mode_label})...")

    # --- Plot 1: accuracy vs epsilon ---
    fig, ax = plt.subplots(figsize=(13, 7))

    sigma_vals = sorted(set(r["sigma"] for r in plot_runs))
    c_vals = sorted(set(r["C"] for r in plot_runs))

    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--"]

    color_map = {s: COLORS[i % len(COLORS)] for i, s in enumerate(sigma_vals)}
    style_map = {c: linestyles[i % len(linestyles)] for i, c in enumerate(c_vals)}
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(sigma_vals)}

    for run in sorted(plot_runs, key=lambda r: (r["sigma"], r["C"])):
        s, c = run["sigma"], run["C"]
        rounds = run["_rounds"]
        if not rounds:
            continue
        epsilons = [rd["global_epsilon"] for rd in rounds]
        accs = [rd["test_acc"] for rd in rounds]
        label = f"σ={s}, C={c}"
        ax.plot(epsilons, accs,
                color=color_map[s],
                linestyle=style_map[c],
                marker=marker_map[s],
                markersize=7,
                linewidth=1.8,
                label=label)

    # Overlay centralized non-private baseline
    baseline_file = None
    for pattern in ["results/stage2/baseline_full_seed*.json",
                    "results/stage2/baseline_subset_seed*.json"]:
        files = glob.glob(pattern)
        if files:
            baseline_file = files[0]
            break

    if baseline_file:
        with open(baseline_file) as f:
            bd = json.load(f)
        baseline_acc = bd.get("best_test_accuracy", 0.0)
        baseline_mode = bd.get("dataset_mode", "unknown")
        if baseline_acc > 0:
            ax.axhline(y=baseline_acc, color="black", linestyle=":",
                       linewidth=1.5,
                       label=f"Centralized non-private ({baseline_mode}, {baseline_acc:.1f}%)")

    ax.set_title(
        f"Privacy–Utility Tradeoff\n"
        f"Accuracy vs Cumulative ε  |  {mode_label}  |  δ=1e-5, Dirichlet α=0.1",
        fontsize=13)
    ax.set_xlabel("Cumulative Privacy Loss ε  (δ=1e-5)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(out_dir, "accuracy_vs_epsilon.png")
    fig.savefig(out1, dpi=300)
    print(f"Saved: {out1}")
    plt.close(fig)

    # --- Plot 2: epsilon vs round ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for run in sorted(plot_runs, key=lambda r: (r["sigma"], r["C"])):
        s, c = run["sigma"], run["C"]
        rounds = run["_rounds"]
        if not rounds:
            continue
        round_nums = [int(rd["round"]) for rd in rounds]
        epsilons = [rd["global_epsilon"] for rd in rounds]
        ax2.plot(round_nums, epsilons,
                 color=color_map[s],
                 linestyle=style_map[c],
                 marker=marker_map[s],
                 label=f"σ={s}, C={c}")
    ax2.set_title(f"Cumulative ε per Communication Round\n{mode_label} | δ=1e-5", fontsize=12)
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("Cumulative ε", fontsize=11)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()
    out2 = os.path.join(out_dir, "epsilon_vs_round.png")
    fig2.savefig(out2, dpi=300)
    print(f"Saved: {out2}")
    plt.close(fig2)

    print("Done.")


if __name__ == "__main__":
    main()
