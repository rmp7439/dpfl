#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Pipeline (Task 4)
Exports vector PGF (LaTeX native), standalone PDF, and 300+ DPI PNG for all DP-FL figures.
Supports full and subset modes, Dirichlet alpha comparison, and atomic output staging.
"""

import argparse
import os
import sys
import json
import csv
import glob
import shutil
import tempfile
from typing import Dict, Any, Tuple, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PUBLICATION-QUALITY PLOTTING CONFIGURATION (LaTeX/PGF compatible)
# -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
    "axes.prop_cycle": cycler(color=["#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#E63946", "#F4A261", "#264653"]),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{pgf}",
    ])
})


class FallbackLatexManager:
    """Provides estimated font metrics when pdflatex is not installed in the OS."""
    def get_width_height_descent(self, text, prop):
        size = prop.get_size_in_points()
        return len(text) * size * 0.5, size, size * 0.2


def check_latex_toolchain() -> Tuple[bool, str]:
    """Checks if pdflatex is available in system PATH."""
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin:
        try:
            import subprocess
            res = subprocess.run([pdflatex_bin, "--version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "pdflatex found"
            return True, first_line
        except Exception:
            return True, f"pdflatex at {pdflatex_bin}"
    return False, "pdflatex not found in PATH"


def validate_png_dpi(png_path: str, min_dpi: int = 300) -> Tuple[bool, Tuple[float, float]]:
    """Validates that a PNG file exists, is non-empty, and has DPI >= min_dpi."""
    if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
        return False, (0.0, 0.0)
    with Image.open(png_path) as im:
        dpi = im.info.get("dpi", (72.0, 72.0))
        # Note: rounding handles float precision differences (e.g. 299.9994)
        is_valid = (round(dpi[0]) >= min_dpi and round(dpi[1]) >= min_dpi)
        return is_valid, dpi


def save_publication_figure(
    fig: plt.Figure,
    base_output_path: str,
    dpi: int = 300,
    staging_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Saves a figure to .pdf, .png (at 300+ DPI), and .pgf formats.
    If staging_dir is provided, writes to staging first, validates, and then replaces destination.
    """
    out_dir = os.path.dirname(base_output_path)
    base_name = os.path.basename(base_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dest_pdf = f"{base_output_path}.pdf"
    dest_png = f"{base_output_path}.png"
    dest_pgf = f"{base_output_path}.pgf"

    if staging_dir:
        os.makedirs(staging_dir, exist_ok=True)
        target_pdf = os.path.join(staging_dir, f"{base_name}.pdf")
        target_png = os.path.join(staging_dir, f"{base_name}.png")
        target_pgf = os.path.join(staging_dir, f"{base_name}.pgf")
    else:
        target_pdf = dest_pdf
        target_png = dest_png
        target_pgf = dest_pgf

    # 1. Vector PDF export
    fig.savefig(target_pdf, format="pdf", dpi=dpi, bbox_inches="tight")

    # 2. High-resolution raster PNG export
    fig.savefig(target_png, format="png", dpi=dpi, bbox_inches="tight")

    # 3. Native LaTeX PGF export
    from matplotlib.backends import backend_pgf
    has_latex, _ = check_latex_toolchain()
    if has_latex:
        fig.savefig(target_pgf, format="pgf", bbox_inches="tight")
    else:
        # Fallback font metric estimator enables clean PGF export without requiring local TeX engine
        orig_get = backend_pgf.LatexManager._get_cached_or_new
        try:
            backend_pgf.LatexManager._get_cached_or_new = lambda: FallbackLatexManager()
            fig.savefig(target_pgf, format="pgf", bbox_inches="tight")
        finally:
            backend_pgf.LatexManager._get_cached_or_new = orig_get

    # Validation
    for p, fmt in [(target_pdf, "PDF"), (target_png, "PNG"), (target_pgf, "PGF")]:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise RuntimeError(f"Generated {fmt} file is missing or empty: {p}")

    valid_dpi, actual_dpi = validate_png_dpi(target_png, min_dpi=max(dpi, 300))
    if not valid_dpi:
        raise ValueError(f"Generated PNG {target_png} has DPI {actual_dpi}, expected at least {max(dpi, 300)}")

    # If staging was used, atomically replace final files
    if staging_dir:
        shutil.copy2(target_pdf, dest_pdf)
        shutil.copy2(target_png, dest_png)
        shutil.copy2(target_pgf, dest_pgf)

    return {"pdf": dest_pdf, "png": dest_png, "pgf": dest_pgf}


def load_run(summary_path: str) -> Optional[Dict[str, Any]]:
    """Loads a summary JSON and its companion rounds CSV."""
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    rounds_path = summary_path.replace("summary.json", "rounds.csv").replace(".json", ".csv")
    if "federation" in rounds_path:
        rounds_path = rounds_path.replace("federation", "rounds").replace(".csv", ".csv")
        if not os.path.exists(rounds_path):
            rounds_path = os.path.join(os.path.dirname(summary_path), "rounds.csv")
    
    rounds = []
    if os.path.exists(rounds_path):
        with open(rounds_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rounds.append({k: float(v) for k, v in row.items()})
    summary["_rounds"] = rounds
    return summary


def load_dirichlet_split_validation(file_path: str) -> Dict[str, Any]:
    """Loads and validates a Dirichlet split validation JSON artifact."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split validation artifact not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["alpha", "class_counts_per_client"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"Split validation artifact {file_path} missing required key '{k}'")

    counts = data["class_counts_per_client"]
    if not counts or not isinstance(counts, dict):
        raise ValueError(f"Artifact {file_path} contains empty or malformed class_counts_per_client")

    return data


def generate_dirichlet_alpha_comparison(
    alpha01_json: str,
    alpha10_json: str,
    base_output_path: str,
    title_suffix: str = "",
    staging_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Generates a publication-quality side-by-side Dirichlet heterogeneity comparison figure
    between alpha=0.1 (non-IID) and alpha=10.0 (near-IID) using shared y-axes and consistent colors.
    """
    data01 = load_dirichlet_split_validation(alpha01_json)
    data10 = load_dirichlet_split_validation(alpha10_json)

    alpha_val1 = data01.get("alpha", 0.1)
    alpha_val2 = data10.get("alpha", 10.0)

    counts1 = data01["class_counts_per_client"]
    counts2 = data10["class_counts_per_client"]

    import numpy as np
    cids1 = sorted(counts1.keys(), key=lambda x: int(x))
    cids2 = sorted(counts2.keys(), key=lambda x: int(x))

    if cids1 != cids2:
        raise ValueError(f"Client IDs differ between artifacts: {cids1} vs {cids2}")

    n_clients = len(cids1)
    n_classes1 = len(counts1[cids1[0]])
    n_classes2 = len(counts2[cids2[0]])
    if n_classes1 != n_classes2:
        raise ValueError(f"Class count mismatch: {n_classes1} vs {n_classes2}")

    x = np.arange(n_classes1)
    bar_width = 0.8 / n_clients
    colors = ["#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#E63946"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)

    # Panel 1: Alpha = 0.1
    for idx, cid in enumerate(cids1):
        color = colors[idx % len(colors)]
        ax1.bar(x + idx * bar_width, counts1[cid], bar_width, label=f"Client {cid}", color=color)

    ax1.set_title(rf"(a) Non-IID Partition ($\alpha = {alpha_val1}$){title_suffix}")
    ax1.set_xlabel("CIFAR-10 Class")
    ax1.set_ylabel("Number of Samples")
    ax1.set_xticks(x + bar_width * (n_clients - 1) / 2)
    ax1.set_xticklabels([str(j) for j in range(n_classes1)])
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Alpha = 10.0
    for idx, cid in enumerate(cids2):
        color = colors[idx % len(colors)]
        ax2.bar(x + idx * bar_width, counts2[cid], bar_width, label=f"Client {cid}", color=color)

    ax2.set_title(rf"(b) Near-IID Partition ($\alpha = {alpha_val2}$){title_suffix}")
    ax2.set_xlabel("CIFAR-10 Class")
    ax2.set_xticks(x + bar_width * (n_clients - 1) / 2)
    ax2.set_xticklabels([str(j) for j in range(n_classes2)])
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Unified Legend on right panel
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    paths = save_publication_figure(fig, base_output_path, staging_dir=staging_dir)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures (LaTeX/PGF, PDF, PNG)")
    parser.add_argument("--subset", action="store_true", help="Use subset artifacts (for validation)")
    parser.add_argument("--full", action="store_true", help="Use full artifacts (for publication)")
    parser.add_argument("--output-dir", default="figures", help="Target directory for output figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster PNG outputs")
    args = parser.parse_args()

    if args.subset and args.full:
        print("ERROR: Cannot specify both --subset and --full")
        sys.exit(1)

    is_subset = args.subset
    if not is_subset and not args.full:
        print("ERROR: Must specify either --subset or --full.")
        sys.exit(1)

    has_latex, latex_info = check_latex_toolchain()
    if has_latex:
        print(f"LaTeX engine verified: {latex_info}")
    else:
        print(f"NOTICE: pdflatex not found in PATH ({latex_info}).")
        print("  PGF figures will be generated using font metric estimation.")
        print("  To install LaTeX on Windows: 'winget install MiKTeX.MiKTeX' or 'choco install miktex'.")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Setup isolated staging directory to protect existing figures against partial failures
    staging_dir = tempfile.mkdtemp(prefix="fig_staging_")

    try:
        if is_subset:
            print("WARNING: Generating VALIDATION figures using subset data. Not for publication.")
            fig_prefix = "val_"
            title_suffix = " (SUBSET VALIDATION)"
        else:
            print("Generating PUBLICATION figures using full-data artifacts.")
            fig_prefix = ""
            title_suffix = ""

        # -------------------------------------------------------------
        # FIGURE 1: Stage 1 Client Label Distribution
        # -------------------------------------------------------------
        print("Generating Figure 1: Stage 1 Client Label Distribution")
        s1_mode = "full" if not is_subset else "subset"
        s1_file = f"results/stage1/split_validation_{s1_mode}.json"
        if os.path.exists(s1_file):
            s1_data = load_dirichlet_split_validation(s1_file)
            counts = s1_data.get("class_counts_per_client", {})
            if counts:
                import numpy as np
                cids = sorted(counts.keys(), key=lambda x: int(x))
                n_clients = len(cids)
                n_classes = len(counts[cids[0]])
                x = np.arange(n_classes)
                width = 0.8 / n_clients
                colors = ["#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#E63946"]

                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                for i, cid in enumerate(cids):
                    ax.bar(x + i * width, counts[cid], width, label=f"Client {cid}", color=colors[i % len(colors)])

                alpha_val = s1_data.get("alpha", 0.1)
                ax.set_title(rf"Client Label Distribution under Dirichlet $\alpha={alpha_val}${title_suffix}")
                ax.set_xlabel("CIFAR-10 Class")
                ax.set_ylabel("Number of Samples")
                ax.set_xticks(x + width * (n_clients - 1) / 2)
                ax.set_xticklabels([str(j) for j in range(n_classes)])
                ax.legend()
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                plt.tight_layout()
                save_publication_figure(fig, os.path.join(out_dir, f"{fig_prefix}fig1_stage1_distribution"), dpi=args.dpi, staging_dir=staging_dir)
                plt.close(fig)
        else:
            print(f"WARNING: Stage 1 split validation file not found at {s1_file}, skipping Figure 1.")

        # -------------------------------------------------------------
        # FIGURE 1b: Dirichlet Heterogeneity Comparison (alpha=0.1 vs alpha=10.0)
        # -------------------------------------------------------------
        if not is_subset:
            print("Generating Figure 1b: Dirichlet Heterogeneity Comparison (alpha=0.1 vs alpha=10.0)")
            alpha01_path = "results/stage1/split_validation_full_alpha0.1.json"
            alpha10_path = "results/stage1/split_validation_full_alpha10.0.json"
            if not os.path.exists(alpha01_path):
                alpha01_path = "results/stage1/split_validation_full.json"
            if os.path.exists(alpha01_path) and os.path.exists(alpha10_path):
                # Save both canonical and fig1b aliases
                generate_dirichlet_alpha_comparison(
                    alpha01_path, alpha10_path,
                    os.path.join(out_dir, "dirichlet_alpha_comparison"),
                    title_suffix=title_suffix, staging_dir=staging_dir
                )
                generate_dirichlet_alpha_comparison(
                    alpha01_path, alpha10_path,
                    os.path.join(out_dir, f"{fig_prefix}fig1b_dirichlet_alpha_comparison"),
                    title_suffix=title_suffix, staging_dir=staging_dir
                )
            else:
                print(f"WARNING: Missing split validation files for comparison: {alpha01_path} or {alpha10_path}")

        # -------------------------------------------------------------
        # FIGURES 2 & 3: Stage 4 DP-FL Convergence and Privacy Accumulation
        # -------------------------------------------------------------
        print("Generating Figures 2 & 3: Stage 4 DP-FL convergence and privacy accumulation")
        stage4_dir = "results/stage4" if not is_subset else None
        if stage4_dir and not os.path.exists(os.path.join(stage4_dir, "summary.json")):
            print(f"ERROR: Stage 4 full run not found in {stage4_dir}")
            sys.exit(1)

        if stage4_dir:
            s4 = load_run(os.path.join(stage4_dir, "summary.json"))
            if s4 and s4.get("_rounds"):
                rounds = s4["_rounds"]
                r_nums = [r["round"] for r in rounds]
                accs = [r["test_acc"] for r in rounds]
                eps = [r["global_epsilon"] for r in rounds]

                central_accs = []
                central_loss = []
                baseline_path = "results/stage2/baseline_full_seed42.csv" if not is_subset else "results/stage2/baseline_subset_seed42.csv"
                if os.path.exists(baseline_path):
                    with open(baseline_path, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            central_accs.append(float(row["test_acc"]))
                            central_loss.append(float(row["train_loss"]))

                # Figure 2a: Test Accuracy Convergence
                fig2a, ax2a = plt.subplots(figsize=(5.5, 3.8))
                ax2a.plot(r_nums, accs, marker='o', label=r"DP-FL (Stage 4, $\sigma=1.0$, $C=1.0$)")
                if central_accs:
                    epochs_per_round = len(central_accs) / len(r_nums)
                    x_central = [i / epochs_per_round + 1 for i in range(len(central_accs))]
                    ax2a.plot(x_central, central_accs, linestyle='--', color="#E63946", alpha=0.75,
                              label="Centralized Baseline (Epochs scaled)")
                ax2a.set_title(rf"Stage 4 DP-FL Accuracy Convergence{title_suffix}")
                ax2a.set_xlabel("Communication Round")
                ax2a.set_ylabel("Test Accuracy (%)")
                ax2a.grid(True)
                ax2a.legend()
                plt.tight_layout()
                save_publication_figure(fig2a, os.path.join(out_dir, f"{fig_prefix}fig2a_stage4_accuracy_convergence"), dpi=args.dpi, staging_dir=staging_dir)
                plt.close(fig2a)

                # Figure 2b: Training Loss Convergence
                if "train_loss" in rounds[0]:
                    dp_loss = [r["train_loss"] for r in rounds]
                    fig2b, ax2b = plt.subplots(figsize=(5.5, 3.8))
                    ax2b.plot(r_nums, dp_loss, marker='o', label=r"DP-FL (Stage 4, $\sigma=1.0$, $C=1.0$)")
                    if central_loss:
                        ax2b.plot(x_central, central_loss, linestyle='--', color="#E63946", alpha=0.75,
                                  label="Centralized Baseline (Epochs scaled)")
                    ax2b.set_title(rf"Stage 4 Training Loss Convergence{title_suffix}")
                    ax2b.set_xlabel("Communication Round")
                    ax2b.set_ylabel("Training Loss (Cross Entropy)")
                    ax2b.grid(True)
                    ax2b.legend()
                    plt.tight_layout()
                    save_publication_figure(fig2b, os.path.join(out_dir, f"{fig_prefix}fig2b_stage4_loss_convergence"), dpi=args.dpi, staging_dir=staging_dir)
                    plt.close(fig2b)

                # Figure 3: Privacy Loss Accumulation
                fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))
                ax3.plot(r_nums, eps, marker='s', color='#2A9D8F', label=r"Privacy Loss ($\epsilon$)")
                ax3.set_title(rf"Stage 4 Privacy Accumulation{title_suffix}")
                ax3.set_xlabel("Communication Round")
                ax3.set_ylabel(r"Cumulative Epsilon ($\epsilon$)")
                ax3.grid(True)
                ax3.legend()
                plt.tight_layout()
                save_publication_figure(fig3, os.path.join(out_dir, f"{fig_prefix}fig3_stage4_epsilon"), dpi=args.dpi, staging_dir=staging_dir)
                plt.close(fig3)

        # -------------------------------------------------------------
        # FIGURE 4: Stage 5 Privacy-Utility Tradeoff Grid
        # -------------------------------------------------------------
        print("Generating Figure 4: Stage 5 privacy-utility tradeoff")
        s5_runs = []
        s5_per_run_dir = "results/stage5/full_15_rounds_fixed_20261009_191553/per_run"
        if not is_subset and not os.path.exists(s5_per_run_dir):
            print(f"ERROR: Required full artifact {s5_per_run_dir} not found. Failing clearly.")
            sys.exit(1)

        if os.path.exists(s5_per_run_dir):
            for d in glob.glob(os.path.join(s5_per_run_dir, "*")):
                r = load_run(os.path.join(d, "summary.json"))
                if r:
                    s5_runs.append(r)

            if s5_runs:
                fig4, ax4 = plt.subplots(figsize=(6.5, 4.0))
                for r in s5_runs:
                    s = r.get("sigma", 0)
                    c = r.get("C", 0)
                    if r.get("_rounds"):
                        eps = [rd["global_epsilon"] for rd in r["_rounds"]]
                        accs = [rd["test_acc"] for rd in r["_rounds"]]
                        ax4.plot(eps, accs, marker='o', label=rf"$\sigma={s}$, $C={c}$")
                ax4.set_title(r"Stage 5 Privacy-Utility Tradeoff ($C \in \{0.1, 1.0\}$, $\sigma \in \{0.5, 1.0, 1.5, 2.0\}$)" + title_suffix)
                ax4.set_xlabel(r"Cumulative Epsilon ($\epsilon$)")
                ax4.set_ylabel("Test Accuracy (%)")
                ax4.grid(True)
                ax4.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
                plt.tight_layout()
                save_publication_figure(fig4, os.path.join(out_dir, f"{fig_prefix}fig4_stage5_tradeoff"), dpi=args.dpi, staging_dir=staging_dir)
                plt.close(fig4)

        # -------------------------------------------------------------
        # FIGURES 5, 5a, 5b, 6: Stage 6 Ablation and Combined Summaries
        # -------------------------------------------------------------
        print("Generating Figures 5 & 6: Stage 6 ablation")
        if not is_subset:
            s6_file = "results/stage6/multiseed_fixed_20261009_211429/ablation_results.json"
        else:
            s6_file = "results/stage6/subset_validation/ablation_results.json"

        if not os.path.exists(s6_file):
            print(f"ERROR: Required artifact {s6_file} not found. Failing clearly.")
            sys.exit(1)

        with open(s6_file, "r", encoding="utf-8") as f:
            s6_data = json.load(f)

        fig5, ax5 = plt.subplots(figsize=(5.5, 3.8))
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            color = '#1D3557' if d['sigma'] == 1.0 else '#E63946'

            if 'final_test_accuracy_mean' in d:
                acc_mean = d['final_test_accuracy_mean']
                acc_std = d.get('final_test_accuracy_std', 0.0)
                eps_mean = d.get('epsilon_mean', d.get('epsilon', 0.0))
                eps_std = d.get('epsilon_std', 0.0)

                ax5.errorbar(eps_mean, acc_mean, xerr=eps_std, yerr=acc_std,
                             label=rf"$\alpha={d['alpha']}$, $\sigma={d['sigma']}$",
                             marker=marker, color=color, markersize=8, capsize=4, linestyle='None')
            else:
                seed_label = f" (seed {d['seed']})" if 'seed' in d else ""
                ax5.scatter(d['epsilon'], d['final_test_accuracy'],
                            label=rf"$\alpha={d['alpha']}$, $\sigma={d['sigma']}${seed_label}",
                            marker=marker, color=color, s=85)

        ax5.set_title(rf"Stage 6 $\alpha$/$\sigma$ Ablation (15 Communication Rounds){title_suffix}")
        ax5.set_xlabel(r"Cumulative Epsilon ($\epsilon$)")
        ax5.set_ylabel("Final Accuracy (%)")
        ax5.grid(True)
        ax5.legend()
        plt.tight_layout()
        save_publication_figure(fig5, os.path.join(out_dir, f"{fig_prefix}fig5_stage6_ablation"), dpi=args.dpi, staging_dir=staging_dir)
        plt.close(fig5)

        # FIGURE 5a: Paired Alpha Comparison under matching conditions
        ms_agg_file = "results/stage6/multiseed/aggregate.json"
        ms_per_run_dir = "results/stage6/multiseed"
        import numpy as np
        if os.path.exists(ms_per_run_dir) and not is_subset:
            print("Generating Figure 5a: Paired Alpha Comparison under matching conditions")
            runs_by_cfg = {}
            for d in os.listdir(ms_per_run_dir):
                sp = os.path.join(ms_per_run_dir, d, "summary.json")
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as f:
                        run_info = json.load(f)
                    k = (run_info["alpha"], run_info["sigma"], run_info["seed"])
                    runs_by_cfg[k] = run_info["final_test_accuracy"]

            if runs_by_cfg:
                fig5a, ax5a = plt.subplots(figsize=(6.5, 4.0))
                seeds = [42, 43, 44]
                x_indices = np.arange(len(seeds))
                b_w = 0.18

                ax5a.bar(x_indices - 1.5 * b_w, [runs_by_cfg.get((0.1, 1.0, s), 0) for s in seeds], b_w, label=r"$\alpha=0.1$, $\sigma=1.0$ (Non-IID)", color="#457B9D")
                ax5a.bar(x_indices - 0.5 * b_w, [runs_by_cfg.get((10.0, 1.0, s), 0) for s in seeds], b_w, label=r"$\alpha=10.0$, $\sigma=1.0$ (Near-IID)", color="#1D3557")
                ax5a.bar(x_indices + 0.5 * b_w, [runs_by_cfg.get((0.1, 2.0, s), 0) for s in seeds], b_w, label=r"$\alpha=0.1$, $\sigma=2.0$ (Non-IID)", color="#F4A261")
                ax5a.bar(x_indices + 1.5 * b_w, [runs_by_cfg.get((10.0, 2.0, s), 0) for s in seeds], b_w, label=r"$\alpha=10.0$, $\sigma=2.0$ (Near-IID)", color="#E63946")

                ax5a.set_title("Stage 6 Paired Heterogeneity Comparison across Matching Seeds (3 Rounds)")
                ax5a.set_xlabel("Random Seed")
                ax5a.set_ylabel("Final Test Accuracy (%)")
                ax5a.set_xticks(x_indices)
                ax5a.set_xticklabels([f"Seed {s}" for s in seeds])
                ax5a.grid(axis='y', linestyle=':', alpha=0.6)
                ax5a.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
                plt.tight_layout()
                save_publication_figure(fig5a, os.path.join(out_dir, f"{fig_prefix}fig5a_stage6_alpha_comparison"), dpi=args.dpi, staging_dir=staging_dir)
                plt.close(fig5a)

        # FIGURE 5b: Stage 6 Multi-Seed Summary
        if os.path.exists(ms_agg_file) and not is_subset:
            print("Generating Figure 5b: Stage 6 Multi-Seed Summary (Mean +/- Std)")
            with open(ms_agg_file, "r", encoding="utf-8") as f:
                ms_agg_data = json.load(f)
            fig5b, ax5b = plt.subplots(figsize=(6.5, 4.0))
            for item in ms_agg_data:
                marker = 'o' if item['alpha'] == 0.1 else 's'
                color = '#1D3557' if item['sigma'] == 1.0 else '#E63946'
                acc_m = item['final_test_accuracy_mean']
                acc_s = item['final_test_accuracy_std']
                eps_m = item['epsilon_mean']
                eps_s = item.get('epsilon_std', 0.0)
                ax5b.errorbar(eps_m, acc_m, xerr=eps_s, yerr=acc_s,
                              label=rf"3-round mean: $\alpha={item['alpha']}$, $\sigma={item['sigma']}$",
                              marker=marker, color=color, markersize=8, capsize=4, linestyle='None')

            for d in s6_data:
                marker = '^' if d['alpha'] == 0.1 else 'D'
                color = '#457B9D' if d['sigma'] == 1.0 else '#F4A261'
                ax5b.scatter(d['epsilon'], d['final_test_accuracy'],
                             label=rf"15-round ref: $\alpha={d['alpha']}$, $\sigma={d['sigma']}$ (seed {d.get('seed', 42)})",
                             marker=marker, color=color, s=80, edgecolors='black', linewidth=1.0)

            ax5b.set_title(r"Stage 6 Privacy-Utility Summary (3-Seed Mean $\pm$ Std vs 15-Round Reference)")
            ax5b.set_xlabel(r"Cumulative Epsilon ($\epsilon$)")
            ax5b.set_ylabel("Test Accuracy (%)")
            ax5b.grid(True)
            ax5b.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
            plt.tight_layout()
            save_publication_figure(fig5b, os.path.join(out_dir, f"{fig_prefix}fig5b_stage6_multiseed_summary"), dpi=args.dpi, staging_dir=staging_dir)
            plt.close(fig5b)

        # FIGURE 6: Combined Plot
        print("Generating Figure 6: Combined Stage 5 and Stage 6 plot")
        fig6, ax6 = plt.subplots(figsize=(6.5, 4.0))
        if s5_runs:
            for i, r in enumerate(s5_runs):
                eps = r.get("epsilon", r["_rounds"][-1]["global_epsilon"] if r.get("_rounds") else 0)
                acc = r.get("final_test_accuracy", r["_rounds"][-1]["test_acc"] if r.get("_rounds") else 0)
                ax6.scatter(eps, acc, color='gray', alpha=0.5, label='Stage 5 Grid (15 rounds)' if i == 0 else "")
        for d in s6_data:
            marker = 'o' if d['alpha'] == 0.1 else 's'
            color = '#1D3557' if d['sigma'] == 1.0 else '#E63946'
            seed_label = f" (seed {d['seed']})" if 'seed' in d else ""
            ax6.scatter(d['epsilon'], d['final_test_accuracy'],
                        label=rf"Stage 6 (15 rounds): $\alpha={d['alpha']}$, $\sigma={d['sigma']}${seed_label}",
                        marker=marker, color=color, s=90, edgecolors='black', linewidth=0.9)
        ax6.set_title(rf"Combined Privacy-Utility Tradeoff across Stages 5 and 6{title_suffix}")
        ax6.set_xlabel(r"Cumulative Epsilon ($\epsilon$)")
        ax6.set_ylabel("Final Accuracy (%)")
        ax6.grid(True)
        handles, labels = ax6.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax6.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        save_publication_figure(fig6, os.path.join(out_dir, f"{fig_prefix}fig6_combined"), dpi=args.dpi, staging_dir=staging_dir)
        plt.close(fig6)

        print(f"All figures generated successfully in: {out_dir}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
