#!/usr/bin/env python3
"""
Baseline Comparison Utility for Centralized CIFAR-10 Experiments (Task 3).
Compares historical reference baseline against fresh rerun artifacts.
Generates machine-readable JSON and human-readable text comparison reports.
"""

import os
import sys
import json
import csv
import math
import hashlib
import argparse
import datetime
from typing import Dict, Any, Tuple, List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_json(filepath: str) -> Dict[str, Any]:
    """Loads and returns parsed JSON from filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_epochs(filepath: str) -> Dict[int, Dict[str, float]]:
    """
    Loads epoch rows from CSV indexed by integer epoch.
    Validates presence of required columns and finite values.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    epochs_data: Dict[int, Dict[str, float]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV {filepath} has no headers")
        
        required_cols = {"epoch", "train_loss", "train_acc", "test_loss", "test_acc"}
        missing_cols = required_cols - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(f"CSV {filepath} missing required columns: {missing_cols}")
            
        for row_idx, row in enumerate(reader, start=1):
            epoch_str = row.get("epoch")
            if epoch_str is None or epoch_str.strip() == "":
                raise ValueError(f"Row {row_idx} in {filepath} has empty epoch")
            try:
                epoch_num = int(epoch_str)
            except ValueError:
                raise ValueError(f"Row {row_idx} in {filepath} has non-integer epoch: '{epoch_str}'")
                
            if epoch_num in epochs_data:
                raise ValueError(f"Duplicate epoch {epoch_num} detected in {filepath}")
                
            metrics: Dict[str, float] = {}
            for col in ["train_loss", "train_acc", "test_loss", "test_acc"]:
                val_str = row.get(col)
                if val_str is None or val_str.strip() == "":
                    raise ValueError(f"Epoch {epoch_num} missing value for '{col}' in {filepath}")
                try:
                    val = float(val_str)
                except ValueError:
                    raise ValueError(f"Epoch {epoch_num} non-numeric value for '{col}': '{val_str}' in {filepath}")
                if not math.isfinite(val):
                    raise ValueError(f"Epoch {epoch_num} non-finite value for '{col}': {val} in {filepath}")
                metrics[col] = val
                
            epochs_data[epoch_num] = metrics
            
    if not epochs_data:
        raise ValueError(f"CSV {filepath} contains no data rows")
        
    return epochs_data


def compare_baseline_runs(
    ref_json_path: str,
    ref_csv_path: str,
    rerun_json_path: str,
    rerun_csv_path: str,
    tolerance: float = 1e-4,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares reference and rerun baseline runs.
    Returns machine-readable comparison dictionary.
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tolerance": tolerance,
        "classification": "INCOMPLETE_COMPARISON",
        "artifacts": {
            "ref_json": {"path": ref_json_path, "sha256": None, "size_bytes": None},
            "ref_csv": {"path": ref_csv_path, "sha256": None, "size_bytes": None},
            "rerun_json": {"path": rerun_json_path, "sha256": None, "size_bytes": None},
            "rerun_csv": {"path": rerun_csv_path, "sha256": None, "size_bytes": None},
        },
        "summary_comparison": {},
        "epoch_comparison": {},
        "trajectory_match": False,
        "diagnostics": []
    }

    # Validate file existence & record checksums
    for key, path in [
        ("ref_json", ref_json_path),
        ("ref_csv", ref_csv_path),
        ("rerun_json", rerun_json_path),
        ("rerun_csv", rerun_csv_path)
    ]:
        if not os.path.exists(path):
            report["diagnostics"].append(f"Missing file: {path}")
            return report
        report["artifacts"][key]["sha256"] = compute_sha256(path)
        report["artifacts"][key]["size_bytes"] = os.path.getsize(path)

    # Load JSON files
    try:
        ref_json = load_json(ref_json_path)
        rerun_json = load_json(rerun_json_path)
    except Exception as e:
        report["diagnostics"].append(f"Failed to load JSON files: {str(e)}")
        return report

    # Load CSV files
    try:
        ref_epochs = load_csv_epochs(ref_csv_path)
        rerun_epochs = load_csv_epochs(rerun_csv_path)
    except Exception as e:
        report["diagnostics"].append(f"Failed to load CSV files: {str(e)}")
        return report

    # Compare Summary Metrics
    summary_metrics_def = [
        ("final_test_accuracy", "%", "Percentage of correctly classified test images at final epoch"),
        ("best_test_accuracy", "%", "Highest test accuracy observed across all epochs"),
        ("number_of_epochs", "epochs", "Total completed training epochs"),
        ("runtime_seconds", "seconds", "Total wall-clock training duration"),
    ]

    summary_diffs = {}
    for metric, units, description in summary_metrics_def:
        ref_val = ref_json.get(metric)
        rerun_val = rerun_json.get(metric)
        if ref_val is not None and rerun_val is not None:
            try:
                ref_num = float(ref_val)
                rerun_num = float(rerun_val)
                abs_diff = abs(rerun_num - ref_num)
                rel_diff = abs_diff / abs(ref_num) if abs(ref_num) > 1e-12 else 0.0
                summary_diffs[metric] = {
                    "ref": ref_num,
                    "rerun": rerun_num,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "exact_match": (ref_num == rerun_num),
                    "units": units,
                    "description": description
                }
            except (ValueError, TypeError):
                report["diagnostics"].append(f"Non-numeric values for summary metric '{metric}'")
        else:
            report["diagnostics"].append(f"Metric '{metric}' missing in reference or rerun JSON")

    # CSV Summary Additions (Final train loss, best test loss)
    ref_final_epoch = max(ref_epochs.keys())
    rerun_final_epoch = max(rerun_epochs.keys())
    summary_diffs["final_train_loss"] = {
        "ref": ref_epochs[ref_final_epoch]["train_loss"],
        "rerun": rerun_epochs[rerun_final_epoch]["train_loss"],
        "abs_diff": abs(rerun_epochs[rerun_final_epoch]["train_loss"] - ref_epochs[ref_final_epoch]["train_loss"]),
        "exact_match": (ref_epochs[ref_final_epoch]["train_loss"] == rerun_epochs[rerun_final_epoch]["train_loss"]),
        "units": "loss",
        "description": "Training loss at the final epoch"
    }

    ref_min_test_loss = min(v["test_loss"] for v in ref_epochs.values())
    rerun_min_test_loss = min(v["test_loss"] for v in rerun_epochs.values())
    summary_diffs["best_test_loss"] = {
        "ref": ref_min_test_loss,
        "rerun": rerun_min_test_loss,
        "abs_diff": abs(rerun_min_test_loss - ref_min_test_loss),
        "exact_match": (ref_min_test_loss == rerun_min_test_loss),
        "units": "loss",
        "description": "Minimum test loss across all epochs"
    }
    report["summary_comparison"] = summary_diffs

    # Compare Per-Epoch CSV Metrics
    ref_epoch_keys = sorted(ref_epochs.keys())
    rerun_epoch_keys = sorted(rerun_epochs.keys())
    if ref_epoch_keys != rerun_epoch_keys:
        report["diagnostics"].append(
            f"Epoch sequences differ: ref has {len(ref_epoch_keys)} epochs ({min(ref_epoch_keys)}..{max(ref_epoch_keys)}), "
            f"rerun has {len(rerun_epoch_keys)} epochs ({min(rerun_epoch_keys)}..{max(rerun_epoch_keys)})"
        )
        return report

    epoch_metrics = ["train_loss", "train_acc", "test_loss", "test_acc"]
    per_epoch_stats: Dict[str, Any] = {}
    all_exact_match = True
    max_overall_abs_diff = 0.0

    for m in epoch_metrics:
        diffs: List[float] = []
        abs_diffs: List[float] = []
        epoch_pairs: List[Dict[str, Any]] = []
        max_diff = -1.0
        max_diff_epoch = -1

        for ep in ref_epoch_keys:
            r_val = ref_epochs[ep][m]
            new_val = rerun_epochs[ep][m]
            diff = new_val - r_val
            abs_d = abs(diff)
            diffs.append(diff)
            abs_diffs.append(abs_d)
            if abs_d > max_diff:
                max_diff = abs_d
                max_diff_epoch = ep
            epoch_pairs.append({
                "epoch": ep,
                "ref": r_val,
                "rerun": new_val,
                "diff": diff,
                "abs_diff": abs_d
            })

        mae = sum(abs_diffs) / len(abs_diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        exact = all(d == 0.0 for d in abs_diffs)
        if not exact:
            all_exact_match = False
        if max_diff > max_overall_abs_diff:
            max_overall_abs_diff = max_diff

        per_epoch_stats[m] = {
            "max_abs_diff": max_diff,
            "max_diff_epoch": max_diff_epoch,
            "mean_abs_diff": mae,
            "rmse": rmse,
            "exact_match": exact,
            "per_epoch_details": epoch_pairs
        }

    report["epoch_comparison"] = per_epoch_stats
    report["trajectory_match"] = all_exact_match

    # Reproducibility Classification
    if all_exact_match:
        report["classification"] = "EXACT_MATCH"
    elif max_overall_abs_diff <= tolerance:
        report["classification"] = "NUMERICALLY_CLOSE"
    else:
        report["classification"] = "MATERIALLY_DIFFERENT"

    # Persist report if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_out = os.path.join(output_dir, f"baseline_full_seed42_vs_rerun_{ts_slug}.json")
        txt_out = os.path.join(output_dir, f"baseline_full_seed42_vs_rerun_{ts_slug}.txt")
        
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        txt_content = format_human_readable_report(report)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(txt_content)
            
        report["saved_json_report"] = json_out
        report["saved_txt_report"] = txt_out

    return report


def format_human_readable_report(report: Dict[str, Any]) -> str:
    """Formats report dictionary into clean console/text layout."""
    lines = []
    lines.append("=" * 78)
    lines.append("CENTRALIZED CIFAR-10 BASELINE REPRODUCIBILITY COMPARISON REPORT")
    lines.append("=" * 78)
    lines.append(f"Timestamp:       {report.get('timestamp')}")
    lines.append(f"Classification:  {report.get('classification')}")
    lines.append(f"Tolerance:       {report.get('tolerance')}")
    lines.append(f"Trajectory Match:{report.get('trajectory_match')}")
    lines.append("")
    
    lines.append("-" * 78)
    lines.append("INPUT ARTIFACTS AND CHECKSUMS")
    lines.append("-" * 78)
    artifacts = report.get("artifacts", {})
    for k, v in artifacts.items():
        lines.append(f"  {k:12s}: {v.get('path')}")
        lines.append(f"               SHA-256: {v.get('sha256')}")
        lines.append(f"               Size:    {v.get('size_bytes')} bytes")
    lines.append("")

    lines.append("-" * 78)
    lines.append("SUMMARY METRICS COMPARISON")
    lines.append("-" * 78)
    lines.append(f"{'Metric':<24s} {'Reference':<14s} {'Rerun':<14s} {'Abs Diff':<12s} {'Exact?':<8s} {'Units'}")
    lines.append("-" * 78)
    summary = report.get("summary_comparison", {})
    for m, d in summary.items():
        ref_str = f"{d['ref']:.4f}" if isinstance(d['ref'], float) else str(d['ref'])
        rerun_str = f"{d['rerun']:.4f}" if isinstance(d['rerun'], float) else str(d['rerun'])
        diff_str = f"{d['abs_diff']:.4f}" if isinstance(d['abs_diff'], float) else str(d['abs_diff'])
        exact_str = "YES" if d.get("exact_match") else "NO"
        lines.append(f"{m:<24s} {ref_str:<14s} {rerun_str:<14s} {diff_str:<12s} {exact_str:<8s} {d.get('units', '')}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PER-EPOCH TRAJECTORY COMPARISON (CSV)")
    lines.append("-" * 78)
    lines.append(f"{'Metric':<14s} {'Max Abs Diff':<14s} {'At Epoch':<10s} {'Mean Abs Diff':<16s} {'RMSE':<12s} {'Exact?'}")
    lines.append("-" * 78)
    epoch_comp = report.get("epoch_comparison", {})
    for m, stats in epoch_comp.items():
        max_d = f"{stats['max_abs_diff']:.6f}"
        max_ep = f"Epoch {stats['max_diff_epoch']}"
        mae = f"{stats['mean_abs_diff']:.6f}"
        rmse = f"{stats['rmse']:.6f}"
        exact = "YES" if stats["exact_match"] else "NO"
        lines.append(f"{m:<14s} {max_d:<14s} {max_ep:<10s} {mae:<16s} {rmse:<12s} {exact}")
    lines.append("")

    if report.get("diagnostics"):
        lines.append("-" * 78)
        lines.append("DIAGNOSTICS & WARNINGS")
        lines.append("-" * 78)
        for diag in report["diagnostics"]:
            lines.append(f"  * {diag}")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline Experiment Runs")
    parser.add_argument("--ref-json", default="results/stage2/baseline_full_seed42.json", help="Path to reference JSON")
    parser.add_argument("--ref-csv", default="results/stage2/baseline_full_seed42.csv", help="Path to reference CSV")
    parser.add_argument("--rerun-json", default="results/stage2/baseline_full_seed42_rerun.json", help="Path to rerun JSON")
    parser.add_argument("--rerun-csv", default="results/stage2/baseline_full_seed42_rerun.csv", help="Path to rerun CSV")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for numerical closeness")
    parser.add_argument("--output-dir", default="results/stage2/comparisons", help="Directory to save comparison reports")
    args = parser.parse_args()

    report = compare_baseline_runs(
        ref_json_path=args.ref_json,
        ref_csv_path=args.ref_csv,
        rerun_json_path=args.rerun_json,
        rerun_csv_path=args.rerun_csv,
        tolerance=args.tolerance,
        output_dir=args.output_dir
    )

    print(format_human_readable_report(report))
    if report.get("saved_json_report"):
        print(f"Machine-readable JSON report: {report['saved_json_report']}")
        print(f"Human-readable text report:   {report['saved_txt_report']}")


if __name__ == "__main__":
    main()
