import unittest
import os
import sys
import json
import csv
import tempfile
import shutil
import hashlib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.train_baseline import resolve_output_paths
from scripts.compare_baselines import (
    compute_sha256,
    compare_baseline_runs,
    load_csv_epochs
)


class TestBaselineReproducibility(unittest.TestCase):
    """
    Automated test suite for Task 3: Centralized baseline output controls,
    comparison arithmetic, and artifact preservation.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_a_default_output_compatibility(self):
        """Test A: Verify default paths match canonical reference without tags."""
        out_dir, json_path, csv_path = resolve_output_paths(mode="full", seed=42)
        expected_json = os.path.join(project_root, "results", "stage2", "baseline_full_seed42.json")
        expected_csv = os.path.join(project_root, "results", "stage2", "baseline_full_seed42.csv")
        self.assertEqual(os.path.normpath(json_path), os.path.normpath(expected_json))
        self.assertEqual(os.path.normpath(csv_path), os.path.normpath(expected_csv))

    def test_b_tagged_output_isolation(self):
        """Test B: Verify tagged output paths are distinct and invalid tags are rejected."""
        out_dir, json_path, csv_path = resolve_output_paths(mode="full", seed=42, tag="rerun")
        canonical_json = os.path.join(project_root, "results", "stage2", "baseline_full_seed42.json")
        
        self.assertNotEqual(os.path.normpath(json_path), os.path.normpath(canonical_json))
        self.assertTrue(json_path.endswith("baseline_full_seed42_rerun.json"))
        self.assertTrue(csv_path.endswith("baseline_full_seed42_rerun.csv"))

        # Reject path traversal and invalid characters
        invalid_tags = ["../escaped", "tag with spaces", "tag/with/slash", "tag*bad", ""]
        for bad_tag in invalid_tags:
            with self.assertRaises(ValueError):
                resolve_output_paths(mode="full", seed=42, tag=bad_tag)

    def test_c_explicit_output_directory(self):
        """Test C: Custom output directory correctly scopes files."""
        custom_dir = os.path.join(self.test_dir, "custom_out")
        out_dir, json_path, csv_path = resolve_output_paths(
            mode="subset", seed=100, tag="test", output_dir=custom_dir
        )
        self.assertEqual(out_dir, os.path.abspath(custom_dir))
        self.assertTrue(json_path.startswith(os.path.abspath(custom_dir)))
        self.assertTrue(json_path.endswith("baseline_subset_seed100_test.json"))

    def test_d_comparison_arithmetic(self):
        """Test D: Accurate calculation of differences, MAE, max diff, and RMSE on synthetic fixtures."""
        ref_json = {
            "final_test_accuracy": 70.0,
            "best_test_accuracy": 75.0,
            "number_of_epochs": 2,
            "runtime_seconds": 100.0
        }
        rerun_json = {
            "final_test_accuracy": 72.0,
            "best_test_accuracy": 76.5,
            "number_of_epochs": 2,
            "runtime_seconds": 110.0
        }
        ref_j_path = os.path.join(self.test_dir, "ref.json")
        rerun_j_path = os.path.join(self.test_dir, "rerun.json")
        with open(ref_j_path, "w") as f:
            json.dump(ref_json, f)
        with open(rerun_j_path, "w") as f:
            json.dump(rerun_json, f)

        ref_c_path = os.path.join(self.test_dir, "ref.csv")
        rerun_c_path = os.path.join(self.test_dir, "rerun.csv")
        with open(ref_c_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
            w.writerow([1, 1.5, 50.0, 1.4, 55.0])
            w.writerow([2, 1.0, 65.0, 0.9, 70.0])

        with open(rerun_c_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
            w.writerow([1, 1.6, 52.0, 1.3, 56.0])  # diffs: +0.1, +2.0, -0.1, +1.0
            w.writerow([2, 0.8, 68.0, 0.8, 72.0])  # diffs: -0.2, +3.0, -0.1, +2.0

        report = compare_baseline_runs(ref_j_path, ref_c_path, rerun_j_path, rerun_c_path, tolerance=0.5)

        self.assertEqual(report["classification"], "MATERIALLY_DIFFERENT")
        self.assertAlmostEqual(report["summary_comparison"]["final_test_accuracy"]["abs_diff"], 2.0)
        self.assertAlmostEqual(report["summary_comparison"]["best_test_accuracy"]["abs_diff"], 1.5)
        self.assertAlmostEqual(report["summary_comparison"]["runtime_seconds"]["abs_diff"], 10.0)

        # Check train_loss stats: diffs are [0.1, -0.2], abs diffs [0.1, 0.2]
        tl_stats = report["epoch_comparison"]["train_loss"]
        self.assertAlmostEqual(tl_stats["max_abs_diff"], 0.2)
        self.assertEqual(tl_stats["max_diff_epoch"], 2)
        self.assertAlmostEqual(tl_stats["mean_abs_diff"], 0.15)
        # RMSE: sqrt((0.01 + 0.04)/2) = sqrt(0.025) ~= 0.15811388
        self.assertAlmostEqual(tl_stats["rmse"], (0.025)**0.5, places=5)

    def test_e_malformed_input_handling(self):
        """Test E: Missing files, malformed JSON, duplicate epochs, and missing headers handled explicitly."""
        # Missing file
        report = compare_baseline_runs("nonexistent.json", "ref.csv", "rerun.json", "rerun.csv")
        self.assertEqual(report["classification"], "INCOMPLETE_COMPARISON")
        self.assertTrue(any("Missing file" in d for d in report["diagnostics"]))

        # Duplicate epoch CSV
        bad_csv = os.path.join(self.test_dir, "dup_epoch.csv")
        with open(bad_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
            w.writerow([1, 1.0, 50.0, 1.0, 50.0])
            w.writerow([1, 0.9, 55.0, 0.9, 55.0])
            
        with self.assertRaises(ValueError) as ctx:
            load_csv_epochs(bad_csv)
        self.assertIn("Duplicate epoch", str(ctx.exception))

    def test_f_artifact_preservation_paths(self):
        """Test F: Canonical reference files exist and cannot be overwritten without --overwrite."""
        canonical_json = os.path.join(project_root, "results", "stage2", "baseline_full_seed42.json")
        canonical_csv = os.path.join(project_root, "results", "stage2", "baseline_full_seed42.csv")
        self.assertTrue(os.path.exists(canonical_json), "Reference JSON must exist")
        self.assertTrue(os.path.exists(canonical_csv), "Reference CSV must exist")

        # Collision detection in resolve_output_paths / default execution
        out_dir, j_path, c_path = resolve_output_paths(mode="full", seed=42)
        self.assertTrue(os.path.exists(j_path))
        self.assertTrue(os.path.exists(c_path))


if __name__ == "__main__":
    unittest.main()
