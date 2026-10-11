import unittest
import os
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TestReproducibilityIntegration(unittest.TestCase):
    """
    Test suite for end-to-end reproducibility of Flower DP-FL simulations (Task 2).
    Validates Test D (identical-seed integration) and Test E (different-seed integration).
    """

    def setUp(self):
        self.dir_a = os.path.join(project_root, "results", "reproducibility", "run_seed42_a")
        self.dir_b = os.path.join(project_root, "results", "reproducibility", "run_seed42_b")
        self.dir_diff = os.path.join(project_root, "results", "reproducibility", "run_seed43")

    def test_identical_seed_integration(self):
        """Test D: Two runs with same seed produce identical initial & final model hashes, weights, epsilon, steps, and metadata."""
        self.assertTrue(os.path.exists(os.path.join(self.dir_a, "summary.json")), "Run A summary missing")
        self.assertTrue(os.path.exists(os.path.join(self.dir_b, "summary.json")), "Run B summary missing")

        with open(os.path.join(self.dir_a, "summary.json"), "r") as fa:
            data_a = json.load(fa)
        with open(os.path.join(self.dir_b, "summary.json"), "r") as fb:
            data_b = json.load(fb)

        # 1. Compare Initial Global Parameter Hashes
        self.assertEqual(data_a["initial_model_hash"], data_b["initial_model_hash"],
                         "Initial model hashes differ between identical seed runs")

        # 2. Compare Final Global Parameter Hashes
        self.assertEqual(data_a["final_model_hash"], data_b["final_model_hash"],
                         "Final model hashes differ between identical seed runs")

        # 3. Compare Final Model Weights from Checkpoint Array-by-Array
        ckpt_a_path = os.path.join(self.dir_a, "checkpoints", "round_2.npz")
        ckpt_b_path = os.path.join(self.dir_b, "checkpoints", "round_2.npz")
        self.assertTrue(os.path.exists(ckpt_a_path))
        self.assertTrue(os.path.exists(ckpt_b_path))

        ckpt_a = np.load(ckpt_a_path)
        ckpt_b = np.load(ckpt_b_path)
        self.assertEqual(len(ckpt_a.files), len(ckpt_b.files))
        for key in ckpt_a.files:
            self.assertTrue(np.array_equal(ckpt_a[key], ckpt_b[key]),
                            f"Model weights for layer {key} differ between identical seed runs")

        # 4. Compare Epsilon & Steps
        self.assertAlmostEqual(data_a["epsilon"], data_b["epsilon"], places=6)
        self.assertEqual(data_a["total_dp_steps"], data_b["total_dp_steps"])
        self.assertEqual(data_a["best_alpha"], data_b["best_alpha"])

        # 5. Compare Per-client DP Step Counts and Sample Rates
        details_a = data_a["client_details"]
        details_b = data_b["client_details"]
        self.assertEqual(set(details_a.keys()), set(details_b.keys()))
        for cid in details_a:
            self.assertEqual(details_a[cid]["cumulative_steps"], details_b[cid]["cumulative_steps"])
            self.assertEqual(details_a[cid]["round_steps"], details_b[cid]["round_steps"])
            self.assertAlmostEqual(details_a[cid]["sample_rate"], details_b[cid]["sample_rate"], places=6)
            self.assertAlmostEqual(details_a[cid]["epsilon"], details_b[cid]["epsilon"], places=6)

        # 6. Compare Participation and Accuracy/Loss Histories
        self.assertEqual(data_a["per_round_test_accuracy"], data_b["per_round_test_accuracy"])
        self.assertEqual(data_a["per_round_train_loss"], data_b["per_round_train_loss"])
        self.assertEqual(data_a["per_round_participation"], data_b["per_round_participation"])

    def test_different_seed_integration(self):
        """Test E: Run with different seed produces different initial & final hashes and different weights."""
        self.assertTrue(os.path.exists(os.path.join(self.dir_a, "summary.json")), "Run A summary missing")
        self.assertTrue(os.path.exists(os.path.join(self.dir_diff, "summary.json")), "Run Diff summary missing")

        with open(os.path.join(self.dir_a, "summary.json"), "r") as fa:
            data_a = json.load(fa)
        with open(os.path.join(self.dir_diff, "summary.json"), "r") as fdiff:
            data_diff = json.load(fdiff)

        # Seeds differ
        self.assertNotEqual(data_a["seed"], data_diff["seed"])

        # Initial model hashes differ
        self.assertNotEqual(data_a["initial_model_hash"], data_diff["initial_model_hash"])

        # Final model hashes differ
        self.assertNotEqual(data_a["final_model_hash"], data_diff["final_model_hash"])

        # Final weights differ
        ckpt_a = np.load(os.path.join(self.dir_a, "checkpoints", "round_2.npz"))
        ckpt_diff = np.load(os.path.join(self.dir_diff, "checkpoints", "round_2.npz"))
        has_diff = any(not np.array_equal(ckpt_a[k], ckpt_diff[k]) for k in ckpt_a.files)
        self.assertTrue(has_diff, "Model weights should differ between runs with different seeds")


if __name__ == "__main__":
    unittest.main()
