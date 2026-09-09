import sys
import os
import json
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

class TestGridArtifacts(unittest.TestCase):
    GRID_JSON = os.path.join(project_root, "results", "stage5", "grid_results.json")
    SIGMAS = [0.5, 1.0, 1.5, 2.0]
    CS = [0.1, 1.0]

    def _load_grid(self):
        if not os.path.exists(self.GRID_JSON):
            self.skipTest("grid_results.json not found — grid has not run yet")
        with open(self.GRID_JSON) as f:
            return json.load(f)

    def test_all_8_configurations_present(self):
        results = self._load_grid()
        self.assertEqual(len(results), 8, f"Expected 8 grid results, got {len(results)}")

    def test_all_configurations_successful(self):
        results = self._load_grid()
        for r in results:
            self.assertEqual(r.get("run_status"), "Success", f"Run failed for sigma={r.get('sigma')}, C={r.get('C')}")

    def test_epsilon_non_decreasing_within_run(self):
        base = os.path.join(project_root, "results", "stage5", "per_run")
        if not os.path.exists(base):
            self.skipTest("per_run directory not found")
        for s in self.SIGMAS:
            for c in self.CS:
                rounds_csv = os.path.join(base, f"sigma_{s}_C_{c}", "rounds.csv")
                if not os.path.exists(rounds_csv):
                    continue
                import csv
                with open(rounds_csv) as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue
                epsilons = [float(r["global_epsilon"]) for r in rows]
                for i in range(1, len(epsilons)):
                    self.assertGreaterEqual(epsilons[i], epsilons[i-1], f"Epsilon decreased at round {i+1} for sigma={s}, C={c}")

    def test_higher_sigma_lower_final_epsilon(self):
        results = self._load_grid()
        eps_map = {(r["sigma"], r["C"]): r["epsilon"] for r in results}
        for c in self.CS:
            for i in range(len(self.SIGMAS) - 1):
                s_low = self.SIGMAS[i]
                s_high = self.SIGMAS[i + 1]
                if (s_low, c) in eps_map and (s_high, c) in eps_map:
                    self.assertGreaterEqual(eps_map[(s_low, c)], eps_map[(s_high, c)], f"sigma={s_high} should have <= epsilon vs sigma={s_low}, C={c}")

    def test_per_run_artifacts_exist(self):
        base = os.path.join(project_root, "results", "stage5", "per_run")
        if not os.path.exists(base):
            self.skipTest("per_run directory not found")
        for s in self.SIGMAS:
            for c in self.CS:
                run_dir = os.path.join(base, f"sigma_{s}_C_{c}")
                summary = os.path.join(run_dir, "summary.json")
                rounds = os.path.join(run_dir, "rounds.csv")
                self.assertTrue(os.path.exists(summary), f"Missing summary.json for sigma={s}, C={c}")
                self.assertTrue(os.path.exists(rounds), f"Missing rounds.csv for sigma={s}, C={c}")

    def test_grid_results_internally_consistent(self):
        results = self._load_grid()
        base = os.path.join(project_root, "results", "stage5", "per_run")
        for r in results:
            s, c = r["sigma"], r["C"]
            summary_path = os.path.join(base, f"sigma_{s}_C_{c}", "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            self.assertAlmostEqual(r["sigma"], summary["sigma"], places=4, msg=f"sigma mismatch in grid vs per-run for sigma={s}, C={c}")
            self.assertAlmostEqual(r["C"], summary["C"], places=4, msg=f"C mismatch in grid vs per-run for sigma={s}, C={c}")
            self.assertEqual(r.get("dataset_mode"), summary.get("dataset_mode"))

    def test_official_stage5_protocol(self):
        if not os.path.exists(self.GRID_JSON):
            self.skipTest("grid_results.json not found")
        results = self._load_grid()
        self.assertEqual(len(results), 8, "Expected exactly 8 official grid runs")
        for r in results:
            self.assertEqual(r.get("dataset_mode"), "full")
            self.assertEqual(r.get("number_of_training_samples"), 50000)
            self.assertEqual(r.get("number_of_test_samples"), 10000)
            self.assertEqual(r.get("number_of_clients"), 5)
            self.assertEqual(r.get("alpha"), 0.1)
            self.assertEqual(r.get("batch_size"), 64)
            self.assertEqual(r.get("local_epochs"), 1)
            self.assertEqual(r.get("optimizer"), "SGD")
            if "learning_rate" in r:
                self.assertEqual(r.get("learning_rate"), 0.05)
            self.assertEqual(r.get("number_of_communication_rounds"), 3)
            self.assertEqual(r.get("delta"), 1e-5)
            self.assertEqual(r.get("run_status"), "Success")

class TestStage6Archive(unittest.TestCase):
    def test_stage6_archive_exists_and_valid(self):
        json_path = os.path.join(project_root, "results", "stage6", "ablation_results.json")
        self.assertTrue(os.path.exists(json_path), "Archived JSON results should exist")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 4, "Should have exactly 4 archived configurations")
        alphas = set(d["alpha"] for d in data)
        sigmas = set(d["sigma"] for d in data)
        self.assertEqual(alphas, {0.1, 10.0})
        self.assertEqual(sigmas, {1.0, 2.0})
        for d in data:
            self.assertEqual(d["C"], 1.0)
            self.assertTrue(d.get("reconstructed_from_console", False), "Missing reconstructed_from_console flag")
            self.assertFalse(d.get("new_experiment_run", True), "Missing or true new_experiment_run flag")
            self.assertEqual(d.get("source_type"), "completed_colab_console_output")
            self.assertIn("final_test_accuracy", d)
            self.assertIn("epsilon", d)
            self.assertEqual(len(d.get("round_accuracy", [])), 3, "Should have 3 rounds of accuracy")

if __name__ == "__main__":
    unittest.main()
