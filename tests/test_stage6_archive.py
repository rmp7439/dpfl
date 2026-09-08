
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
import os
import json
import unittest

class TestStage6Archive(unittest.TestCase):
    def test_stage6_archive_exists_and_valid(self):
        json_path = os.path.join("results", "stage6", "final_results_from_console.json")
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
            self.assertTrue(d["reconstructed_from_console"])
            self.assertFalse(d["new_experiment_run"])
            self.assertEqual(d["source_type"], "completed_colab_console_output")
            self.assertIn("final_test_accuracy", d)
            self.assertIn("epsilon", d)
            self.assertEqual(len(d["round_accuracy"]), 3, "Should have 3 rounds of accuracy")

if __name__ == '__main__':
    unittest.main()
