import sys
import os
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

from src.config import SUBSET_CONFIG, FULL_CONFIG, STAGE5_SIGMAS, STAGE5_C_VALUES, STAGE6_ALPHAS, STAGE6_SIGMAS, STAGE6_C

class TestExperimentConfiguration(unittest.TestCase):
    def test_subset_config(self):
        self.assertEqual(SUBSET_CONFIG["num_samples"], 1000)
        self.assertIn("batch_size", SUBSET_CONFIG)
        self.assertIn("fed_lr", SUBSET_CONFIG)
        self.assertIn("dp_lr", SUBSET_CONFIG)
        self.assertIn("noise_multiplier", SUBSET_CONFIG)
        self.assertIn("max_grad_norm", SUBSET_CONFIG)

    def test_full_config(self):
        self.assertEqual(FULL_CONFIG["num_samples"], 50000)
        self.assertIn("batch_size", FULL_CONFIG)
        self.assertIn("fed_lr", FULL_CONFIG)
        self.assertIn("dp_lr", FULL_CONFIG)
        self.assertIn("noise_multiplier", FULL_CONFIG)
        self.assertIn("max_grad_norm", FULL_CONFIG)
        
    def test_stage5_matrix_is_correct(self):
        self.assertEqual(STAGE5_SIGMAS, [0.5, 1.0, 1.5, 2.0])
        self.assertEqual(STAGE5_C_VALUES, [0.1, 1.0])
        
    def test_stage6_matrix_is_correct(self):
        self.assertEqual(STAGE6_ALPHAS, [0.1, 10.0])
        self.assertEqual(STAGE6_SIGMAS, [1.0, 2.0])
        self.assertEqual(STAGE6_C, 1.0)

if __name__ == "__main__":
    unittest.main()
