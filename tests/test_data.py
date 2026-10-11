
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
"""
Test suite for DP-FL data operations.
Tests cover:
  - Dirichlet partition correctness
"""

import unittest
import math
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleCNN


# =========================================================
# 1. Dirichlet partition tests
# =========================================================

class TestDirichletSplit(unittest.TestCase):

    def _make_dataset(self, n=500):
        """Simple synthetic dataset with labels 0-9."""
        from torch.utils.data import TensorDataset
        x = torch.zeros(n, 3, 32, 32)
        y = torch.tensor([i % 10 for i in range(n)])
        return TensorDataset(x, y)

    def test_all_samples_assigned_exactly_once(self):
        """Every index appears in exactly one client — no missing, no duplicate."""
        from data import dirichlet_split
        import numpy as np

        n = 500
        x = torch.zeros(n, 3, 32, 32)
        y_raw = [i % 10 for i in range(n)]
        y = torch.tensor(y_raw)

        # Build a minimal Subset-like object
        class FakeDataset:
            def __init__(self, labels):
                self.targets = labels
                self.indices = list(range(len(labels)))
            def __len__(self):
                return len(self.targets)

        ds = FakeDataset(y_raw)
        num_clients = 3
        np.random.seed(42)
        client_indices, labels = dirichlet_split(ds, num_clients, alpha=0.1)

        all_idx = []
        for v in client_indices.values():
            all_idx.extend(v)

        self.assertEqual(len(all_idx), n, "Total assigned != dataset size")
        self.assertEqual(len(set(all_idx)), n, "Duplicates detected")

    def test_deterministic_with_same_seed(self):
        """Same seed → same partition."""
        from data import dirichlet_split

        class FakeDataset:
            def __init__(self):
                self.targets = [i % 10 for i in range(300)]
                self.indices = list(range(300))

        def run(seed):
            np.random.seed(seed)
            return dirichlet_split(FakeDataset(), 3, alpha=0.1)[0]

        a = run(7)
        b = run(7)
        for cid in range(3):
            self.assertEqual(sorted(a[cid]), sorted(b[cid]))

    def test_heterogeneous_distribution(self):
        """alpha=0.1 should create more heterogeneous distribution than alpha=10."""
        from data import dirichlet_split

        class FakeDataset:
            def __init__(self):
                n = 1000
                self.targets = [i % 10 for i in range(n)]
                self.indices = list(range(n))

        def class_std(client_indices, labels, num_clients=3, num_classes=10):
            counts = np.zeros((num_clients, num_classes))
            for cid, idxs in client_indices.items():
                for idx in idxs:
                    counts[cid, labels[idx]] += 1
            return float(np.mean(np.std(counts, axis=0)))

        np.random.seed(42)
        ds = FakeDataset()
        ci_low, labels = dirichlet_split(ds, 3, alpha=0.1)
        std_low = class_std(ci_low, labels)

        np.random.seed(42)
        ci_high, labels = dirichlet_split(ds, 3, alpha=10.0)
        std_high = class_std(ci_high, labels)

        self.assertGreater(std_low, std_high,
            "alpha=0.1 should be more heterogeneous than alpha=10")

    def test_full_cifar10_dirichlet_split_and_heterogeneity_reference(self):
        """Verify full CIFAR-10 alpha=0.1 and alpha=10.0 split against known references."""
        from data import get_cifar10, dirichlet_split
        
        # 1. alpha=0.1 reference check
        np.random.seed(42)
        dataset = get_cifar10(subset_size=None)
        client_indices_01, labels = dirichlet_split(dataset, num_clients=5, alpha=0.1)
        
        sizes_01 = [len(client_indices_01[i]) for i in range(5)]
        expected_sizes_01 = [12944, 10514, 11934, 9029, 5579]
        self.assertEqual(sizes_01, expected_sizes_01, "alpha=0.1 client sample counts do not match reference")
        
        # Calculate heterogeneity metric (mean of per-class standard deviations)
        counts_01 = np.zeros((5, 10))
        for cid, idxs in client_indices_01.items():
            for idx in idxs:
                counts_01[cid, labels[idx]] += 1
        het_01 = float(np.mean(np.std(counts_01, axis=0)))
        
        # Reference value is approximately 1699.28
        self.assertAlmostEqual(het_01, 1699.280992, places=2,
                               msg=f"alpha=0.1 heterogeneity {het_01} does not match reference ~1699.28")

        # 2. alpha=10.0 check
        np.random.seed(42)
        client_indices_10, labels = dirichlet_split(dataset, num_clients=5, alpha=10.0)
        sizes_10 = [len(client_indices_10[i]) for i in range(5)]
        expected_sizes_10 = [8856, 10009, 10553, 10399, 10183]
        self.assertEqual(sizes_10, expected_sizes_10, "alpha=10.0 client sample counts do not match expected")
        
        counts_10 = np.zeros((5, 10))
        for cid, idxs in client_indices_10.items():
            for idx in idxs:
                counts_10[cid, labels[idx]] += 1
        het_10 = float(np.mean(np.std(counts_10, axis=0)))
        
        # Verify lower heterogeneity for alpha=10.0 compared to alpha=0.1
        self.assertLess(het_10, het_01, "alpha=10.0 should have lower heterogeneity than alpha=0.1")
        self.assertAlmostEqual(het_10, 236.879573, places=2)

    def test_split_validation_artifacts_schema_and_compatibility(self):
        """Verify saved validation JSON artifacts match required schema and references."""
        for filename, expected_alpha, expected_sizes in [
            ("split_validation_full.json", 0.1, [12944, 10514, 11934, 9029, 5579]),
            ("split_validation_full_alpha0.1.json", 0.1, [12944, 10514, 11934, 9029, 5579]),
            ("split_validation_full_alpha10.0.json", 10.0, [8856, 10009, 10553, 10399, 10183]),
        ]:
            path = os.path.join(project_root, "results", "stage1", filename)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = json.load(f)
                
            self.assertEqual(data["dataset_mode"], "full")
            self.assertEqual(data["number_of_samples"], 50000)
            self.assertEqual(data["number_of_clients"], 5)
            self.assertEqual(data["num_clients"], 5)
            self.assertEqual(data["alpha"], expected_alpha)
            self.assertEqual(data["seed"], 42)
            self.assertTrue(data["all_samples_assigned_exactly_once"])
            self.assertEqual(data["missing_samples"], 0)
            self.assertEqual(data["duplicated_samples"], 0)
            
            actual_sizes = [data["samples_per_client"][str(i)] for i in range(5)]
            self.assertEqual(actual_sizes, expected_sizes)
            if "client_sample_distribution" in data:
                self.assertEqual(data["client_sample_distribution"], expected_sizes)

    def test_cli_argument_validation(self):
        """Verify invalid alpha and client count values raise ValueError."""
        # Test alpha <= 0
        from data import main
        import sys
        
        orig_argv = sys.argv
        try:
            sys.argv = ["data.py", "--alpha", "0.0"]
            with self.assertRaises(ValueError):
                main()
                
            sys.argv = ["data.py", "--alpha", "-1.0"]
            with self.assertRaises(ValueError):
                main()
                
            sys.argv = ["data.py", "--num-clients", "0"]
            with self.assertRaises(ValueError):
                main()
                
            sys.argv = ["data.py", "--num-clients", "-5"]
            with self.assertRaises(ValueError):
                main()
        finally:
            sys.argv = orig_argv


if __name__ == "__main__":
    unittest.main()
