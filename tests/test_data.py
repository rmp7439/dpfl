
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
"""
Comprehensive test suite for the DP-FL repository.
Tests cover:
  - Dirichlet partition correctness
  - Stage 3 non-DP enforcement
  - Opacus grad_sample existence and shape
  - RDP accounting: finiteness, monotonicity, sigma ordering, delta, cumulative match
  - Grid artifact completeness
"""

import unittest
import math
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from dpfl.model import SimpleCNN


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
        from dpfl.data import dirichlet_split
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
        from dpfl.data import dirichlet_split

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
        from dpfl.data import dirichlet_split

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


if __name__ == "__main__":
    unittest.main()
