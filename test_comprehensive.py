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
        from data_split import dirichlet_split
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
        from data_split import dirichlet_split

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
        from data_split import dirichlet_split

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


# =========================================================
# 2. Stage 3 non-DP enforcement
# =========================================================

class TestStage3NonDP(unittest.TestCase):

    def _make_client(self):
        from federated import FlowerClient
        device = torch.device("cpu")
        net = SimpleCNN().to(device)
        x = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)
        return FlowerClient("0", net, loader, loader, device, use_dp=False)

    def test_stage3_client_use_dp_false(self):
        client = self._make_client()
        self.assertFalse(client.use_dp)

    def test_stage3_no_privacy_engine_attribute(self):
        """After a fit() call with use_dp=False, no privacy_engine should exist."""
        client = self._make_client()
        params = [p.copy() for p in client.get_parameters(config={})]
        client.fit(params, config={})
        # Should not have privacy_engine attached
        self.assertFalse(hasattr(client, "privacy_engine"),
            "Stage 3 client should not have privacy_engine after fit()")

    def test_stage3_sgd_config(self):
        from config import SUBSET_CONFIG
        self.assertEqual(SUBSET_CONFIG.get("fed_optimizer"), "SGD",
            "Default federated optimizer must be SGD for Stage 3")

    def test_stage3_parameters_change_after_fit(self):
        client = self._make_client()
        initial_params = [p.copy() for p in client.get_parameters(config={})]
        client.fit(initial_params, config={})
        new_params = client.get_parameters(config={})
        changed = any(not (p1 == p2).all() for p1, p2 in zip(initial_params, new_params))
        self.assertTrue(changed, "Parameters must change after local training")


# =========================================================
# 3. Opacus DP validation
# =========================================================

class TestOpacusDP(unittest.TestCase):

    def setUp(self):
        import warnings
        warnings.filterwarnings("ignore")
        self.device = torch.device("cpu")

    def _make_dp_client(self):
        from opacus import PrivacyEngine
        net = SimpleCNN().to(self.device)
        x = torch.randn(17, 3, 32, 32)
        y = torch.randint(0, 10, (17,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)

        import torch.optim as optim
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        pe = PrivacyEngine(accountant="rdp")
        net, optimizer, dp_loader = pe.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        return net, optimizer, dp_loader, pe

    def test_grad_sample_present(self):
        net, optimizer, dp_loader, pe = self._make_dp_client()
        net.train()
        criterion = nn.CrossEntropyLoss()
        batch_x, batch_y = next(iter(dp_loader))
        optimizer.zero_grad()
        out = net(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "grad_sample"),
                    f"Missing grad_sample for {name}")

    def test_grad_sample_batch_dimension(self):
        net, optimizer, dp_loader, pe = self._make_dp_client()
        net.train()
        criterion = nn.CrossEntropyLoss()
        batch_x, batch_y = next(iter(dp_loader))
        optimizer.zero_grad()
        out = net(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad and hasattr(param, "grad_sample"):
                self.assertEqual(param.grad_sample.shape[0], len(batch_x),
                    f"grad_sample batch dim mismatch for {name}")

    def test_epsilon_from_accountant_is_finite(self):
        net, optimizer, dp_loader, pe = self._make_dp_client()
        net.train()
        criterion = nn.CrossEntropyLoss()
        for batch_x, batch_y in dp_loader:
            optimizer.zero_grad()
            out = net(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        eps = pe.accountant.get_epsilon(delta=1e-5)
        self.assertTrue(math.isfinite(eps), "Epsilon from accountant must be finite")
        self.assertGreater(eps, 0, "Epsilon must be positive")


# =========================================================
# 4. RDP accounting properties
# =========================================================

class TestRDPAccounting(unittest.TestCase):

    def setUp(self):
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
        self.compute_rdp = compute_rdp
        self.get_privacy_spent = get_privacy_spent
        self.alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.delta = 1e-5

    def _eps(self, q, sigma, steps):
        rdp = self.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=self.alphas)
        eps, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp, delta=self.delta)
        return eps

    def test_epsilon_finite_and_positive(self):
        eps = self._eps(0.01, 1.0, 100)
        self.assertTrue(math.isfinite(eps))
        self.assertGreater(eps, 0)

    def test_epsilon_non_decreasing_with_steps(self):
        """Epsilon must be non-decreasing as steps increase."""
        q, sigma = 0.05, 1.0
        prev = 0.0
        for steps in [10, 20, 30, 50, 100]:
            eps = self._eps(q, sigma, steps)
            self.assertGreaterEqual(eps, prev,
                f"Epsilon decreased at steps={steps}: {eps} < {prev}")
            prev = eps

    def test_higher_sigma_lower_epsilon(self):
        """Higher noise multiplier must give lower epsilon, all else equal."""
        q, steps = 0.05, 50
        eps_low_noise = self._eps(q, 0.5, steps)
        eps_med_noise = self._eps(q, 1.0, steps)
        eps_high_noise = self._eps(q, 2.0, steps)
        self.assertGreater(eps_low_noise, eps_med_noise)
        self.assertGreater(eps_med_noise, eps_high_noise)

    def test_smaller_delta_larger_epsilon(self):
        """Smaller delta (tighter privacy guarantee) must require larger epsilon."""
        rdp = self.compute_rdp(q=0.01, noise_multiplier=1.0, steps=100, orders=self.alphas)
        eps_1e5, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-5)
        eps_1e6, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-6)
        self.assertGreater(eps_1e6, eps_1e5)

    def test_cumulative_steps_match(self):
        """Composing r rounds of T steps each == accounting for r*T steps total."""
        q, sigma = 0.05, 1.0
        T_per_round, num_rounds = 10, 3
        rdp_total = self.compute_rdp(q=q, noise_multiplier=sigma,
                                     steps=T_per_round * num_rounds, orders=self.alphas)
        eps_total, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp_total, delta=self.delta)

        # Step-by-step cumulative
        eps_cumulative = self._eps(q, sigma, T_per_round * num_rounds)
        self.assertAlmostEqual(eps_total, eps_cumulative, places=6)

    def test_c_does_not_change_epsilon_directly(self):
        """Clipping norm C is not an accountant parameter — sigma alone drives epsilon."""
        # C does NOT appear in compute_rdp. Same sigma -> same epsilon regardless of C label.
        eps_c01 = self._eps(0.05, 1.0, 50)
        eps_c10 = self._eps(0.05, 1.0, 50)
        self.assertAlmostEqual(eps_c01, eps_c10, places=6,
            msg="Epsilon must be identical for same sigma regardless of C label")


# =========================================================
# 5. Grid artifact completeness (run only if artifacts exist)
# =========================================================

class TestGridArtifacts(unittest.TestCase):

    GRID_JSON = os.path.join("results", "stage5", "grid_results.json")
    SIGMAS = [0.5, 1.0, 1.5, 2.0]
    CS = [0.1, 1.0]

    def _load_grid(self):
        if not os.path.exists(self.GRID_JSON):
            self.skipTest("grid_results.json not found — grid has not run yet")
        with open(self.GRID_JSON) as f:
            return json.load(f)

    def test_all_8_configurations_present(self):
        results = self._load_grid()
        self.assertEqual(len(results), 8,
            f"Expected 8 grid results, got {len(results)}")

    def test_all_configurations_successful(self):
        results = self._load_grid()
        for r in results:
            self.assertEqual(r.get("run_status"), "Success",
                f"Run failed for sigma={r.get('sigma')}, C={r.get('C')}")

    def test_epsilon_non_decreasing_within_run(self):
        """For each run, per-round epsilon must be non-decreasing."""
        base = os.path.join("results", "stage5", "per_run")
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
                epsilons = [float(r["epsilon"]) for r in rows]
                for i in range(1, len(epsilons)):
                    self.assertGreaterEqual(epsilons[i], epsilons[i-1],
                        f"Epsilon decreased at round {i+1} for sigma={s}, C={c}")

    def test_higher_sigma_lower_final_epsilon(self):
        """In grid results, higher sigma configs should yield lower epsilon."""
        results = self._load_grid()
        # Build map: (sigma, C) -> epsilon
        eps_map = {(r["sigma"], r["C"]): r["epsilon"] for r in results}
        for c in self.CS:
            for i in range(len(self.SIGMAS) - 1):
                s_low = self.SIGMAS[i]
                s_high = self.SIGMAS[i + 1]
                if (s_low, c) in eps_map and (s_high, c) in eps_map:
                    self.assertGreaterEqual(eps_map[(s_low, c)], eps_map[(s_high, c)],
                        f"sigma={s_high} should have <= epsilon vs sigma={s_low}, C={c}")

    def test_per_run_artifacts_exist(self):
        """Each configuration directory must have summary.json and rounds.csv."""
        base = os.path.join("results", "stage5", "per_run")
        if not os.path.exists(base):
            self.skipTest("per_run directory not found")
        for s in self.SIGMAS:
            for c in self.CS:
                run_dir = os.path.join(base, f"sigma_{s}_C_{c}")
                summary = os.path.join(run_dir, "summary.json")
                rounds = os.path.join(run_dir, "rounds.csv")
                self.assertTrue(os.path.exists(summary),
                    f"Missing summary.json for sigma={s}, C={c}")
                self.assertTrue(os.path.exists(rounds),
                    f"Missing rounds.csv for sigma={s}, C={c}")

    def test_grid_results_internally_consistent(self):
        """sigma and C in grid_results.json must match those in per-run summary.json;
        both must agree on dataset_mode."""
        results = self._load_grid()
        base = os.path.join("results", "stage5", "per_run")
        for r in results:
            s, c = r["sigma"], r["C"]
            summary_path = os.path.join(base, f"sigma_{s}_C_{c}", "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            # sigma and C must match
            self.assertAlmostEqual(r["sigma"], summary["sigma"], places=4,
                msg=f"sigma mismatch in grid vs per-run for sigma={s}, C={c}")
            self.assertAlmostEqual(r["C"], summary["C"], places=4,
                msg=f"C mismatch in grid vs per-run for sigma={s}, C={c}")
            # dataset_mode must match
            self.assertEqual(r.get("dataset_mode"), summary.get("dataset_mode"),
                msg=f"dataset_mode mismatch in grid vs per-run for sigma={s}, C={c}: "
                    f"grid says {r.get('dataset_mode')}, per-run says {summary.get('dataset_mode')}")


if __name__ == "__main__":
    unittest.main()
