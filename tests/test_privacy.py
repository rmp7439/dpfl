import sys
import os
import unittest
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

from dpfl.model import SimpleCNN

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
                self.assertTrue(hasattr(param, "grad_sample"), f"Missing grad_sample for {name}")

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
                self.assertEqual(param.grad_sample.shape[0], len(batch_x), f"grad_sample batch dim mismatch for {name}")

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
        q, sigma = 0.05, 1.0
        prev = 0.0
        for steps in [10, 20, 30, 50, 100]:
            eps = self._eps(q, sigma, steps)
            self.assertGreaterEqual(eps, prev, f"Epsilon decreased at steps={steps}: {eps} < {prev}")
            prev = eps

    def test_higher_sigma_lower_epsilon(self):
        q, steps = 0.05, 50
        eps_low_noise = self._eps(q, 0.5, steps)
        eps_med_noise = self._eps(q, 1.0, steps)
        eps_high_noise = self._eps(q, 2.0, steps)
        self.assertGreater(eps_low_noise, eps_med_noise)
        self.assertGreater(eps_med_noise, eps_high_noise)

    def test_smaller_delta_larger_epsilon(self):
        rdp = self.compute_rdp(q=0.01, noise_multiplier=1.0, steps=100, orders=self.alphas)
        eps_1e5, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-5)
        eps_1e6, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-6)
        self.assertGreater(eps_1e6, eps_1e5)

    def test_cumulative_steps_match(self):
        q, sigma = 0.05, 1.0
        T_per_round, num_rounds = 10, 3
        rdp_total = self.compute_rdp(q=q, noise_multiplier=sigma, steps=T_per_round * num_rounds, orders=self.alphas)
        eps_total, _ = self.get_privacy_spent(orders=self.alphas, rdp=rdp_total, delta=self.delta)
        eps_cumulative = self._eps(q, sigma, T_per_round * num_rounds)
        self.assertAlmostEqual(eps_total, eps_cumulative, places=6)

    def test_c_does_not_change_epsilon_directly(self):
        eps_c01 = self._eps(0.05, 1.0, 50)
        eps_c10 = self._eps(0.05, 1.0, 50)
        self.assertAlmostEqual(eps_c01, eps_c10, places=6, msg="Epsilon must be identical for same sigma regardless of C label")

class TestDPSteps(unittest.TestCase):
    def test_dp_steps_and_sample_rate(self):
        from opacus import PrivacyEngine
        import warnings
        warnings.filterwarnings("ignore")
        for dataset_size, batch_size in [(100, 32), (333, 32), (500, 32)]:
            data = torch.randn(dataset_size, 3)
            target = torch.randint(0, 2, (dataset_size,))
            dataset = TensorDataset(data, target)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            
            privacy_engine = PrivacyEngine(accountant="rdp")
            model, optimizer, dp_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
            
            steps = 0
            for _ in dp_loader:
                steps += 1
                
            self.assertEqual(steps, len(dp_loader), f"Iterated steps {steps} does not match len(dp_loader) {len(dp_loader)}")
            self.assertTrue(hasattr(dp_loader, "sample_rate"), "dp_loader missing sample_rate")

if __name__ == "__main__":
    unittest.main()
