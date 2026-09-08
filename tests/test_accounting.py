
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
import unittest
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
import math

class TestStage5Accounting(unittest.TestCase):
    def setUp(self):
        self.alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.delta = 1e-5
        
    def test_epsilon_is_finite(self):
        rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=100, orders=self.alphas)
        epsilon, best_alpha = get_privacy_spent(orders=self.alphas, rdp=rdp, delta=self.delta)
        
        self.assertTrue(math.isfinite(epsilon))
        self.assertTrue(epsilon > 0)
        
    def test_epsilon_strictly_increases_with_steps(self):
        q = 0.05
        sigma = 1.0
        
        rdp_10 = compute_rdp(q=q, noise_multiplier=sigma, steps=10, orders=self.alphas)
        eps_10, _ = get_privacy_spent(orders=self.alphas, rdp=rdp_10, delta=self.delta)
        
        rdp_20 = compute_rdp(q=q, noise_multiplier=sigma, steps=20, orders=self.alphas)
        eps_20, _ = get_privacy_spent(orders=self.alphas, rdp=rdp_20, delta=self.delta)
        
        self.assertTrue(eps_20 > eps_10)
        
    def test_increasing_sigma_decreases_epsilon(self):
        q = 0.05
        steps = 50
        
        rdp_low_noise = compute_rdp(q=q, noise_multiplier=0.5, steps=steps, orders=self.alphas)
        eps_low, _ = get_privacy_spent(orders=self.alphas, rdp=rdp_low_noise, delta=self.delta)
        
        rdp_high_noise = compute_rdp(q=q, noise_multiplier=1.5, steps=steps, orders=self.alphas)
        eps_high, _ = get_privacy_spent(orders=self.alphas, rdp=rdp_high_noise, delta=self.delta)
        
        self.assertTrue(eps_high < eps_low)
        
    def test_cumulative_accounting_matches_total(self):
        q = 0.05
        sigma = 1.0
        steps_per_round = 10
        num_rounds = 3
        
        # Calculate cumulative step-by-step
        epsilons = []
        for r in range(1, num_rounds + 1):
            total_steps = r * steps_per_round
            rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=total_steps, orders=self.alphas)
            eps, _ = get_privacy_spent(orders=self.alphas, rdp=rdp, delta=self.delta)
            epsilons.append(eps)
            
        # Calculate total directly
        rdp_total = compute_rdp(q=q, noise_multiplier=sigma, steps=steps_per_round * num_rounds, orders=self.alphas)
        eps_total, _ = get_privacy_spent(orders=self.alphas, rdp=rdp_total, delta=self.delta)
        
        # They should match exactly because the formula is identical
        self.assertAlmostEqual(epsilons[-1], eps_total, places=5)
        
    def test_delta_is_correctly_applied(self):
        rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=100, orders=self.alphas)
        eps_5, _ = get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-5)
        eps_6, _ = get_privacy_spent(orders=self.alphas, rdp=rdp, delta=1e-6)
        
        # A smaller delta (tighter guarantee) requires a larger epsilon (looser bound)
        self.assertTrue(eps_6 > eps_5)

if __name__ == "__main__":
    unittest.main()
