import sys
import os
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

from model import SimpleCNN
from config import CONFIG
from scripts.run_federated import FlowerClient
import scripts.run_federated as federated

class TestStage3Federated(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.net = SimpleCNN().to(self.device)
        
        # Create mock data
        x = torch.randn(10, 3, 32, 32)
        y = torch.randint(0, 10, (10,))
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=2)
        
    def test_plain_client_creation(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        self.assertFalse(client.use_dp)
        
    def test_parameter_serialization(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        params = client.get_parameters(config={})
        self.assertTrue(isinstance(params, list))
        client.set_parameters(params)
        
    def test_local_training_step_no_dp(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        initial_params = [p.copy() for p in client.get_parameters(config={})]
        new_params, num_samples, metrics = client.fit(initial_params, config={})
        self.assertEqual(num_samples, 10)
        self.assertEqual(metrics, {})
        changed = any(not (p1 == p2).all() for p1, p2 in zip(initial_params, new_params))
        self.assertTrue(changed, "Parameters should change after local training")

    def test_stage3_no_privacy_engine_attribute(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        params = [p.copy() for p in client.get_parameters(config={})]
        client.fit(params, config={})
        self.assertFalse(hasattr(client, "privacy_engine"), "Stage 3 client should not have privacy_engine after fit()")

class TestStage3Federation(unittest.TestCase):
    def test_client_is_non_private(self):
        federated.CLIENT_INDICES = {0: [0, 1, 2]}
        federated.GLOBAL_TRAINSET = [0, 1, 2]
        federated.GLOBAL_TESTSET = [0, 1, 2]
        
        client = federated.client_fn("0")
        self.assertFalse(client.use_dp, "Stage 3 client MUST NOT use DP by default")
        
    def test_optimizer_is_sgd_by_default(self):
        opt_name = CONFIG.get("fed_optimizer", "SGD")
        self.assertEqual(opt_name, "SGD", "Stage 3 must use SGD for official run")

if __name__ == "__main__":
    unittest.main()
