import sys
import os
import unittest
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

from src.model import SimpleCNN
from src.config import FULL_CONFIG, SUBSET_CONFIG
from scripts.run_federated import FlowerClient, get_evaluate_fn, fit_metrics_aggregation_fn
import scripts.run_federated as federated
from src.data import dirichlet_split
import numpy as np


class TestFederatedPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.net = SimpleCNN().to(self.device)
        
        # Create mock data
        x = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))
        self.dataset = TensorDataset(x, y)
        self.dataset.targets = y.numpy().tolist()
        self.loader = DataLoader(self.dataset, batch_size=2)

    def test_config_deepcopy_isolation(self):
        # Ensure deepcopy prevents mutation leakage
        c1 = deepcopy(SUBSET_CONFIG)
        c2 = deepcopy(SUBSET_CONFIG)
        c1["num_clients"] = 999
        self.assertNotEqual(c1["num_clients"], c2["num_clients"])
        self.assertEqual(SUBSET_CONFIG["num_clients"], 3)

    def test_dirichlet_split_determinism(self):
        np.random.seed(42)
        split1, labels1 = dirichlet_split(self.dataset, num_clients=3, alpha=0.1)
        np.random.seed(42)
        split2, labels2 = dirichlet_split(self.dataset, num_clients=3, alpha=0.1)
        
        for k in split1.keys():
            self.assertEqual(split1[k], split2[k])
        self.assertTrue((labels1 == labels2).all())

    def test_client_creation_and_config(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False, run_config={"batch_size": 16})
        self.assertFalse(client.use_dp)
        self.assertEqual(client.run_config["batch_size"], 16)

    def test_dp_client_creation(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=True, run_config={"noise_multiplier": 1.5, "max_grad_norm": 2.0})
        self.assertTrue(client.use_dp)
        
        # Test DP fit step
        initial_params = [p.copy() for p in client.get_parameters(config={})]
        new_params, num_samples, metrics = client.fit(initial_params, config={})
        self.assertEqual(num_samples, 20)
        self.assertEqual(metrics["sigma"], 1.5)
        self.assertEqual(metrics["C"], 2.0)
        self.assertIn("dp_steps", metrics)
        
        # Verify weights actually changed
        changed = any(not (p1 == p2).all() for p1, p2 in zip(initial_params, new_params))
        self.assertTrue(changed)

    def test_get_evaluate_fn(self):
        # verify get_evaluate_fn accepts run_config and works correctly
        eval_fn = get_evaluate_fn(self.dataset, self.device, run_config={"batch_size": 4})
        
        # Mock parameters
        params = [val.cpu().numpy() for val in self.net.state_dict().values()]
        
        loss, metrics = eval_fn(1, params, {})
        self.assertIsInstance(loss, float)
        self.assertIn("accuracy", metrics)

    def test_fit_metrics_aggregation(self):
        results = [
            (10, {"client_id": "0", "sample_rate": 0.1, "dp_steps": 10, "sigma": 1.0, "C": 1.0, "train_loss": 0.5}),
            (10, {"client_id": "1", "sample_rate": 0.1, "dp_steps": 10, "sigma": 1.0, "C": 1.0, "train_loss": 0.6})
        ]
        agg_metrics = fit_metrics_aggregation_fn(results)
        self.assertIn("client_stats", agg_metrics)
        stats = json.loads(agg_metrics["client_stats"])
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["client_id"], "0")

    def test_client_fn_factory(self):
        federated.CLIENT_INDICES = {0: list(range(10))}
        federated.GLOBAL_TRAINSET = self.dataset
        federated.GLOBAL_TESTSET = self.dataset
        
        class MockContext:
            def __init__(self):
                self.node_id = 0
                self.node_config = {"partition-id": 0}
                
        ctx = MockContext()
        client_fn = federated.client_fn_factory(run_config={"batch_size": 2, "num_clients": 1})
        client = client_fn(ctx)
        
        self.assertIsNotNone(client)
        # Should be a numpy client wrapper
        self.assertTrue(hasattr(client, "fit"))

if __name__ == "__main__":
    unittest.main()
