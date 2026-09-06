import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from federated import FlowerClient
from model import SimpleCNN

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
        # Create a Stage 3 client (no DP)
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        self.assertFalse(client.use_dp)
        
    def test_parameter_serialization(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        params = client.get_parameters(config={})
        self.assertTrue(isinstance(params, list))
        self.assertTrue(isinstance(params[0], type(params[0]))) # numpy array check conceptually
        
        # Mutate and set
        client.set_parameters(params)
        
    def test_local_training_step_no_dp(self):
        client = FlowerClient("0", self.net, self.loader, self.loader, self.device, use_dp=False)
        initial_params = [p.copy() for p in client.get_parameters(config={})]
        
        # Run fit (which uses plain SGD)
        new_params, num_samples, metrics = client.fit(initial_params, config={})
        
        self.assertEqual(num_samples, 10)
        self.assertEqual(metrics, {})
        
        # Check if parameters actually updated
        changed = False
        for p1, p2 in zip(initial_params, new_params):
            if not (p1 == p2).all():
                changed = True
                break
        self.assertTrue(changed, "Parameters should change after local training")

if __name__ == "__main__":
    unittest.main()
