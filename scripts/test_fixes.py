import unittest
import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, FitRes, Status, Code
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_federated import CheckpointingFedAvg

class TestClientParticipation(unittest.TestCase):
    def setUp(self):
        self.strategy = CheckpointingFedAvg(run_config={"num_clients": 5}, accept_failures=False)
        self.parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
        
    def test_missing_client(self):
        # 4 clients instead of 5
        results = []
        for i in range(4):
            results.append((None, FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=self.parameters,
                num_examples=10,
                metrics={"client_id": str(i)}
            )))
        
        with self.assertRaisesRegex(RuntimeError, "expected 5 clients, but got 4"):
            self.strategy.aggregate_fit(1, results, [])
            
    def test_missing_client_ids(self):
        # 5 clients, but one ID is duplicated (missing client_id '4')
        results = []
        for i in [0, 1, 2, 3, 3]:
            results.append((None, FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=self.parameters,
                num_examples=10,
                metrics={"client_id": str(i)}
            )))
            
        with self.assertRaisesRegex(RuntimeError, "expected client IDs"):
            self.strategy.aggregate_fit(1, results, [])

    def test_client_failures(self):
        # test failure
        results = []
        for i in range(4):
            results.append((None, FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=self.parameters,
                num_examples=10,
                metrics={"client_id": str(i)}
            )))
            
        failures = [(None, Exception("Client failed"))]
        with self.assertRaisesRegex(RuntimeError, "1 client\(s\) failed"):
            self.strategy.aggregate_fit(1, results, failures)

if __name__ == '__main__':
    unittest.main()
