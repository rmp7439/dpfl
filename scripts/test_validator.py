import sys
import os
import json
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from scripts.validate_artifact import validate_artifact

class TestValidator(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "dataset_mode": "full",
            "number_of_communication_rounds": 15,
            "num_clients": 5,
            "alpha": 0.1,
            "sigma": 1.0,
            "C": 1.0,
            "delta": 1e-5,
            "seed": 42,
            "dp_enabled": True,
            "per_round_test_accuracy": [0.1] * 15,
            "per_round_participation": [
                {
                    "round": i + 1,
                    "expected_clients": ["0", "1", "2", "3", "4"],
                    "actual_clients": ["0", "1", "2", "3", "4"],
                    "num_success": 5,
                    "num_failures": 0
                } for i in range(15)
            ],
            "epsilon": 2.5,
            "best_alpha": 4.5,
            "client_details": {
                str(i): {
                    "cumulative_steps": 100,
                    "epsilon": 2.0,
                    "sample_rate": 0.05,
                    "round_steps": 10
                } for i in range(5)
            }
        }
        
    def write_temp(self, data, name):
        path = os.path.join(os.path.dirname(__file__), name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def tearDown(self):
        for f in ["valid.json", "invalid_rounds.json", "invalid_participation.json"]:
            path = os.path.join(os.path.dirname(__file__), f)
            if os.path.exists(path):
                os.remove(path)

    def test_valid_artifact(self):
        path = self.write_temp(self.valid_data, "valid.json")
        res = validate_artifact(path, {"sigma": 1.0, "C": 1.0})
        self.assertIsNotNone(res)

    def test_invalid_rounds(self):
        data = dict(self.valid_data)
        data["number_of_communication_rounds"] = 10
        path = self.write_temp(data, "invalid_rounds.json")
        with self.assertRaisesRegex(ValueError, "not 15"):
            validate_artifact(path, {"sigma": 1.0})

    def test_invalid_participation(self):
        data = dict(self.valid_data)
        data["per_round_participation"][0]["actual_clients"] = ["0", "1", "2", "3"]
        path = self.write_temp(data, "invalid_participation.json")
        with self.assertRaisesRegex(ValueError, "expected 5 clients, got 4"):
            validate_artifact(path, {"sigma": 1.0})
            
if __name__ == "__main__":
    unittest.main()
