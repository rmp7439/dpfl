import unittest
from federated import client_fn
import torch
from config import CONFIG

class TestStage3Federation(unittest.TestCase):
    def test_client_is_non_private(self):
        # By default USE_DP should be False, but we explicitly test the client_fn
        import federated
        # Mock CLIENT_INDICES and datasets
        federated.CLIENT_INDICES = {0: [0, 1, 2]}
        federated.GLOBAL_TRAINSET = [0, 1, 2]
        federated.GLOBAL_TESTSET = [0, 1, 2]
        
        client = client_fn("0")
        self.assertFalse(client.use_dp, "Stage 3 client MUST NOT use DP by default")
        
    def test_optimizer_is_sgd_by_default(self):
        # We test that the config defines SGD
        opt_name = CONFIG.get("fed_optimizer", "SGD")
        self.assertEqual(opt_name, "SGD", "Stage 3 must use SGD for official run")

if __name__ == "__main__":
    unittest.main()
