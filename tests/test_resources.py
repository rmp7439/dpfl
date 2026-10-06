import unittest
from src.config import get_client_resources

class TestResourceAllocation(unittest.TestCase):
    def test_cpu_resource_config_valid(self):
        res = get_client_resources(num_clients=3, use_dp=False, gpu_available=False)
        self.assertEqual(res["num_cpus"], 1.0)
        self.assertEqual(res["num_gpus"], 0.0)

    def test_gpu_resource_config_valid(self):
        res = get_client_resources(num_clients=3, use_dp=False, gpu_available=True)
        self.assertEqual(res["num_cpus"], 1.0)
        self.assertGreater(res["num_gpus"], 0.0)
        
    def test_gpu_resource_nonzero_when_enabled(self):
        res_dp = get_client_resources(num_clients=5, use_dp=True, gpu_available=True)
        self.assertGreater(res_dp["num_gpus"], 0.0)

    def test_gpu_resource_does_not_exceed_physical(self):
        res = get_client_resources(num_clients=5, use_dp=True, gpu_available=True)
        # Even if there are 5 clients, a single client must not ask for more than 1.0 GPU
        self.assertLessEqual(res["num_gpus"], 1.0)

    def test_client_resource_values_numeric_positive(self):
        res = get_client_resources(num_clients=3, use_dp=True, gpu_available=True)
        self.assertIsInstance(res["num_cpus"], (int, float))
        self.assertIsInstance(res["num_gpus"], (int, float))
        self.assertGreaterEqual(res["num_cpus"], 0)
        self.assertGreaterEqual(res["num_gpus"], 0)

    def test_subset_and_full_derive_valid_settings(self):
        # Subset (3 clients)
        res_subset_non_dp = get_client_resources(num_clients=3, use_dp=False, gpu_available=True)
        res_subset_dp = get_client_resources(num_clients=3, use_dp=True, gpu_available=True)
        self.assertGreater(res_subset_non_dp["num_gpus"], 0.0)
        self.assertGreater(res_subset_dp["num_gpus"], 0.0)
        
        # Full (5 clients)
        res_full_non_dp = get_client_resources(num_clients=5, use_dp=False, gpu_available=True)
        res_full_dp = get_client_resources(num_clients=5, use_dp=True, gpu_available=True)
        self.assertGreater(res_full_non_dp["num_gpus"], 0.0)
        self.assertGreater(res_full_dp["num_gpus"], 0.0)

    def test_configuration_is_deterministic(self):
        res1 = get_client_resources(num_clients=5, use_dp=True, gpu_available=True)
        res2 = get_client_resources(num_clients=5, use_dp=True, gpu_available=True)
        self.assertEqual(res1, res2)

if __name__ == '__main__':
    unittest.main()
