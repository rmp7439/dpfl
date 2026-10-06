# Configuration for DP-FL experiments

SUBSET_CONFIG = {
    # Data parameters
    "num_samples": 1000,
    "batch_size": 32,
    
    # Federated Learning (Flower)
    "num_clients": 3,
    "num_rounds": 2,
    "local_epochs": 1,
    "fed_lr": 0.01,
    "fed_optimizer": "SGD",
    
    # Data Split (Dirichlet)
    "alpha": 0.1,
    
    # Centralized Baseline Training
    "central_epochs": 2,
    "central_lr": 0.001,
    
    # DP-SGD (Opacus) parameters
    "dp_lr": 0.01,
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
}

STAGE5_SIGMAS = [0.5, 1.0, 1.5, 2.0]
STAGE5_C_VALUES = [0.1, 1.0]

STAGE6_ALPHAS = [0.1, 10.0]
STAGE6_SIGMAS = [1.0, 2.0]
STAGE6_C = 1.0

FULL_CONFIG = {
    # Data parameters
    "num_samples": 50000,
    "batch_size": 64,
    
    # Federated Learning (Flower)
    "num_clients": 5,          # 5 clients for diversity, manageable on CPU
    "num_rounds": 3,           # 3 communication rounds (CPU-feasible)
    "local_epochs": 1,         # 1 local epoch per round
    "fed_lr": 0.05,            # Higher LR for SGD to converge faster
    "fed_optimizer": "SGD",
    
    # Data Split (Dirichlet)
    "alpha": 0.1,
    
    # Centralized Baseline Training
    "central_epochs": 30,      # 30 epochs for reasonable convergence on CPU
    "central_lr": 0.001,
    
    # DP-SGD (Opacus) parameters
    "dp_lr": 0.05,
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
}

# Default to SUBSET. Scripts must explicitly import and use FULL_CONFIG when running full data.
CONFIG = SUBSET_CONFIG

def get_client_resources(num_clients: int, use_dp: bool, gpu_available: bool) -> dict:
    """
    Centralized resource allocation for Flower virtual clients.
    Ensures safe concurrency on a single GPU without oversubscription.
    """
    if not gpu_available:
        return {"num_cpus": 1.0, "num_gpus": 0.0}
        
    # GPU Mode
    # A single T4 (16GB) can safely hold 2 DP clients concurrently.
    if use_dp:
        # Allocating 0.5 allows exactly 2 to run concurrently per GPU.
        # This prevents out-of-memory errors on 5-client full runs on a single T4.
        return {"num_cpus": 1.0, "num_gpus": 0.5}
    else:
        # Non-DP can safely run more concurrently. 
        # Allocating 0.33 allows exactly 3 clients concurrently on 1 GPU.
        return {"num_cpus": 1.0, "num_gpus": 0.33}
