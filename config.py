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
