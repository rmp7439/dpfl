# Small-subset config for fast iteration and debugging
# This configuration is used for testing the end-to-end flow before scaling up.

CONFIG = {
    # Data parameters
    "num_samples": 1000,       # Tiny subset of CIFAR-10 images
    "batch_size": 32,          # Small batch size
    
    # Federated Learning (Flower)
    "num_clients": 3,          # 3 simulated clients for better visualization
    "num_rounds": 2,           # Tiny number of federation rounds
    "local_epochs": 1,         # Minimal local training
    
    # Data Split (Dirichlet)
    "alpha": 0.1,              # Concentration parameter for Dirichlet split
    
    # Centralized Baseline Training
    "central_epochs": 2,       # Tiny number of epochs for small-subset testing
    "central_lr": 0.001,       # Learning rate
    
    # DP-SGD (Opacus) parameters
    "dp_lr": 0.00025,
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
}
