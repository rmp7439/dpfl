# Small-subset config for fast iteration and debugging
# This configuration is used for testing the end-to-end flow before scaling up.

CONFIG = {
    # Data parameters
    "num_samples": 1000,       # Tiny subset of CIFAR-10 images
    "batch_size": 32,          # Small batch size
    
    # Federated Learning (Flower)
    "num_clients": 2,          # 2-3 simulated clients
    "num_rounds": 2,           # Tiny number of federation rounds
    "local_epochs": 1,         # Minimal local training
    
    # DP-SGD (Opacus) parameters
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
}
