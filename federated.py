import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
import argparse
import numpy as np
from typing import Dict, List, Tuple

from config import CONFIG
from model import SimpleCNN
from train_baseline import get_data, train, test
from data_split import dirichlet_split
import math
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

# Global data placeholders for simulation
GLOBAL_TRAINSET = None
GLOBAL_TESTSET = None
CLIENT_INDICES = None

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, test_loader, device):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = CONFIG.get("dp_lr", 0.00025)
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        local_epochs = CONFIG.get("local_epochs", 1)
        
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        self.net, optimizer, train_loader = privacy_engine.make_private(
            module=self.net,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=CONFIG.get("noise_multiplier", 1.0),
            max_grad_norm=CONFIG.get("max_grad_norm", 1.0),
        )
        
        for epoch in range(1, local_epochs + 1):
            train(self.net, self.device, train_loader, optimizer, epoch)
            
        self.net = self.net._module
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test(self.net, self.device, self.test_loader)
        return float(0.0), len(self.test_loader.dataset), {"accuracy": acc}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    
    # Get the client's subset of data
    client_id = int(cid)
    indices = CLIENT_INDICES[client_id]
    
    client_dataset = torch.utils.data.Subset(GLOBAL_TRAINSET, indices)
    
    train_loader = DataLoader(client_dataset, batch_size=CONFIG.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(GLOBAL_TESTSET, batch_size=CONFIG.get("batch_size", 32), shuffle=False)
    
    return FlowerClient(cid, net, train_loader, test_loader, device)

def evaluate_metrics_aggregation_fn(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics over clients."""
    if not results:
        return {}
    
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in results])
    
    return {"accuracy": weighted_acc / total_examples}

def main():
    global GLOBAL_TRAINSET, GLOBAL_TESTSET, CLIENT_INDICES
    
    parser = argparse.ArgumentParser(description="Run Flower Federation")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    parser.add_argument("--sigma", type=float, help="Noise multiplier (sigma) for DP-SGD")
    parser.add_argument("--C", type=float, help="Max grad norm (C) for DP-SGD")
    args = parser.parse_args()
    
    if args.sigma is not None:
        CONFIG["noise_multiplier"] = args.sigma
    if args.C is not None:
        CONFIG["max_grad_norm"] = args.C
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = CONFIG.get("num_samples", 1000)
    num_clients = CONFIG.get("num_clients", 3)
    alpha = CONFIG.get("alpha", 0.1)
    
    if args.full:
        print(f"Running Federation on FULL CIFAR-10, clients: {num_clients}, alpha: {alpha}")
        GLOBAL_TRAINSET, GLOBAL_TESTSET = get_data(subset_size=None)
        num_rounds = 10 # match baseline epochs
    else:
        print(f"Running Federation on subset ({num_samples} samples), clients: {num_clients}, alpha: {alpha}")
        GLOBAL_TRAINSET, GLOBAL_TESTSET = get_data(subset_size=num_samples)
        num_rounds = CONFIG.get("num_rounds", 2)
        
    CLIENT_INDICES, _ = dirichlet_split(GLOBAL_TRAINSET, num_clients, alpha)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    # Calculate RDP
    alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    max_epsilon = 0
    best_alpha_global = 0
    
    for i in range(num_clients):
        dataset_size = len(CLIENT_INDICES[i])
        batch_size = CONFIG.get("batch_size", 32)
        steps_per_epoch = math.ceil(dataset_size / batch_size)
        sample_rate = 1.0 / steps_per_epoch
        steps_per_round = steps_per_epoch * CONFIG.get("local_epochs", 1)
        total_steps = steps_per_round * num_rounds
        
        rdp = compute_rdp(q=sample_rate, noise_multiplier=CONFIG.get("noise_multiplier", 1.0), steps=total_steps, orders=alphas)
        epsilon, best_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=1e-5)
        if epsilon > max_epsilon:
            max_epsilon = epsilon
            best_alpha_global = best_alpha
            
    final_acc = 0.0
    if history.metrics_distributed and "accuracy" in history.metrics_distributed:
        final_acc = history.metrics_distributed["accuracy"][-1][1]
        
    print("\n--- FINAL RESULTS ---")
    print(f"sigma={CONFIG.get('noise_multiplier', 1.0)}, C={CONFIG.get('max_grad_norm', 1.0)}")
    print(f"Privacy Guarantee: epsilon={max_epsilon:.4f} (delta=1e-5, alpha={best_alpha_global})")
    print(f"Final Aggregated Accuracy: {final_acc:.2f}%")
    print(f"RESULT_CSV:{CONFIG.get('noise_multiplier', 1.0)},{CONFIG.get('max_grad_norm', 1.0)},1e-5,{best_alpha_global},{max_epsilon:.4f},{final_acc:.4f}")

if __name__ == "__main__":
    main()
