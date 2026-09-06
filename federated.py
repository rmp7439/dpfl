import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
import argparse
import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
import datetime
import time
import math
from typing import Dict, List, Tuple

from config import CONFIG
from model import SimpleCNN
from train_baseline import get_data, train, test
from data_split import dirichlet_split
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

# Global data placeholders for simulation
GLOBAL_TRAINSET = None
GLOBAL_TESTSET = None
CLIENT_INDICES = None
USE_DP = False

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, test_loader, device, use_dp=False):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_dp = use_dp
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = CONFIG.get("local_epochs", 1)
        
        if self.use_dp:
            lr = CONFIG.get("dp_lr", 0.00025)
            optimizer = optim.Adam(self.net.parameters(), lr=lr)
            
            from opacus import PrivacyEngine
            import warnings
            
            warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.*")
            
            privacy_engine = PrivacyEngine(accountant="rdp")
            self.net, optimizer, train_loader = privacy_engine.make_private(
                module=self.net,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=CONFIG.get("noise_multiplier", 1.0),
                max_grad_norm=CONFIG.get("max_grad_norm", 1.0),
            )
            
            for epoch in range(1, local_epochs + 1):
                train(self.net, self.device, train_loader, optimizer, epoch)
                
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"[Client {self.cid}] Opacus Accountant Epsilon: {epsilon:.4f}")
            
            # Validation logic for grad_sample
            grad_sample_valid = True
            grad_shapes = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    if not hasattr(param, "grad_sample"):
                        grad_sample_valid = False
                    else:
                        grad_shapes[name] = list(param.grad_sample.shape)
                        
            # Save validation to a file if it's the first client's first round
            if not os.path.exists("results/stage4/validation.json"):
                os.makedirs("results/stage4", exist_ok=True)
                val_data = {
                    "grad_sample_present": grad_sample_valid,
                    "grad_sample_shapes": grad_shapes,
                    "clipping_C": CONFIG.get("max_grad_norm", 1.0),
                    "noise_sigma": CONFIG.get("noise_multiplier", 1.0)
                }
                with open("results/stage4/validation.json", "w") as f:
                    json.dump(val_data, f, indent=4)
                    
            self.net = self.net._module
        else:
            lr = CONFIG.get("fed_lr", 0.01)
            opt_name = CONFIG.get("fed_optimizer", "SGD")
            if opt_name == "SGD":
                optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(self.net.parameters(), lr=lr)
                
            for epoch in range(1, local_epochs + 1):
                train(self.net, self.device, self.train_loader, optimizer, epoch)
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        te_loss, acc = test(self.net, self.device, self.test_loader)
        return float(te_loss), len(self.test_loader.dataset), {"accuracy": acc}

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
    
    return FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP)

def evaluate_metrics_aggregation_fn(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics over clients."""
    if not results:
        return {}
    
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in results])
    
    return {"accuracy": weighted_acc / total_examples}

def load_baseline_result(mode, seed):
    baseline_path = os.path.join("results", "stage2", f"baseline_{mode}_seed{seed}.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            return json.load(f)
    return None

def main():
    global GLOBAL_TRAINSET, GLOBAL_TESTSET, CLIENT_INDICES, USE_DP
    
    parser = argparse.ArgumentParser(description="Run Flower Federation")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    parser.add_argument("--enable-dp", action="store_true", help="Enable Opacus DP-SGD (Stage 4/5)")
    parser.add_argument("--sigma", type=float, help="Noise multiplier (sigma) for DP-SGD")
    parser.add_argument("--C", type=float, help="Max grad norm (C) for DP-SGD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="results/stage4", help="Output directory for results")
    args = parser.parse_args()
    
    USE_DP = args.enable_dp
    
    if args.sigma is not None:
        CONFIG["noise_multiplier"] = args.sigma
    if args.C is not None:
        CONFIG["max_grad_norm"] = args.C
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    num_samples = CONFIG.get("num_samples", 1000)
    num_clients = CONFIG.get("num_clients", 3)
    alpha = CONFIG.get("alpha", 0.1)
    
    mode = "full" if args.full else "subset"
    
    if args.full:
        print(f"Running Federation on FULL CIFAR-10, clients: {num_clients}, alpha: {alpha}")
        GLOBAL_TRAINSET, GLOBAL_TESTSET = get_data(subset_size=None)
        
        # Determine rounds based on device capability
        if torch.cuda.is_available():
            num_rounds = 15 # match baseline epochs if on GPU
        else:
            num_rounds = 3  # heavily reduce rounds on CPU to prevent endless execution
            print("WARNING: GPU is not available. Reduced num_rounds to 3 to prevent indefinite execution.")
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
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    import opacus
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Opacus Version: {opacus.__version__}")
    print(f"Flower Version: {fl.__version__}")
    
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    final_acc = 0.0
    acc_history = []
    if history.metrics_distributed and "accuracy" in history.metrics_distributed:
        acc_history = [acc for _, acc in history.metrics_distributed["accuracy"]]
        final_acc = acc_history[-1] if acc_history else 0.0
        
    if not USE_DP:
        # Persistence for Stage 3
        out_dir = os.path.join("results", "stage3")
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"federation_{mode}"
        
        result_data = {
            "seed": args.seed,
            "dataset_mode": mode,
            "number_of_training_samples": len(GLOBAL_TRAINSET),
            "number_of_test_samples": len(GLOBAL_TESTSET),
            "number_of_clients": num_clients,
            "alpha": alpha,
            "batch_size": CONFIG.get("batch_size", 32),
            "local_epochs": CONFIG.get("local_epochs", 1),
            "optimizer": CONFIG.get("fed_optimizer", "SGD"),
            "learning_rate": CONFIG.get("fed_lr", 0.01),
            "number_of_communication_rounds": num_rounds,
            "samples_per_client": {k: len(v) for k, v in CLIENT_INDICES.items()},
            "per_round_test_accuracy": acc_history,
            "final_test_accuracy": final_acc,
            "best_test_accuracy": max(acc_history) if acc_history else 0.0,
            "execution_status": "Success",
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.json"), "w") as f:
            json.dump(result_data, f, indent=4)
            
        with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "test_acc"])
            for i, acc in enumerate(acc_history):
                writer.writerow([i+1, acc])
                
        # Comparison with Stage 2 baseline
        baseline = load_baseline_result(mode, args.seed)
        
        plt.figure(figsize=(10, 6))
        rounds = range(1, len(acc_history) + 1)
        plt.plot(rounds, acc_history, label="Federated (Stage 3)", marker='o')
        
        if baseline is not None:
            baseline_csv_path = os.path.join("results", "stage2", f"baseline_{mode}_seed{args.seed}.csv")
            baseline_accs = []
            if os.path.exists(baseline_csv_path):
                with open(baseline_csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        baseline_accs.append(float(row["test_acc"]))
            if baseline_accs:
                # Plot baseline up to matching epochs/rounds
                b_rounds = range(1, len(baseline_accs) + 1)
                plt.plot(b_rounds, baseline_accs, label="Centralized Baseline (Stage 2)", linestyle='--', color='gray')
                
        plt.xlabel('Communication Round / Epoch')
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'Federated vs Centralized Learning ({mode.capitalize()})\nNon-IID (alpha={alpha})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"convergence_{mode}_seed{args.seed}.png"))
        
        print("\n--- STAGE 3 FINAL RESULTS ---")
        print(f"Final Aggregated Accuracy: {final_acc:.2f}%")
        print(f"Results saved to {out_dir}")
        
    else:
        # Calculate RDP for Stage 4/5
        alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        # We need the maximum steps across clients for bounding epsilon
        max_steps_per_epoch = 0
        max_sample_rate = 0
        for i in range(num_clients):
            dataset_size = len(CLIENT_INDICES[i])
            batch_size = CONFIG.get("batch_size", 32)
            steps_per_epoch = math.ceil(dataset_size / batch_size)
            sample_rate = 1.0 / steps_per_epoch
            if steps_per_epoch > max_steps_per_epoch:
                max_steps_per_epoch = steps_per_epoch
                max_sample_rate = sample_rate
                
        steps_per_round = max_steps_per_epoch * CONFIG.get("local_epochs", 1)
        
        # Calculate per-round epsilon
        round_stats = []
        for r, acc in enumerate(acc_history):
            round_idx = r + 1
            total_steps_r = steps_per_round * round_idx
            rdp = compute_rdp(q=max_sample_rate, noise_multiplier=CONFIG.get("noise_multiplier", 1.0), steps=total_steps_r, orders=alphas)
            epsilon_r, best_alpha_r = get_privacy_spent(orders=alphas, rdp=rdp, delta=1e-5)
            round_stats.append({
                "round": round_idx,
                "dp_steps": total_steps_r,
                "epsilon": epsilon_r,
                "best_alpha": best_alpha_r,
                "test_acc": acc
            })
            
        final_epsilon = round_stats[-1]["epsilon"] if round_stats else 0
        final_best_alpha = round_stats[-1]["best_alpha"] if round_stats else 0
        total_steps = steps_per_round * num_rounds
        
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"dp_{mode}"
        
        result_data = {
            "dataset_mode": mode,
            "number_of_training_samples": len(GLOBAL_TRAINSET),
            "number_of_test_samples": len(GLOBAL_TESTSET),
            "number_of_clients": num_clients,
            "alpha": alpha,
            "sigma": CONFIG.get("noise_multiplier", 1.0),
            "C": CONFIG.get("max_grad_norm", 1.0),
            "delta": 1e-5,
            "sample_rate": max_sample_rate,
            "total_dp_steps": total_steps,
            "batch_size": CONFIG.get("batch_size", 32),
            "local_epochs": CONFIG.get("local_epochs", 1),
            "optimizer": "Adam",
            "model": "SimpleCNN",
            "device": str(device),
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "number_of_communication_rounds": num_rounds,
            "per_round_test_accuracy": acc_history,
            "final_test_accuracy": final_acc,
            "best_test_accuracy": max(acc_history) if acc_history else 0.0,
            "epsilon": final_epsilon,
            "best_alpha": final_best_alpha,
            "runtime_seconds": runtime,
            "run_status": "Success",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(result_data, f, indent=4)
            
        with open(os.path.join(out_dir, "rounds.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "dp_steps", "epsilon", "best_alpha", "test_acc"])
            for stat in round_stats:
                writer.writerow([stat["round"], stat["dp_steps"], stat["epsilon"], stat["best_alpha"], stat["test_acc"]])
                
        # Comparison with Stage 2 and Stage 3 - we disable automatic plotting in grid search mode if output_dir is customized
        if args.output_dir == "results/stage4":
            with open(os.path.join(out_dir, "runtime.json"), "a") as f:
                f.write(json.dumps({"mode": mode, "runtime_seconds": runtime}) + "\n")
                
            baseline = load_baseline_result(mode, args.seed)
            stage3_path = os.path.join("results", "stage3", f"federation_{mode}_seed{args.seed}.json")
            stage3_accs = []
            if os.path.exists(stage3_path):
                with open(os.path.join("results", "stage3", f"federation_{mode}_seed{args.seed}.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        stage3_accs.append(float(row["test_acc"]))
                        
            plt.figure(figsize=(10, 6))
            rounds = range(1, len(acc_history) + 1)
            plt.plot(rounds, acc_history, label=f"DP-FedAvg (Stage 4, ε={final_epsilon:.2f})", marker='o')
            
            if stage3_accs:
                s3_rounds = range(1, len(stage3_accs) + 1)
                plt.plot(s3_rounds, stage3_accs, label="FedAvg (Stage 3)", linestyle='-.', marker='x')
                
            if baseline is not None:
                baseline_csv_path = os.path.join("results", "stage2", f"baseline_{mode}_seed{args.seed}.csv")
                baseline_accs = []
                if os.path.exists(baseline_csv_path):
                    with open(baseline_csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            baseline_accs.append(float(row["test_acc"]))
                if baseline_accs:
                    b_rounds = range(1, len(baseline_accs) + 1)
                    plt.plot(b_rounds, baseline_accs, label="Centralized (Stage 2)", linestyle='--', color='gray')
                    
            plt.xlabel('Communication Round / Epoch')
            plt.ylabel('Test Accuracy (%)')
            plt.title(f'Privacy vs Utility Comparison ({mode.capitalize()})\nNon-IID (alpha={alpha})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"comparison_{mode}_seed{args.seed}.png"))
                
        print("\n--- DP-SGD FINAL RESULTS ---")
        print(f"sigma={CONFIG.get('noise_multiplier', 1.0)}, C={CONFIG.get('max_grad_norm', 1.0)}")
        print(f"Privacy Guarantee: epsilon={final_epsilon:.4f} (delta=1e-5, alpha={final_best_alpha})")
        print(f"Final Aggregated Accuracy: {final_acc:.2f}%")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
