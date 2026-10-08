
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Context
from typing import Dict, List, Tuple, Optional, Union
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

from src.config import SUBSET_CONFIG, FULL_CONFIG, get_client_resources
from src.model import SimpleCNN
from scripts.train_baseline import get_data, train, test
from src.data import dirichlet_split
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

# Global data placeholders for simulation
GLOBAL_TRAINSET = None
GLOBAL_TESTSET = None
CLIENT_INDICES = None
USE_DP = False

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, test_loader, device, use_dp=False, run_config=None):
        self.run_config = run_config or {}
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_dp = use_dp
        self.setup_time = 0.0
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        import time
        start_fit = time.time()
        self.set_parameters(parameters)
        local_epochs = self.run_config.get("local_epochs", 1)
        
        if self.use_dp:
            dp_setup_start = time.time()
            lr = self.run_config.get("dp_lr", 0.01)
            opt_name = self.run_config.get("fed_optimizer", "SGD")
            if opt_name == "SGD":
                optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.Adam(self.net.parameters(), lr=lr)
            
            from opacus import PrivacyEngine
            import warnings
            
            warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.*")
            
            privacy_engine = PrivacyEngine(accountant="rdp")
            self.net, optimizer, train_loader = privacy_engine.make_private(
                module=self.net,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.run_config.get("noise_multiplier", 1.0),
                max_grad_norm=self.run_config.get("max_grad_norm", 1.0),
            )
            dp_setup_time = time.time() - dp_setup_start
            
            actual_sample_rate = getattr(train_loader, "sample_rate", -1.0)
            actual_steps = len(train_loader)
            
            grad_sample_valid = False
            grad_shapes = {}
            actual_batch_size = None
            opt_class = type(optimizer).__name__
            
            out_dir = self.run_config.get("out_dir", "results/stage4")
            if not os.path.exists(os.path.join(out_dir, "validation.json")):
                batch_x, batch_y = next(iter(train_loader))
                actual_batch_size = len(batch_x)
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(self.net(batch_x.to(self.device)), batch_y.to(self.device))
                loss.backward()
                for name, param in self.net.named_parameters():
                    if param.requires_grad and hasattr(param, "grad_sample"):
                        grad_sample_valid = True
                        grad_shapes[name] = list(param.grad_sample.shape)
                optimizer.zero_grad()
            
            train_start = time.time()
            total_loss = 0.0
            for epoch in range(1, local_epochs + 1):
                epoch_loss, _ = train(self.net, self.device, train_loader, optimizer, epoch)
                total_loss += epoch_loss
            train_time = time.time() - train_start
            
            avg_train_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
                
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"[Client {self.cid}] Opacus Accountant Epsilon: {epsilon:.4f}")
            
            if not os.path.exists(os.path.join(out_dir, "validation.json")):
                os.makedirs(out_dir, exist_ok=True)
                val_data = {
                    "grad_sample_present": grad_sample_valid,
                    "grad_sample_shapes": grad_shapes,
                    "clipping_C": self.run_config.get("max_grad_norm", 1.0),
                    "noise_sigma": self.run_config.get("noise_multiplier", 1.0),
                    "actual_batch_size_probed": actual_batch_size,
                    "optimizer_class": opt_class
                }
                with open(os.path.join(out_dir, "validation.json"), "w") as f:
                    json.dump(val_data, f, indent=4)
                    
            metrics = {
                "client_id": str(self.cid),
                "sample_rate": float(actual_sample_rate),
                "dp_steps": int(actual_steps * local_epochs),
                "sigma": float(self.run_config.get("noise_multiplier", 1.0)),
                "C": float(self.run_config.get("max_grad_norm", 1.0)),
                "setup_time": self.setup_time,
                "dp_setup_time": dp_setup_time,
                "train_time": train_time,
                "fit_total_time": time.time() - start_fit,
                "train_loss": avg_train_loss
            }
            self.net = self.net._module
        else:
            lr = self.run_config.get("fed_lr", 0.01)
            opt_name = self.run_config.get("fed_optimizer", "SGD")
            if opt_name == "SGD":
                optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(self.net.parameters(), lr=lr)
                
            train_start = time.time()
            total_loss = 0.0
            for epoch in range(1, local_epochs + 1):
                epoch_loss, _ = train(self.net, self.device, self.train_loader, optimizer, epoch)
                total_loss += epoch_loss
            train_time = time.time() - train_start
            
            avg_train_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
            metrics = {
                "client_id": str(self.cid),
                "setup_time": self.setup_time,
                "train_time": train_time,
                "fit_total_time": time.time() - start_fit,
                "train_loss": avg_train_loss
            }
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics


        
    def evaluate(self, parameters, config):
        import time
        start_eval = time.time()
        self.set_parameters(parameters)
        te_loss, acc = test(self.net, self.device, self.test_loader)
        eval_time = time.time() - start_eval
        return float(te_loss), len(self.test_loader.dataset), {"accuracy": acc, "eval_time": eval_time}

def client_fn_factory(run_config):
    def client_fn(context: Context) -> fl.client.Client:
        """Create a Flower client representing a single organization."""
        import time
        start_setup = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SimpleCNN().to(device)
        
        # Get the client's subset of data
        try:
            client_id = int(context.node_config["partition-id"])
        except (KeyError, AttributeError, TypeError):
            client_id = int(context.node_id)
            
        cid = str(client_id)
        indices = CLIENT_INDICES[client_id]
        
        client_dataset = torch.utils.data.Subset(GLOBAL_TRAINSET, indices)
        
        train_loader = DataLoader(client_dataset, batch_size=run_config.get("batch_size", 32), shuffle=True)
        test_loader = DataLoader(GLOBAL_TESTSET, batch_size=run_config.get("batch_size", 32), shuffle=False)
        
        setup_time = time.time() - start_setup
        client = FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP, run_config=run_config)
        client.setup_time = setup_time
        return client.to_client()
    return client_fn


def fit_metrics_aggregation_fn(results: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    if not results or not results[0][1].get("client_id"):
        return {}
    client_stats = []
    round_timings = []
    for _, m in results:
        client_stats.append({
            "client_id": m["client_id"],
            "sample_rate": m.get("sample_rate", 0),
            "dp_steps": m.get("dp_steps", 0),
            "sigma": m.get("sigma", 0),
            "C": m.get("C", 0),
            "train_loss": m.get("train_loss", 0.0)
        })
        round_timings.append({
            "client_id": m["client_id"],
            "setup_time": m.get("setup_time", 0),
            "dp_setup_time": m.get("dp_setup_time", 0),
            "train_time": m.get("train_time", 0),
            "fit_total_time": m.get("fit_total_time", 0)
        })
    os.makedirs("results/profiling", exist_ok=True)
    with open("results/profiling/timings.jsonl", "a") as f:
        f.write(json.dumps({"type": "fit_metrics", "timings": round_timings}) + "\n")
        
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_loss = sum([num_examples * m.get("train_loss", 0.0) for num_examples, m in results])
    avg_train_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
    
    return {"client_stats": json.dumps(client_stats), "train_loss": avg_train_loss}

def get_evaluate_fn(testset, device, run_config):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        import time
        start_eval = time.time()
        net = SimpleCNN().to(device)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        
        test_loader = DataLoader(testset, batch_size=run_config.get("batch_size", 32), shuffle=False)
        loss, acc = test(net, device, test_loader)
        
        eval_time = time.time() - start_eval
        os.makedirs("results/profiling", exist_ok=True)
        with open("results/profiling/timings.jsonl", "a") as f:
            f.write(json.dumps({"type": "evaluate_metrics", "timings": [eval_time]}) + "\n")
            
        return float(loss), {"accuracy": acc}
    return evaluate

class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, run_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config or {}
        
    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            out_dir = self.run_config.get("out_dir", "results/checkpoints")
            ckpt_dir = os.path.join(out_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            np.savez(os.path.join(ckpt_dir, f"round_{server_round}.npz"), *ndarrays)
            
            client_stats = []
            for _, fit_res in results:
                client_stats.append({
                    "client_id": fit_res.metrics.get("client_id", ""),
                    "sample_rate": fit_res.metrics.get("sample_rate", 0),
                    "dp_steps": fit_res.metrics.get("dp_steps", 0),
                    "sigma": fit_res.metrics.get("sigma", 0),
                    "C": fit_res.metrics.get("C", 0),
                    "train_loss": fit_res.metrics.get("train_loss", 0.0)
                })
            with open(f"results/checkpoints/round_{server_round}_stats.json", "w") as f:
                json.dump(client_stats, f)
                
        return aggregated_parameters, aggregated_metrics


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
    parser.add_argument("--alpha", type=float, help="Dirichlet alpha for non-IID split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--rounds", type=int, help="Number of communication rounds (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()
    
    USE_DP = args.enable_dp
    
    from copy import deepcopy
    run_config = deepcopy(FULL_CONFIG if args.full else SUBSET_CONFIG)
    run_config["out_dir"] = args.output_dir or ("results/stage4" if args.enable_dp else "results/stage3")
    
        
    if args.sigma is not None:
        run_config["noise_multiplier"] = args.sigma
    if args.C is not None:
        run_config["max_grad_norm"] = args.C
    if args.rounds is not None:
        if args.rounds <= 0:
            raise ValueError("--rounds must be a positive integer")
        run_config["num_rounds"] = args.rounds
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    num_samples = run_config.get("num_samples", 1000)
    num_clients = run_config.get("num_clients", 3)
    
    if args.alpha is not None:
        if args.alpha <= 0:
            raise ValueError("alpha must be > 0")
        run_config["alpha"] = args.alpha
        
    alpha = run_config.get("alpha", 0.1)
    
    mode = "full" if args.full else "subset"
    
    if args.full:
        print(f"Running Federation on FULL CIFAR-10, clients: {num_clients}, alpha: {alpha}")
        GLOBAL_TRAINSET, GLOBAL_TESTSET = get_data(subset_size=None)
    else:
        print(f"Running Federation on subset ({num_samples} samples), clients: {num_clients}, alpha: {alpha}")
        GLOBAL_TRAINSET, GLOBAL_TESTSET = get_data(subset_size=num_samples)
        
    num_rounds = run_config.get("num_rounds", 2)
        
    CLIENT_INDICES, _ = dirichlet_split(GLOBAL_TRAINSET, num_clients, alpha)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    strategy = CheckpointingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_fn=get_evaluate_fn(GLOBAL_TESTSET, device, run_config),
        run_config=run_config,
    )
    
    # Checkpoints are saved by CheckpointingFedAvg. 
    # Resume logic is omitted because Flower's start_simulation does not natively 
    # support resuming the round counter. Injecting initial_parameters would restart 
    # the simulation from round 1, silently duplicating privacy accounting composition.
    # See docs/COMPUTE_OPTIMIZATION_REPORT.md for details.
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    import opacus
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Opacus Version: {opacus.__version__}")
    print(f"Flower Version: {fl.__version__}")
    
    start_time = time.time()
    
    try:
        import ray._private.utils
        ray._private.utils.set_kill_child_on_death_win32 = lambda *args, **kwargs: None
    except Exception:
        pass
        
    gpu_available = torch.cuda.is_available()
    client_resources = get_client_resources(num_clients, USE_DP, gpu_available)
    print(f"\n--- Resource Configuration ---")
    print(f"Execution Device: {device}")
    print(f"Number of Virtual Clients: {num_clients}")
    print(f"Detected Physical GPUs: {torch.cuda.device_count() if gpu_available else 0}")
    print(f"Client Resource Allocation: {client_resources}\n")
        
    history = fl.simulation.start_simulation(
        client_fn=client_fn_factory(run_config),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    final_acc = 0.0
    acc_history = []
    
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        acc_history = [acc for r, acc in history.metrics_centralized["accuracy"] if r > 0]
        final_acc = acc_history[-1] if acc_history else 0.0
    elif history.metrics_distributed and "accuracy" in history.metrics_distributed:
        acc_history = [acc for r, acc in history.metrics_distributed["accuracy"] if r > 0]
        final_acc = acc_history[-1] if acc_history else 0.0
        
    train_loss_history = []
    if history.metrics_distributed_fit and "train_loss" in history.metrics_distributed_fit:
        train_loss_history = [loss for r, loss in history.metrics_distributed_fit["train_loss"] if r > 0]
        
    if not USE_DP:
        # Persistence for Stage 3
        out_dir = args.output_dir or os.path.join("results", "stage3")
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"federation_{mode}"
        
        result_data = {
            "dp_enabled": False,
            "model": "SimpleCNN",
            "seed": args.seed,
            "dataset_mode": mode,
            "num_train_samples": len(GLOBAL_TRAINSET),
            "number_of_test_samples": len(GLOBAL_TESTSET),
            "num_clients": num_clients,
            "alpha": alpha,
            "batch_size": run_config.get("batch_size", 32),
            "local_epochs": run_config.get("local_epochs", 1),
            "optimizer": run_config.get("fed_optimizer", "SGD"),
            "learning_rate": run_config.get("fed_lr", 0.01),
            "number_of_communication_rounds": num_rounds,
            "samples_per_client": {k: len(v) for k, v in CLIENT_INDICES.items()},
            "per_round_test_accuracy": acc_history,
            "per_round_train_loss": train_loss_history,
            "final_test_accuracy": final_acc,
            "best_test_accuracy": max(acc_history) if acc_history else 0.0,
            "execution_status": "Success",
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.json"), "w") as f:
            json.dump(result_data, f, indent=4)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(result_data, f, indent=4)
            
        with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "train_loss", "test_acc"])
            for i, acc in enumerate(acc_history):
                t_loss = train_loss_history[i] if i < len(train_loss_history) else 0.0
                writer.writerow([i+1, t_loss, acc])
                
        with open(os.path.join(out_dir, "rounds.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "train_loss", "test_acc"])
            for i, acc in enumerate(acc_history):
                t_loss = train_loss_history[i] if i < len(train_loss_history) else 0.0
                writer.writerow([i+1, t_loss, acc])
                

        
        print("\n--- STAGE 3 FINAL RESULTS ---")
        print(f"Final Aggregated Accuracy: {final_acc:.2f}%")
        print(f"Results saved to {out_dir}")
        
    else:
        # Calculate RDP for Stage 4/5 based on actual client accounting
        alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        client_rdp_state = {str(i): np.zeros(len(alphas)) for i in range(num_clients)}
        client_cumulative_steps = {str(i): 0 for i in range(num_clients)}
        
        round_stats = []
        fit_metrics = history.metrics_distributed_fit.get("client_stats", [])
        
        for r_idx, acc in enumerate(acc_history):
            round_num = r_idx + 1
            stats_str = None
            for r, s in fit_metrics:
                if r == round_num:
                    stats_str = s
                    break
                    
            if not stats_str:
                # If flwr simulation history is missing client_stats, attempt to recover from checkpoint
                chk_path = f"results/checkpoints/round_{round_num}_stats.json"
                if os.path.exists(chk_path):
                    with open(chk_path, "r") as f:
                        client_stats = json.load(f)
                else:
                    continue
            else:
                client_stats = json.loads(stats_str)
            round_epsilons = {}
            
            for c_stat in client_stats:
                cid = str(c_stat["client_id"])
                q = float(c_stat["sample_rate"])
                steps = int(c_stat["dp_steps"])
                sigma = float(c_stat["sigma"])
                
                rdp_increment = compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=alphas)
                client_rdp_state[cid] += rdp_increment
                client_cumulative_steps[cid] += steps
                
                eps, best_alpha = get_privacy_spent(orders=alphas, rdp=client_rdp_state[cid], delta=1e-5)
                round_epsilons[cid] = {
                    "epsilon": eps, 
                    "best_alpha": best_alpha, 
                    "cumulative_steps": client_cumulative_steps[cid], 
                    "sample_rate": q, 
                    "round_steps": steps
                }
                
            worst_client = max(round_epsilons.keys(), key=lambda k: round_epsilons[k]["epsilon"])
            worst_eps = round_epsilons[worst_client]["epsilon"]
            worst_alpha = round_epsilons[worst_client]["best_alpha"]
            
            round_stats.append({
                "round": round_num,
                "global_epsilon": worst_eps,
                "best_alpha": worst_alpha,
                "test_acc": acc,
                "train_loss": train_loss_history[r_idx] if r_idx < len(train_loss_history) else 0.0,
                "worst_client": worst_client,
                "client_details": round_epsilons
            })
            
        final_epsilon = round_stats[-1]["global_epsilon"] if round_stats else 0
        final_best_alpha = round_stats[-1]["best_alpha"] if round_stats else 0
        total_steps = max(client_cumulative_steps.values()) if client_cumulative_steps else 0
        max_sample_rate = max([round_stats[-1]["client_details"][cid]["sample_rate"] for cid in client_cumulative_steps]) if round_stats else 0
        
        out_dir = args.output_dir or "results/stage4"
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"dp_{mode}"
        
        result_data = {
            "dataset_mode": mode,
            "num_train_samples": len(GLOBAL_TRAINSET),
            "number_of_test_samples": len(GLOBAL_TESTSET),
            "num_clients": num_clients,
            "alpha": alpha,
            "sigma": run_config.get("noise_multiplier", 1.0),
            "C": run_config.get("max_grad_norm", 1.0),
            "delta": 1e-5,
            "sample_rate": max_sample_rate,
            "total_dp_steps": total_steps,
            "batch_size": run_config.get("batch_size", 32),
            "local_epochs": run_config.get("local_epochs", 1),
            "optimizer": run_config.get("fed_optimizer", "SGD"),
            "learning_rate": run_config.get("dp_lr", 0.01),
            "model": "SimpleCNN",
            "device": str(device),
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "number_of_communication_rounds": num_rounds,
            "seed": args.seed,
            "per_round_test_accuracy": acc_history,
            "per_round_train_loss": train_loss_history,
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
            writer.writerow(["round", "max_dp_steps", "global_epsilon", "best_alpha", "train_loss", "test_acc"])
            for stat in round_stats:
                max_steps = max([d["round_steps"] for d in stat["client_details"].values()]) if "client_details" in stat else 0
                writer.writerow([stat["round"], max_steps, stat["global_epsilon"], stat["best_alpha"], stat["train_loss"], stat["test_acc"]])
                

                
        print("\n--- DP-SGD FINAL RESULTS ---")
        print(f"sigma={run_config.get('noise_multiplier', 1.0)}, C={run_config.get('max_grad_norm', 1.0)}")
        print(f"Privacy Guarantee: epsilon={final_epsilon:.4f} (delta=1e-5, alpha={final_best_alpha})")
        print(f"Final Aggregated Accuracy: {final_acc:.2f}%")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()