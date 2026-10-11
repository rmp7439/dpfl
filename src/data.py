import sys
import os
import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_cifar10(subset_size=None):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    if subset_size is not None and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    
    return dataset

def dirichlet_split(dataset, num_clients, alpha):
    """
    Splits a dataset among clients using a Dirichlet distribution over classes.
    """
    if isinstance(dataset, Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        labels = np.array(dataset.targets)
    
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(c_idx)).astype(int)
        
        diff = len(c_idx) - counts.sum()
        if diff > 0:
            for i in np.random.choice(num_clients, diff, p=proportions, replace=True):
                counts[i] += 1
                
        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(c_idx[start:end].tolist())
            start = end

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    return client_indices, labels

def plot_class_distribution(client_indices, labels, num_classes, alpha, filename):
    """
    Plots the class distribution for each client.
    """
    num_clients = len(client_indices)
    client_class_counts = np.zeros((num_clients, num_classes))
    
    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        for c in range(num_classes):
            client_class_counts[client_id, c] = np.sum(client_labels == c)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(num_clients)
    client_ids = [f"Client {i+1}" for i in range(num_clients)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for c in range(num_classes):
        ax.bar(client_ids, client_class_counts[:, c], bottom=bottom, label=f"Class {c}", color=colors[c])
        bottom += client_class_counts[:, c]
        
    ax.set_title(f"Class Distribution per Client (Dirichlet, alpha={alpha})")
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Client ID")
    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if os.path.exists(filename):
        print(f"[Collision Policy] Existing distribution plot found at {filename}. Updating plot.")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")
    
    return client_class_counts

def validate_and_save_json(client_indices, labels, num_classes, client_class_counts, alpha, num_samples, is_full, json_path, seed=42):
    # Check coverage and duplication
    all_assigned_indices = []
    for indices in client_indices.values():
        all_assigned_indices.extend(indices)
        
    unique_assigned = set(all_assigned_indices)
    expected_indices = set(range(len(labels)))
    
    missing = expected_indices - unique_assigned
    duplicates = len(all_assigned_indices) - len(unique_assigned)
    
    exactly_once = (len(missing) == 0) and (duplicates == 0)
    
    # Simple heterogeneity metric: standard deviation of class counts across clients
    # High std dev means high heterogeneity
    heterogeneity_score = float(np.mean(np.std(client_class_counts, axis=0)))
    
    validation_data = {
        "dataset_mode": "full" if is_full else "subset",
        "number_of_samples": len(labels),
        "number_of_clients": len(client_indices),
        "num_clients": len(client_indices),
        "alpha": alpha,
        "seed": seed,
        "samples_per_client": {str(k): len(v) for k, v in client_indices.items()},
        "client_sample_distribution": [len(client_indices[k]) for k in range(len(client_indices))],
        "class_counts_per_client": {str(k): client_class_counts[k].tolist() for k in range(len(client_indices))},
        "all_samples_assigned_exactly_once": exactly_once,
        "missing_samples": len(missing),
        "duplicated_samples": duplicates,
        "heterogeneity_metric_mean_std": heterogeneity_score
    }
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as ef:
                old_info = json.load(ef)
            print(f"[Collision Policy] Existing artifact found at {json_path} (alpha={old_info.get('alpha')}, seed={old_info.get('seed')}, clients={old_info.get('number_of_clients')}). Updating file.")
        except Exception:
            print(f"[Collision Policy] Existing file found at {json_path}. Updating file.")
            
    with open(json_path, "w") as f:
        json.dump(validation_data, f, indent=4)
        
    print(f"Saved validation data to {json_path}")
    assert exactly_once, "Data split failed: missing or duplicated samples!"

def main():
    parser = argparse.ArgumentParser(description="Run Dirichlet Data Split")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha parameter (default: 0.1)")
    parser.add_argument("--num-clients", type=int, default=None, help="Number of clients (default from config)")
    args = parser.parse_args()

    try:
        from src.config import FULL_CONFIG, SUBSET_CONFIG
    except ModuleNotFoundError:
        from config import FULL_CONFIG, SUBSET_CONFIG
    active_config = FULL_CONFIG if args.full else SUBSET_CONFIG
    
    alpha = args.alpha if args.alpha is not None else active_config.get("alpha", 0.1)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
        
    default_clients = active_config.get("num_clients", 5 if args.full else 3)
    num_clients = args.num_clients if args.num_clients is not None else default_clients
    if num_clients <= 0:
        raise ValueError(f"num_clients must be a positive integer, got {num_clients}")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    mode = "full" if args.full else "subset"
    print(f"Running Dirichlet split on {mode.upper()} CIFAR-10, clients: {num_clients}, alpha: {alpha}, seed: {args.seed}")
    
    if args.full:
        dataset = get_cifar10(subset_size=None)
    else:
        dataset = get_cifar10(subset_size=SUBSET_CONFIG["num_samples"])

    client_indices, labels = dirichlet_split(dataset, num_clients, alpha)
    
    out_dir = os.path.join("results", "stage1")
    os.makedirs(out_dir, exist_ok=True)
    
    # Parameter-specific paths
    param_json = os.path.join(out_dir, f"split_validation_{mode}_alpha{alpha}.json")
    param_png = os.path.join(out_dir, f"client_class_distribution_{mode}_alpha{alpha}.png")
    
    client_class_counts = plot_class_distribution(client_indices, labels, 10, alpha, param_png)
    validate_and_save_json(client_indices, labels, 10, client_class_counts, alpha, len(labels), args.full, param_json, seed=args.seed)
    
    # Preserve canonical alpha=0.1 filenames if standard client count and seed 42
    is_canonical = (abs(alpha - 0.1) < 1e-6) and (num_clients == default_clients) and (args.seed == 42)
    if is_canonical:
        canonical_json = os.path.join(out_dir, f"split_validation_{mode}.json")
        canonical_png = os.path.join(out_dir, f"client_class_distribution_{mode}.png")
        import shutil
        shutil.copyfile(param_json, canonical_json)
        shutil.copyfile(param_png, canonical_png)
        print(f"Preserved canonical files: {canonical_json}, {canonical_png}")
    
if __name__ == "__main__":
    main()
