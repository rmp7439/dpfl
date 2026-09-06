import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
import argparse
import json
from config import CONFIG

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

def plot_class_distribution(client_indices, labels, num_classes, filename):
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
        
    ax.set_title(f"Class Distribution per Client (Dirichlet, alpha={CONFIG.get('alpha', 0.1)})")
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Client ID")
    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    return client_class_counts

def validate_and_save_json(client_indices, labels, num_classes, client_class_counts, alpha, num_samples, is_full):
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
        "alpha": alpha,
        "seed": 42,
        "samples_per_client": {k: len(v) for k, v in client_indices.items()},
        "class_counts_per_client": {k: client_class_counts[k].tolist() for k in range(len(client_indices))},
        "all_samples_assigned_exactly_once": exactly_once,
        "missing_samples": len(missing),
        "duplicated_samples": duplicates,
        "heterogeneity_metric_mean_std": heterogeneity_score
    }
    
    json_filename = f"stage1_validation_{'full' if is_full else 'subset'}.json"
    with open(json_filename, "w") as f:
        json.dump(validation_data, f, indent=4)
        
    print(f"Saved validation data to {json_filename}")
    assert exactly_once, "Data split failed: missing or duplicated samples!"

def main():
    parser = argparse.ArgumentParser(description="Run Dirichlet Data Split")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = CONFIG.get("num_samples", 1000)
    num_clients = CONFIG.get("num_clients", 3)
    alpha = CONFIG.get("alpha", 0.1)
    
    if args.full:
        print(f"Running Dirichlet split on FULL CIFAR-10, clients: {num_clients}, alpha: {alpha}")
        dataset = get_cifar10(subset_size=None)
        filename = "full_dirichlet_split.png"
    else:
        print(f"Running Dirichlet split on subset ({num_samples} samples), clients: {num_clients}, alpha: {alpha}")
        dataset = get_cifar10(subset_size=num_samples)
        filename = "subset_dirichlet_split.png"

    client_indices, labels = dirichlet_split(dataset, num_clients, alpha)
    client_class_counts = plot_class_distribution(client_indices, labels, 10, filename)
    
    validate_and_save_json(client_indices, labels, 10, client_class_counts, alpha, num_samples, args.full)
    
if __name__ == "__main__":
    main()
