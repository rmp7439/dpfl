import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
import argparse
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
    plot_class_distribution(client_indices, labels, 10, filename)
    
if __name__ == "__main__":
    main()
