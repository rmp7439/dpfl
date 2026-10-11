
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import argparse
import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
import datetime
import time
import re
import random

from config import SUBSET_CONFIG, FULL_CONFIG
from model import SimpleCNN

def get_data(subset_size=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    if subset_size is not None and subset_size < len(trainset):
        indices = np.random.choice(len(trainset), subset_size, replace=False)
        trainset = Subset(trainset, indices)
        
        # also subset testset proportionally
        test_size = int(subset_size * 0.2)
        test_indices = np.random.choice(len(testset), test_size, replace=False)
        testset = Subset(testset, test_indices)
        
    return trainset, testset

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f}\tAccuracy: {acc:.2f}%")
    return avg_loss, acc

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")
    return test_loss, acc

def resolve_output_paths(mode: str, seed: int, tag: str = None, output_dir: str = None):
    """
    Safely resolves output JSON and CSV paths for centralized baseline.
    Validates tags to prevent directory traversal and illegal characters.
    """
    if tag is not None:
        tag = tag.strip()
        if not tag:
            raise ValueError("Provided --tag cannot be empty or whitespace.")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", tag):
            raise ValueError(f"Invalid --tag '{tag}': must contain only alphanumeric characters, underscores, or hyphens.")

    if output_dir is not None:
        out_dir = os.path.abspath(output_dir)
    else:
        out_dir = os.path.join(project_root, "results", "stage2")

    prefix = f"baseline_{mode}_seed{seed}"
    if tag:
        filename_base = f"{prefix}_{tag}"
    else:
        filename_base = prefix

    json_path = os.path.join(out_dir, f"{filename_base}.json")
    csv_path = os.path.join(out_dir, f"{filename_base}.csv")
    return out_dir, json_path, csv_path

def main():
    parser = argparse.ArgumentParser(description="Run Centralized Baseline")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tag", type=str, default=None, help="Tag to append to output filenames (e.g. 'rerun')")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for results")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    args = parser.parse_args()

    mode = "full" if args.full else "subset"
    out_dir, json_path, csv_path = resolve_output_paths(mode, args.seed, args.tag, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Collision detection before expensive training begins
    if not args.overwrite:
        if os.path.exists(json_path):
            raise FileExistsError(f"Destination JSON output already exists: {json_path}. Use --overwrite to allow replacement.")
        if os.path.exists(csv_path):
            raise FileExistsError(f"Destination CSV output already exists: {csv_path}. Use --overwrite to allow replacement.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from copy import deepcopy
    run_config = deepcopy(FULL_CONFIG if args.full else SUBSET_CONFIG)
    run_config["seed"] = args.seed
    
    if args.full:
        print("Running centralized baseline on FULL CIFAR-10")
        trainset, testset = get_data(subset_size=None)
    else:
        print(f"Running centralized baseline on subset ({run_config.get('num_samples', 1000)} samples)")
        trainset, testset = get_data(subset_size=run_config.get("num_samples", 1000))
        
    batch_size = run_config.get("batch_size", 64 if args.full else 32)
    epochs = run_config.get("central_epochs", 30 if args.full else 2)
    lr = run_config.get("central_lr", 0.001)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    
    best_test_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, device, train_loader, optimizer, epoch)
        te_loss, te_acc = test(model, device, test_loader)
        scheduler.step()
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            
    runtime = time.time() - start_time
    
    # Save JSON config & results
    result_data = {
        "dataset_mode": mode,
        "num_train_samples": len(trainset),
        "number_of_training_samples": len(trainset),
        "number_of_test_samples": len(testset),
        "seed": args.seed,
        "tag": args.tag,
        "model_name": "SimpleCNN+GroupNorm",
        "optimizer": "Adam+CosineAnnealingLR",
        "learning_rate": lr,
        "batch_size": batch_size,
        "number_of_epochs": epochs,
        "final_test_accuracy": history["test_acc"][-1],
        "best_test_accuracy": best_test_acc,
        "convergence_definition": f"Best test accuracy achieved within {epochs} epochs",
        "device": str(device),
        "runtime_seconds": runtime,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Atomic write to temporary files then rename
    tmp_json = json_path + ".tmp"
    tmp_csv = csv_path + ".tmp"
    try:
        with open(tmp_json, "w") as f:
            json.dump(result_data, f, indent=4)
            
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
            for i in range(epochs):
                writer.writerow([i+1, history["train_loss"][i], history["train_acc"][i], 
                                 history["test_loss"][i], history["test_acc"][i]])
                                 
        os.replace(tmp_json, json_path)
        os.replace(tmp_csv, csv_path)
    finally:
        if os.path.exists(tmp_json):
            try:
                os.remove(tmp_json)
            except Exception:
                pass
        if os.path.exists(tmp_csv):
            try:
                os.remove(tmp_csv)
            except Exception:
                pass
    
    print(f"\nFinal Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Results saved to: {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
