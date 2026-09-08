
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

from dpfl.config import SUBSET_CONFIG, FULL_CONFIG
from dpfl.model import SimpleCNN

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

def main():
    parser = argparse.ArgumentParser(description="Run Centralized Baseline")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    if args.full:
        print("Running centralized baseline on FULL CIFAR-10")
        CONFIG = FULL_CONFIG
        trainset, testset = get_data(subset_size=None)
    else:
        CONFIG = SUBSET_CONFIG
        print(f"Running centralized baseline on subset ({CONFIG.get('num_samples', 1000)} samples)")
        trainset, testset = get_data(subset_size=CONFIG.get("num_samples", 1000))
        
    batch_size = CONFIG.get("batch_size", 32)
    epochs = CONFIG.get("central_epochs", 2)
    lr = CONFIG.get("central_lr", 0.001)
    
    mode = "full" if args.full else "subset"

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
    
    # Persistence
    out_dir = os.path.join("results", "stage2")
    os.makedirs(out_dir, exist_ok=True)
    
    prefix = f"baseline_{mode}"
    
    # Save JSON config & results
    result_data = {
        "dataset_mode": mode,
        "number_of_training_samples": len(trainset),
        "number_of_test_samples": len(testset),
        "seed": args.seed,
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
    
    with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.json"), "w") as f:
        json.dump(result_data, f, indent=4)
        
    # Save CSV history
    with open(os.path.join(out_dir, f"{prefix}_seed{args.seed}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for i in range(epochs):
            writer.writerow([i+1, history["train_loss"][i], history["train_acc"][i], 
                             history["test_loss"][i], history["test_acc"][i]])
                             
    # Plots
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label='Train Loss')
    plt.plot(epochs_range, history["test_loss"], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{mode.capitalize()} Baseline Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label='Train Acc')
    plt.plot(epochs_range, history["test_acc"], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{mode.capitalize()} Baseline Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_plots_seed{args.seed}.png"))
    
    print(f"\nFinal Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
