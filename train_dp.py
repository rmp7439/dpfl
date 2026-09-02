import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import argparse
import numpy as np

from config import CONFIG
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
    return acc

def main():
    parser = argparse.ArgumentParser(description="Run Centralized DP Baseline")
    parser.add_argument("--full", action="store_true", help="Run on full CIFAR-10 instead of subset")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = CONFIG.get("batch_size", 32)
    epochs = CONFIG.get("central_epochs", 2)
    lr = CONFIG.get("dp_lr", 0.00025)
    num_samples = CONFIG.get("num_samples", 1000)
    
    if args.full:
        print("Running centralized DP baseline on FULL CIFAR-10")
        trainset, testset = get_data(subset_size=None)
        # For full dataset we want more epochs normally, but let's stick to config or scale
        epochs = 10  # Override to 10 for a decent baseline
    else:
        print(f"Running centralized DP baseline on subset ({num_samples} samples)")
        trainset, testset = get_data(subset_size=num_samples)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=CONFIG.get("noise_multiplier", 1.0),
        max_grad_norm=CONFIG.get("max_grad_norm", 1.0),
    )
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        
if __name__ == "__main__":
    main()
