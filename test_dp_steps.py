import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

def run_test(dataset_size, batch_size):
    data = torch.randn(dataset_size, 3)
    target = torch.randint(0, 2, (dataset_size,))
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )
    
    steps = 0
    for _ in dp_loader:
        steps += 1
        
    print(f"Dataset: {dataset_size}, Batch: {batch_size}")
    print(f"  len(loader) = {len(loader)}")
    print(f"  len(dp_loader) = {len(dp_loader)}")
    print(f"  Actual steps taken = {steps}")
    print(f"  Sample rate = {dp_loader.sample_rate}")
    
    assert steps == len(dp_loader), f"Iterated steps {steps} does not match len(dp_loader) {len(dp_loader)}"
    assert hasattr(dp_loader, "sample_rate"), "dp_loader missing sample_rate"

run_test(100, 32)
run_test(333, 32)
run_test(500, 32)
print("test_dp_steps.py passed.")

