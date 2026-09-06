import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from model import SimpleCNN
import warnings
warnings.simplefilter("always")

device = torch.device("cpu")
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = torch.randn(32, 3, 32, 32)
target = torch.randint(0, 10, (32,))
dataset = TensorDataset(data, target)
loader = DataLoader(dataset, batch_size=16)

privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    # Let's inspect per-sample gradients
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample"):
            print(f"{name} has grad_sample of shape {param.grad_sample.shape}")
        else:
            print(f"{name} has NO grad_sample")
    
    optimizer.step()
    break
print("Success!")
