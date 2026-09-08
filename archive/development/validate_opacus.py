import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from model import SimpleCNN
import warnings

def main():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    original_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dataset_size = 1000
    batch_size = 32
    data = torch.randn(dataset_size, 3, 32, 32)
    target = torch.randint(0, 10, (dataset_size,))
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch_size)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=original_optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    print("--- VALIDATION START ---")
    print(f"make_private() succeeded: {True}")
    print(f"returned DataLoader is DPDataLoader: {isinstance(dp_loader, DPDataLoader)}")
    print(f"DPDataLoader sample_rate: {dp_loader.sample_rate}")
    
    # 1 / len(data_loader)
    one_over_len = 1.0 / len(loader)
    # batch_size / dataset_size
    bs_over_ds = batch_size / dataset_size
    print(f"1 / len(original_loader): {one_over_len}")
    print(f"batch_size / dataset_size: {bs_over_ds}")

    print(f"optimizer is DPOptimizer: {isinstance(optimizer, DPOptimizer)}")

    model.train()
    
    pre_step_epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    print(f"Accountant epsilon before step: {pre_step_epsilon}")
    
    batch_sizes = []
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for i, (x, y) in enumerate(dp_loader):
            if i >= 2:
                break
            
            x, y = x.to(device), y.to(device)
            actual_bs = x.shape[0]
            batch_sizes.append(actual_bs)
            
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            
            all_have_grad_sample = True
            all_match_bs = True
            all_finite = True
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not hasattr(param, "grad_sample"):
                        all_have_grad_sample = False
                    else:
                        if param.grad_sample.shape[0] != actual_bs:
                            all_match_bs = False
                        if not torch.isfinite(param.grad_sample).all():
                            all_finite = False
            
            print(f"Batch {i}: size={actual_bs}")
            print(f"  all_have_grad_sample: {all_have_grad_sample}")
            print(f"  all_match_bs: {all_match_bs}")
            print(f"  all_finite: {all_finite}")
            
            optimizer.step()
            
            post_step_epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"  Accountant epsilon after step: {post_step_epsilon}")
            
        print("Warnings caught:")
        for warn in w:
            print(f" - {warn.category.__name__}: {warn.message}")

if __name__ == '__main__':
    main()
