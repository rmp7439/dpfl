import sys
import time
from config import CONFIG

def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: GPU is not available! A GPU is highly recommended for later training stages.")
            print("Running on CPU for now...")
        else:
            print("SUCCESS: PyTorch GPU (CUDA) is available.")
    except ImportError:
        print("ERROR: PyTorch is not installed. Did you run `pip install -r requirements.txt`?")
        sys.exit(1)

def dummy_run():
    print(f"Loading small-subset config: {CONFIG['num_samples']} samples, {CONFIG['num_clients']} clients...")
    
    start_time = time.time()
    
    # Simulate a tiny dummy training run
    for round_num in range(1, CONFIG['num_rounds'] + 1):
        print(f"--- Round {round_num} ---")
        for client_id in range(1, CONFIG['num_clients'] + 1):
            print(f"  Client {client_id} training for {CONFIG['local_epochs']} epoch(s) on small batch...")
            # Simulate processing time for subset
            time.sleep(0.5)
            
    end_time = time.time()
    print(f"Dummy run completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    print("--- DP-FL Setup Check ---")
    check_gpu()
    print("-------------------------")
    dummy_run()
    print("--- Checkpoint Passed ---")
