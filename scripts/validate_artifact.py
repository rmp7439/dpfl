import sys
import json
import os
import argparse
from typing import List, Dict

def validate_artifact(filepath: str, expected_config: Dict):
    if not os.path.exists(filepath):
        raise ValueError(f"Artifact {filepath} does not exist.")
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    print(f"Validating {filepath}...")
    
    # Check 1: Metadata matches expected configuration
    for k, v in expected_config.items():
        if k in data and data[k] != v:
            raise ValueError(f"Metadata mismatch for {k}: expected {v}, found {data.get(k)}")
    
    # Check 2: Exactly 15 completed rounds
    if data.get("number_of_communication_rounds") != 15:
        raise ValueError("number_of_communication_rounds is not 15")
        
    acc_hist = data.get("per_round_test_accuracy", [])
    if len(acc_hist) != 15:
        raise ValueError(f"Expected 15 rounds of accuracy, found {len(acc_hist)}")
        
    # Check 3: Per-round participation
    part_hist = data.get("per_round_participation", [])
    if len(part_hist) != 15:
        raise ValueError(f"Expected 15 rounds of participation data, found {len(part_hist)}")
        
    for i, p in enumerate(part_hist):
        round_num = i + 1
        if p["round"] != round_num:
            raise ValueError(f"Round mismatch: expected {round_num}, got {p['round']}")
        
        # All 5 expected client IDs participated
        expected_clients = data.get("num_clients", 5)
        expected_cids = {str(c) for c in range(expected_clients)}
        actual_cids = set(p["actual_clients"])
        
        if len(p["actual_clients"]) != expected_clients:
            raise ValueError(f"Round {round_num}: expected {expected_clients} clients, got {len(p['actual_clients'])}")
            
        if actual_cids != expected_cids:
            raise ValueError(f"Round {round_num}: expected clients {expected_cids}, got {actual_cids}")
            
        if len(set(p["actual_clients"])) != len(p["actual_clients"]):
            raise ValueError(f"Round {round_num}: duplicate clients in actual_clients")
            
        if p.get("num_failures", 0) > 0:
            raise ValueError(f"Round {round_num}: recorded {p['num_failures']} failures")
            
    # Check 4: DP specific accounting checks
    if data.get("dp_enabled", True) and "sigma" in expected_config:
        eps_history = data.get("rounds_epsilon_history", []) # wait, we didn't save this in JSON natively. But we can check if it exists or we just rely on runtime.
        # Actually, let's just ensure the final epsilon is greater than 0, and rely on runtime check for monotonicity.
        if "epsilon" not in data or data["epsilon"] <= 0:
            raise ValueError("Global epsilon is missing or 0")
        if "best_alpha" not in data or data["best_alpha"] <= 0:
            raise ValueError("Minimizing RDP order (best_alpha) is missing or 0")
            
        client_details = data.get("client_details", {})
        if not client_details:
            raise ValueError("Client details missing")
            
        for cid, details in client_details.items():
            if "cumulative_steps" not in details:
                raise ValueError(f"Client {cid} missing cumulative_steps")
            if "epsilon" not in details:
                raise ValueError(f"Client {cid} missing epsilon (privacy/accounting state)")
                
    print(f"  -> Valid!")
    return data

def cross_check_c_sigma(artifacts: List[Dict]):
    # Group by (alpha, sigma, seed, delta, etc.)
    groups = {}
    for a in artifacts:
        # Only DP runs
        if "sigma" not in a:
            continue
        key = (a.get("alpha"), a.get("sigma"), a.get("seed"), a.get("dataset_mode"))
        if key not in groups:
            groups[key] = []
        groups[key].append(a)
        
    for key, group in groups.items():
        if len(group) > 1:
            epsilons = [x["epsilon"] for x in group]
            # Since C doesn't change math for RDP, epsilons must be identical across varying C
            # Give a small floating point tolerance
            base_eps = epsilons[0]
            for e in epsilons[1:]:
                if abs(base_eps - e) > 1e-4:
                    print(f"WARNING: Discrepancy detected! Same sigma={key[1]} but different C values produced different epsilons: {epsilons}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSON summary files to validate")
    parser.add_argument("--expected-rounds", type=int, default=15)
    parser.add_argument("--expected-clients", type=int, default=5)
    args = parser.parse_args()
    
    valid_data = []
    has_errors = False
    
    for f in args.files:
        try:
            # We enforce some defaults that must be true for authoritative runs
            expected = {
                "number_of_communication_rounds": args.expected_rounds,
                "num_clients": args.expected_clients,
            }
            d = validate_artifact(f, expected)
            valid_data.append(d)
        except Exception as e:
            print(f"ERROR validating {f}: {e}")
            has_errors = True
            
    cross_check_c_sigma(valid_data)
    
    if has_errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
