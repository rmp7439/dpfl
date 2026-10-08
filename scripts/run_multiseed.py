import os
import sys
import subprocess
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Seed Stage 6 Experiments")
    parser.add_argument("--subset", action="store_true", help="Run on subset (for testing)")
    parser.add_argument("--sigmas", type=str, default="1.0", help="Comma-separated sigmas (default: 1.0)")
    args = parser.parse_args()
    
    seeds = [42, 43, 44]
    alphas = [0.1, 10.0]
    sigmas = [float(s.strip()) for s in args.sigmas.split(',')]
    C = 1.0
    
    base_dir = os.path.join("results", "stage6", "multiseed")
    os.makedirs(base_dir, exist_ok=True)
    
    all_summaries = []
    
    for sigma in sigmas:
        for alpha in alphas:
            for seed in seeds:
                # Format: alpha_{alpha}_sigma_{sigma}_C_{C}_seed_{seed}
                # To match exactly: alpha_0.1_sigma_1.0_C_1.0_seed_42 or alpha_10_sigma_1.0_C_1.0_seed_42
                # (Notice alpha_10 vs alpha_10.0, we will format it so 10.0 becomes 10)
                alpha_str = str(int(alpha)) if int(alpha) == alpha else str(alpha)
                sigma_str = str(int(sigma)) if int(sigma) == sigma else str(sigma)
                C_str = str(int(C)) if int(C) == C else str(C)
                
                # actually Python float format: 1.0 -> 1.0, we should ensure 1.0 stays 1.0 if the prompt asks for 1.0
                # "alpha_10_sigma_1.0_C_1.0_seed_42"
                alpha_str = "10" if alpha == 10.0 else str(alpha)
                sigma_str = "1.0" if sigma == 1.0 else str(sigma)
                if sigma == 2.0: sigma_str = "2.0"
                C_str = "1.0" if C == 1.0 else str(C)
                
                out_dir_name = f"alpha_{alpha_str}_sigma_{sigma_str}_C_{C_str}_seed_{seed}"
                out_dir = os.path.join(base_dir, out_dir_name)
                os.makedirs(out_dir, exist_ok=True)
                
                print(f"\n--- Running alpha={alpha}, sigma={sigma}, seed={seed} ---")
                
                cmd = [
                    sys.executable, "scripts/run_federated.py",
                    "--enable-dp",
                    "--alpha", str(alpha),
                    "--sigma", str(sigma),
                    "--C", str(C),
                    "--seed", str(seed),
                    "--rounds", "3",
                    "--output-dir", out_dir
                ]
                if not args.subset:
                    cmd.append("--full")
                    
                subprocess.run(cmd, check=True)
                
                summary_path = os.path.join(out_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        all_summaries.append(json.load(f))
                        
    # Aggregate
    # We want to group by (alpha, sigma, C)
    aggregate_results = []
    for sigma in sigmas:
        for alpha in alphas:
            group = [s for s in all_summaries if s.get("alpha") == alpha and s.get("sigma") == sigma]
            if not group:
                continue
                
            final_accs = [s.get("final_test_accuracy", 0.0) for s in group]
            best_accs = [s.get("best_test_accuracy", 0.0) for s in group]
            
            # Privacy values should be identical across seeds, so we take from the first run
            rep = group[0]
            
            agg = {
                "alpha": alpha,
                "sigma": sigma,
                "C": C,
                "delta": 1e-5,
                "seeds": seeds,
                "number_of_runs": len(group),
                "final_accuracy_mean": float(np.mean(final_accs)),
                "final_accuracy_std": float(np.std(final_accs)),
                "best_accuracy_mean": float(np.mean(best_accs)),
                "best_accuracy_std": float(np.std(best_accs)),
                "epsilon": rep.get("epsilon"),
                "RDP_order": rep.get("best_alpha")
            }
            aggregate_results.append(agg)
            
    with open(os.path.join(base_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate_results, f, indent=4)
        
    print(f"\nSaved multiseed aggregate results to {os.path.join(base_dir, 'aggregate.json')}")

if __name__ == "__main__":
    main()
