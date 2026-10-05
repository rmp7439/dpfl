import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Estimate DP-FL Experiment Runtime")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--configs", type=int, default=1, help="Number of hyperparameter configurations")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds per configuration")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients per round")
    args = parser.parse_args()

    timings_path = "results/profiling/timings.jsonl"
    
    if not os.path.exists(timings_path):
        print(f"Error: Profiling data not found at {timings_path}.")
        print("Please run a benchmark with profiling enabled first.")
        return

    # Analyze timings
    train_times = []
    eval_times = []
    
    with open(timings_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "fit_metrics":
                for client in data["timings"]:
                    # We look at the train time + DP setup time + client setup
                    total_fit = client.get("fit_total_time", 0)
                    train_times.append(total_fit)
            elif data["type"] == "evaluate_metrics":
                for t in data["timings"]:
                    eval_times.append(t)
                    
    if not train_times:
        print("No training timings found in profiling data.")
        return
        
    avg_client_train = sum(train_times) / len(train_times)
    
    # Check if centralized eval was used (1 eval per round) or distributed (multiple per round)
    # But since we just want to project cost, we'll assume the current evaluation setup is preserved.
    # Actually, let's just average all eval times and multiply by 1 if centralized, or by clients if distributed.
    # To be safe, we'll estimate based on average client eval time.
    avg_client_eval = sum(eval_times) / len(eval_times) if eval_times else 0

    # In our optimized version, eval is 1x per round. If we have N evals for R rounds, we can detect it.
    evals_per_round = len(eval_times) / (len(train_times) / len(set([i%args.clients for i in range(len(train_times))])))
    # A safer way to estimate round time is to just assume sequential client execution (Ray VCE default behavior if num_cpus is limited)
    # We will assume sequential execution for safety in estimation.
    est_round_time = (avg_client_train * args.clients) + avg_client_eval

    # Print Report
    print("="*50)
    print(" EXPERIMENT COMPUTE BUDGET ESTIMATION")
    print("="*50)
    print(f"Based on measured profiling data from {len(train_times)} client tasks:")
    print(f" - Avg Client Training Time: {avg_client_train:.2f} s")
    print(f" - Avg Server/Client Eval Time: {avg_client_eval:.2f} s")
    print(f" - Estimated Time per Round (Sequential): {est_round_time:.2f} s")
    
    total_rounds = args.rounds * args.configs * args.seeds
    total_time_seconds = est_round_time * total_rounds
    total_hours = total_time_seconds / 3600.0
    
    print("\nProjected Experiment Request:")
    print(f" - Configurations: {args.configs}")
    print(f" - Seeds: {args.seeds}")
    print(f" - Rounds per Run: {args.rounds}")
    print(f" - Total Rounds: {total_rounds}")
    print("-" * 50)
    print(f"ESTIMATED GPU/CPU TIME: {total_hours:.2f} hours")
    print(f"                        ({total_time_seconds:.0f} seconds)")
    print("="*50)

if __name__ == "__main__":
    main()
