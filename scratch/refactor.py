import re

with open('scripts/run_federated.py', 'r') as f:
    text = f.read()

# 1. Remove global CONFIG assignment
text = text.replace('CONFIG = SUBSET_CONFIG\n', '')

# 2. Update FlowerClient
text = text.replace(
    'def __init__(self, cid, net, train_loader, test_loader, device, use_dp=False):',
    'def __init__(self, cid, net, train_loader, test_loader, device, use_dp=False, run_config=None):\n        self.run_config = run_config or {}'
)

# 3. Inside FlowerClient replace CONFIG with self.run_config
class_pattern = re.compile(r'(class FlowerClient.*?)(def client_fn_factory)', re.DOTALL)
m = class_pattern.search(text)
if m:
    client_code = m.group(1)
    client_code = client_code.replace('CONFIG.get', 'self.run_config.get')
    text = text[:m.start()] + client_code + text[m.end(1):]

# 4. Update client_fn_factory
text = text.replace(
    'def client_fn_factory(cid_str: str):',
    'def client_fn_factory(cid_str: str, run_config: dict):'
)

text = text.replace(
    'return FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP).to_client()',
    'return FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP, run_config=run_config).to_client()'
)

# 5. Fix start_simulation call to pass run_config
text = text.replace(
    'client_fn=client_fn_factory',
    'client_fn=lambda cid: client_fn_factory(cid, run_config)'
)

# 6. Update global statement in main
text = text.replace('global GLOBAL_TRAINSET, GLOBAL_TESTSET, CLIENT_INDICES, USE_DP, CONFIG', 'global GLOBAL_TRAINSET, GLOBAL_TESTSET, CLIENT_INDICES, USE_DP')

# 7. Update main configuration initialization
init_old = """    if args.full:
        CONFIG = FULL_CONFIG
        
    if args.enable_dp:
        CONFIG["noise_multiplier"] = args.sigma
        CONFIG["max_grad_norm"] = args.C
        
    if args.rounds is not None:
        CONFIG["num_rounds"] = args.rounds
        
    if args.alpha is not None:
        CONFIG["alpha"] = args.alpha
        
    # Store out_dir in global CONFIG for client-side artifact dumping
    CONFIG["out_dir"] = args.output_dir"""

init_new = """    import copy
    run_config = copy.deepcopy(FULL_CONFIG) if args.full else copy.deepcopy(SUBSET_CONFIG)
    
    if args.enable_dp:
        run_config["noise_multiplier"] = args.sigma
        run_config["max_grad_norm"] = args.C
        
    if args.rounds is not None:
        run_config["num_rounds"] = args.rounds
        
    if args.alpha is not None:
        run_config["alpha"] = args.alpha
        
    run_config["out_dir"] = args.output_dir"""

text = text.replace(init_old, init_new)

# 8. Replace remaining CONFIG with run_config inside main
main_match = re.search(r'(def main.*?)$', text, re.DOTALL)
if main_match:
    main_code = main_match.group(1)
    main_code = main_code.replace('CONFIG', 'run_config')
    text = text[:main_match.start()] + main_code

with open('scripts/run_federated.py', 'w') as f:
    f.write(text)
print('Refactored run_federated.py successfully.')
