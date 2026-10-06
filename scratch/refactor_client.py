import re

with open('scripts/run_federated.py', 'r') as f:
    text = f.read()

# Fix client_fn to be client_fn_factory
old_client_fn = """def client_fn(context: Context) -> fl.client.Client:
    \"\"\"Create a Flower client representing a single organization.\"\"\"
    import time
    start_setup = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    
    # Get the client's subset of data
    try:
        client_id = int(context.node_config["partition-id"])
    except (KeyError, AttributeError, TypeError):
        client_id = int(context.node_id)
        
    cid = str(client_id)
    indices = CLIENT_INDICES[client_id]
    
    client_dataset = torch.utils.data.Subset(GLOBAL_TRAINSET, indices)
    
    train_loader = DataLoader(client_dataset, batch_size=self.run_config.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(GLOBAL_TESTSET, batch_size=self.run_config.get("batch_size", 32), shuffle=False)
    
    setup_time = time.time() - start_setup
    client = FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP)
    client.setup_time = setup_time
    return client.to_client()"""

new_client_fn = """def client_fn_factory(run_config):
    def client_fn(context: Context) -> fl.client.Client:
        \"\"\"Create a Flower client representing a single organization.\"\"\"
        import time
        start_setup = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SimpleCNN().to(device)
        
        # Get the client's subset of data
        try:
            client_id = int(context.node_config["partition-id"])
        except (KeyError, AttributeError, TypeError):
            client_id = int(context.node_id)
            
        cid = str(client_id)
        indices = CLIENT_INDICES[client_id]
        
        client_dataset = torch.utils.data.Subset(GLOBAL_TRAINSET, indices)
        
        train_loader = DataLoader(client_dataset, batch_size=run_config.get("batch_size", 32), shuffle=True)
        test_loader = DataLoader(GLOBAL_TESTSET, batch_size=run_config.get("batch_size", 32), shuffle=False)
        
        setup_time = time.time() - start_setup
        client = FlowerClient(cid, net, train_loader, test_loader, device, use_dp=USE_DP, run_config=run_config)
        client.setup_time = setup_time
        return client.to_client()
    return client_fn"""

text = text.replace(old_client_fn, new_client_fn)
text = text.replace('client_fn=client_fn,', 'client_fn=client_fn_factory(run_config),')
text = text.replace('client_fn=lambda cid: client_fn_factory(cid, run_config)', 'client_fn=client_fn_factory(run_config)')

with open('scripts/run_federated.py', 'w') as f:
    f.write(text)

print('Success')
