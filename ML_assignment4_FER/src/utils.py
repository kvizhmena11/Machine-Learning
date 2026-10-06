import torch

def run_sanity_check(model_class, device):
    print(f"--- Running Sanity Check for {model_class.__name__} ---")
    try:
        model = model_class().to(device)
        x = torch.randn(2, 1, 48, 48).to(device)
        out = model(x)
        print("-> Forward Pass Verification: PASSED")
        
        loss = out.sum()
        loss.backward()
        print("-> Backward Gradient Computation: PASSED")
        
        # 1-Batch Overfit
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x_b = torch.randn(4, 1, 48, 48).to(device)
        y_b = torch.randint(0, 7, (4,)).to(device)
        crit = torch.nn.CrossEntropyLoss()
        
        init_loss = crit(model(x_b), y_b).item()
        for _ in range(200):
            optimizer.zero_grad()
            l = crit(model(x_b), y_b)
            l.backward()
            optimizer.step()
        final_loss = crit(model(x_b), y_b).item()
        print(f"-> 1-Batch Overfit Test: Initial Loss: {init_loss:.4f} -> Final Loss: {final_loss:.4f}")
        print("--- Sanity Check Complete ---")
    except Exception as e:
        print(f"-> Sanity Check FAILED: {str(e)}")
