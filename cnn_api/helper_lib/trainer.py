
import torch
from torch import nn

def train_model(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str = "cpu",
                epochs: int = 5):
    model.to(device)
    model.train()  # Enable training behavior (Dropout/BatchNorm)
    for epoch in range(1, epochs+1):
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        print(f"[{epoch}/{epochs}] train_loss={running/len(loader.dataset):.4f}")
    return model
