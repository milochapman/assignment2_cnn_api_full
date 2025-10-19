
import torch
from torch import nn
from typing import Tuple

@torch.no_grad()
def evaluate_model(model, loader, criterion, device: str = "cpu") -> Tuple[float, float]:
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total
