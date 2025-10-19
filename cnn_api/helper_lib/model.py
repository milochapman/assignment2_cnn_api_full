
import torch.nn as nn

class SimpleCNN64(nn.Module):
    """
    Input: 64x64x3
    Conv(3->16, 3x3, s=1, p=1) -> ReLU -> MaxPool(2)
    Conv(16->32, 3x3, s=1, p=1) -> ReLU -> MaxPool(2)
    Flatten -> FC(100) -> ReLU -> FC(10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_model(model_name: str):
    if model_name.upper() in ["CNN", "SIMPLECNN64"]:
        return SimpleCNN64(num_classes=10)
    raise ValueError(f"Unknown model_name: {model_name}")
