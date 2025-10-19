
import torch
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_data_loader(data_dir: str, batch_size: int = 64, train: bool = True):
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader
