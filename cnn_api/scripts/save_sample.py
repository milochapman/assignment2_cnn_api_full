
from torchvision import datasets, transforms
from PIL import Image
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="sample.jpg")
    args = parser.parse_args()

    ds = datasets.CIFAR10(root="./data", train=False, download=True,
                          transform=transforms.ToTensor())
    img, label = ds[args.index]
    # denormalize not needed because ToTensor gives [0,1]; just convert to PIL
    pil = transforms.ToPILImage()(img)
    pil.save(args.out)
    print(f"Saved {args.out} with label index {label}")

if __name__ == "__main__":
    main()
