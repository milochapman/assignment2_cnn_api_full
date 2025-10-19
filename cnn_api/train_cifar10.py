
import argparse, torch, torch.nn as nn, torch.optim as optim, os
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", default="weights/cnn_cifar10.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.weights), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader = get_data_loader(args.data_dir, batch_size=args.batch_size, train=True)
    test_loader  = get_data_loader(args.data_dir, batch_size=256, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, criterion, optimizer, device=device, epochs=args.epochs)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.3f}")

    torch.save(model.state_dict(), args.weights)
    print("Saved weights to", args.weights)

if __name__ == "__main__":
    main()
