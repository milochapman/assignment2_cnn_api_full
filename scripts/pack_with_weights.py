
import argparse, pathlib, shutil, zipfile, os

def pack(weights_path: str, out_zip: str):
    weights = pathlib.Path(weights_path).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    # project root assumed to be the parent of cnn_api directory if run from repo root
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src = repo_root  # whole assignment folder
    # where to place weights in bundle
    dst_weights = repo_root / "cnn_api" / "weights" / "cnn_cifar10.pt"
    # copy weights into project
    dst_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dst_weights)
    # make zip
    out_path = pathlib.Path(out_zip).expanduser().resolve()
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in repo_root.rglob("*"):
            z.write(p, p.relative_to(repo_root.parent))
    print("Packed ->", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to cnn_cifar10.pt")
    ap.add_argument("--out", default="./assignment2_cnn_api_with_weights.zip")
    args = ap.parse_args()
    pack(args.weights, args.out)
