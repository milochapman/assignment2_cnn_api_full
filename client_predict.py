import argparse, requests, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    p.add_argument("--image", default="../assets/sample.jpg")
    args = p.parse_args()

    with open(args.image, "rb") as f:
        r = requests.post(args.url, files={"file": (os.path.basename(args.image), f, "image/jpeg")}, timeout=30)
    print(r.json())

if __name__ == "__main__":
    main()