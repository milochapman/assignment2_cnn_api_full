
import requests, sys, os

URL = os.environ.get("URL", "http://127.0.0.1:8000")

def main():
    # health
    r = requests.get(f"{URL}/health", timeout=10)
    print("/health:", r.json())

    # use a local sample image if available
    img_path = "sample.jpg"
    if not os.path.exists(img_path):
        print("sample.jpg not found, please run scripts/save_sample.py first.")
        sys.exit(0)
    with open(img_path, "rb") as f:
        files = {"file": ("sample.jpg", f, "image/jpeg")}
        r = requests.post(f"{URL}/predict", files=files, timeout=30)
    print("/predict:", r.json())

if __name__ == "__main__":
    main()
