
# CNN API (Assignment 2)

## Train
```bash
python -m venv .venv && source .venv/bin/activate   # on macOS/Linux
pip install -r requirements.txt
python train_cifar10.py --epochs 5
```

## Run API
```bash
uvicorn app.main:app --reload --port 8000
```

## Predict
```bash
curl -F "file=@sample.jpg" http://localhost:8000/predict
```

## Docker
```bash
docker build -t cnn-api .
docker run --rm -p 8000:8000 cnn-api
```
