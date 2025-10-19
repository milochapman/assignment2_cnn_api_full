
.PHONY: train api docker build run test

train:
	uv run python cnn_api/train_cifar10.py --epochs 2

api:
	uv run uvicorn cnn_api/app.main:app --host 0.0.0.0 --port 8000

build:
	cd cnn_api && docker build -t cnn-api .

run:
	docker run --rm -p 8000:8000 cnn-api

test:
	python client_predict.py --image assets/sample.jpg

lock:
	cd cnn_api && bash scripts/generate_lockfiles.sh
