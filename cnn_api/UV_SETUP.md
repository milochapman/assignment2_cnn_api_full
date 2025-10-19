# uv Workflow (Project Local Setup)

## Install uv (skip if already installed)
macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows (PowerShell):
```powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

## 1) Create lockfile and sync env (creates `.venv`)
```bash
uv lock
uv sync
```
Pin Python version (optional):
```bash
uv python pin 3.11
uv sync
```

## 2) Train
```bash
uv run python train_cifar10.py --epochs 2
```

## 3) Run API
```bash
uv run uvicorn app.main:app --reload --port 8000
```

## 4) Call the API
```bash
python scripts/save_sample.py --index 0 --out sample.jpg
curl -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

## 5) Export a frozen requirements file (optional)
```bash
uv export --frozen --format requirements-txt > requirements.lock.txt
```

---

# Dockerfile (uv-based) — `Dockerfile.uv`
```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency definition first (layer caching)
COPY pyproject.toml ./
# If you have uv.lock, include it for reproducible builds:
# COPY uv.lock ./

RUN uv sync --no-dev

# Copy the rest of the project
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:
```bash
docker build -f Dockerfile.uv -t cnn-api-uv .
docker run --rm -p 8000:8000 cnn-api-uv
```