# Assignment 2 — CNN + FastAPI + Docker (Ready to Run)

This repository contains a complete, ready-to-run solution for the assignment:
- Part 1: Implement the specified CNN architecture in PyTorch
- Part 2: Train on CIFAR‑10, serve an inference API via FastAPI, and provide a Docker deployment
- Part 3: Theory questions (with derivations)

## Quick Start for Reviewers

### Option A — Docker (no local Python setup)
```bash
git clone <YOUR_REPO_URL>
cd assignment2_cnn_api_full/cnn_api
docker build -t cnn-api .
docker run --rm -p 8000:8000 cnn-api
# In a second terminal (use the sample image provided):
curl -F "file=@../assets/sample.jpg" http://localhost:8000/predict
```

### Option B — uv (Python package manager)
```bash
git clone <YOUR_REPO_URL>
cd assignment2_cnn_api_full/cnn_api
uv lock && uv sync
# If weights are not included, quickly train to create them:
# uv run python train_cifar10.py --epochs 2
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -F "file=@../assets/sample.jpg" http://127.0.0.1:8000/predict
```

## Project Layout
- `cnn_api/` — runnable code (PyTorch model, training script, FastAPI app, Dockerfiles)
- `docs/` — assignment PDF and lecture materials (original attachments)
- `assets/` — `sample.jpg` used for quick API smoke tests
- `THEORY.md` — answers and derivations for the theory questions
- `API_SPEC.md` — endpoint documentation
- `RUBRIC_CHECKLIST.md` — self-check aligned to the grading rubric
- `client_predict.py` — small helper client to call `/predict`
- `Makefile` — convenience targets (`make train`, `make api`, `make build`, `make run`, `make test`)

## Notes
- To maximize “clone → run” for graders, include `cnn_api/weights/cnn_cifar10.pt` in the repository. If missing, run a short training first.
- For strict reproducibility, commit `uv.lock` after `uv lock`.

---

## Lockfiles (Reproducibility)
Generate and commit lockfiles (recommended):
```bash
cd cnn_api
bash scripts/generate_lockfiles.sh
# Windows:
# powershell -ExecutionPolicy Bypass -File scripts\generate_lockfiles.ps1
```
This produces:
- `uv.lock`
- `requirements.lock.txt` (exported from the lockfile)
