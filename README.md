# Assignment 2 — CNN + FastAPI + Docker

Three parts:

- **Part 1** — CNN in PyTorch (given spec)  
- **Part 2** — Train on CIFAR-10, serve `/predict` with FastAPI, provide Docker  
- **Part 3** — Theory answers with short derivations (`THEORY.md`)

---

## Quick start

### A) Docker（不装 Python 的最省心方式）

```bash
git clone <YOUR_REPO_URL>
cd assignment2_cnn_api_full/cnn_api
docker build -t cnn-api .
docker run --rm -p 8000:8000 cnn-api
