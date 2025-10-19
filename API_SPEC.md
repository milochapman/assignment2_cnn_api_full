# API Specification

## 1) Health Check
- Method: `GET /health`
- Response: `{"status":"ok"}`

## 2) Image Classification
- Method: `POST /predict`
- Form field: `file` (upload JPG/PNG)
- Response example:
```json
{
  "class_index": 3,
  "label": "cat",
  "prob": 0.8765
}
```
**Model**: 64×64 input CNN (Conv-ReLU-Pool ×2 → Flatten → FC(100)-ReLU → FC(10)).  
If `weights/cnn_cifar10.pt` is not present, the API still runs but predictions are untrained.