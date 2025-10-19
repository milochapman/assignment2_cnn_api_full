# app/main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import transforms
from helper_lib.model import get_model

# 可读的标签（CIFAR-10）
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

app = FastAPI(title="CNN Image Classifier", version="1.0.0")

# 与训练一致的预处理
tfm = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化并尝试加载已训练权重
model = get_model("CNN")
try:
    state = torch.load("weights/cnn_cifar10.pt", map_location=device)
    model.load_state_dict(state)
    print("Loaded weights from weights/cnn_cifar10.pt")
except Exception as e:
    # 没有权重也允许启动，但会提示（预测将不可靠）
    print("Warning: could not load weights, using untrained model. Error:", e)

model.to(device)
model.eval()


@app.get("/")
def root():
    """欢迎路由，方便直接访问根路径时不返回 404。"""
    return {"message": "CNN API is running. See /health and /docs."}


@app.get("/health")
def health():
    """健康检查：用于 Docker/部署侧探活。"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    上传一张 JPG/PNG，返回预测类别与概率。
    - 表单字段名必须为 file
    """
    # 读取并转为 PIL.Image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 预处理 -> [1, 3, 64, 64]
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(probs.argmax(dim=1).item())
        prob = float(probs[0, idx].item())

    return {
        "class_index": idx,
        "label": CIFAR10_LABELS[idx],
        "prob": round(prob, 4)
    }
