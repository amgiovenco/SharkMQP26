from typing import Optional, Dict, Any
from pathlib import Path
import json
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# ================================
# Paths
# ================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "resnet18"

WEIGHTS_PATH = MODEL_DIR / "resnet18.pth"
ENCODER_PATH = MODEL_DIR / "label_encoder.json"


# ================================
# Transform (和训练完全一致)
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================
# Load Label Encoder
# ================================
def load_idx2label():
    with open(ENCODER_PATH, "r") as f:
        label2idx = json.load(f)
    return {int(v): k for k, v in label2idx.items()}


# ================================
# Singleton Model
# ================================
_model = None
_idx2label = None


def get_model(device):
    global _model, _idx2label

    if _idx2label is None:
        _idx2label = load_idx2label()

    if _model is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(_idx2label))

        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")

        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)
        model.eval()

        _model = model

    _model.to(device)
    return _model, _idx2label


# ================================
# CSV -> Melt Curve Extraction
# ================================
def extract_curve(filepath, sample_index):
    df = pd.read_csv(filepath)

    # melt_block 处理后格式应该是：
    # 每一行一个 sample
    row = df.iloc[sample_index]

    temperature = row["Temperature (°C)"]
    melting_curve = row["Melting Curve Data"]

    # 如果是字符串形式 list
    if isinstance(temperature, str):
        temperature = json.loads(temperature)
    if isinstance(melting_curve, str):
        melting_curve = json.loads(melting_curve)

    return temperature, melting_curve


# ================================
# 关键：复刻训练时 matplotlib 风格
# ================================
def curve_to_image(temperature, melting_curve):

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(temperature, melting_curve)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=450)
    plt.close(fig)

    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image


# ================================
# Public Interface
# ================================
def ml_inference(filepath: str,
                 sample_index: int = 0,
                 device: Optional[str] = None) -> Dict[str, Any]:

    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        temperature, melting_curve = extract_curve(filepath, sample_index)

        image = curve_to_image(temperature, melting_curve)
        image_tensor = transform(image).unsqueeze(0).to(device)

        model, idx2label = get_model(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(-probs)[:10]

        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            predictions.append({
                "rank": rank,
                "species": idx2label[int(idx)],
                "confidence": float(probs[idx])
            })

        return {
            "success": True,
            "predictions": predictions,
            "sample_index": sample_index,
            "curve_data": {
                "frequencies": list(map(float, temperature)),
                "signal": list(map(float, melting_curve))
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "predictions": [],
            "sample_index": sample_index
        }
