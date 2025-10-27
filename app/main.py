from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import io
from PIL import Image
import torch
import torchvision.transforms as T
import os

from src.model.cnn_model import create_model
from src.utils.config import load_config

app = FastAPI(title="Flower Classification API")

# Lazy model
_model = None
_class_names: List[str] = []
_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    global _model, _class_names
    if _model is None:
        config = load_config()
        num_classes = config.get("NUM_CLASSES", 17)
        _model = create_model(num_classes=num_classes)
        model_path = os.path.join(config.get("MODEL_SAVE_PATH", "models"), "flower_model.pth")
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            _model.load_state_dict(state)
        _model.eval()
        # Derive class names from training mapping if available
        # Fallback to generic names
        _class_names = [str(i) for i in range(num_classes)]
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_model()
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        x = _transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            score, idx = torch.max(probs, dim=0)
        return {"label": _class_names[int(idx)], "score": float(score)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
