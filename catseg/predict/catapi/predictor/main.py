from ultralytics import YOLO
from fastapi import FastAPI
from typing import Optional
from PIL import Image
import numpy as np
import base64
import io
import os

app = FastAPI()

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best.pt")
    return YOLO(model_path)


@app.get("/health_check")
def read_root():
    return {"Ping": "Pong"}

@app.post("/predict")
def predict(file: Optional[dict]):
    try:
        model = load_model()
        image_bytes = base64.b64decode(file["image"])
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
    except Exception as e:
        return {"error": str(e)}
    
    try:  
        results = model(img_array)
        
        # Check if any detection is a cat (assuming class 0 is cat)
        is_cat = False
        confidence = 0.0
        
        for result in results:
            if len(result.boxes) > 0:
                # Get class predictions and confidences
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Find cat predictions (class 0)
                cat_mask = classes == 0
                if any(cat_mask):
                    is_cat = True
                    confidence = float(max(confs[cat_mask]))
        
        return {"is_cat": is_cat, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}
