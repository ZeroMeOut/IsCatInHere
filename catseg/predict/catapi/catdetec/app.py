from pydantic import BaseModel
from ultralytics import YOLO
from fastapi import FastAPI
from typing import Optional
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = YOLO("best.pt")

class PredictionResponse(BaseModel):
    is_cat: bool
    confidence: float

@app.get("/health_check")
def read_root():
    return {"Ping": "Pong"}

@app.post("/predict", response_model=PredictionResponse)
def predict(file: Optional[dict]):
    image_bytes = file["input_image"]
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_array = np.array(image)
    
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
    
    return PredictionResponse(is_cat=is_cat, confidence=confidence)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)