from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ultralytics import YOLO
from mangum import Mangum
import numpy as np
import cv2

app = FastAPI()
handler = Mangum(app)
model = YOLO("best.pt")

class PredictionResponse(BaseModel):
    is_cat: bool
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)