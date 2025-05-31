from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import darknet  # make sure this is in /app/darknet

app = FastAPI()

# Paths
CONFIG_PATH = "cfg/yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
DATA_FILE = "cfg/coco.data"

# Load YOLO
try:
    net = darknet.load_network(
        CONFIG_PATH.encode("utf-8"),
        DATA_FILE.encode("utf-8"),
        WEIGHTS_PATH.encode("utf-8"),
        batch_size=1
    )
except AttributeError:
    raise RuntimeError("Darknet Python bindings not built or incorrect. Check Dockerfile.")

class ImageRequest(BaseModel):
    image_path: str

@app.post("/process-image")
async def process_image(request: ImageRequest):
    try:
        img = cv2.imread(request.image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image path")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = darknet.network_width(net)
        height = darknet.network_height(net)

        img_resized = cv2.resize(img_rgb, (width, height))
        darknet_img = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_img, img_resized.tobytes())

        detections = darknet.detect_image(net, darknet.make_metadata(DATA_FILE.encode("utf-8")), darknet_img, thresh=0.25)

        pixel_to_cm = 0.0295
        target_width, target_height = 950, 950
        positions = []

        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            area = w * h
            if area < 5000 or area > 1000000:
                continue

            x_br = target_width - x
            y_br = target_height - y

            positions.append({
                "x_cm": round(x_br * pixel_to_cm, 2),
                "y_cm": round(y_br * pixel_to_cm, 2),
                "surface_cm2": round(area * (pixel_to_cm ** 2), 2),
                "class": label.decode() if isinstance(label, bytes) else label,
                "confidence": round(confidence, 3)
            })

        darknet.free_image(darknet_img)

        return {"item_positions": positions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
