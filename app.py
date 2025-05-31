from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import darknet  # ensure darknet python binding is installed and accessible

app = FastAPI()

# Paths to config files
CONFIG_PATH = "cfg/yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
DATA_FILE = "cfg/coco.data"

net = None
meta = None

# Load YOLO model at startup
@app.on_event("startup")
def load_model():
    global net, meta
    net = darknet.load_net_custom(CONFIG_PATH.encode('utf-8'), WEIGHTS_PATH.encode('utf-8'), 0, 1)
    meta = darknet.load_meta(DATA_FILE.encode('utf-8'))

# Request model
class ImageRequest(BaseModel):
    image_path: str  # Local path to image

@app.post("/process-image")
async def process_image(request: ImageRequest):
    try:
        img = cv2.imread(request.image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image path or unable to read image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        net_width = darknet.network_width(net)
        net_height = darknet.network_height(net)

        img_resized = cv2.resize(img_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)
        darknet_img = darknet.make_image(net_width, net_height, 3)

        try:
            darknet.copy_image_from_bytes(darknet_img, img_resized.tobytes())
            detections = darknet.detect_image(net, meta, darknet_img, thresh=0.25)
        finally:
            darknet.free_image(darknet_img)

        target_width, target_height = 950, 950  # Adjust to your workspace size
        pixel_to_cm = 0.0295

        positions = []
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            area = w * h
            if area < 5000 or area > 1000000:
                continue

            x_br = target_width - x
            y_br = target_height - y

            label_str = label.decode("utf-8") if isinstance(label, bytes) else label

            positions.append({
                "x_cm": round(x_br * pixel_to_cm, 2),
                "y_cm": round(y_br * pixel_to_cm, 2),
                "surface_cm2": round(area * (pixel_to_cm ** 2), 2),
                "class": label_str,
                "confidence": round(confidence, 3)
            })

        return {"item_positions": positions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For local testing
if __name__ == "__main__":
    import asyncio

    async def test_image():
        test_request = ImageRequest(image_path="test.jpg")  # Replace with your test image
        result = await process_image(test_request)
        print(result)

    asyncio.run(test_image())


