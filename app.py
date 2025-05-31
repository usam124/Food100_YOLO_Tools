from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import darknet  # ensure darknet python binding is installed and accessible

app = FastAPI()

# Load YOLO model
CONFIG_PATH = "cfg/yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
DATA_FILE = "cfg/coco.data"

net = darknet.load_net_custom(CONFIG_PATH.encode('utf-8'), WEIGHTS_PATH.encode('utf-8'), 0, 1)
meta = darknet.load_meta(DATA_FILE.encode('utf-8'))

class ImageRequest(BaseModel):
    image_path: str  # path to image on disk or URL (if extended)

@app.post("/process-image")
async def process_image(request: ImageRequest):
    try:
        img = cv2.imread(request.image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image path or unable to read image")

        # Prepare darknet image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        net_width = darknet.network_width(net)
        net_height = darknet.network_height(net)

        img_resized = cv2.resize(img_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)
        darknet_img = darknet.make_image(net_width, net_height, 3)
        darknet.copy_image_from_bytes(darknet_img, img_resized.tobytes())

        # Run detection
        detections = darknet.detect_image(net, meta, darknet_img, thresh=0.25)

        # Target coordinate system and scale
        target_width, target_height = 950, 950  # your workspace dimension in pixels
        pixel_to_cm = 0.0295  # scale

        positions = []

        for label, confidence, bbox in detections:
            x, y, w, h = bbox  # darknet bbox: center_x, center_y, width, height
            area = w * h
            if area < 5000 or area > 1000000:
                continue  # filter by area

            # Convert bbox center (x,y) to bottom-right origin coordinate system:
            # bottom-right origin means (0,0) is at bottom-right corner of the image,
            # so x_br = width - x, y_br = height - y (where width/height are the target system sizes)

            x_br = target_width - x
            y_br = target_height - y

            # Midpoint position from bottom-right corner:
            # Your bbox center is already midpoint; if you want midpoint of bbox bottom-right corner,
            # or midpoint between bottom-right and center, please clarify.
            # Assuming you want the midpoint of bbox as origin is bottom-right (x_br, y_br):

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


# Local test run
if __name__ == "__main__":
    import asyncio
    async def test_image():
        test_request = ImageRequest(image_path="test.jpg")  # replace with your test image path
        result = await process_image(test_request)
        print(result)
    asyncio.run(test_image())

