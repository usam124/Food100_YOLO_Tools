from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import cv2
import numpy as np
import darknet  # make sure darknet python wrapper is importable

app = FastAPI()

# Load Darknet network once on startup
config_path = "./cfg/yolov4.cfg"
weights_path = "./weights/yolov4.weights"
data_file = "./cfg/coco.data"

network, class_names, class_colors = darknet.load_network(
    config_path,
    data_file,
    weights_path,
    batch_size=1
)

def detect_image(image_path):
    # Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Image not found or invalid")

    # Convert image to darknet format
    darknet_image = darknet.make_image(darknet.network_width(network), darknet.network_height(network), 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (darknet.network_width(network), darknet.network_height(network)),
                            interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())

    # Run detection
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
    darknet.free_image(darknet_image)

    # Format detection results to bounding boxes & positions
    results = []
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        # Convert from darknet center x,y,w,h to corner points
        xmin = int(x - w/2)
        ymin = int(y - h/2)
        xmax = int(x + w/2)
        ymax = int(y + h/2)

        results.append({
            "class": label,
            "confidence": float(confidence),
            "bbox": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            },
            "center": {
                "x": int(x),
                "y": int(y)
            }
        })
    return results

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Save uploaded file to disk temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run detection
        detections = detect_image(temp_file_path)

        # Clean up
        os.remove(temp_file_path)

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
