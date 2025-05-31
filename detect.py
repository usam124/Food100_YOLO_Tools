import torch
from PIL import Image

# Load YOLOv5s model from ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image file (put test.jpg in the repo)
img = Image.open('meal-photo6.jpg')

# Run detection
results = model(img)

# Save results images to 'runs/detect/exp'
results.save()

print("Detection complete. Check runs/detect/exp folder for output.")
