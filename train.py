import torch
import os

if torch.cuda.is_available():
    print ("CUDA is available!")
    device = 0
# There are currently some unsupported operators for the MPS
# framework. Some of these issues can be circumvented, but M1 performance
# is not super great. Therefore, let's skip it.
#
#elif torch.backends.mps.is_available():
#    print ("MPS is available!")
#    device = "mps"
else:
    print ("Using CPU device as a fallback")
    device = "cpu"
    
print (f"We run training on the following device: {device}")

from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO(os.path.join("pretrained", "yolov8n.pt"))
results = model.train(data="lego-dataset.yaml", epochs=200, imgsz=640, device=device)

# Evaluate the model's performance on the validation set
results = model.val()
