
# YOLOv8 Training and Live Demo

This repository contains scripts to train a YOLOv8 model on a custom dataset and demonstrate live object detection using the trained model. The scripts are divided into two main components: `train.py` and `demo.py`.

## Installation

First, install the required packages:
```bash
conda create -n yolo python=3.10 # or whatever virtual env you like
pip install torch torchvision opencv-python ultralytics numpy supervision gdown label-studio
```

Sometimes, installation can be annoyingly slow due to on-the-fly compilation of some packages.
You can enable parallel compilation by:
```bash
MAKEFLAGS="-j8" pip install ultralytics  
```

Now, you can download a simple lego brick dataset:
```bash
mkdir datasets
cd datasets
gdown 1MjdDdlhCJxajtUJTuSne2c98ImaAItIC
unzip lego-dataset.zip
```

## Creating your own dataset

You can easily create your own dataset by running:

```bash
label-studio
```

Follow the following steps to annotate your own images with bounding boxes:
1. Sign-up (local storage of accounts)
2. Create a project
3. Remove the two existing classes (airplane, ...) and add your own classes
4. Select a bounding box labeling setup
5. Upload your images
6. Annotate them by first clicking on the class (below the image) and then drawing the bounding box (alternatively you can type the number of a class for a quicker annotation workflow)
7. Export as YOLO

## Training

1. **Pretrained Model:** Ensure you have a pretrained YOLO model file (e.g., `yolov8n.pt`) located in the `pretrained/` directory. We store it in this repository for convenience.
2. **Dataset Configuration:** Prepare your dataset configuration file (e.g., `lego-dataset.yaml`), which describes the paths to your training and validation datasets.

Run the training script as follows:

```bash
python train.py
```

This will check for available hardware (CUDA or CPU) and train the model for 200 epochs using the specified image size and device. 

## Running the Live Detection Demo (`demo.py`)

This script demonstrates live object detection using a webcam feed and a trained YOLOv8 model.

- `--webcam-resolution`: Specifies the resolution of the webcam feed (default: `[1280, 720]`).
- `--confidence`: Sets the confidence threshold for detections (default: `0.15`).
- `--model`: Specifies the path to the trained YOLOv8 model file (**required**).

1. Ensure your webcam is connected and recognized by OpenCV.
2. Execute the demo script with the required model argument:

```bash
python demo.py --model path/to/trained-yolov8-model.pt
```

Adjust the `--webcam-resolution` and `--confidence` parameters as needed to optimize performance according to your requirements.

## Interactive Features:

- The detection window will display real-time annotations of detected objects with labels and confidence scores.
- Press `ESC` to exit the demo interface.

## Notes

- **Hardware Acceleration:** For optimal performance, it’s recommended to run the scripts on a machine with a compatible NVIDIA GPU and properly configured CUDA drivers.
- **Adjustments:** Modify training parameters and adjust command-line arguments to fit specific use cases and hardware capabilities.

For further details and troubleshooting, refer to the [official Ultralytics YOLO documentation](https://github.com/ultralytics/yolov5/wiki)