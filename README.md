# Custom Car Object Detection Pipeline

An end-to-end computer vision pipeline for custom object detection built on YOLOv8 and YOLO11. This project provides a complete workflow from dataset ingestion and annotation conversion (CSV to YOLO format) to model training, evaluation, inference, ONNX export, and an interactive Gradio web application.

---

## Benchmark Results (Car Detection Model)

Trained on 284 training images and validated on 71 images (117 car instances):

| Metric | Score |
|---|---|
| Precision (Box) | 98.0% |
| Recall (Box) | 99.1% |
| mAP @ 0.50 | 99.3% |
| mAP @ 0.50:0.95 | 65.8% |
| Inference Latency | ~32 ms / frame (CPU) |

Checkpoints are saved at `runs/train/car_detector/weights/best.pt`.

---

## Key Features

- **Multi-Architecture Support**: Native support for YOLOv8 and YOLO11 architectures (`n`, `s`, `m`, `l`, `x` variants).
- **Automated Hardware Acceleration**: Automatically switches between CUDA GPU, Apple Silicon MPS, or multi-threaded CPU.
- **Automated Dataset Converter**: Dedicated ingestion script converting CSV bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`) to normalized YOLO format (`0 x_center y_center width height`).
- **Production Export Pipeline**: One-command export to ONNX, TorchScript, OpenVINO, TFLite, and CoreML.
- **Interactive Web Application**: Gradio-based interface for image and video inference, real-time threshold adjustments, and tabular detection coordinates.
- **Modular Codebase**: Strict separation between configuration, data management, training, evaluation, inference, and UI.

---

## Project Structure

```text
custom-object-detection/
|-- .gitignore                  # Git ignore rules for checkpoints, caches, and runs
|-- README.md                   # Comprehensive project documentation
|-- requirements.txt            # Python dependencies
|-- config/
|   `-- data.yaml               # YOLO dataset configuration and class definitions
|-- data/
|   |-- ingest_car_dataset.py   # Dataset ingestion & CSV-to-YOLO converter
|   |-- sample_dataset_generator.py # Synthetic dataset generator in YOLO format
|   |-- images/                 # Image repository (train / val / test)
|   `-- labels/                 # Annotation text files (train / val)
|-- src/
|   |-- __init__.py
|   |-- train.py                # Model training script with early stopping & metrics
|   |-- evaluate.py             # Validation script (Precision, Recall, mAP50, mAP50-95)
|   |-- predict.py              # CLI inference on images, folders, videos, or webcam
|   |-- export.py               # Serialization to ONNX, TorchScript, and TFLite
|   `-- utils.py                # Dataset validation and visualization helpers
|-- app/
|   `-- app.py                  # Interactive Gradio web interface
`-- tests/
    `-- test_pipeline.py        # Automated test suite
```

---

## Installation

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd custom-object-detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quickstart

### Step 1: Ingest and Convert Dataset
To convert the car dataset from CSV bounding boxes to YOLO format and split into train/val sets:
```bash
python data/ingest_car_dataset.py
```

### Step 2: Train the Detector
Train a lightweight YOLO model using the ingested dataset:
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 10 --imgsz 640 --batch 16 --name car_detector
```
Artifacts and best weights are saved automatically to `runs/train/car_detector/weights/best.pt`.

### Step 3: Evaluate Model Performance
Validate trained checkpoints and calculate metrics:
```bash
python src/evaluate.py --weights runs/train/car_detector/weights/best.pt --data config/data.yaml --imgsz 640
```

### Step 4: Run Inference
Test your model on new images, directories, or a webcam feed:
```bash
# On test images folder:
python src/predict.py --weights runs/train/car_detector/weights/best.pt --source data/images/test --name test_cars

# On single image:
python src/predict.py --weights runs/train/car_detector/weights/best.pt --source data/images/test/vid_5_26660.jpg

# Live webcam (index 0):
python src/predict.py --weights runs/train/car_detector/weights/best.pt --source 0
```

### Step 5: Export Model to ONNX
Export trained weights for deployment on mobile, edge devices, or cloud runtimes:
```bash
python src/export.py --weights runs/train/car_detector/weights/best.pt --format onnx --imgsz 640
```

---

## Interactive Web UI

Launch the local web dashboard for visual testing and demonstration:
```bash
python app/app.py
```
Open your browser and navigate to `http://127.0.0.1:7860`.

Features:
- Drag-and-drop image upload
- Live threshold tuning (Confidence cutoff and NMS IoU)
- Real-time bounding box visualization
- Tabular bounding box coordinates export

---

## Running Tests

Run the test suite to ensure dataset generators and inference pipelines function correctly:
```bash
python -m unittest tests/test_pipeline.py
```

---

## License

This project is licensed under the MIT License.
