# Custom Object Detection Pipeline

An end-to-end computer vision pipeline for custom object detection built on YOLOv8 and YOLO11. This project provides a complete workflow from synthetic/custom dataset generation and model training to evaluation, inference, ONNX export, and an interactive Gradio web application.

---

## Key Features

- **Multi-Architecture Support**: Native support for YOLOv8 and YOLO11 architectures (`n`, `s`, `m`, `l`, `x` variants).
- **Automated Hardware Acceleration**: Automatically switches between CUDA GPU, Apple Silicon MPS, or multi-threaded CPU.
- **Out-of-the-Box Dataset Generator**: Generates synthetic annotated samples with standardized YOLO bounding boxes (`class_id x_center y_center width height`) for immediate training and verification.
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
|   |-- sample_dataset_generator.py # Synthetic dataset generator in YOLO format
|   |-- images/                 # Image repository (train / val)
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

### Step 1: Generate Sample Dataset
To start training immediately without downloading external datasets, generate synthetic training and validation images:
```bash
python data/sample_dataset_generator.py --train-count 60 --val-count 20 --img-size 416
```
This generates labeled geometric shapes (circle, rectangle, triangle) in `data/images/` and `data/labels/`.

### Step 2: Train the Detector
Train a lightweight YOLO model using the generated dataset:
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 10 --imgsz 416 --batch 8
```
Artifacts and best weights are saved automatically to `runs/train/custom_detector/weights/best.pt`.

### Step 3: Evaluate Model Performance
Validate trained checkpoints and calculate metrics:
```bash
python src/evaluate.py --weights runs/train/custom_detector/weights/best.pt --data config/data.yaml --imgsz 416
```

### Step 4: Run Inference
Test your model on new images, directories, or a webcam feed:
```bash
# On validation folder:
python src/predict.py --weights runs/train/custom_detector/weights/best.pt --source data/images/val

# On single image:
python src/predict.py --weights runs/train/custom_detector/weights/best.pt --source path/to/image.jpg

# Live webcam (index 0):
python src/predict.py --weights runs/train/custom_detector/weights/best.pt --source 0
```

### Step 5: Export Model to ONNX
Export trained weights for deployment on mobile, edge devices, or cloud runtimes:
```bash
python src/export.py --weights runs/train/custom_detector/weights/best.pt --format onnx --imgsz 416
```

---

## Using Custom Datasets

To train on your own real-world images (e.g., from Roboflow, CVAT, or LabelImg):

1. Organize your files in the YOLO layout:
   ```text
   data/
   |-- images/
   |   |-- train/  (jpg/png files)
   |   `-- val/    (jpg/png files)
   `-- labels/
       |-- train/  (txt files with matching filenames)
       `-- val/    (txt files with matching filenames)
   ```

2. Each label `.txt` file must contain one row per object:
   ```text
   <class_id> <x_center> <y_center> <width> <height>
   ```
   Values must be normalized between `0.0` and `1.0`.

3. Update `config/data.yaml` with your class names:
   ```yaml
   path: ../data
   train: images/train
   val: images/val
   names:
     0: helmet
     1: vest
     2: boots
   ```

4. Launch training:
   ```bash
   python src/train.py --data config/data.yaml --epochs 50 --imgsz 640 --batch 16
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
