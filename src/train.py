"""
Training script for Custom YOLO Object Detection.

Supports YOLOv8 and YOLO11 architectures with configurable hyperparameters,
early stopping, automatic device detection, and validation logging.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Local imports
try:
    from src.utils import get_optimal_device, validate_dataset
except ImportError:
    from utils import get_optimal_device, validate_dataset


def train_model(
    data_yaml,
    model_name="yolov8n.pt",
    epochs=50,
    batch_size=16,
    img_size=640,
    device=None,
    project="runs/train",
    experiment_name="custom_detector",
    patience=10,
    pretrained=True
):
    """
    Train a custom YOLO model.

    Args:
        data_yaml (str or Path): Path to the dataset configuration file.
        model_name (str): YOLO model weights or architecture (e.g., yolov8n.pt, yolo11n.pt).
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        img_size (int): Image size in pixels.
        device (str, optional): Target compute device ('0', 'cpu', 'mps').
        project (str): Directory where training runs are saved.
        experiment_name (str): Subdirectory name for this training run.
        patience (int): Epochs to wait for early stopping.
        pretrained (bool): Whether to use pretrained backbone weights.

    Returns:
        ultralytics.engine.results.Results: Training results object.
    """
    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    # Dataset check
    if not validate_dataset(data_path):
        print("[WARNING] Dataset validation indicated missing directories or zero images.")
        print("[INFO] Attempting to continue if images exist...")

    # Determine hardware device
    chosen_device = device if device is not None else get_optimal_device()

    print("\n" + "=" * 60)
    print(f"[INFO] Initializing YOLO training pipeline")
    print(f"  Base Model       : {model_name}")
    print(f"  Dataset Config   : {data_path}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Image Size       : {img_size}")
    print(f"  Compute Device   : {chosen_device}")
    print(f"  Early Stopping   : patience={patience}")
    print(f"  Output Directory : {project}/{experiment_name}")
    print("=" * 60 + "\n")

    # Load model
    model = YOLO(model_name)

    project_path = str(Path(project).resolve())

    # Launch training
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=chosen_device,
        project=project_path,
        name=experiment_name,
        patience=patience,
        pretrained=pretrained,
        plots=True,
        verbose=True
    )

    best_weights = Path(project_path) / experiment_name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("[SUCCESS] Training completed successfully!")
    if best_weights.exists():
        print(f"[OUTPUT] Best model weights saved at: {best_weights.resolve()}")
    print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO object detector.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "data.yaml"),
        help="Path to data.yaml dataset config file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model architecture (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--project", type=str, default="runs/train", help="Run save directory.")
    parser.add_argument("--name", type=str, default="custom_detector", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        experiment_name=args.name,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
