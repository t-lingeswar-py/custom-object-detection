"""
Evaluation and Validation script for Custom YOLO Object Detection.

Computes precision, recall, mAP@0.50, mAP@0.50:0.95 and class-level statistics.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

try:
    from src.utils import get_optimal_device
except ImportError:
    from utils import get_optimal_device


def evaluate_model(weights_path, data_yaml, img_size=640, batch_size=16, device=None, project="runs/val", name="eval"):
    """
    Evaluate a trained YOLO model on validation dataset.

    Args:
        weights_path (str or Path): Path to trained weights (.pt).
        data_yaml (str or Path): Path to dataset yaml configuration.
        img_size (int): Image size in pixels.
        batch_size (int): Validation batch size.
        device (str, optional): Target compute device.
        project (str): Directory for validation outputs.
        name (str): Evaluation run name.

    Returns:
        dict: Validation metrics summary.
    """
    weights = Path(weights_path).resolve()
    data_path = Path(data_yaml).resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    chosen_device = device if device is not None else get_optimal_device()

    print("\n" + "=" * 60)
    print(f"[INFO] Evaluating YOLO Model")
    print(f"  Weights          : {weights}")
    print(f"  Dataset Config   : {data_path}")
    print(f"  Image Size       : {img_size}")
    print(f"  Compute Device   : {chosen_device}")
    print("=" * 60 + "\n")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        imgsz=img_size,
        batch=batch_size,
        device=chosen_device,
        project=project,
        name=name,
        verbose=True
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print("\n" + "=" * 60)
    print("           VALIDATION PERFORMANCE METRICS SUMMARY           ")
    print("=" * 60)
    print(f"  Precision (B)       : {precision:.4f}")
    print(f"  Recall (B)          : {recall:.4f}")
    print(f"  mAP @ 0.50          : {map50:.4f}")
    print(f"  mAP @ 0.50:0.95     : {map50_95:.4f}")
    print("=" * 60 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to trained model weights (.pt)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "data.yaml"),
        help="Path to data.yaml dataset config file."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--project", type=str, default="runs/val", help="Run save directory.")
    parser.add_argument("--name", type=str, default="eval", help="Evaluation name.")

    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
