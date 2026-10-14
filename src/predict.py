"""
Inference and Prediction script for Custom YOLO Object Detection.

Supports inference on single images, directories, video files, or webcam feeds.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

try:
    from src.utils import get_optimal_device
except ImportError:
    from utils import get_optimal_device


def run_prediction(
    weights_path,
    source,
    conf_thresh=0.25,
    iou_thresh=0.45,
    img_size=640,
    device=None,
    save=True,
    show=False,
    project="runs/detect",
    name="predict"
):
    """
    Run inference on a specified input source.

    Args:
        weights_path (str or Path): Path to trained model weights (.pt).
        source (str or int): Image path, folder path, video path, or webcam index (0).
        conf_thresh (float): Minimum confidence threshold (0.0 to 1.0).
        iou_thresh (float): NMS IoU threshold (0.0 to 1.0).
        img_size (int): Image size in pixels.
        device (str, optional): Target compute device.
        save (bool): Whether to save annotated outputs.
        show (bool): Whether to display results in an interactive OpenCV window.
        project (str): Directory where detections will be saved.
        name (str): Subdirectory name for this prediction run.

    Returns:
        list: Ultralytics Results list.
    """
    weights = Path(weights_path).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights}")

    chosen_device = device if device is not None else get_optimal_device()

    print("\n" + "=" * 60)
    print(f"[INFO] Running Object Detection Inference")
    print(f"  Weights          : {weights}")
    print(f"  Source           : {source}")
    print(f"  Confidence Cutoff: {conf_thresh}")
    print(f"  NMS IoU Cutoff   : {iou_thresh}")
    print(f"  Compute Device   : {chosen_device}")
    print(f"  Save Output      : {save} ({project}/{name})")
    print("=" * 60 + "\n")

    model = YOLO(str(weights))

    # Convert numeric string to integer for webcam (e.g., '0' -> 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    project_path = str(Path(project).resolve())

    results = model.predict(
        source=source,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        device=chosen_device,
        save=save,
        show=show,
        project=project_path,
        name=name,
        verbose=True
    )

    total_detections = 0
    for i, res in enumerate(results):
        num_boxes = len(res.boxes)
        total_detections += num_boxes
        detected_classes = [res.names[int(cls)] for cls in res.boxes.cls]
        print(f"[RESULT] Frame/Image {i + 1}: {num_boxes} objects detected: {detected_classes}")

    print("\n" + "=" * 60)
    print(f"[SUMMARY] Total detected objects across batch: {total_detections}")
    if save:
        output_dir = Path(project_path) / name
        print(f"[OUTPUT] Saved annotated results to: {output_dir.resolve()}")
    print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection inference.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to model weights file (.pt)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/val",
        help="Input image, folder, video file, or webcam index '0'."
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated results.")
    parser.add_argument("--show", action="store_true", default=False, help="Show OpenCV window.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Run save directory.")
    parser.add_argument("--name", type=str, default="predict", help="Prediction run name.")

    args = parser.parse_args()

    run_prediction(
        weights_path=args.weights,
        source=args.source,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
