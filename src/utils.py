"""
Utility functions for dataset validation, configuration loading,
and visualization for custom object detection.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch


def load_yaml_config(config_path):
    """
    Safely load a YAML configuration file.

    Args:
        config_path (str or Path): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def get_optimal_device():
    """
    Determine the optimal compute device available (CUDA, MPS, or CPU).

    Returns:
        str: Device identifier string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA GPU detected: {device_name}")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Apple Silicon MPS detected.")
        return "mps"
    else:
        print("[INFO] No GPU accelerator found. Using CPU.")
        return "cpu"


def validate_dataset(data_yaml_path):
    """
    Validate that the dataset structure and referenced paths exist.

    Args:
        data_yaml_path (str or Path): Path to the dataset YAML file.

    Returns:
        bool: True if dataset structure is valid.
    """
    data = load_yaml_config(data_yaml_path)
    base_dir = Path(data_yaml_path).parent

    path_str = data.get("path", ".")
    dataset_root = (base_dir / path_str).resolve()
    if not dataset_root.exists():
        dataset_root = (base_dir.parent / path_str).resolve()
    if not dataset_root.exists():
        dataset_root = Path(path_str).resolve()

    train_rel = data.get("train", "images/train")
    val_rel = data.get("val", "images/val")

    train_path = (dataset_root / train_rel).resolve()
    val_path = (dataset_root / val_rel).resolve()

    print(f"[CHECK] Verifying dataset at root: {dataset_root.resolve()}")

    if not train_path.exists():
        print(f"[WARNING] Training directory missing: {train_path}")
        return False
    if not val_path.exists():
        print(f"[WARNING] Validation directory missing: {val_path}")
        return False

    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))

    print(f"[OK] Training images found: {len(train_images)}")
    print(f"[OK] Validation images found: {len(val_images)}")
    print(f"[OK] Classes registered: {data.get('names')}")

    return len(train_images) > 0 and len(val_images) > 0


def draw_bounding_boxes(image_np, detections, class_names=None):
    """
    Draw bounding boxes on an image numpy array.

    Args:
        image_np (np.ndarray): Image in RGB format.
        detections (list): List of dicts containing 'box' (x1, y1, x2, y2),
                          'confidence', and 'class_id'.
        class_names (dict or list, optional): Mapping of class_id to name.

    Returns:
        np.ndarray: Annotated RGB image.
    """
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for det in detections:
        box = det["box"]
        conf = det.get("confidence", 1.0)
        cls_id = det.get("class_id", 0)

        label_name = class_names[cls_id] if class_names and cls_id in class_names else str(cls_id)
        label_text = f"{label_name} {conf:.2f}"

        x1, y1, x2, y2 = [int(v) for v in box]

        color = (0, 255, 0)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            img_bgr,
            label_text,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
