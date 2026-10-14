"""
Sample Synthetic Dataset Generator for YOLO Object Detection.

This script generates synthetic images containing geometric shapes (circle,
rectangle, triangle) with corresponding ground-truth bounding box labels
in standard YOLO format (class_id x_center y_center width height).
"""

import argparse
import math
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw


def generate_single_sample(image_size, classes, max_objects=5):
    """
    Generate a single synthetic image with random shapes and annotations.
    
    Args:
        image_size (tuple): Width and height of the image.
        classes (list): List of class names.
        max_objects (int): Maximum number of shapes to place.
        
    Returns:
        tuple: (PIL.Image, list of YOLO label strings)
    """
    width, height = image_size
    bg_color = (
        random.randint(230, 255),
        random.randint(230, 255),
        random.randint(230, 255)
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    labels = []
    num_objects = random.randint(1, max_objects)

    for _ in range(num_objects):
        class_id = random.randint(0, len(classes) - 1)
        class_name = classes[class_id]

        obj_w = random.randint(30, int(width * 0.35))
        obj_h = random.randint(30, int(height * 0.35))

        x0 = random.randint(5, width - obj_w - 5)
        y0 = random.randint(5, height - obj_h - 5)
        x1 = x0 + obj_w
        y1 = y0 + obj_h

        color = (
            random.randint(20, 200),
            random.randint(20, 200),
            random.randint(20, 200)
        )

        if class_name == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=2)
        elif class_name == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=2)
        elif class_name == "triangle":
            pt1 = (x0 + obj_w // 2, y0)
            pt2 = (x0, y1)
            pt3 = (x1, y1)
            draw.polygon([pt1, pt2, pt3], fill=color, outline=(0, 0, 0))

        # Calculate normalized YOLO coordinates: x_center, y_center, width, height
        x_center = (x0 + (obj_w / 2.0)) / width
        y_center = (y0 + (obj_h / 2.0)) / height
        norm_w = obj_w / float(width)
        norm_h = obj_h / float(height)

        # Clamping to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return img, labels


def create_dataset(output_dir, train_count=60, val_count=20, img_size=(416, 416)):
    """
    Generate training and validation splits.
    
    Args:
        output_dir (str or Path): Destination root directory.
        train_count (int): Number of training images.
        val_count (int): Number of validation images.
        img_size (tuple): Image dimensions (width, height).
    """
    classes = ["circle", "rectangle", "triangle"]
    output_path = Path(output_dir).resolve()

    splits = {
        "train": train_count,
        "val": val_count
    }

    for split_name, count in splits.items():
        img_dir = output_path / "images" / split_name
        lbl_dir = output_path / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {count} samples for split: {split_name}...")

        for i in range(count):
            img, labels = generate_single_sample(img_size, classes)
            file_stem = f"sample_{split_name}_{i:04d}"

            img_file = img_dir / f"{file_stem}.jpg"
            lbl_file = lbl_dir / f"{file_stem}.txt"

            img.save(img_file, format="JPEG", quality=95)
            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

    print(f"Dataset generated successfully at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO dataset for testing.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent),
        help="Path to output data directory."
    )
    parser.add_argument("--train-count", type=int, default=60, help="Number of training samples.")
    parser.add_argument("--val-count", type=int, default=20, help="Number of validation samples.")
    parser.add_argument("--img-size", type=int, default=416, help="Image size (square pixels).")

    args = parser.parse_args()
    create_dataset(args.output, args.train_count, args.val_count, (args.img_size, args.img_size))
