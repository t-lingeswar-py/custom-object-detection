"""
Model Export script for YOLO Object Detection.

Exports trained PyTorch models (.pt) into production-ready formats
such as ONNX, TorchScript, OpenVINO, CoreML, and TFLite.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(
    weights_path,
    export_format="onnx",
    img_size=640,
    half=False,
    dynamic=False,
    device="cpu"
):
    """
    Export a YOLO model to the target serialization format.

    Args:
        weights_path (str or Path): Path to trained .pt weights.
        export_format (str): Desired format ('onnx', 'torchscript', 'openvino', 'tflite', etc.).
        img_size (int): Model input dimension.
        half (bool): FP16 half-precision quantization.
        dynamic (bool): Support dynamic input batch/dimensions (ONNX).
        device (str): Device to perform export on.

    Returns:
        str: Path of exported model file.
    """
    weights = Path(weights_path).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights}")

    print("\n" + "=" * 60)
    print(f"[INFO] Exporting YOLO Model")
    print(f"  Weights          : {weights}")
    print(f"  Target Format    : {export_format}")
    print(f"  Image Size       : {img_size}")
    print(f"  FP16 Half        : {half}")
    print(f"  Dynamic Input    : {dynamic}")
    print("=" * 60 + "\n")

    model = YOLO(str(weights))
    exported_path = model.export(
        format=export_format,
        imgsz=img_size,
        half=half,
        dynamic=dynamic,
        device=device
    )

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Model successfully exported to: {exported_path}")
    print("=" * 60 + "\n")

    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or other formats.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to trained model weights (.pt)."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "openvino", "engine", "coreml", "tflite"],
        help="Export format."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension.")
    parser.add_argument("--half", action="store_true", default=False, help="Use FP16 half-precision.")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic axes (ONNX).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export ('cpu', '0').")

    args = parser.parse_args()

    export_model(
        weights_path=args.weights,
        export_format=args.format,
        img_size=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        device=args.device
    )


if __name__ == "__main__":
    main()
