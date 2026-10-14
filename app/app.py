"""
Interactive Web Application for Custom Object Detection using Gradio.

Allows users to upload images, test webcam input, tune confidence/IoU
thresholds, switch model weights, and view structured bounding box detections.
"""

import os
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# Global cache for loaded models
LOADED_MODELS = {}


def get_model(weights_path):
    """
    Load or retrieve cached YOLO model instance.
    """
    if weights_path not in LOADED_MODELS:
        LOADED_MODELS[weights_path] = YOLO(weights_path)
    return LOADED_MODELS[weights_path]


def detect_objects_in_image(input_image, weights_path, conf_thresh, iou_thresh):
    """
    Run detection on an input image.

    Args:
        input_image (PIL.Image or np.ndarray): Input image.
        weights_path (str): Path or model name (e.g. yolov8n.pt, runs/train/.../best.pt).
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): NMS IoU threshold.

    Returns:
        tuple: (annotated_image, pandas.DataFrame detection summary)
    """
    if input_image is None:
        return None, pd.DataFrame(columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])

    try:
        model = get_model(weights_path)
    except Exception as e:
        error_msg = f"[ERROR] Failed to load model '{weights_path}': {str(e)}"
        print(error_msg)
        return input_image, pd.DataFrame([{"Class": "Error", "Confidence": 0.0, "X1": 0, "Y1": 0, "X2": 0, "Y2": 0}])

    # Run inference
    results = model.predict(
        source=input_image,
        conf=float(conf_thresh),
        iou=float(iou_thresh),
        verbose=False
    )

    first_result = results[0]
    # Plot detections (returns BGR numpy array)
    annotated_bgr = first_result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    records = []
    for box in first_result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = first_result.names.get(cls_id, str(cls_id))
        conf_val = round(float(box.conf[0].item()), 4)
        xyxy = box.xyxy[0].tolist()

        records.append({
            "Class": cls_name,
            "Confidence": conf_val,
            "X1": round(xyxy[0], 1),
            "Y1": round(xyxy[1], 1),
            "X2": round(xyxy[2], 1),
            "Y2": round(xyxy[3], 1)
        })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])

    return annotated_rgb, df


def build_app():
    """
    Construct the Gradio UI layout and event bindings.
    """
    available_weights = [
        "runs/train/car_detector/weights/best.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt"
    ]

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    )

    with gr.Blocks(theme=custom_theme, title="Custom Object Detection Studio") as demo:
        gr.Markdown(
            """
            # Custom Object Detection Studio
            High-performance object detection pipeline powered by YOLO.
            Upload an image, adjust detection parameters, and inspect detections in real-time.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Input Image")

                model_selector = gr.Dropdown(
                    choices=available_weights,
                    value="runs/train/car_detector/weights/best.pt",
                    label="Model Weights",
                    info="Choose a pretrained YOLO checkpoint or custom trained best.pt"
                )

                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum probability score to accept detection"
                )

                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="IoU (NMS) Threshold",
                    info="Intersection over Union threshold for Non-Maximum Suppression"
                )

                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(type="numpy", label="Detections Visualizer")
                output_table = gr.Dataframe(
                    headers=["Class", "Confidence", "X1", "Y1", "X2", "Y2"],
                    label="Detection Results Table"
                )

        detect_btn.click(
            fn=detect_objects_in_image,
            inputs=[input_img, model_selector, conf_slider, iou_slider],
            outputs=[output_img, output_table]
        )

        gr.Markdown(
            """
            ### Pipeline Details
            - Framework: Ultralytics YOLOv8 / YOLO11
            - Bounding Box Format: [X1, Y1, X2, Y2] pixel coordinates
            - Hardware Acceleration: CUDA / MPS / CPU automatic fallback
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
