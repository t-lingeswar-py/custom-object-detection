"""
Unit and Integration tests for Custom Object Detection Pipeline.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure parent and project root are in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.sample_dataset_generator import generate_single_sample
from src.utils import load_yaml_config, get_optimal_device
from ultralytics import YOLO


class TestObjectDetectionPipeline(unittest.TestCase):

    def test_sample_generator(self):
        """Test synthetic sample generation and label format."""
        classes = ["circle", "rectangle", "triangle"]
        img, labels = generate_single_sample((200, 200), classes, max_objects=3)

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (200, 200))
        self.assertGreaterEqual(len(labels), 1)

        # Check YOLO label format: class_id x_center y_center width height
        for label in labels:
            parts = label.split()
            self.assertEqual(len(parts), 5)
            class_id = int(parts[0])
            self.assertIn(class_id, [0, 1, 2])
            coords = [float(v) for v in parts[1:]]
            for c in coords:
                self.assertGreaterEqual(c, 0.0)
                self.assertLessEqual(c, 1.0)

    def test_yaml_config_loader(self):
        """Test loading dataset yaml configuration."""
        yaml_path = PROJECT_ROOT / "config" / "data.yaml"
        self.assertTrue(yaml_path.exists())

        config = load_yaml_config(yaml_path)
        self.assertIn("names", config)
        self.assertIn("train", config)
        self.assertIn("val", config)

    def test_optimal_device_detection(self):
        """Test compute device resolution."""
        device = get_optimal_device()
        self.assertIn(device, ["0", "cpu", "mps"])

    def test_model_inference_dummy(self):
        """Test base YOLO forward pass on dummy image."""
        model = YOLO("yolov8n.pt")
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        results = model.predict(source=dummy_img, verbose=False)

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].boxes)


if __name__ == "__main__":
    unittest.main()
