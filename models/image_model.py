"""
Medical image analysis with lightweight heuristics.

This keeps deployment small and avoids TensorFlow at request startup,
which is a poor fit for Vercel's serverless environment.
"""

import numpy as np
from PIL import Image


class ImagePredictor:
    def __init__(self):
        """Initialize image prediction settings."""
        self.img_size = (224, 224)

    def preprocess_image(self, image_path):
        """Preprocess image for simple statistical analysis."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

    def predict(self, image_path):
        """Estimate image risk using contrast and intensity cues."""
        img_array = self.preprocess_image(image_path)
        grayscale = img_array.mean(axis=2)

        brightness = float(grayscale.mean())
        contrast = float(grayscale.std())
        hotspot_ratio = float((grayscale > 0.75).mean())
        dark_ratio = float((grayscale < 0.25).mean())

        risk = (
            0.35 * contrast +
            0.30 * hotspot_ratio +
            0.20 * dark_ratio +
            0.15 * abs(0.5 - brightness)
        )
        risk = float(np.clip(risk, 0.0, 1.0))

        confidence = 0.70 + min(contrast, 0.25)
        confidence = float(np.clip(confidence, 0.70, 0.92))

        return risk, confidence

    def get_gradcam_visualization(self, image_path):
        """Placeholder retained for API compatibility."""
        return None
