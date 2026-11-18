"""Shared, label-preserving image transformations for the experiments."""

import cv2
import numpy as np

#加噪声
def photometric(image, rng):
    """Apply reproducible mild exposure, noise, and optional blur to RGB pixels."""
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.9, 1.1)
    sigma = rng.uniform(2, 5)
    pixels = (image.astype(np.float32) - 127.5) * contrast + 127.5
    pixels *= brightness
    # Reuse one spatial noise field across channels to avoid strong color speckle.
    noise = rng.standard_normal(image.shape[:2], dtype=np.float32)
    pixels += noise[..., None] * sigma
    result = np.clip(pixels, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        result = cv2.GaussianBlur(result, (3, 3), 0.6)
    return result
