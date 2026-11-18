import cv2
import numpy as np

from .common import photometric

#加噪+旋转
def augment(image, boxes, rng, geometric=False):
    image = photometric(image, rng)
    if not geometric:
        return image, boxes.copy()

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-3, 3), 1)
    if not len(boxes):
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE), boxes.copy()

    centers = boxes[:, :2] * (width, height)
    half_sizes = boxes[:, 2:] * (width, height) / 2
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    corners = centers[:, None, :] + half_sizes[:, None, :] * offsets
    rotated = corners @ matrix[:, :2].T + matrix[:, 2]
    lower, upper = rotated.min(axis=1), rotated.max(axis=1)
    if np.any(lower < 0) or np.any(upper > (width, height)):
        # Keep the photometric result if rotation would cut off a formula.
        return image, boxes.copy()

    transformed = np.concatenate(((lower + upper) / 2, upper - lower), axis=1)
    transformed /= (width, height, width, height)
    image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return image, transformed
