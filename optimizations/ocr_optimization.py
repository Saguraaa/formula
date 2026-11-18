"""OCR-only training augmentation."""
from .common import photometric


def augment(image, rng):
    """Perturb crop appearance without changing its text or geometry."""
    return photometric(image, rng)
