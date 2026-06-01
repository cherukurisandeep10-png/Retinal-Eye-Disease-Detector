"""Image preprocessing for the retinal disease classifier."""

import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = (224, 224)


def preprocess_image(image) -> np.ndarray:
    """Resize, enhance contrast, and normalize a fundus image for the CNN."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BILINEAR)
    image = ImageOps.autocontrast(image, cutoff=1)

    return np.asarray(image, dtype=np.float32) / 255.0
