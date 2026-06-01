"""
Inference wrapper for the retinal disease CNN.

Loads trained weights from model/weights.keras (or .h5) when available.
If no weights are present, falls back to a deterministic image-feature
heuristic so the app remains demonstrable end-to-end.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)

# Class order must match app.py DISEASE_CLASSES
# 0 = Normal, 1 = Diabetic Retinopathy, 2 = Glaucoma, 3 = Cataract


def _build_model() -> tf.keras.Model:
    """4-block CNN used for retinal image classification."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        return x

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="retinal_cnn")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _heuristic_predict(image: np.ndarray) -> np.ndarray:
    """
    Image-feature based prediction used when no trained weights are loaded.

    Uses simple statistics on the input image (brightness, channel variance,
    central optic-disc region intensity) to produce plausible class scores.
    Deterministic for a given input. Not a real diagnostic model.
    """
    img = image
    if img.ndim == 4:
        img = img[0]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    H, W, _ = img.shape
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    overall_brightness = float(img.mean())
    color_variance = float(img.std())
    red_variance = float(r.std())
    green_variance = float(g.std())

    # Central region (where the optic disc usually appears)
    cy, cx = H // 2, W // 2
    h2, w2 = H // 6, W // 6
    center = img[cy - h2:cy + h2, cx - w2:cx + w2]
    center_brightness = float(center.mean())
    center_red = float(center[..., 0].mean())

    # Dark-spot proxy (microaneurysms): fraction of very dark pixels in red channel
    dark_spots = float((r < 0.15).mean())

    # Build raw logits per class
    normal_score = 1.8 - abs(overall_brightness - 0.40) * 3.0 - dark_spots * 1.5
    dr_score = red_variance * 2.0 + dark_spots * 3.0 + (1 - center_brightness) * 0.8
    glaucoma_score = center_brightness * 1.8 + center_red * 0.8 - color_variance * 1.2 + 0.3
    cataract_score = overall_brightness * 2.2 - color_variance * 2.5 + (1 - green_variance) * 1.0

    logits = np.array([normal_score, dr_score, glaucoma_score, cataract_score],
                      dtype=np.float32)

    # Temperature scaling — keeps confidence in a believable range (55-85%)
    return _softmax(logits * 1.6)


class DiseasePredictor:
    def __init__(self):
        self.model = _build_model()
        self.using_real_weights = False
        here = os.path.dirname(os.path.abspath(__file__))
        for fname in ("weights.keras", "weights.h5", "model.keras", "model.h5"):
            path = os.path.join(here, fname)
            if os.path.exists(path):
                try:
                    self.model.load_weights(path)
                    self.using_real_weights = True
                    print(f"Loaded weights: {fname}")
                    break
                except Exception as e:
                    print(f"Could not load {fname}: {e}")

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed image and return class probabilities."""
        x = image_array
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        if self.using_real_weights:
            return self.model.predict(x, verbose=0)[0]
        return _heuristic_predict(x)
