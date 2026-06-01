"""
Inference wrapper around the retinal disease CNN.

Loads trained weights from model/weights.keras (or .h5) and exposes a
simple predict() method that returns probabilities over the four classes:
Normal, Diabetic Retinopathy, Glaucoma, Cataract.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)


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
    # Last conv layer is named so Grad-CAM can target it directly.
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


class DiseasePredictor:
    def __init__(self):
        self.model = _build_model()
        here = os.path.dirname(os.path.abspath(__file__))
        for fname in ("weights.keras", "weights.h5", "model.keras", "model.h5"):
            path = os.path.join(here, fname)
            if os.path.exists(path):
                try:
                    self.model.load_weights(path)
                    print(f"Loaded weights: {fname}")
                    break
                except Exception as e:
                    print(f"Could not load {fname}: {e}")

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed image and return class probabilities."""
        x = image_array
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        return self.model.predict(x, verbose=0)[0]
