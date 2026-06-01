"""Grad-CAM visualization for the retinal CNN."""

import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower() and len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam_heatmap(model, processed_image: np.ndarray,
                             class_index: int, layer_name: str | None = None):
    """Generate a Grad-CAM overlay on the input image for a given target class."""
    img = processed_image
    if img.ndim == 3:
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
    else:
        img_batch = img.astype(np.float32)
        img = img[0]

    if layer_name is None:
        try:
            model.get_layer("last_conv")
            layer_name = "last_conv"
        except Exception:
            layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        cam = np.ones(conv_outputs.shape[1:3], dtype=np.float32)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    cam_img = Image.fromarray(cam).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0

    heat = np.zeros((cam_arr.shape[0], cam_arr.shape[1], 3), dtype=np.float32)
    heat[..., 0] = cam_arr
    heat[..., 1] = np.clip(cam_arr - 0.5, 0, 1)
    heat[..., 2] = 0.0

    base = (img * 255.0).astype(np.float32) / 255.0
    overlay = np.clip(base * 0.55 + heat * 0.45, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))
