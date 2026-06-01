"""
Training script for the retinal disease classifier.

Expects a dataset folder with one subfolder per class:

    data/
        cataract/
        diabetic_retinopathy/
        glaucoma/
        normal/

Usage:
    python train.py --data_dir data --epochs 30 --batch_size 32

Best weights are saved to model/weights.keras and loaded automatically
by the Streamlit app on next launch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers

from model.predict import _build_model, INPUT_SHAPE

CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

FOLDER_ALIASES = {
    "normal": "Normal",
    "no_dr": "Normal",
    "healthy": "Normal",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "diabetic-retinopathy": "Diabetic Retinopathy",
    "diabetic retinopathy": "Diabetic Retinopathy",
    "dr": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "cataract": "Cataract",
}


def normalize_dataset(src: Path, dst: Path) -> Path:
    """Map dataset folder names to the canonical class names used by the app."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    found = {}
    for child in src.iterdir():
        if not child.is_dir():
            continue
        key = child.name.strip().lower().replace("-", "_").replace(" ", "_")
        canon = FOLDER_ALIASES.get(key) or FOLDER_ALIASES.get(child.name.strip().lower())
        if canon is None:
            print(f"  [skip] Unknown class folder: {child.name}")
            continue
        target = dst / canon
        target.mkdir(exist_ok=True)
        for img in child.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                link = target / img.name
                try:
                    os.symlink(img.resolve(), link)
                except OSError:
                    shutil.copy2(img, link)
        found[canon] = len(list(target.iterdir()))

    print("  Class counts:", found)
    absent = [c for c in CLASSES if c not in found]
    if absent:
        raise SystemExit(f"Classes not found: {absent}")
    return dst


def build_datasets(data_dir: Path, image_size, batch_size, val_split, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training", seed=seed,
        image_size=image_size, batch_size=batch_size,
        label_mode="categorical", class_names=CLASSES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation", seed=seed,
        image_size=image_size, batch_size=batch_size,
        label_mode="categorical", class_names=CLASSES,
    )

    augment = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")
    rescale = tf.keras.Sequential([layers.Rescaling(1.0 / 255)], name="rescale")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y),
                        num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds


def compute_class_weights(data_dir: Path) -> dict[int, float]:
    counts = np.array([len(list((data_dir / c).iterdir())) for c in CLASSES],
                      dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    cw = {i: float(w) for i, w in enumerate(weights)}
    print("  Class weights:", cw)
    return cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    args = ap.parse_args()

    src = Path(args.data_dir).resolve()
    if not src.exists():
        raise SystemExit(f"Dataset folder not found: {src}")

    print(">> Normalizing dataset folder structure...")
    norm_dir = normalize_dataset(src, Path("data_normalized").resolve())

    print(">> Building data pipelines...")
    train_ds, val_ds = build_datasets(norm_dir, INPUT_SHAPE[:2],
                                      args.batch_size, args.val_split, args.seed)

    print(">> Computing class weights...")
    class_weights = compute_class_weights(norm_dir)

    print(">> Building model...")
    model = _build_model()
    model.optimizer.learning_rate.assign(args.learning_rate)
    model.summary()

    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "weights.keras"

    cbs = [
        callbacks.ModelCheckpoint(filepath=str(ckpt_path),
                                  monitor="val_accuracy",
                                  save_best_only=True, mode="max", verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1),
    ]

    print(">> Training...")
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        class_weight=class_weights,
                        callbacks=cbs,
                        verbose=2)

    print(">> Evaluating best model...")
    best = tf.keras.models.load_model(ckpt_path)
    val_loss, val_acc = best.evaluate(val_ds, verbose=0)
    print(f"   val_loss={val_loss:.4f}  val_accuracy={val_acc:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps({
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "epochs_run": len(history.history["loss"]),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "class_order": CLASSES,
    }, indent=2))

    print(f">> Saved weights: {ckpt_path}")


if __name__ == "__main__":
    main()
