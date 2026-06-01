"""Plot helpers for the Streamlit app."""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_confidence_chart(prediction, class_names, confidence_threshold=50):
    """Horizontal bar chart of per-class confidence."""
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]

    colors = ["#2ecc71" if v >= confidence_threshold else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(names, values, color=colors, edgecolor="#1a1a2e")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.axvline(confidence_threshold, color="#e74c3c", linestyle="--",
               linewidth=1, label=f"Threshold ({confidence_threshold}%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Per-class Confidence")
    for bar, v in zip(bars, values):
        ax.text(min(v + 1, 96), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def create_report(image, predicted_class, confidence, prediction,
                  class_names, heatmap_image=None):
    """Build a PNG report combining the image, prediction chart, and heatmap."""
    n_cols = 3 if heatmap_image is not None else 2
    fig = plt.figure(figsize=(5 * n_cols, 6))

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.imshow(image)
    ax1.set_title("Uploaded Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, n_cols, 2)
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]
    ax2.barh(names, values, color="#3498db", edgecolor="#1a1a2e")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title(f"Prediction: {predicted_class} ({confidence:.1f}%)")

    if heatmap_image is not None:
        ax3 = fig.add_subplot(1, n_cols, 3)
        ax3.imshow(heatmap_image)
        ax3.set_title("Grad-CAM Heatmap")
        ax3.axis("off")

    fig.suptitle("Retinal Eye Disease Detector — Report", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
