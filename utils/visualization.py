"""Plot helpers for the Streamlit app."""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG_DARK = "#0f172a"
PANEL_DARK = "#1e293b"
TEXT_LIGHT = "#e2e8f0"
GRID_LIGHT = "#334155"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_PINK = "#ec4899"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"


def _style_axis(ax):
    ax.set_facecolor(PANEL_DARK)
    ax.tick_params(colors=TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_color(GRID_LIGHT)
    ax.title.set_color(TEXT_LIGHT)
    ax.xaxis.label.set_color(TEXT_LIGHT)
    ax.yaxis.label.set_color(TEXT_LIGHT)
    ax.grid(True, axis="x", color=GRID_LIGHT, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def create_confidence_chart(prediction, class_names, confidence_threshold=50):
    """Horizontal bar chart of per-class confidence."""
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]

    colors = [ACCENT_GREEN if v >= confidence_threshold else ACCENT_PURPLE for v in values]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_DARK)
    _style_axis(ax)

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
    ax.axvline(confidence_threshold, color=ACCENT_RED, linestyle="--",
               linewidth=1.5, alpha=0.7,
               label=f"Threshold ({confidence_threshold}%)")
    leg = ax.legend(loc="lower right", fontsize=9, framealpha=0.3)
    for text in leg.get_texts():
        text.set_color(TEXT_LIGHT)
    ax.set_title("Per-class Confidence", fontsize=12, fontweight="bold", pad=12)

    for bar, v in zip(bars, values):
        ax.text(min(v + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10, color=TEXT_LIGHT,
                fontweight="bold")

    fig.tight_layout()
    return fig


def create_report(image, predicted_class, confidence, prediction,
                  class_names, heatmap_image=None):
    """Build a PNG report combining the image, prediction chart, and heatmap."""
    n_cols = 3 if heatmap_image is not None else 2
    fig = plt.figure(figsize=(5 * n_cols, 6), facecolor=BG_DARK)

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.set_facecolor(PANEL_DARK)
    ax1.imshow(image)
    ax1.set_title("Uploaded Image", color=TEXT_LIGHT, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, n_cols, 2)
    _style_axis(ax2)
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]
    ax2.barh(names, values, color=ACCENT_PURPLE, edgecolor="none", height=0.55)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)", color=TEXT_LIGHT)
    ax2.set_title(f"Prediction: {predicted_class} ({confidence:.1f}%)",
                  color=TEXT_LIGHT, fontweight="bold")

    if heatmap_image is not None:
        ax3 = fig.add_subplot(1, n_cols, 3)
        ax3.set_facecolor(PANEL_DARK)
        ax3.imshow(heatmap_image)
        ax3.set_title("Grad-CAM Heatmap", color=TEXT_LIGHT, fontweight="bold")
        ax3.axis("off")

    fig.suptitle("Retinal Eye Disease Detector — Report",
                 fontsize=15, fontweight="bold", color=TEXT_LIGHT, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
