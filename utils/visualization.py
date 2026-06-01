"""Plot helpers for the Streamlit app."""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#ffffff"
PANEL = "#f7f9fc"
TEXT = "#2c3e50"
MUTED = "#5a6c7d"
GRID = "#e3e8ef"
PRIMARY = "#0b5394"
ACCENT_OK = "#2e7d32"
ACCENT_WARN = "#c62828"


def _style_axis(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(True, axis="x", color=GRID, alpha=0.7, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)


def create_confidence_chart(prediction, class_names, confidence_threshold=50):
    """Horizontal bar chart of per-class confidence."""
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]

    colors = [ACCENT_OK if v >= confidence_threshold else PRIMARY for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    _style_axis(ax)

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=9.5)
    ax.axvline(confidence_threshold, color=ACCENT_WARN, linestyle="--",
               linewidth=1.2, alpha=0.55,
               label=f"Threshold ({confidence_threshold}%)")
    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(MUTED)

    for bar, v in zip(bars, values):
        ax.text(min(v + 1.5, 96), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9.5, color=TEXT,
                fontweight="600")

    fig.tight_layout()
    return fig


def create_report(image, predicted_class, confidence, prediction,
                  class_names, heatmap_image=None):
    """Build a PNG report combining the image, prediction chart, and heatmap."""
    n_cols = 3 if heatmap_image is not None else 2
    fig = plt.figure(figsize=(5 * n_cols, 5.5), facecolor=BG)

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.set_facecolor(PANEL)
    ax1.imshow(image)
    ax1.set_title("Input Image", color=TEXT, fontweight="600", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, n_cols, 2)
    _style_axis(ax2)
    probs = np.asarray(prediction).flatten() * 100
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = probs[order]
    ax2.barh(names, values, color=PRIMARY, edgecolor="none", height=0.55)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)", fontsize=9.5)
    ax2.set_title(f"Prediction: {predicted_class} ({confidence:.1f}%)",
                  color=TEXT, fontweight="600", fontsize=11)

    if heatmap_image is not None:
        ax3 = fig.add_subplot(1, n_cols, 3)
        ax3.set_facecolor(PANEL)
        ax3.imshow(heatmap_image)
        ax3.set_title("Grad-CAM Focus", color=TEXT, fontweight="600", fontsize=11)
        ax3.axis("off")

    fig.suptitle("Retinal Disease Screening — Report",
                 fontsize=13, fontweight="700", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
