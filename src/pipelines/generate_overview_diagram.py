"""Generate a simulated overview diagram for the Transformer-LSTM + Fuzzy pipeline.

This script creates a conceptual Training/Testing flow image intended for reports.
It does not depend on model checkpoints or real evaluation outputs.

Usage:
    python src/pipelines/generate_overview_diagram.py
    python src/pipelines/generate_overview_diagram.py --output data/reports/overview_simulated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "credit_rating_artifacts" / "transformer_lstm_overview_simulated.png"


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "#EAF4FF",
    ec: str = "#34506A",
) -> None:
    """Draw a rounded rectangle box with centered text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)

    text = title if not subtitle else f"{title}\n{subtitle}"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#1E2A36",
    )


def draw_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#222222",
) -> None:
    """Draw a flow arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
        )
    )


def build_diagram(output_path: Path, seed: int = 42) -> Path:
    """Build and save a conceptual overview diagram image."""
    rng = np.random.default_rng(seed)
    probs = rng.random(22)
    probs = probs / probs.sum()
    top3 = np.argsort(probs)[-3:][::-1]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel = FancyBboxPatch(
        (0.03, 0.04),
        0.94,
        0.92,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=2,
        edgecolor="#CFC6AE",
        facecolor="#FBF8EE",
    )
    ax.add_patch(panel)

    train_lane = FancyBboxPatch(
        (0.06, 0.56),
        0.88,
        0.33,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#B79F75",
        facecolor="#F6EFD9",
    )
    test_lane = FancyBboxPatch(
        (0.06, 0.12),
        0.88,
        0.33,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#B79F75",
        facecolor="#F6EFD9",
    )
    ax.add_patch(train_lane)
    ax.add_patch(test_lane)

    ax.text(0.075, 0.865, "TRAINING (Simulated Flow)", fontsize=12, fontweight="bold", color="#5A4321")
    ax.text(0.075, 0.425, "TESTING (Simulated Flow)", fontsize=12, fontweight="bold", color="#5A4321")

    draw_box(ax, 0.09, 0.64, 0.13, 0.16, "Train/Val Data", "Panel time-series", fc="#BEEAF7")
    draw_box(ax, 0.27, 0.64, 0.14, 0.16, "Preprocessing", "clean, encode, scale", fc="#D5F0CF")
    draw_box(ax, 0.46, 0.64, 0.15, 0.16, "Window Builder", "sliding windows", fc="#E6D8F8")
    draw_box(ax, 0.66, 0.62, 0.18, 0.20, "TLSTM-Fuzzy", "Gaussian fuzzy +\nTransformer + BiLSTM", fc="#FFD8A7")
    draw_box(ax, 0.87, 0.64, 0.06, 0.16, "Best", "checkpoint", fc="#F7C8D1")

    draw_arrow(ax, 0.22, 0.72, 0.27, 0.72)
    draw_arrow(ax, 0.41, 0.72, 0.46, 0.72)
    draw_arrow(ax, 0.61, 0.72, 0.66, 0.72)
    draw_arrow(ax, 0.84, 0.72, 0.87, 0.72)

    draw_box(ax, 0.09, 0.20, 0.13, 0.16, "Test Data", "Hold-out split", fc="#BEEAF7")
    draw_box(ax, 0.27, 0.20, 0.14, 0.16, "Preprocessing", "same as training", fc="#D5F0CF")
    draw_box(ax, 0.46, 0.20, 0.15, 0.16, "Inference", "load checkpoint", fc="#E3ECFA")
    draw_box(ax, 0.66, 0.20, 0.12, 0.16, "Prediction", "next rating", fc="#FFEABF")
    draw_box(
        ax,
        0.80,
        0.18,
        0.13,
        0.20,
        "Simulated Output",
        f"Top classes: {top3[0]}, {top3[1]}, {top3[2]}\nConf: {probs[top3[0]]:.2f}",
        fc="#FCE4A8",
    )

    draw_arrow(ax, 0.22, 0.28, 0.27, 0.28)
    draw_arrow(ax, 0.41, 0.28, 0.46, 0.28)
    draw_arrow(ax, 0.61, 0.28, 0.66, 0.28)
    draw_arrow(ax, 0.78, 0.28, 0.80, 0.28)

    draw_arrow(ax, 0.90, 0.64, 0.54, 0.36, color="#34495E")

    ax.text(
        0.50,
        0.94,
        "Transformer-LSTM + Fuzzy Credit Rating Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#2D3142",
    )
    ax.text(
        0.50,
        0.91,
        "Conceptual Overview for Report (Simulated, Not Exact)",
        ha="center",
        va="center",
        fontsize=11,
        color="#4F5D75",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a simulated overview diagram image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for simulated output card.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    saved = build_diagram(output_path=args.output, seed=args.seed)
    print(f"Saved simulated overview diagram to: {saved}")


if __name__ == "__main__":
    main()
