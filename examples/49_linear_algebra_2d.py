from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from examples.paths import get_plots_path

FONTNAME = "Comic Sans MS"


def generate_vectors_svg():
    """ Generate raw svg file with basic primitives """
    x_vector = np.array([2, 2], dtype=float)
    y_vector = np.array([3, 6], dtype=float)

    figure, axis = plt.subplots(figsize=(7, 7))

    # Limits + equal scale
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    axis.set_aspect("equal", adjustable="box")

    # Axes through origin
    axis.spines["left"].set_position("zero")
    axis.spines["bottom"].set_position("zero")
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)

    # Axes styling
    axis.spines["left"].set_color("black")
    axis.spines["bottom"].set_color("black")
    axis.spines["left"].set_linewidth(2)
    axis.spines["bottom"].set_linewidth(2)

    # Ticks every 1 unit
    ticks = np.arange(-10, 11, 1)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)

    # Grid
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="0.85", linewidth=1)

    # Tick marks + labels font
    axis.tick_params(axis="both", which="major", length=6, width=1, colors="black")
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname(FONTNAME)
        label.set_fontsize(10)

    # --- Vectors (arrows from origin) ---
    axis.quiver(
        0, 0, x_vector[0], x_vector[1],
        angles="xy", scale_units="xy", scale=1,
        width=0.008
    )
    axis.quiver(
        0, 0, y_vector[0], y_vector[1],
        angles="xy", scale_units="xy", scale=1,
        width=0.008
    )

    # Labels near arrow tips
    axis.text(
        x_vector[0] * 1.05, x_vector[1] * 1.05, "x",
        fontname=FONTNAME, fontsize=14, ha="left", va="bottom"
    )
    axis.text(
        y_vector[0] * 1.05, y_vector[1] * 1.05, "y",
        fontname=FONTNAME, fontsize=14, ha="left", va="bottom"
    )

    plt.savefig(Path(get_plots_path(), "49_linear_algebra.svg"))
    plt.close()


if __name__ == "__main__":
    generate_vectors_svg()
