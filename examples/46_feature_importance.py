from pathlib import Path
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from examples.paths import get_plots_path

FONTNAME = "Comic Sans MS"
FONTDICT = {"fontsize": 14, "fontname": FONTNAME}
CMAP_COOL_WARM = "coolwarm"
X_TICKS = [1, 2, 3]
Y_TICKS = [1, 2, 3]
Y_TICKS_SCALED = [100, 200, 300]
TARGET_TICKS = [10, 20, 30]


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "How important is the feature"
    elif mode == "rus":
        title = "Насколько важен признак"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title


def generate_3d_plot(
    ax,
    f: np.array,
    s: np.array,
    t: np.array,
    y_ticks: list,
    y_min: float,
    y_max: float,
    title: str,
    c_to_show: Union[List, None] = None
):
    """Draw 3D scatter and regression surface for two features and a target."""
    # Fit linear regression model
    regression_features = np.column_stack([f, s])
    linear_model = LinearRegression()
    linear_model.fit(regression_features, t)

    print(title)
    coef_f, coef_s = linear_model.coef_
    print(f"{linear_model.intercept_:.1f} + {coef_f:.1f} + {coef_s:.1f}")

    # Create a dense grid for smooth regression surface
    n = 10
    x_grid, y_grid = np.meshgrid(
        np.linspace(min(f), max(f), n),
        np.linspace(min(s), max(s), n),
    )
    grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    z_grid = linear_model.predict(grid).reshape(x_grid.shape)

    ax.scatter(f, s, t, c=np.ravel(t), cmap=CMAP_COOL_WARM, s=70, alpha=1.0, edgecolors="black",
               linewidth=0.3, vmin=0, vmax=35, zorder=2)

    # Regression surface
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=CMAP_COOL_WARM,
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        vmin=0,
        vmax=35,
        zorder=1,
    )
    if c_to_show is not None:
        for c_f, c_s, c_t in c_to_show:
            ax.text(c_f, c_s, c_t, f"[{int(c_f)}, {int(c_s)}, {int(c_t)}]",
                    fontsize=12, color='grey', fontname=FONTNAME, zorder=3)

    ax.view_init(elev=20, azim=155)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(y_ticks)
    ax.set_zticks(TARGET_TICKS)

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(5, 35)

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r"$x_1$", fontdict={"fontname": FONTNAME, "fontsize": 10})
    ax.set_ylabel(r"$x_2$", fontdict={"fontname": FONTNAME, "fontsize": 10})
    ax.set_zlabel(r"$y$", fontdict={"fontname": FONTNAME, "fontsize": 10})

    if title:
        ax.set_title(title, fontdict={"fontname": FONTNAME, "fontsize": 12})


def plot_basement_for_feature_importance(mode: str = "eng"):
    """Generate a picture that can be used for 'feature importance' explanation."""
    title = annotations_by_language(mode)

    # Dataset where second feature is useless
    f = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    s = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    t = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    # Show some labels explicitly
    c_to_show = [[1, 1, 10], [1 , 2, 10], [2, 1, 20]]

    # Dataset where both features are equally important
    f_ = np.array([1, 2, 1, 1, 2, 3, 2, 3, 2])
    s_ = np.array([2, 1, 2, 3, 2, 1, 3, 2, 3])
    c_to_show_ = [[2, 1, 10], [1, 2, 10], [1, 3, 20]]

    # Two aligned 3D subplots
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(18, 4),
        subplot_kw={"projection": "3d"},
    )
    ax_with_useless_feature, ax_with_useful_feature, ax_scaled = axs

    generate_3d_plot(ax_with_useless_feature, f, s, t, y_ticks=Y_TICKS, y_min=0.5, y_max=3.5,
                     title="Feature s adds no information", c_to_show=c_to_show)
    generate_3d_plot(ax_with_useful_feature, f_, s_, t, y_ticks=Y_TICKS, y_min=0.5, y_max=3.5,
                     title="Both features are informative", c_to_show=c_to_show_)
    c_to_show_ = [[2, 100, 10], [1, 200, 10], [1, 300, 20]]
    generate_3d_plot(ax_scaled, f_, s_*100, t, y_ticks=Y_TICKS_SCALED, y_min=50, y_max=350,
                     title="Both features are informative but scaled", c_to_show=c_to_show_)

    plt.suptitle(title, fontsize=15, fontname=FONTNAME)
    raw_svg_file = Path(get_plots_path(), f"46_feature_importance_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_basement_for_feature_importance("rus")
    plot_basement_for_feature_importance("eng")
