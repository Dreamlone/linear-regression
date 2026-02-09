from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Callable

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset


# --- Config ---
MIN_COEFFICIENT_BORDER = -5.0
MAX_COEFFICIENT_BORDER = 5.0
GRID_SIZE = 30
FONTNAME = "Comic Sans MS"
DPI = 200

# Given optima (scaled space, as in your previous scripts)
BEST_MSE_B0 = 0.0033627746453212693
BEST_MSE_B1 = 0.8670593738668325

BEST_TUKEY_B0 = -0.039596891903385265
BEST_TUKEY_B1 = 0.8531769045512251


@dataclass
class RusAnnotations:
    left_title: str = "Ландшафт квадратичной функции потерь (MSE)"
    right_title: str = "Ландшафт функции потерь Тьюки (Tukey biweight)"
    model_title: str = "Две оптимальные модели"
    x_axis: str = "Сдвиг стандартизированный ($b_0$)"
    y_axis: str = "Наклон стандартизированный ($b_1$)"
    model_x: str = "Количество комнат, стандартизированное (x)"
    model_y: str = "Стоимость, стандартизированная (y)"


@dataclass
class EngAnnotations:
    left_title: str = "MSE landscape"
    right_title: str = "Tukey biweight landscape"
    model_title: str = "Data and two optimal models"
    x_axis: str = "Intercept, scaled ($b_0$)"
    y_axis: str = "Slope, scaled ($b_1$)"
    model_x: str = "Rooms, scaled (x)"
    model_y: str = "Price, scaled (y)"


def annotations_by_language(mode: str) -> Union[RusAnnotations, EngAnnotations]:
    if mode == "rus":
        return RusAnnotations()
    if mode == "eng":
        return EngAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def load_scaled_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"], dtype=float)
    target = np.array(dataset["price"], dtype=float)

    features_train, target_train, _, _ = take_sample_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(features_train.reshape(-1, 1))

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target_train.reshape(-1, 1))

    return features_scaled, target_scaled


def tukey_biweight_mean_loss(residual: np.ndarray, tukey_c: float) -> float:
    """
    Tukey biweight (bisquare) rho loss (mean):
      u = r / c
      rho(r) = (c^2/6) * (1 - (1 - u^2)^3) , if |u| < 1
               (c^2/6)                     , otherwise
    """
    residual = np.ravel(residual).astype(float)
    c_value = float(tukey_c)
    if c_value <= 0:
        raise ValueError("tukey_c must be positive.")

    u = residual / c_value
    abs_u = np.abs(u)

    base = (c_value ** 2) / 6.0
    inside = abs_u < 1.0

    loss = np.empty_like(u, dtype=float)
    loss[~inside] = base
    loss[inside] = base * (1.0 - (1.0 - (u[inside] ** 2)) ** 3)

    return float(np.mean(loss))


def compute_loss_surface(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    grid_size: int = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b0_values = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, int(grid_size), dtype=float)
    b1_values = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, int(grid_size), dtype=float)

    surface = np.zeros((len(b1_values), len(b0_values)), dtype=float)

    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    for j, b1 in enumerate(b1_values):
        for i, b0 in enumerate(b0_values):
            predicted = float(b0) + float(b1) * x_values
            residual = predicted - y_values
            surface[j, i] = float(loss_fn(residual))

    return surface, b0_values, b1_values


def plot_surface_with_colorbar(
    fig,
    ax,
    surface: np.ndarray,
    b0_values: np.ndarray,
    b1_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cbar_title: str,
    cmap_name: str = "coolwarm",
):
    b0_grid, b1_grid = np.meshgrid(b0_values, b1_values)

    vmin = float(np.nanmin(surface))
    vmax = float(np.nanmax(surface))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(surface))

    ax.plot_surface(
        b0_grid,
        b1_grid,
        surface,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        alpha=0.85,
        zorder=2,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(surface)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.82, pad=0.15)
    cbar.ax.set_title(cbar_title, pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})
    cbar.ax.tick_params(labelsize=9)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname(FONTNAME)

    ax.set_xlim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax.set_ylim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)

    ax.set_xlabel(x_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax.set_ylabel(y_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax.set_zlabel("")  # keep uncluttered
    ax.set_title(title, fontsize=14, fontdict={"fontname": FONTNAME}, x=0.75, y=1.08)

    ax.view_init(elev=28, azim=125)
    ax.invert_xaxis()


def render_center_model(
    ax,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    annotations: Union[RusAnnotations, EngAnnotations],
    b0_mse: float,
    b1_mse: float,
    b0_tukey: float,
    b1_tukey: float,
):
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    ax.scatter(x_values, y_values, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([-2.0, 2.0], dtype=float)
    y_line_mse = float(b0_mse) + float(b1_mse) * x_line
    y_line_tukey = float(b0_tukey) + float(b1_tukey) * x_line

    ax.plot(x_line, y_line_mse, linewidth=2.6, zorder=3, label="Оптимум по MSE")
    ax.plot(x_line, y_line_tukey, linewidth=2.6, zorder=3, label="Оптимум по Tukey biweight")

    ax.set_xlabel(annotations.model_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.model_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3.2, 3.2)
    ax.grid(color="grey", alpha=0.3, zorder=1)
    ax.set_title(annotations.model_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.02)

    legend = ax.legend(loc="upper left", frameon=True, fontsize=10)
    for text in legend.get_texts():
        text.set_fontname(FONTNAME)


def show_mse_vs_tukey_static(
    mode: str = "rus",
    tukey_c: float = 4.685,
    point_b0: float = 1.2,
    point_b1: float = -0.5,
):
    annotations = annotations_by_language(mode)
    features_scaled, target_scaled = load_scaled_data()

    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    def mse_loss_fn(residual: np.ndarray) -> float:
        return float(np.mean(np.ravel(residual).astype(float) ** 2))

    def tukey_loss_fn(residual: np.ndarray) -> float:
        return tukey_biweight_mean_loss(residual=np.ravel(residual).astype(float), tukey_c=float(tukey_c))

    mse_surface, b0_values, b1_values = compute_loss_surface(
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        loss_fn=mse_loss_fn,
        grid_size=GRID_SIZE,
    )
    tukey_surface, b0_values_t, b1_values_t = compute_loss_surface(
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        loss_fn=tukey_loss_fn,
        grid_size=GRID_SIZE,
    )

    # Point marker (same point shown on both landscapes)
    pred_point = float(point_b0) + float(point_b1) * x_values
    mse_point = float(np.mean((pred_point - y_values) ** 2))
    tukey_point = tukey_biweight_mean_loss(residual=(pred_point - y_values), tukey_c=float(tukey_c))

    fig = plt.figure(figsize=(22, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.35])
    gs.update(wspace=0.28)

    ax_left_land = fig.add_subplot(gs[0, 0], projection="3d")
    ax_center_model = fig.add_subplot(gs[0, 1])
    ax_right_land = fig.add_subplot(gs[0, 2], projection="3d")

    ax_left_land.computed_zorder = False
    ax_right_land.computed_zorder = False

    # Left: MSE landscape
    plot_surface_with_colorbar(
        fig=fig,
        ax=ax_left_land,
        surface=mse_surface,
        b0_values=b0_values,
        b1_values=b1_values,
        title=annotations.left_title,
        x_label=annotations.x_axis,
        y_label=annotations.y_axis,
        cbar_title="MSE",
        cmap_name="coolwarm",
    )
    ax_left_land.scatter(
        float(point_b0),
        float(point_b1),
        float(mse_point),
        s=85,
        c="gold",
        edgecolor="black",
        linewidth=0.9,
        depthshade=False,
        zorder=10,
    )

    # Center: data + two optimal models
    render_center_model(
        ax=ax_center_model,
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        annotations=annotations,
        b0_mse=float(BEST_MSE_B0),
        b1_mse=float(BEST_MSE_B1),
        b0_tukey=float(BEST_TUKEY_B0),
        b1_tukey=float(BEST_TUKEY_B1),
    )

    # Right: Tukey landscape
    plot_surface_with_colorbar(
        fig=fig,
        ax=ax_right_land,
        surface=tukey_surface,
        b0_values=b0_values_t,
        b1_values=b1_values_t,
        title=annotations.right_title,
        x_label=annotations.x_axis,
        y_label=annotations.y_axis,
        cbar_title="Tukey biweight",
        cmap_name="coolwarm",
    )
    ax_right_land.scatter(
        float(point_b0),
        float(point_b1),
        float(tukey_point),
        s=85,
        c="gold",
        edgecolor="black",
        linewidth=0.9,
        depthshade=False,
        zorder=10,
    )

    # Save (prefix 60_)
    raw_svg = Path(get_plots_path(), f"60_mse_vs_tukey_static_{mode}.svg")
    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    out_png = Path(get_plots_path(), f"60_mse_vs_tukey_static_{mode}.png")
    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name="template.svg",
        dpi=DPI,
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    # point_b0/point_b1 are only for the highlighted point on both landscapes
    show_mse_vs_tukey_static(mode="rus", tukey_c=2.5, point_b0=-2.0, point_b1=1.0)
