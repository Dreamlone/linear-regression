from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset


MIN_COEFFICIENT_BORDER = -2.0
MAX_COEFFICIENT_BORDER = 2.0
GRID_SIZE = 15
FONTNAME = "Comic Sans MS"
DPI = 200

SLICE_XLIM = (-2.5, 2.5)
SLICE_YLIM = (0.0, 10.0)


@dataclass
class RusAnnotations:
    suptitle: str = ("Градиент в многомерном пространстве - это вектор, составленный из частных производных\n"
                     "Антиградиент - направление наискорейшего убывания функции")

    landscape_title = (
        r"$-\nabla_{(b_0,b_1)}\,\mathrm{MSE}"
        r"=\left(-\frac{\partial\,\mathrm{MSE}}{\partial b_0},"
        r"\ -\frac{\partial\,\mathrm{MSE}}{\partial b_1}\right)$"
    )
    landscape_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    landscape_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"

    left_title: str = r"$-\frac{\partial\,\mathrm{MSE}}{\partial b_0}$ - производная в срезе по $b_0$ при фиксированном $b_1$"
    left_x_axis: str = "Сдвиг стандартизированный ($b_0$)"
    left_y_axis: str = "Ошибка модели (MSE)"

    right_title: str = r"$-\frac{\partial\,\mathrm{MSE}}{\partial b_1}$ - производная в срезе по $b_1$ при фиксированном $b_0$"
    right_x_axis: str = "Наклон стандартизированный ($b_1$)"
    right_y_axis: str = "Ошибка модели (MSE)"

@dataclass
class EngAnnotations:
    suptitle: str = ""
    landscape_title: str = "MSE landscape and two slices through current point"
    landscape_x_axis: str = "Intercept, scaled ($b_0$)"
    landscape_y_axis: str = "Slope, scaled ($b_1$)"

    left_title: str = "Slice over $b_0$ (fixed $b_1$)"
    left_x_axis: str = "Intercept, scaled ($b_0$)"
    left_y_axis: str = "Model error (MSE)"

    right_title: str = "Slice over $b_1$ (fixed $b_0$)"
    right_x_axis: str = "Slope, scaled ($b_1$)"
    right_y_axis: str = "Model error (MSE)"

def annotations_by_language(mode: str) -> Union[RusAnnotations, EngAnnotations]:
    if mode == "rus":
        return RusAnnotations()
    if mode == "eng":
        return EngAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def load_scaled_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"])
    target = np.array(dataset["price"])
    x_values, y_values, _, _ = take_sample_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x_values.reshape(-1, 1))

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y_values.reshape(-1, 1))

    return features_scaled, target_scaled


def generate_df_coefficients_vs_error(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> pd.DataFrame:
    rows = []
    intercept_grid = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size, dtype=float)
    slope_grid = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size, dtype=float)

    for intercept in intercept_grid:
        for slope in slope_grid:
            predicted_scaled = intercept + slope * features_scaled
            metric_value = mean_squared_error(target_scaled, predicted_scaled)
            rows.append([float(intercept), float(slope), float(metric_value)])

    return pd.DataFrame(rows, columns=["intercept", "slope", "metric"])


def robust_griddata(
    points_x: np.ndarray,
    points_y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    for method in ["cubic", "linear", "nearest"]:
        surface = griddata((points_x, points_y), values, (grid_x, grid_y), method=method)
        if surface is not None and not np.all(np.isnan(surface)):
            return surface
    return griddata((points_x, points_y), values, (grid_x, grid_y), method="nearest")


def compute_surface_from_dataframe(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff_b0 = np.array(dataframe["intercept"], dtype=float)
    coeff_b1 = np.array(dataframe["slope"], dtype=float)
    metric_values = np.array(dataframe["metric"], dtype=float)

    intercept_values = np.sort(dataframe["intercept"].unique().astype(float))
    slope_values = np.sort(dataframe["slope"].unique().astype(float))

    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
    errors_surface = robust_griddata(coeff_b0, coeff_b1, metric_values, intercept_grid, slope_grid)
    return errors_surface, intercept_values, slope_values


def mse_and_gradients(
    intercept_value: float,
    slope_value: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> Tuple[float, float, float]:
    """
    MSE(b0, b1) = (1/n) sum (y - (b0 + b1*x))^2

    residual = y_hat - y

    dMSE/db0 = (2/n) sum(residual)
    dMSE/db1 = (2/n) sum(residual * x)
    """
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    predicted = float(intercept_value) + float(slope_value) * x_values
    residual = predicted - y_values

    n = float(len(x_values))
    mse_value = float(np.mean(residual ** 2))

    grad_b0 = float((2.0 / n) * np.sum(residual))
    grad_b1 = float((2.0 / n) * np.sum(residual * x_values))

    return mse_value, grad_b0, grad_b1


def compute_slice_over_b0(
    b0_values: np.ndarray,
    slope_fixed: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    b0_values = np.ravel(b0_values).astype(float)
    return np.array(
        [mse_and_gradients(float(b0), float(slope_fixed), features_scaled, target_scaled)[0] for b0 in b0_values],
        dtype=float,
    )


def compute_slice_over_b1(
    b1_values: np.ndarray,
    intercept_fixed: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    b1_values = np.ravel(b1_values).astype(float)
    return np.array(
        [mse_and_gradients(float(intercept_fixed), float(b1), features_scaled, target_scaled)[0] for b1 in b1_values],
        dtype=float,
    )


def plot_mse_surface(
    ax_landscape,
    errors_surface: np.ndarray,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    cmap_name: str = "coolwarm",
    alpha: float = 0.85,
    antialiased: bool = True,
):
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)

    vmin = float(np.nanmin(errors_surface))
    vmax = float(np.nanmax(errors_surface))
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(errors_surface))

    ax_landscape.plot_surface(
        intercept_grid,
        slope_grid,
        errors_surface,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=antialiased,
        shade=False,
        alpha=alpha,
        zorder=2,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(errors_surface)

    ax_landscape.set_ylim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax_landscape.set_xlim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    return mappable


def create_axes_1x3(fig):
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 2.3, 1.35])
    gs.update(wspace=0.28)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_landscape = fig.add_subplot(gs[0, 1], projection="3d")
    ax_right = fig.add_subplot(gs[0, 2])

    for ax in (ax_left, ax_right):
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.set_xlim(*SLICE_XLIM)
        ax.set_ylim(*SLICE_YLIM)
        ax.grid(color="grey", alpha=0.3)

    return ax_left, ax_landscape, ax_right


def draw_antigrad_arrow_on_1d_slice(
    ax,
    x0: float,
    y0: float,
    grad: float,
    x_min: float,
    x_max: float,
    color: str = "black",
):
    if abs(float(grad)) < 1e-9:
        return

    direction = -1.0 if grad > 0.0 else 1.0

    max_dx = 0.95
    dx_len = max_dx * min(1.0, abs(float(grad)) / (abs(float(grad)) + 1.0))

    x1 = float(x0) + direction * float(dx_len)
    x1 = float(np.clip(x1, x_min, x_max))

    if abs(x1 - float(x0)) < 1e-9:
        return

    ax.annotate(
        "",
        xy=(x1, float(y0)),
        xytext=(float(x0), float(y0)),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        zorder=10,
        annotation_clip=True,
    )


def draw_antigrad_arrows_on_landscape(
    ax_land,
    b0: float,
    b1: float,
    mse_value: float,
    grad_b0: float,
    grad_b1: float
):
    anti_b0 = -float(grad_b0)
    anti_b1 = -float(grad_b1)

    length_b0 = 0.9
    length_b1 = 0.9
    length_comb = 1.15

    if abs(anti_b0) > 1e-9:
        ax_land.quiver(
            float(b0), float(b1), float(mse_value),
            float(anti_b0), 0.0, 0.0,
            normalize=True,
            length=float(length_b0),
            color="black",
            linewidth=2.2,
            arrow_length_ratio=0.32,
            pivot="tail",
            zorder=12,
        )

    if abs(anti_b1) > 1e-9:
        ax_land.quiver(
            float(b0), float(b1), float(mse_value),
            0.0, float(anti_b1), 0.0,
            normalize=True,
            length=float(length_b1),
            color="red",
            linewidth=2.2,
            arrow_length_ratio=0.32,
            pivot="tail",
            zorder=12,
        )

    if float(np.hypot(anti_b0, anti_b1)) > 1e-9:
        ax_land.quiver(
            float(b0), float(b1), float(mse_value),
            float(anti_b0), float(anti_b1), 0.0,
            normalize=True,
            length=float(length_comb),
            color="gold",
            linewidth=2.6,
            arrow_length_ratio=0.34,
            pivot="tail",
            zorder=13,
        )


def show_two_slices_static(
    mode: str = "rus",
    point_b0: float = -1.2,
    point_b1: float = -0.5,
    dense_points: int = 260,
    surface_grid_size: int = GRID_SIZE,
):
    annotations = annotations_by_language(mode)

    features_scaled, target_scaled = load_scaled_data()

    dataframe = generate_df_coefficients_vs_error(
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        grid_size=int(surface_grid_size),
    )
    errors_surface, intercept_values, slope_values = compute_surface_from_dataframe(dataframe)

    # Dense slices
    b0_dense = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, int(dense_points), dtype=float)
    b1_dense = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, int(dense_points), dtype=float)

    slice_b0_mse = compute_slice_over_b0(b0_dense, float(point_b1), features_scaled, target_scaled)
    slice_b1_mse = compute_slice_over_b1(b1_dense, float(point_b0), features_scaled, target_scaled)

    mse_point, grad_b0, grad_b1 = mse_and_gradients(float(point_b0), float(point_b1), features_scaled, target_scaled)

    fig = plt.figure(figsize=(20, 6))
    ax_left, ax_landscape, ax_right = create_axes_1x3(fig)
    ax_landscape.computed_zorder = False

    # ---- Center: surface + point + TWO slice curves ----
    mappable = plot_mse_surface(
        ax_landscape=ax_landscape,
        errors_surface=errors_surface,
        intercept_values=intercept_values,
        slope_values=slope_values,
        cmap_name="coolwarm",
    )
    cbar = fig.colorbar(mappable, ax=ax_landscape, shrink=0.82, pad=0.18)
    cbar.ax.set_title("MSE", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.invert_xaxis()

    ax_landscape.set_xlabel(annotations.landscape_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_ylabel(annotations.landscape_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_zlabel("")  # no z label
    ax_landscape.set_title(
        annotations.landscape_title,
        fontsize=15,
        fontdict={"fontname": FONTNAME},
        x=0.75,
        y=1.2,
    )

    ax_landscape.plot(
        b0_dense,
        np.full_like(b0_dense, float(point_b1)),
        slice_b0_mse,
        color="black",
        linewidth=2.1,
        zorder=10,
    )

    ax_landscape.plot(
        np.full_like(b1_dense, float(point_b0)),
        b1_dense,
        slice_b1_mse,
        color="red",
        linewidth=2.1,
        zorder=10,
    )

    ax_landscape.scatter(
        float(point_b0),
        float(point_b1),
        float(mse_point),
        c="gold",
        s=85,
        edgecolor="black",
        linewidth=0.9,
        depthshade=False,
        zorder=14,
    )

    draw_antigrad_arrows_on_landscape(
        ax_land=ax_landscape,
        b0=float(point_b0),
        b1=float(point_b1),
        mse_value=float(mse_point),
        grad_b0=float(grad_b0),
        grad_b1=float(grad_b1)
    )

    ax_left.plot(b0_dense, slice_b0_mse, color="black", linewidth=2.2, zorder=2)
    ax_left.scatter(
        [float(point_b0)],
        [float(mse_point)],
        c="gold",
        s=70,
        edgecolor="black",
        linewidth=0.9,
        zorder=6,
    )
    draw_antigrad_arrow_on_1d_slice(
        ax=ax_left,
        x0=float(point_b0),
        y0=float(mse_point),
        grad=float(grad_b0),
        x_min=float(SLICE_XLIM[0]),
        x_max=float(SLICE_XLIM[1]),
        color="black",
    )
    ax_left.set_title(annotations.left_title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.02)
    ax_left.set_xlabel(annotations.left_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_left.set_ylabel(annotations.left_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_right.plot(b1_dense, slice_b1_mse, color="red", linewidth=2.2, zorder=2)
    ax_right.scatter(
        [float(point_b1)],
        [float(mse_point)],
        c="gold",
        s=70,
        edgecolor="black",
        linewidth=0.9,
        zorder=6,
    )
    draw_antigrad_arrow_on_1d_slice(
        ax=ax_right,
        x0=float(point_b1),
        y0=float(mse_point),
        grad=float(grad_b1),
        x_min=float(SLICE_XLIM[0]),
        x_max=float(SLICE_XLIM[1]),
        color="red",
    )
    ax_right.set_title(annotations.right_title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.02)
    ax_right.set_xlabel(annotations.right_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_right.set_ylabel(annotations.right_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    fig.suptitle(
        annotations.suptitle,
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        va="top",
        x=0.52,
        y=1.15,
    )
    raw_svg = Path(get_plots_path(), f"58_two_slices_static_{mode}.svg")
    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    out_png = Path(get_plots_path(), f"58_two_slices_static_{mode}.png")
    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name="template_small.svg",
        dpi=DPI,
    )

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    show_two_slices_static(mode="rus")
