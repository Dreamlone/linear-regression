from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 15
FONTNAME = "Comic Sans MS"


@dataclass
class RusAnnotations:
    title: str = ("Временно упростим задачу, оптимизируем только один коэффициент $b_0$"
                  "\nЕсли MSE при увеличении $b_0$ падает, то продолжаем двигаться в ту же сторону, если "
                  "MSE растет, то надо идти в другую")
    landscape_title: str = "Ландшафт функционала ошибки"
    landscape_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    landscape_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    slice_title: str = "Срез значений $b_0$ при фиксированном значении $b_1$\n"
    slice_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    slice_y_axis: str = "Ошибка модели (MSE)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"


@dataclass
class EngAnnotations:
    title: str = ""
    landscape_title: str = ""
    landscape_x_axis: str = ""
    landscape_y_axis: str = ""
    slice_title: str = ""
    slice_x_axis: str = ""
    slice_y_axis: str = ""
    scatter_title: str = ""
    scatter_x_axis: str = "Number of the rooms in the apartment, scaled (x)"
    scatter_y_axis: str = "Price, scaled (y)"


def annotations_by_language(mode: str):
    if mode == "eng":
        annotations = EngAnnotations()
    elif mode == "rus":
        annotations = RusAnnotations()
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return annotations


def compute_optimal_slope(features_scaled: np.ndarray, target_scaled: np.ndarray) -> float:
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    x_mean = float(np.mean(x_values))
    y_mean = float(np.mean(y_values))

    x_centered = x_values - x_mean
    y_centered = y_values - y_mean

    denominator = float(np.sum(x_centered ** 2))
    if denominator == 0.0:
        raise ValueError("Cannot compute slope: all x values are identical after scaling.")

    return float(np.sum(x_centered * y_centered) / denominator)


def generate_df_coefficients_vs_error():
    print("Generating the dataset with metrics")
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x.reshape(-1, 1))
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    rows = []
    for intercept in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
        for slope in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
            predicted_scaled = intercept + slope * features_scaled
            metric_value = mean_squared_error(target_scaled, predicted_scaled)
            rows.append([intercept, slope, metric_value])

    dataframe = pd.DataFrame(rows, columns=["intercept", "slope", "metric"])
    print(dataframe.head(5))
    return dataframe, features_scaled, target_scaled


def plot_mse_surface(
    ax_landscape,
    errors,
    dataframe,
    cmap_name: str = "coolwarm",
    alpha: float = 0.8,
    antialiased: bool = True,
):
    intercept_values = np.sort(dataframe["intercept"].unique())
    slope_values = np.sort(dataframe["slope"].unique())
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)

    vmin = float(np.nanmin(errors))
    vmax = float(np.nanmax(errors))
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(errors))

    ax_landscape.plot_surface(
        intercept_grid,
        slope_grid,
        errors,
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
    mappable.set_array(errors)

    ax_landscape.set_ylim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax_landscape.set_xlim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    return mappable


def add_slice_plane(
    ax_landscape,
    intercept_values: np.ndarray,
    slope_fixed: float,
    slice_errors: np.ndarray,
    z_base: float,
):
    intercept_values = np.ravel(intercept_values).astype(float)
    slice_errors = np.ravel(slice_errors).astype(float)

    for mask, plane_alpha in zip([intercept_values >= 0.0, intercept_values < 0.0], [0.8, 0.15]):
        x_half = intercept_values[mask]
        z_half = slice_errors[mask]

        x_plane = np.vstack([x_half, x_half])
        y_plane = np.vstack([np.full_like(x_half, slope_fixed), np.full_like(x_half, slope_fixed)])
        z_plane = np.vstack([np.full_like(x_half, z_base), z_half])

        ax_landscape.plot_surface(
            x_plane,
            y_plane,
            z_plane,
            color="grey",
            alpha=plane_alpha,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            shade=False,
            zorder=4,
        )

    ax_landscape.plot(
        intercept_values,
        np.full_like(intercept_values, slope_fixed),
        slice_errors,
        color="black",
        linewidth=1.5,
        zorder=6,
    )


def create_axes(fig):
    gs = fig.add_gridspec(1, 3)
    gs.update(wspace=0.4)
    ax_landscape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_slice = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    ax_slice.tick_params(axis="both", which="major", labelsize=8)
    ax_model.tick_params(axis="both", which="major", labelsize=8)
    return ax_landscape, ax_slice, ax_model


def render_landscape(
    ax_landscape,
    fig,
    errors,
    dataframe,
    annotations: Union[EngAnnotations, RusAnnotations],
):
    mappable = plot_mse_surface(ax_landscape, errors, dataframe, cmap_name="coolwarm")
    cbar = fig.colorbar(mappable, ax=ax_landscape, shrink=0.8, pad=0.2)
    cbar.ax.set_title("MSE", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.invert_xaxis()

    ax_landscape.set_xlabel(annotations.landscape_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_ylabel(annotations.landscape_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_title(
        annotations.landscape_title,
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        x=0.75,
        y=1.485,
    )
    return mappable


def add_curved_arrow(
    ax_slice,
    x_from: float,
    y_from: float,
    x_to: float,
    y_to: float,
    arc_rad: float,
    color: str
):
    ax_slice.annotate(
        "",
        xy=(x_to, y_to),
        xytext=(x_from, y_from),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 1.5,
            "connectionstyle": f"arc3,rad={arc_rad}",
        },
        zorder=5,
    )


def show_one_slice(mode: str = "eng"):
    annotations = annotations_by_language(mode)

    dataframe, features_scaled, target_scaled = generate_df_coefficients_vs_error()
    coeff_b0 = np.array(dataframe["intercept"])
    coeff_b1 = np.array(dataframe["slope"])
    metric_values = np.array(dataframe["metric"])

    intercept_values = np.sort(dataframe["intercept"].unique())
    slope_values = np.sort(dataframe["slope"].unique())
    intercept_range, slope_range = np.meshgrid(intercept_values, slope_values)

    fig = plt.figure(figsize=(16, 4))
    ax_landscape, ax_slice, ax_model = create_axes(fig)
    ax_landscape.computed_zorder = False

    errors = griddata((coeff_b0, coeff_b1), metric_values,(intercept_range, slope_range), method="cubic")

    slope_i = compute_optimal_slope(features_scaled, target_scaled)

    # Two example intercepts (yellow and green)
    intercept_yellow = float(intercept_values[4])
    intercept_green = float(intercept_values[-3])

    # Exact slice values at fixed slope_i (no interpolation)
    slice_errors = np.array([mean_squared_error(target_scaled, intercept_value + slope_i * features_scaled)
                             for intercept_value in intercept_values], dtype=float)

    # Exact MSE at the two highlighted points (consistent with the slice above)
    idx_yellow = int(np.where(intercept_values == intercept_yellow)[0][0])
    idx_green = int(np.where(intercept_values == intercept_green)[0][0])
    mse_yellow = float(slice_errors[idx_yellow])
    mse_green = float(slice_errors[idx_green])

    render_landscape(ax_landscape=ax_landscape, fig=fig, errors=errors, dataframe=dataframe, annotations=annotations)

    # Highlight two points on the 3D slice curve
    ax_landscape.scatter(intercept_green, slope_i, mse_green, c="#15D600", s=70, edgecolor="black",
                         linewidth=0.8, depthshade=False, zorder=7)
    ax_landscape.scatter(intercept_yellow, slope_i, mse_yellow, c="gold", s=70, edgecolor="black",
                         linewidth=0.8, depthshade=False, zorder=7)

    z_base = float(np.nanmin(errors))
    add_slice_plane(ax_landscape=ax_landscape, intercept_values=intercept_values, slope_fixed=float(slope_i),
                    slice_errors=slice_errors, z_base=z_base)

    # Slice plot
    ax_slice.plot(intercept_values, slice_errors, color="black", linewidth=2.0, zorder=2)
    ax_slice.scatter(intercept_values, slice_errors, marker="x", color="black", s=25, zorder=3)
    ax_slice.scatter(intercept_yellow, mse_yellow, c="gold", s=70, edgecolor="black", linewidth=0.8, zorder=7)
    ax_slice.scatter(intercept_green, mse_green, c="#15D600", s=70, edgecolor="black", linewidth=0.8, zorder=7)

    ax_slice.grid(color="grey", alpha=0.3, zorder=1)
    ax_slice.set_title(f"{annotations.slice_title}$b_1$={slope_i:.2f}", fontsize=14,
                       fontdict={"fontname": FONTNAME}, y=1.1)
    ax_slice.set_xlabel(annotations.slice_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_slice.set_ylabel(annotations.slice_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})

    # Green point neighbors: intercept_values[-4] (left) and intercept_values[-2] (right)
    idx_green_left = idx_green - 1
    x_green_left = float(intercept_values[idx_green_left])
    f_green_left = float(slice_errors[idx_green_left])

    add_curved_arrow(ax_slice, intercept_green, mse_green, x_green_left, f_green_left, arc_rad=0.25, color="#15D600")

    # Yellow point neighbors: intercept_values[idx_yellow-1] (left) and intercept_values[idx_yellow+1] (right)
    idx_yellow_left = idx_yellow - 1
    idx_yellow_right = idx_yellow + 1
    x_yellow_left = float(intercept_values[idx_yellow_left])
    x_yellow_right = float(intercept_values[idx_yellow_right])
    f_yellow_left = float(slice_errors[idx_yellow_left])
    f_yellow_right = float(slice_errors[idx_yellow_right])

    add_curved_arrow(ax_slice, intercept_yellow, mse_yellow, x_yellow_right, f_yellow_right, arc_rad=-0.25, color="gold")

    ##########################
    # Proper configure ticks #
    ##########################
    ticks = [-2, x_yellow_left, -1, intercept_yellow, 0, 1, x_green_left, intercept_green, 2]
    ax_slice.set_xticks(ticks)
    labels = []
    for value in ticks:
        if (np.isclose(value, x_green_left, atol=1e-6)
                or np.isclose(value, x_yellow_left, atol=1e-6)
                or np.isclose(value, intercept_green, atol=1e-6)
                or np.isclose(value, intercept_yellow, atol=1e-6)):
            labels.append(f"{value:.1f}")
        else:
            labels.append(f"{value:.0f}")
    ax_slice.set_xticklabels(labels)
    for label, value in zip(ax_slice.get_xticklabels(), ticks):
        label.set_fontname(FONTNAME)
        if (np.isclose(value, x_green_left, atol=1e-6)
                or np.isclose(value, x_yellow_left, atol=1e-6)
                or np.isclose(value, intercept_green, atol=1e-6)
                or np.isclose(value, intercept_yellow, atol=1e-6)):
            label.set_fontsize(5)
            label.set_fontweight("bold")
        else:
            label.set_fontsize(8)

    # Y ticks with custom formatting and per-tick font sizes
    y_ticks = [0, 0.5, mse_yellow, f_green_left, 2, mse_green, 2.5, 3, 3.5, 4, 4.5]
    ax_slice.set_yticks(y_ticks)
    y_labels = []
    for value in y_ticks:
        y_labels.append(f"{value:.1f}")

    ax_slice.set_yticklabels(y_labels)

    for label, value in zip(ax_slice.get_yticklabels(), y_ticks):
        label.set_fontname(FONTNAME)
        if (np.isclose(value, f_green_left, atol=1e-6)
                or np.isclose(value, mse_yellow, atol=1e-6)
                or np.isclose(value, mse_green, atol=1e-6)):
            label.set_fontsize(5)
            label.set_fontweight("bold")
        else:
            label.set_fontsize(8)

    ax_slice.text(intercept_green - 0.2, mse_green + 0.05, f"{mse_green:.1f}",
                  fontsize=10, fontname=FONTNAME, ha="center", va="bottom", color="#15D600", zorder=6)
    # Green backward
    diff_green_backward = (mse_green - f_green_left) / (intercept_green - x_green_left)
    calculations = (
        rf"$\frac{{{mse_green:.1f} - {f_green_left:.1f}}}{{{intercept_green:.1f} - {x_green_left:.1f}}}"
        rf"= {diff_green_backward:.1f}$"
    )
    ax_slice.text(intercept_green - 0.65, mse_green + 0.3, calculations, fontsize=10, fontname=FONTNAME, ha="center",
                  va="bottom", zorder=6)

    # The equation
    eq = r"$\frac{f_i-f_{i-1}}{x_i-x_{i-1}}$"
    ax_slice.text(0, 4.0, eq, fontsize=14, fontname=FONTNAME, ha="center", va="bottom",
                  zorder=6, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2.0})

    ax_slice.text(intercept_yellow + 0.2, mse_yellow + 0.05, f"{mse_yellow:.1f}",
                  fontsize=10, fontname=FONTNAME, ha="center", va="bottom", color="gold", zorder=6)
    # Yellow backward
    diff_yellow_backward = (mse_yellow - f_yellow_left) / (intercept_yellow - x_yellow_left)

    calculations = (
        rf"$\frac{{{mse_yellow:.1f} - {f_yellow_left:.1f}}}{{{intercept_yellow:.1f} - {x_yellow_left:.1f}}}"
        rf"= {diff_yellow_backward:.1f}$"
    )

    ax_slice.text(
        x_yellow_left - 0.3,
        mse_yellow - 0.45,
        calculations,
        fontsize=10,
        fontname=FONTNAME,
        ha="center",
        va="bottom",
        zorder=6,
    )
    ax_model.scatter(features_scaled, target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    circle = Circle((x_green_left, f_green_left), radius=0.1, fill=False, edgecolor="#15D600",
                    linewidth=1, linestyle="--", zorder=8)
    ax_slice.add_patch(circle)
    circle = Circle((x_yellow_right, f_yellow_right), radius=0.1, fill=False, edgecolor="gold", linewidth=1,
                    linestyle="--", zorder=8)
    ax_slice.add_patch(circle)
    ax_slice.axvline(0, linestyle='--', c='black', linewidth=1.5, zorder=5)

    x_line = np.array([[-1.5], [1.5]])
    predicted_green_line = intercept_green + slope_i * x_line
    predicted_green_lin_new = x_green_left + slope_i * x_line
    predicted_yellow_line = intercept_yellow + slope_i * x_line
    predicted_yellow_line_new = x_yellow_right + slope_i * x_line

    ax_model.plot(np.ravel(x_line), np.ravel(predicted_green_line), c="black", linewidth=2.5, zorder=2)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_green_line), c="#15D600", linewidth=2, zorder=3, alpha=0.8)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_green_lin_new), '--',
                  c="#15D600", linewidth=1, zorder=3, alpha=0.5)

    ax_model.plot(np.ravel(x_line), np.ravel(predicted_yellow_line), c="black", linewidth=2.5, zorder=2)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_yellow_line), c="gold", linewidth=2, zorder=3, alpha=0.5)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_yellow_line_new), '--',
                  c="gold", linewidth=1, zorder=3, alpha=0.5)

    # Text labels for the model lines
    x_text = -1.85
    y_text_green = -0.3
    y_text_yellow = -2.5

    ax_model.text(x_text, y_text_green, rf"$\hat{{y}} = {intercept_green:.2f} + {slope_i:.2f}x$",
                  fontname=FONTNAME, fontsize=10, color="#15D600", zorder=10)
    ax_model.text(x_text, y_text_yellow, rf"$\hat{{y}} = {intercept_yellow:.2f} + {slope_i:.2f}x$",
                  fontname=FONTNAME, fontsize=10, color="gold", zorder=10)

    ax_model.set_xlabel(annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(annotations.scatter_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)

    fig.suptitle(annotations.title, fontsize=16, fontdict={"fontname": FONTNAME}, va="top", x=0.55, y=1.4)

    raw_svg_file = Path(get_plots_path(), f"53_one_dimension_slice_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    path_to_final_path = Path(get_plots_path(), f"53_one_dimension_slice_{mode}.png")
    save_plot_according_to_template(
        raw_svg_file,
        path_to_final_path,
        template_name="template_small.svg",
        dpi=200,
    )


if __name__ == "__main__":
    show_one_slice("rus")
