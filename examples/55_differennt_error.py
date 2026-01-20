import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import imageio.v2 as imageio
import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -3
MAX_COEFFICIENT_BORDER = 3
GRID_SIZE = 15
ANIMATION_DURATION: int = 1100
FONTNAME = "Comic Sans MS"
CMAP_BY_METRIC = {"mae": "coolwarm", "mape": "coolwarm"}

# Try 1.5–2.5 to make the non-convex behavior more visible (especially with outliers).
TUKEY_C = 1.5


def tukey_biweight_loss(target, predicted, tukey_c: float = TUKEY_C) -> float:
    """
    Tukey biweight (bisquare) loss averaged over samples.
    target, predicted: arrays of shape (n, 1) or (n,)
    """
    residuals = np.ravel(predicted) - np.ravel(target)
    scaled = residuals / float(tukey_c)

    rho = np.empty_like(scaled, dtype=float)
    mask = np.abs(scaled) < 1.0

    # rho(r) = (c^2/6) * (1 - (1 - (r/c)^2)^3) for |r| <= c
    # rho(r) = (c^2/6)                           for |r| >  c
    c_squared_over_six = (float(tukey_c) ** 2) / 6.0

    inside = scaled[mask]
    rho[mask] = c_squared_over_six * (1.0 - (1.0 - inside ** 2) ** 3)
    rho[~mask] = c_squared_over_six

    return float(np.mean(rho))


@dataclass
class RusAnnotations:
    landscape_title: str = "Ландшафт функционала ошибки"
    best_model: str = "Лучшая модель"
    map_title: str = "Исследованные комбинации"
    map_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    map_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"


@dataclass
class EngAnnotations:
    landscape_title: str = ""
    best_model: str = ""
    map_title: str = ""
    map_x_axis: str = r"Intercept scaled ($b_0$)"
    map_y_axis: str = r"Slope scaled ($b_1$)"
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
            metric_value = tukey_biweight_loss(target_scaled, predicted_scaled, tukey_c=TUKEY_C)
            rows.append([intercept, slope, metric_value])

    dataframe = pd.DataFrame(rows, columns=["intercept", "slope", "metric"])
    print(dataframe.head(5))
    return dataframe, features_scaled, target_scaled


def plot_metric_surface(
    ax_landscape,
    errors,
    dataframe,
    cmap_name: str = "coolwarm",
    alpha: float = 0.95,
    linewidth: float = 0.0,
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


def create_axes(fig):
    gs = fig.add_gridspec(1, 3)
    gs.update(wspace=0.4)
    ax_landscape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_map = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    return ax_landscape, ax_map, ax_model


def render_landscape(
    ax_landscape,
    fig,
    errors,
    dataframe,
    annotations
):
    mappable = plot_metric_surface(ax_landscape, errors, dataframe, cmap_name="coolwarm")
    cbar = fig.colorbar(mappable, ax=ax_landscape, shrink=0.8, pad=0.2)
    cbar.ax.set_title(f"Tukey (c={TUKEY_C})", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.set_xlabel(annotations.map_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_ylabel(annotations.map_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_title(annotations.landscape_title, fontsize=14, fontdict={"fontname": FONTNAME},
                           x=0.75, y=1.485)
    return mappable


def render_model(
    ax_model, features_scaled, target_scaled, annotations,
    intercept_i, slope_i
):
    ax_model.scatter(features_scaled, target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    x_line_flat = [-1.5, 1.5]

    predicted_current = intercept_i + slope_i * x_line
    ax_model.plot(x_line_flat, np.ravel(predicted_current), c="red", zorder=3)

    ax_model.set_xlabel(annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(annotations.scatter_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)


def show_one_slice(mode: str = "eng"):
    annotations = annotations_by_language(mode)

    dataframe, features_scaled, target_scaled = generate_df_coefficients_vs_error()
    coeff_b0 = np.array(dataframe["intercept"])
    coeff_b1 = np.array(dataframe["slope"])
    metric_values = np.array(dataframe["metric"])
    intercept_range, slope_range = np.meshgrid(np.unique(coeff_b0), np.unique(coeff_b1))

    fig = plt.figure(figsize=(16, 4))
    ax_landscape, ax_map, ax_model = create_axes(fig)
    ax_landscape.computed_zorder = False

    # Note: for non-convex/noisy surfaces, "linear" can be more stable than "cubic".
    errors = griddata(
        (coeff_b0, coeff_b1),
        metric_values,
        (intercept_range, slope_range),
        method="cubic",
    )

    intercept_i = 0.0
    slope_i = 1.1

    mappable = render_landscape(ax_landscape=ax_landscape,
                                fig=fig, errors=errors,
                                dataframe=dataframe, annotations=annotations)
    render_model(
        ax_model=ax_model, features_scaled=features_scaled, target_scaled=target_scaled,
        annotations=annotations, intercept_i=intercept_i, slope_i=slope_i)

    fig.suptitle("Рассматриваем срез функционала ошибки", fontsize=16, fontdict={"fontname": FONTNAME},
                 va="top", x=0.5, y=1.2)
    ax_map.set_title(annotations.map_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)

    raw_svg_file = Path(get_plots_path(), f"55_one_dimension_slice_tukey_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    path_to_final_path = Path(get_plots_path(), f"55_one_dimension_slice_tukey_{mode}.png")
    save_plot_according_to_template(
        raw_svg_file,
        path_to_final_path,
        template_name="template_small.svg",
        dpi=200,
    )


if __name__ == "__main__":
    show_one_slice("rus")
