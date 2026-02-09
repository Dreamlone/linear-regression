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
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 15
ANIMATION_DURATION: int = 1100
FONTNAME = "Comic Sans MS"
CMAP_BY_METRIC = {"mae": "coolwarm", "mape": "coolwarm"}


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

    @staticmethod
    def get_title(method: str):
        if method == "brute":
            title = "Оптимизация коэффициентов полным перебором"
        else:
            title = "Оптимизация коэффициентов случайным поиском"

        return title


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

    @staticmethod
    def get_title(method: str):
        if method == "brute":
            title = ""
        else:
            title = ""

        return title


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
    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x.reshape(-1, 1))
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    rows = []
    for intercept in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
        for slope in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
            predicted_scaled = intercept + slope * features_scaled
            metric_value = root_mean_squared_error(target_scaled, predicted_scaled)
            rows.append([intercept, slope, metric_value])

    dataframe = pd.DataFrame(rows, columns=["intercept", "slope", "metric"])
    print(dataframe.head(5))
    return dataframe, features_scaled, target_scaled


def plot_rmse_surface(
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

    surface = ax_landscape.plot_surface(
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


def render_landscape_marker(ax_landscape, intercept_i, slope_i, rmse_i):
    x_min, x_max = ax_landscape.get_xlim()
    y_min, y_max = ax_landscape.get_ylim()
    z_min, z_max = ax_landscape.get_zlim()

    z_span = z_max - z_min
    point_z = rmse_i + 0.06 * z_span
    text_z = point_z + 0.09 * z_span

    ax_landscape.plot(
        [intercept_i, intercept_i],
        [slope_i, slope_i],
        [point_z, z_min],
        '--',
        c="red",
        linewidth=1,
        zorder=1,
    )
    ax_landscape.plot(
        [intercept_i, x_max],
        [slope_i, slope_i],
        [point_z, point_z],
        '--',
        c="red",
        linewidth=1,
        zorder=1,
    )
    ax_landscape.plot(
        [intercept_i, intercept_i],
        [slope_i, y_min],
        [point_z, point_z],
        '--',
        c="red",
        linewidth=1,
        zorder=1,
    )

    ax_landscape.scatter(
        intercept_i,
        slope_i,
        point_z,
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=5,
    )
    ax_landscape.text(
        intercept_i,
        slope_i,
        text_z,
        f"{rmse_i:.1f}",
        color="red",
        fontsize=12,
        fontname=FONTNAME,
        zorder=6,
    )


def render_landscape(
    ax_landscape,
    fig,
    errors,
    dataframe,
    annotations,
    intercept_i,
    slope_i,
    rmse_i,
):
    mappable = plot_rmse_surface(ax_landscape, errors, dataframe, cmap_name="coolwarm")
    cbar = fig.colorbar(mappable, ax=ax_landscape, shrink=0.8, pad=0.2)
    cbar.ax.set_title("RMSE", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.invert_xaxis()

    ax_landscape.set_xlabel(annotations.map_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_ylabel(annotations.map_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_title(annotations.landscape_title, fontsize=14, fontdict={"fontname": FONTNAME},
                           x=0.75, y=1.485)

    render_landscape_marker(
        ax_landscape=ax_landscape,
        intercept_i=intercept_i,
        slope_i=slope_i,
        rmse_i=rmse_i,
    )
    return mappable


def render_map(
    ax_map,
    coeff_grid_points,
    annotations,
    intercept_i,
    slope_i,
    rmse_i,
    explored_coeff_b0,
    explored_coeff_b1,
    explored_rmse,
    mappable,
):
    ax_map.grid(color="grey", alpha=0.3, zorder=1)
    ax_map.scatter(
        [p[0] for p in coeff_grid_points],
        [p[1] for p in coeff_grid_points],
        c="grey",
        marker="x",
        s=50,
        zorder=2,
    )

    if len(explored_rmse) > 0:
        ax_map.scatter(
            explored_coeff_b0,
            explored_coeff_b1,
            c=explored_rmse,
            cmap=mappable.cmap,
            norm=mappable.norm,
            s=100,
            edgecolor="black",
            zorder=3,
        )

        old_colors = mappable.to_rgba(explored_rmse)
        for b0_old, b1_old, rmse_old, color_old in zip(
            explored_coeff_b0, explored_coeff_b1, explored_rmse, old_colors
        ):
            ax_map.text(
                b0_old + 0.14,
                b1_old + 0.08,
                f"{rmse_old:.1f}",
                fontsize=12,
                color=tuple(color_old),
                fontname=FONTNAME,
                zorder=4,
            )

    ax_map.scatter(intercept_i, slope_i, c="red", s=100, zorder=3)
    ax_map.text(
        intercept_i + 0.14,
        slope_i + 0.08,
        f"{rmse_i:.1f}",
        fontsize=12,
        color="red",
        fontname=FONTNAME,
        zorder=5,
    )
    ax_map.set_xlabel(annotations.map_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_map.set_ylabel(annotations.map_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_map.set_ylim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax_map.set_xlim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax_map.invert_yaxis()


def render_model(
    ax_model, features_scaled, target_scaled, annotations,
    intercept_i, slope_i, explored_coeff_b0, explored_coeff_b1, explored_rmse, mappable
):
    ax_model.scatter(features_scaled, target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    x_line_flat = [-1.5, 1.5]

    if len(explored_rmse) > 0:
        old_colors = mappable.to_rgba(explored_rmse)
        for b0_old, b1_old, color_old in zip(explored_coeff_b0, explored_coeff_b1, old_colors):
            predicted_old = b0_old + b1_old * x_line
            ax_model.plot(
                x_line_flat,
                np.ravel(predicted_old),
                linestyle="--",
                color=tuple(color_old),
                alpha=0.5,
                zorder=2,
            )

    predicted_current = intercept_i + slope_i * x_line
    ax_model.plot(x_line_flat, np.ravel(predicted_current), c="red", zorder=3)

    ax_model.set_xlabel(annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(annotations.scatter_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)


def generate_frame(intercept_i, slope_i, coeff_b0, coeff_b1, metric_values, intercept_range, slope_range,
                   features_scaled, target_scaled, dataframe, annotations, coeff_grid_points, explored_coeff_b0,
                   explored_coeff_b1, explored_rmse, method):
    fig = plt.figure(figsize=(16, 4))
    ax_landscape, ax_map, ax_model = create_axes(fig)
    ax_landscape.computed_zorder = False

    errors = griddata(
        (coeff_b0, coeff_b1),
        metric_values,
        (intercept_range, slope_range),
        method="cubic",
    )

    predicted_scaled_full = intercept_i + slope_i * features_scaled
    rmse_i = root_mean_squared_error(target_scaled, predicted_scaled_full)

    mappable = render_landscape(
        ax_landscape=ax_landscape,
        fig=fig,
        errors=errors,
        dataframe=dataframe,
        annotations=annotations,
        intercept_i=intercept_i,
        slope_i=slope_i,
        rmse_i=rmse_i,
    )

    render_map(
        ax_map=ax_map,
        coeff_grid_points=coeff_grid_points,
        annotations=annotations,
        intercept_i=intercept_i,
        slope_i=slope_i,
        rmse_i=rmse_i,
        explored_coeff_b0=explored_coeff_b0,
        explored_coeff_b1=explored_coeff_b1,
        explored_rmse=explored_rmse,
        mappable=mappable,
    )

    render_model(
        ax_model=ax_model, features_scaled=features_scaled, target_scaled=target_scaled,
        annotations=annotations, intercept_i=intercept_i, slope_i=slope_i,
        explored_coeff_b0=explored_coeff_b0, explored_coeff_b1=explored_coeff_b1,
        explored_rmse=explored_rmse, mappable=mappable
    )

    fig.suptitle(
        annotations.get_title(method),
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        va="top",
        x=0.5,
        y=1.2,
    )

    return ax_map, rmse_i


def show_optimal_b_search(mode: str = "eng", method: str = "brute"):
    iteration_number_per_coefficient = 3
    if method == "brute":
        coefficients = np.linspace(
            MIN_COEFFICIENT_BORDER + 0.5,
            MAX_COEFFICIENT_BORDER - 0.5,
            iteration_number_per_coefficient,
        )
        coeff_grid_points = list(product(coefficients, coefficients))
    else:
        seed = 2007
        rng = np.random.default_rng(seed)

        b0_s = rng.uniform(
            MIN_COEFFICIENT_BORDER + 0.5,
            MAX_COEFFICIENT_BORDER - 0.5,
            iteration_number_per_coefficient * iteration_number_per_coefficient
        )
        b1_s = rng.uniform(
            MIN_COEFFICIENT_BORDER + 0.5,
            MAX_COEFFICIENT_BORDER - 0.5,
            iteration_number_per_coefficient * iteration_number_per_coefficient
        )

        coeff_grid_points = [[b0_s[i], b1_s[i]] for i in range(len(b0_s))]
        coeff_grid_points.sort()

    annotations = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    dataframe, features_scaled, target_scaled = generate_df_coefficients_vs_error()
    coeff_b0 = np.array(dataframe["intercept"])
    coeff_b1 = np.array(dataframe["slope"])
    metric_values = np.array(dataframe["metric"])
    intercept_range, slope_range = np.meshgrid(np.unique(coeff_b0), np.unique(coeff_b1))

    image_files = []
    image_index = 0
    explored_coeff_b0 = []
    explored_coeff_b1 = []
    explored_rmse = []

    for intercept_i, slope_i in coeff_grid_points:
        ax_map, rmse_i = generate_frame(intercept_i, slope_i, coeff_b0, coeff_b1,
                                        metric_values, intercept_range, slope_range,
                                        features_scaled, target_scaled, dataframe,
                                        annotations, coeff_grid_points, explored_coeff_b0,
                                        explored_coeff_b1, explored_rmse, method)
        ax_map.set_title(annotations.map_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)

        raw_svg_file = Path(tmp_dir, f"52_optimization_direct_{mode}.svg")
        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close()

        path_to_final_path = Path(tmp_dir, f"52_optimization_direct_{mode}_{image_index}.png")
        save_plot_according_to_template(
            raw_svg_file,
            path_to_final_path,
            template_name="template_small.svg",
            dpi=200,
        )

        image_files.append(path_to_final_path)
        image_index += 1

        explored_coeff_b0.append(intercept_i)
        explored_coeff_b1.append(slope_i)
        explored_rmse.append(rmse_i)

    # Choose the best model and display it
    min_rmse_id = np.argmin(np.array(explored_rmse))
    best_b0 = explored_coeff_b0[min_rmse_id]
    best_b1 = explored_coeff_b1[min_rmse_id]

    ax_map, _ = generate_frame(best_b0, best_b1, coeff_b0, coeff_b1,
                               metric_values, intercept_range, slope_range,
                               features_scaled, target_scaled, dataframe,
                               annotations, coeff_grid_points, explored_coeff_b0,
                               explored_coeff_b1, explored_rmse, method)
    ax_map.set_title(annotations.best_model, fontsize=14, fontdict={"fontname": FONTNAME}, c='red', y=1.1)
    raw_svg_file = Path(tmp_dir, f"52_optimization_direct_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    path_to_final_path = Path(tmp_dir, f"52_optimization_direct_{mode}_{image_index}.png")
    save_plot_according_to_template(
        raw_svg_file,
        path_to_final_path,
        template_name="template_small.svg",
        dpi=200,
    )
    # Let's make the final frame a bit longer
    image_files.append(path_to_final_path)
    image_files.append(path_to_final_path)
    image_files.append(path_to_final_path)

    gif_path = Path(get_plots_path(), f"52_optimization_direct_{method}_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    show_optimal_b_search("rus", "brute")
    show_optimal_b_search("rus", "random")
