import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import imageio.v2 as imageio
from matplotlib.lines import Line2D

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 15
FONTNAME = "Comic Sans MS"
ANIM_DURATION = 1100
DPI = 120


@dataclass
class RusAnnotations:
    title: str = "Количество итераций (оценок MSE)"
    landscape_title: str = "Ландшафт функционала ошибки"
    landscape_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    landscape_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    slice_title: str = "Направление движения"
    slice_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    slice_y_axis: str = "Ошибка модели (MSE)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"
    latest_point: str = "Предыдущая итерация"


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
    latest_point: str = ""


def annotations_by_language(mode: str):
    if mode == "eng":
        return EngAnnotations()
    if mode == "rus":
        return RusAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


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
            metric_value = mean_squared_error(target_scaled, predicted_scaled)
            rows.append([intercept, slope, metric_value])

    dataframe = pd.DataFrame(rows, columns=["intercept", "slope", "metric"])
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

    ax_slice.tick_params(axis="both", which="major", labelsize=11)
    ax_model.tick_params(axis="both", which="major", labelsize=11)

    ax_slice.set_ylim(0, 4.5)
    ax_slice.set_xlim(-2.2, 2.2)
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


def compute_slice_errors(
    intercept_values: np.ndarray,
    slope_fixed: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            mean_squared_error(target_scaled, intercept_value + slope_fixed * features_scaled)
            for intercept_value in intercept_values
        ],
        dtype=float,
    )


def find_minimum_by_raw_differences(
    slice_errors: np.ndarray,
    start_index: int,
    max_iterations: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Discrete search using only raw differences between neighbour MSE values.

    Behaviour:
      - First move is forced to the right if possible.
      - Walk while MSE decreases.
      - If a step makes it worse: reverse and step back.
      - If stepping back returns to an already visited point, stop
        (removes extra oscillation frames).
    """
    n_points = int(len(slice_errors))
    if not (0 <= start_index < n_points):
        raise ValueError("start_index is out of bounds.")

    current_index = int(start_index)
    direction = +1

    visited = {current_index}
    evaluated = {current_index}
    eval_count = 1

    path_indices: List[int] = [current_index]
    move_directions: List[int] = []
    mse_evaluations: List[int] = [eval_count]

    def append_step(new_index: int, move_dir: int):
        nonlocal current_index, eval_count
        current_index = int(new_index)
        path_indices.append(current_index)
        move_directions.append(int(move_dir))

        if current_index not in evaluated:
            evaluated.add(current_index)
            eval_count += 1
        visited.add(current_index)
        mse_evaluations.append(eval_count)

    # forced first move to the right
    first_next = current_index + 1
    if first_next < n_points:
        from_index = current_index
        append_step(first_next, +1)
        diff = float(slice_errors[current_index] - slice_errors[from_index])
        direction = -1 if diff > 0.0 else +1
    else:
        direction = -1

    for _ in range(max_iterations):
        from_index = current_index
        next_index = from_index + direction

        if next_index < 0 or next_index >= n_points:
            direction *= -1
            next_index = from_index + direction
            if next_index < 0 or next_index >= n_points:
                break

        append_step(next_index, direction)
        diff = float(slice_errors[current_index] - slice_errors[from_index])

        if diff <= 0.0:
            continue

        # worse -> reverse and step back
        direction *= -1
        back_index = current_index + direction

        if back_index < 0 or back_index >= n_points:
            break

        if back_index in visited:
            append_step(back_index, direction)
            break

        append_step(back_index, direction)

    return path_indices, move_directions, mse_evaluations


@dataclass
class FrameContext:
    dataframe: pd.DataFrame
    errors_surface: np.ndarray
    intercept_values: np.ndarray
    slice_errors: np.ndarray
    slope_value: float
    features_scaled: np.ndarray
    target_scaled: np.ndarray
    annotations: Union[EngAnnotations, RusAnnotations]


def draw_next_direction_arrow(ax_slice, next_direction: int):
    if next_direction == 0:
        return

    y_axes = 1.08
    x_left = 0.15
    x_right = 0.85

    if next_direction > 0:
        xy = (x_right, y_axes)
        xytext = (x_left, y_axes)
    else:
        xy = (x_left, y_axes)
        xytext = (x_right, y_axes)

    ax_slice.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2.0},
        annotation_clip=False,
        zorder=20,
    )


def compute_optimal_intercept_for_fixed_slope(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    slope_fixed: float,
) -> float:
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)
    return float(np.mean(y_values) - slope_fixed * np.mean(x_values))


def render_frame(
    context: FrameContext,
    intercept_index: int,
    prev_intercept_index: Optional[int],
    output_png: Path,
    color: str,
    next_direction: int,
    number_of_mse_evaluations: int,
):
    intercept_value = float(context.intercept_values[intercept_index])
    mse_value = float(context.slice_errors[intercept_index])

    fig = plt.figure(figsize=(16, 4))
    ax_landscape, ax_slice, ax_model = create_axes(fig)
    ax_landscape.computed_zorder = False

    render_landscape(
        ax_landscape=ax_landscape,
        fig=fig,
        errors=context.errors_surface,
        dataframe=context.dataframe,
        annotations=context.annotations,
    )

    ax_landscape.scatter(
        intercept_value,
        context.slope_value,
        mse_value,
        c=color,
        s=70,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=7,
    )

    z_base = float(np.nanmin(context.errors_surface))
    add_slice_plane(
        ax_landscape=ax_landscape,
        intercept_values=context.intercept_values,
        slope_fixed=float(context.slope_value),
        slice_errors=context.slice_errors,
        z_base=z_base,
    )

    # Slice plot
    ax_slice.plot(context.intercept_values, context.slice_errors, color="black", linewidth=2.0, zorder=2)
    ax_slice.scatter(context.intercept_values, context.slice_errors, marker="x", color="black", s=25, zorder=3)
    ax_slice.scatter(intercept_value, mse_value, c=color, s=70, edgecolor="black", linewidth=0.8, zorder=7)

    # Previous visited point highlighted by a circle
    if prev_intercept_index is not None:
        prev_x = float(context.intercept_values[int(prev_intercept_index)])
        prev_y = float(context.slice_errors[int(prev_intercept_index)])
        circle_prev = Circle((prev_x, prev_y), radius=0.1, fill=False, edgecolor=color, linewidth=1, zorder=8)
        ax_slice.add_patch(circle_prev)
        legend_handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.0,
            markersize=8,
            label=context.annotations.latest_point,
        )
        ax_slice.legend(
            handles=[legend_handle],
            loc="upper center",
            frameon=True,
            fontsize=9,
        )

    ax_slice.grid(color="grey", alpha=0.3, zorder=1)
    ax_slice.set_title(
        context.annotations.slice_title,
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        y=1.1,
    )
    ax_slice.set_xlabel(context.annotations.slice_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_slice.set_ylabel(context.annotations.slice_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    opt_b0 = compute_optimal_intercept_for_fixed_slope(
        context.features_scaled, context.target_scaled, context.slope_value
    )
    ax_slice.axvline(opt_b0, linestyle="--", c="black", linewidth=1.5, zorder=5, alpha=0.6)

    draw_next_direction_arrow(ax_slice, next_direction)

    # Model plot
    ax_model.scatter(context.features_scaled, context.target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    predicted_model = intercept_value + context.slope_value * x_line
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c="black", linewidth=2.5, zorder=2)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c=color, linewidth=2, zorder=3, alpha=0.8)

    # Previous model line (same color)
    if prev_intercept_index is not None:
        prev_intercept_value = float(context.intercept_values[int(prev_intercept_index)])
        predicted_prev = prev_intercept_value + context.slope_value * x_line
        ax_model.plot(np.ravel(x_line), np.ravel(predicted_prev), c=color, linewidth=1, zorder=3, alpha=0.9)

    ax_model.set_xlabel(context.annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(context.annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(context.annotations.scatter_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)

    fig.suptitle(
        f"{context.annotations.title}: {number_of_mse_evaluations}",
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        color="red",
        va="top",
        x=0.52,
        y=1.2,
    )

    raw_svg_file = output_png.with_suffix(".svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(raw_svg_file, output_png, template_name="template_small.svg", dpi=DPI)


def show_one_slice(mode: str = "eng"):
    max_number_of_iterations = 25
    annotations = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir = get_tmp_animation_directory()

    dataframe, features_scaled, target_scaled = generate_df_coefficients_vs_error()
    coeff_b0 = np.array(dataframe["intercept"])
    coeff_b1 = np.array(dataframe["slope"])
    metric_values = np.array(dataframe["metric"])

    intercept_values = np.sort(dataframe["intercept"].unique())
    slope_values = np.sort(dataframe["slope"].unique())
    intercept_range, slope_range = np.meshgrid(intercept_values, slope_values)

    errors_surface = griddata(
        (coeff_b0, coeff_b1),
        metric_values,
        (intercept_range, slope_range),
        method="cubic",
    )

    slope_i = compute_optimal_slope(features_scaled, target_scaled)
    slice_errors = compute_slice_errors(intercept_values, slope_i, features_scaled, target_scaled)

    context = FrameContext(
        dataframe=dataframe,
        errors_surface=errors_surface,
        intercept_values=intercept_values,
        slice_errors=slice_errors,
        slope_value=float(slope_i),
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        annotations=annotations,
    )

    image_files: List[Path] = []
    frame_idx = 0

    for start_index, color in zip([3, 12], ["gold", "#15D600"]):
        path_indices, move_directions, mse_evaluations = find_minimum_by_raw_differences(
            slice_errors=slice_errors,
            start_index=start_index,
            max_iterations=max_number_of_iterations,
        )

        for step_idx, intercept_index in enumerate(path_indices):
            prev_index = int(path_indices[step_idx - 1]) if step_idx > 0 else None
            next_dir = int(move_directions[step_idx]) if step_idx < len(move_directions) else 0

            frame_png = Path(tmp_dir, f"54_frame_{frame_idx}.png")
            render_frame(
                context=context,
                intercept_index=int(intercept_index),
                prev_intercept_index=prev_index,
                output_png=frame_png,
                color=color,
                next_direction=next_dir,
                number_of_mse_evaluations=int(mse_evaluations[step_idx]),
            )
            image_files.append(frame_png)
            frame_idx += 1

        # Pause frames at the end: keep showing previous point/line too
        last_eval = int(mse_evaluations[-1])
        pause_prev = int(path_indices[-2]) if len(path_indices) > 1 else None
        for _ in [1, 2]:
            frame_png = Path(tmp_dir, f"54_frame_{frame_idx}.png")
            render_frame(
                context=context,
                intercept_index=int(path_indices[-1]),
                prev_intercept_index=None,
                output_png=frame_png,
                color="grey",
                next_direction=0,
                number_of_mse_evaluations=last_eval,
            )
            image_files.append(frame_png)
            frame_idx += 1

    gif_path = Path(get_plots_path(), f"54_one_dimension_slice_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    show_one_slice("rus")
