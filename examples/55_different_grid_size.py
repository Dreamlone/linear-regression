import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional

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

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
FONTNAME = "Comic Sans MS"
ANIM_DURATION = 1100
DPI = 120

# Fixed limits so plots don't "jump"
SLICE_XLIM = (-2.2, 2.2)
SLICE_YLIM = (0.0, 4.5)


@dataclass
class RusAnnotations:
    iter_title: str = "Количество итераций (оценок MSE)"
    landscape_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    landscape_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    slice_x_axis: str = "Сдвиг стандартизированный ($b_0$)"
    slice_y_axis: str = "Ошибка модели (MSE)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"


@dataclass
class EngAnnotations:
    iter_title: str = "Number of iterations (MSE evals)"
    landscape_x_axis: str = ""
    landscape_y_axis: str = ""
    slice_x_axis: str = ""
    slice_y_axis: str = ""
    scatter_title: str = ""
    scatter_x_axis: str = "Number of the rooms in the apartment, scaled (x)"
    scatter_y_axis: str = "Price, scaled (y)"


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


def load_scaled_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x.reshape(-1, 1))

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    return features_scaled, target_scaled


def generate_df_coefficients_vs_error(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    grid_size: int,
) -> pd.DataFrame:
    rows = []
    for intercept in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size):
        for slope in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size):
            predicted_scaled = intercept + slope * features_scaled
            metric_value = mean_squared_error(target_scaled, predicted_scaled)
            rows.append([intercept, slope, metric_value])
    return pd.DataFrame(rows, columns=["intercept", "slope", "metric"])


def robust_griddata(
    points_x: np.ndarray,
    points_y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    # For small grids (e.g., 4x4), cubic is unreliable / not supported
    for method in ["linear", "nearest"]:
        surface = griddata((points_x, points_y), values, (grid_x, grid_y), method=method)
        if surface is not None and not np.all(np.isnan(surface)):
            return surface
    return griddata((points_x, points_y), values, (grid_x, grid_y), method="nearest")


def compute_surface_from_dataframe(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff_b0 = np.array(dataframe["intercept"], dtype=float)
    coeff_b1 = np.array(dataframe["slope"], dtype=float)
    metric_values = np.array(dataframe["metric"], dtype=float)

    intercept_values = np.sort(dataframe["intercept"].unique().astype(float))
    slope_values = np.sort(dataframe["slope"].unique().astype(float))

    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
    errors_surface = robust_griddata(coeff_b0, coeff_b1, metric_values, intercept_grid, slope_grid)

    return errors_surface, intercept_values, slope_values


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


def plot_mse_surface(
    ax_landscape,
    errors_surface: np.ndarray,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    cmap,
    norm,
    alpha: float = 0.8,
    antialiased: bool = True,
):
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
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

    ax_landscape.set_ylim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)
    ax_landscape.set_xlim(MIN_COEFFICIENT_BORDER - 0.5, MAX_COEFFICIENT_BORDER + 0.5)


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


def find_minimum_by_raw_differences(
    slice_errors: np.ndarray,
    start_index: int,
    max_iterations: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Discrete search using only raw differences between neighbour MSE values.

    IMPORTANT:
      - "No change" in MSE is treated as WORSE (plateau counts as deterioration),
        i.e. only strictly negative diff means improvement.

    Behaviour:
      - First move is forced to the right if possible.
      - Walk while MSE strictly decreases.
      - If a step makes it worse OR equal: reverse and step back.
      - If stepping back returns to an already visited point, stop (no extra oscillation frames).
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

    # Forced first move to the right if possible
    first_next = current_index + 1
    if first_next < n_points:
        from_index = current_index
        append_step(first_next, +1)
        diff = float(slice_errors[current_index] - slice_errors[from_index])

        # diff < 0 -> improved, else (>=0) -> worse or same, reverse
        direction = -1 if diff >= 0.0 else +1
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

        # Only strict improvement keeps direction
        if diff < 0.0:
            continue

        # Worse OR equal -> reverse and step back
        direction *= -1
        back_index = current_index + direction

        if back_index < 0 or back_index >= n_points:
            break

        # If we would return to an already visited point -> do it once and stop (no extra oscillations)
        if back_index in visited:
            append_step(back_index, direction)
            break

        append_step(back_index, direction)

    return path_indices, move_directions, mse_evaluations


@dataclass
class RowContext:
    grid_size: int
    dataframe: pd.DataFrame
    errors_surface: np.ndarray
    intercept_values: np.ndarray
    slope_values: np.ndarray
    slice_errors: np.ndarray


@dataclass
class FrameContext:
    rows: Dict[int, RowContext]
    slope_value: float
    features_scaled: np.ndarray
    target_scaled: np.ndarray
    annotations: Union[EngAnnotations, RusAnnotations]


def create_axes_2x3_with_cbar(fig):
    """
    Layout: 2 rows x (landscape, cbar, slice, model).
    This keeps the colorbar narrow and guarantees it doesn't overlap columns 2 & 3.
    """
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.05, 0.045, 1.05, 1.05],  # narrow cbar column
        wspace=0.35,
        hspace=0.35,
    )

    ax_land_top = fig.add_subplot(gs[0, 0], projection="3d")
    ax_slice_top = fig.add_subplot(gs[0, 2])
    ax_model_top = fig.add_subplot(gs[0, 3])

    ax_land_bot = fig.add_subplot(gs[1, 0], projection="3d")
    ax_slice_bot = fig.add_subplot(gs[1, 2])
    ax_model_bot = fig.add_subplot(gs[1, 3])

    cax = fig.add_subplot(gs[:, 1])  # shared colorbar axis
    pos = cax.get_position()
    dx = 0.02
    cax.set_position([pos.x0 - dx, pos.y0, pos.width, pos.height])

    for ax_slice in [ax_slice_top, ax_slice_bot]:
        ax_slice.tick_params(axis="both", which="major", labelsize=11)
        ax_slice.set_xlim(*SLICE_XLIM)
        ax_slice.set_ylim(*SLICE_YLIM)

    for ax_model in [ax_model_top, ax_model_bot]:
        ax_model.tick_params(axis="both", which="major", labelsize=11)

    return (ax_land_top, ax_slice_top, ax_model_top), (ax_land_bot, ax_slice_bot, ax_model_bot), cax


def compute_optimal_intercept_for_fixed_slope(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    slope_fixed: float,
) -> float:
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)
    return float(np.mean(y_values) - slope_fixed * np.mean(x_values))


def render_row(
    axes_triplet,
    context: FrameContext,
    row: RowContext,
    step_index: int,
    path_indices: List[int],
    mse_evaluations: List[int],
    cmap,
    norm,
    landscape_title_template: str,
) -> int:
    ax_landscape, ax_slice, ax_model = axes_triplet
    ax_landscape.computed_zorder = False

    last_step = len(path_indices) - 1
    step_index = min(step_index, last_step)

    # --- NEW: detect "converged" state for this row ---
    converged = (step_index >= last_step)

    current_index = int(path_indices[step_index])
    # --- NEW: hide previous point once converged ---
    prev_index: Optional[int] = int(path_indices[step_index - 1]) if (step_index > 0 and not converged) else None
    eval_count = int(mse_evaluations[step_index])

    active_color = "gold"
    frozen_color = "grey"
    point_color = frozen_color if converged else active_color

    intercept_value = float(row.intercept_values[current_index])
    mse_value = float(row.slice_errors[current_index])

    # ---- Landscape ----
    plot_mse_surface(
        ax_landscape=ax_landscape,
        errors_surface=row.errors_surface,
        intercept_values=row.intercept_values,
        slope_values=row.slope_values,
        cmap=cmap,
        norm=norm,
    )

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.invert_xaxis()

    ax_landscape.set_xlabel(context.annotations.landscape_x_axis, fontdict={"fontsize": 8, "fontname": FONTNAME})
    ax_landscape.set_ylabel(context.annotations.landscape_y_axis, fontdict={"fontsize": 8, "fontname": FONTNAME})
    ax_landscape.set_zlabel("MSE", fontdict={"fontsize": 9, "fontname": FONTNAME})

    ax_landscape.set_title(
        landscape_title_template.format(row.grid_size),
        fontsize=12,
        fontdict={"fontname": FONTNAME},
        x=0.70,
        y=1.00,
    )

    ax_landscape.scatter(
        intercept_value,
        context.slope_value,
        mse_value,
        c=point_color,
        s=70,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=7,
    )

    z_base = float(np.nanmin(row.errors_surface))
    add_slice_plane(
        ax_landscape=ax_landscape,
        intercept_values=row.intercept_values,
        slope_fixed=float(context.slope_value),
        slice_errors=row.slice_errors,
        z_base=z_base,
    )

    # ---- Slice ----
    ax_slice.plot(row.intercept_values, row.slice_errors, color="black", linewidth=2.0, zorder=2)
    ax_slice.scatter(row.intercept_values, row.slice_errors, marker="x", color="black", s=25, zorder=3)
    ax_slice.scatter(intercept_value, mse_value, c=point_color, s=70, edgecolor="black", linewidth=0.8, zorder=7)

    # --- NEW: prev circle only while not converged ---
    if prev_index is not None:
        prev_x = float(row.intercept_values[prev_index])
        prev_y = float(row.slice_errors[prev_index])
        circle_prev = Circle((prev_x, prev_y), radius=0.1, fill=False, edgecolor=active_color, linewidth=1, zorder=8)
        ax_slice.add_patch(circle_prev)

    ax_slice.grid(color="grey", alpha=0.3, zorder=1)
    ax_slice.set_xlabel(context.annotations.slice_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_slice.set_ylabel(context.annotations.slice_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})

    opt_b0 = compute_optimal_intercept_for_fixed_slope(
        context.features_scaled, context.target_scaled, context.slope_value
    )
    ax_slice.axvline(opt_b0, linestyle="--", c="black", linewidth=1.5, zorder=5, alpha=0.6)

    ax_slice.set_xlim(*SLICE_XLIM)
    ax_slice.set_ylim(*SLICE_YLIM)

    ax_slice.set_title(
        f"{context.annotations.iter_title}: {eval_count}",
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        color="red",
        y=1.05,
    )

    # ---- Model ----
    ax_model.scatter(context.features_scaled, context.target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    predicted_model = intercept_value + context.slope_value * x_line

    if converged:
        # --- NEW: frozen state -> grey line only, no previous model line ---
        ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c="grey", linewidth=2.5, zorder=3, alpha=0.9)
    else:
        # active state: black backbone + gold overlay
        ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c="black", linewidth=2.5, zorder=2)
        ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c=active_color, linewidth=2, zorder=3, alpha=0.8)

        # previous model line (only while active)
        if prev_index is not None:
            prev_intercept_value = float(row.intercept_values[prev_index])
            predicted_prev = prev_intercept_value + context.slope_value * x_line
            ax_model.plot(np.ravel(x_line), np.ravel(predicted_prev), c=active_color, linewidth=1, zorder=3, alpha=0.9)

    ax_model.set_xlabel(context.annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(context.annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(context.annotations.scatter_title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)

    return eval_count


def add_shared_colorbar(cax, cmap, norm):
    cax.clear()
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar = plt.colorbar(mappable, cax=cax)
    cbar.ax.set_title("MSE", pad=6, fontdict={"fontsize": 10, "fontname": FONTNAME})
    cbar.ax.tick_params(labelsize=9)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname(FONTNAME)


def render_frame_2x3(
    context: FrameContext,
    step_index: int,
    output_png: Path,
    raw_svg_file: Path,
    run_top: Tuple[List[int], List[int], List[int]],
    run_bottom: Tuple[List[int], List[int], List[int]],
    top_grid_size: int,
    bottom_grid_size: int,
):
    path_top, _, evals_top = run_top
    path_bottom, _, evals_bottom = run_bottom

    fig = plt.figure(figsize=(18, 9))
    axes_top, axes_bottom, cax = create_axes_2x3_with_cbar(fig)

    # Shared color scale for both landscapes
    vmin = float(min(np.nanmin(context.rows[top_grid_size].errors_surface), np.nanmin(context.rows[bottom_grid_size].errors_surface)))
    vmax = float(max(np.nanmax(context.rows[top_grid_size].errors_surface), np.nanmax(context.rows[bottom_grid_size].errors_surface)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    render_row(
        axes_triplet=axes_top,
        context=context,
        row=context.rows[top_grid_size],
        step_index=step_index,
        path_indices=path_top,
        mse_evaluations=evals_top,
        cmap=cmap,
        norm=norm,
        landscape_title_template="Ландшафт функционала ошибки\nразмер сетки={}",
    )

    render_row(
        axes_triplet=axes_bottom,
        context=context,
        row=context.rows[bottom_grid_size],
        step_index=step_index,
        path_indices=path_bottom,
        mse_evaluations=evals_bottom,
        cmap=cmap,
        norm=norm,
        landscape_title_template="размер сетки={}",
    )

    add_shared_colorbar(cax=cax, cmap=cmap, norm=norm)

    fig.suptitle(
        "Влияние размера сетки на сходимость",
        fontsize=20,
        fontdict={"fontname": FONTNAME},
        color="black",
        va="top",
        x=0.53,
        y=1.01,
    )

    # IMPORTANT: overwrite the same SVG file each frame (no per-frame SVGs in tmp dir)
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(raw_svg_file, output_png, template_name="template.svg", dpi=DPI)


def show_animation(mode: str = "rus"):
    annotations = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir = get_tmp_animation_directory()

    features_scaled, target_scaled = load_scaled_data()
    slope_i = compute_optimal_slope(features_scaled, target_scaled)

    top_grid_size = 4
    bottom_grid_size = 50

    rows: Dict[int, RowContext] = {}
    for grid_size in [top_grid_size, bottom_grid_size]:
        dataframe = generate_df_coefficients_vs_error(features_scaled, target_scaled, grid_size=grid_size)
        errors_surface, intercept_values, slope_values = compute_surface_from_dataframe(dataframe)
        slice_errors = compute_slice_errors(intercept_values, slope_i, features_scaled, target_scaled)
        rows[grid_size] = RowContext(
            grid_size=grid_size,
            dataframe=dataframe,
            errors_surface=errors_surface,
            intercept_values=intercept_values,
            slope_values=slope_values,
            slice_errors=slice_errors,
        )

    context = FrameContext(
        rows=rows,
        slope_value=float(slope_i),
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        annotations=annotations,
    )

    # One start only: index 0 for both
    max_number_of_iterations = 500
    run_top = find_minimum_by_raw_differences(rows[top_grid_size].slice_errors, start_index=0, max_iterations=max_number_of_iterations)
    run_bottom = find_minimum_by_raw_differences(rows[bottom_grid_size].slice_errors, start_index=0, max_iterations=max_number_of_iterations)

    total_frames = max(len(run_top[0]), len(run_bottom[0]))
    pause_frames = 3

    raw_svg_file = Path(tmp_dir, "55_different_grid_size.svg")
    image_files: List[Path] = []

    for frame_idx in range(total_frames + pause_frames):
        step_index = min(frame_idx, total_frames - 1)

        frame_png = Path(tmp_dir, f"55_different_grid_size_{frame_idx}.png")
        render_frame_2x3(
            context=context,
            step_index=step_index,
            output_png=frame_png,
            raw_svg_file=raw_svg_file,  # always the same SVG filename
            run_top=run_top,
            run_bottom=run_bottom,
            top_grid_size=top_grid_size,
            bottom_grid_size=bottom_grid_size,
        )
        image_files.append(frame_png)

    gif_path = Path(get_plots_path(), f"55_different_grid_size_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    show_animation("rus")
