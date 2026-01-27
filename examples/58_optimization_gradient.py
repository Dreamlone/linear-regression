import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional

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
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset


MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 15
ANIMATION_DURATION: int = 1100
FONTNAME = "Comic Sans MS"


@dataclass
class RusAnnotations:
    landscape_title: str = "Ландшафт функционала ошибки"
    best_model: str = "Лучшая модель"
    map_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    map_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"

    @staticmethod
    def get_title() -> str:
        return "Оптимизация коэффициентов градиентным спуском"


@dataclass
class EngAnnotations:
    landscape_title: str = ""
    best_model: str = ""
    map_x_axis: str = r"Intercept scaled ($b_0$)"
    map_y_axis: str = r"Slope scaled ($b_1$)"
    scatter_title: str = ""
    scatter_x_axis: str = "Number of the rooms in the apartment, scaled (x)"
    scatter_y_axis: str = "Price, scaled (y)"

    @staticmethod
    def get_title() -> str:
        return ""


def annotations_by_language(mode: str):
    if mode == "eng":
        return EngAnnotations()
    if mode == "rus":
        return RusAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def generate_df_coefficients_vs_error(
    grid_size: int = GRID_SIZE,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"])
    target = np.array(dataset["price"])
    x_values, y_values, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x_values.reshape(-1, 1))

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y_values.reshape(-1, 1))

    rows = []
    intercept_grid = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size)
    slope_grid = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_size)

    for intercept in intercept_grid:
        for slope in slope_grid:
            predicted_scaled = intercept + slope * features_scaled
            metric_value = root_mean_squared_error(target_scaled, predicted_scaled)
            rows.append([float(intercept), float(slope), float(metric_value)])

    dataframe = pd.DataFrame(rows, columns=["intercept", "slope", "metric"])
    return dataframe, features_scaled, target_scaled


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


def create_axes(fig):
    gs = fig.add_gridspec(1, 3)
    gs.update(wspace=0.4)
    ax_landscape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_map = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    return ax_landscape, ax_map, ax_model


def rmse_and_gradients(
    intercept_value: float,
    slope_value: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    RMSE(b0, b1) = sqrt( mean( (y - (b0 + b1*x))^2 ) )

    residual = y_hat - y
    MSE = mean(residual^2)
    RMSE = sqrt(MSE)

    grad_RMSE:
      dRMSE/db0 = (1/(n*RMSE)) sum(residual)
      dRMSE/db1 = (1/(n*RMSE)) sum(residual*x)
    """
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    predicted = float(intercept_value) + float(slope_value) * x_values
    residual = predicted - y_values

    mse_value = float(np.mean(residual ** 2))
    rmse_value = float(np.sqrt(mse_value))
    denom = float(max(rmse_value, eps))
    n = float(len(x_values))

    grad_b0 = float((1.0 / (n * denom)) * np.sum(residual))
    grad_b1 = float((1.0 / (n * denom)) * np.sum(residual * x_values))

    return rmse_value, grad_b0, grad_b1


def clip_to_bounds(value: float) -> float:
    return float(np.clip(value, MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER))


def gradient_descent_path(
    start_b0: float,
    start_b1: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    grad_tol: float,
    step_tol: float,
    backtracking_max_tries: int,
    backtracking_shrink: float,
) -> List[Tuple[float, float]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not (0.0 < backtracking_shrink < 1.0):
        raise ValueError("backtracking_shrink must be in (0, 1).")

    current_b0 = clip_to_bounds(float(start_b0))
    current_b1 = clip_to_bounds(float(start_b1))
    path: List[Tuple[float, float]] = [(current_b0, current_b1)]

    for _ in range(max_iterations):
        rmse_current, grad_b0, grad_b1 = rmse_and_gradients(
            current_b0, current_b1, features_scaled, target_scaled
        )
        grad_norm = float(np.hypot(grad_b0, grad_b1))

        # Stop 1: gradient norm small
        if grad_norm < grad_tol:
            break

        # Backtracking line-search: ensure RMSE decreases
        lr_try = float(learning_rate)
        accepted = False
        next_b0, next_b1 = current_b0, current_b1

        for _ in range(backtracking_max_tries):
            candidate_b0 = clip_to_bounds(current_b0 - lr_try * grad_b0)
            candidate_b1 = clip_to_bounds(current_b1 - lr_try * grad_b1)

            rmse_candidate, _, _ = rmse_and_gradients(
                candidate_b0, candidate_b1, features_scaled, target_scaled
            )

            if rmse_candidate <= rmse_current:
                next_b0, next_b1 = candidate_b0, candidate_b1
                accepted = True
                break

            lr_try *= backtracking_shrink

        # Stop 2: cannot find improvement
        if not accepted:
            break

        step_size = float(np.hypot(next_b0 - current_b0, next_b1 - current_b1))
        current_b0, current_b1 = next_b0, next_b1
        path.append((current_b0, current_b1))

        # Stop 3: step tiny
        if step_size < step_tol:
            break

    return path


def plot_rmse_surface(
    ax_landscape,
    errors_surface: np.ndarray,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    cmap_name: str = "coolwarm",
    alpha: float = 0.8,
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


def render_landscape_marker(ax_landscape, intercept_value: float, slope_value: float, rmse_value: float):
    x_min, x_max = ax_landscape.get_xlim()
    y_min, y_max = ax_landscape.get_ylim()
    z_min, z_max = ax_landscape.get_zlim()

    z_span = z_max - z_min
    point_z = float(rmse_value) + 0.06 * z_span
    text_z = point_z + 0.09 * z_span

    ax_landscape.plot([intercept_value, intercept_value], [slope_value, slope_value], [point_z, z_min],
                      "--", c="red", linewidth=1, zorder=1)
    ax_landscape.plot([intercept_value, x_max], [slope_value, slope_value], [point_z, point_z],
                      "--", c="red", linewidth=1, zorder=1)
    ax_landscape.plot([intercept_value, intercept_value], [slope_value, y_min], [point_z, point_z],
                      "--", c="red", linewidth=1, zorder=1)

    ax_landscape.scatter(
        float(intercept_value),
        float(slope_value),
        float(point_z),
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=5,
    )
    ax_landscape.text(
        float(intercept_value),
        float(slope_value),
        float(text_z),
        f"{rmse_value:.2f}",
        color="red",
        fontsize=12,
        fontname=FONTNAME,
        zorder=6,
    )


def render_landscape(
    ax_landscape,
    fig,
    errors_surface: np.ndarray,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    annotations,
    intercept_value: float,
    slope_value: float,
    rmse_value: float,
):
    mappable = plot_rmse_surface(
        ax_landscape=ax_landscape,
        errors_surface=errors_surface,
        intercept_values=intercept_values,
        slope_values=slope_values,
        cmap_name="coolwarm",
    )

    cbar = fig.colorbar(mappable, ax=ax_landscape, shrink=0.8, pad=0.2)
    cbar.ax.set_title("RMSE", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_landscape.view_init(elev=28, azim=125)
    ax_landscape.invert_xaxis()

    ax_landscape.set_xlabel(annotations.map_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_ylabel(annotations.map_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_landscape.set_title(
        annotations.landscape_title,
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        x=0.75,
        y=1.485,
    )

    render_landscape_marker(ax_landscape, intercept_value, slope_value, rmse_value)
    return mappable


def render_map(
    ax_map,
    annotations,
    current_b0: float,
    current_b1: float,
    current_rmse: float,
    explored_b0: List[float],
    explored_b1: List[float],
    explored_rmse: List[float],
    mappable,
    iteration_count: int,
):
    ax_map.grid(color="grey", alpha=0.3, zorder=1)

    # Explored points (trajectory history)
    if len(explored_rmse) > 0:
        ax_map.scatter(
            explored_b0,
            explored_b1,
            c=explored_rmse,
            cmap=mappable.cmap,
            norm=mappable.norm,
            s=110,
            edgecolor="black",
            zorder=3,
        )
        ax_map.plot(explored_b0, explored_b1, "-", color="black", linewidth=1.0, alpha=0.35, zorder=2)

    # Current point (red)
    ax_map.scatter(float(current_b0), float(current_b1), c="red", s=140, zorder=4, edgecolor="black")
    ax_map.text(
        float(current_b0) + 0.15,
        float(current_b1) + 0.06,
        f"{current_rmse:.2f}",
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

    ax_map.set_title(
        f"Количество итераций: {iteration_count}",
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        y=1.1,
    )


def render_model(
    ax_model,
    features_scaled,
    target_scaled,
    annotations,
    current_b0: float,
    current_b1: float,
    explored_b0: List[float],
    explored_b1: List[float],
    explored_rmse: List[float],
    mappable,
):
    ax_model.scatter(features_scaled, target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    x_line_flat = [-1.5, 1.5]

    # Previous models as dashed lines (faint), colored by their RMSE
    if len(explored_rmse) > 0:
        old_colors = mappable.to_rgba(explored_rmse)
        for b0_old, b1_old, color_old in zip(explored_b0, explored_b1, old_colors):
            predicted_old = float(b0_old) + float(b1_old) * x_line
            ax_model.plot(
                x_line_flat,
                np.ravel(predicted_old),
                linestyle="--",
                color=tuple(color_old),
                alpha=0.35,
                zorder=2,
            )

    # Current model (red)
    predicted_current = float(current_b0) + float(current_b1) * x_line
    ax_model.plot(x_line_flat, np.ravel(predicted_current), c="red", linewidth=2.5, zorder=3)

    ax_model.set_xlabel(annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(annotations.scatter_title, fontsize=14, fontdict={"fontname": FONTNAME}, y=1.1)


def generate_frame(
    current_b0: float,
    current_b1: float,
    errors_surface: np.ndarray,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    annotations,
    explored_b0: List[float],
    explored_b1: List[float],
    explored_rmse: List[float],
    iteration_count: int,
    map_title_override: Optional[str] = None,
):
    rmse_value, _, _ = rmse_and_gradients(current_b0, current_b1, features_scaled, target_scaled)

    fig = plt.figure(figsize=(16, 4))
    ax_landscape, ax_map, ax_model = create_axes(fig)
    ax_landscape.computed_zorder = False

    mappable = render_landscape(
        ax_landscape=ax_landscape,
        fig=fig,
        errors_surface=errors_surface,
        intercept_values=intercept_values,
        slope_values=slope_values,
        annotations=annotations,
        intercept_value=current_b0,
        slope_value=current_b1,
        rmse_value=rmse_value,
    )

    render_map(
        ax_map=ax_map,
        annotations=annotations,
        current_b0=current_b0,
        current_b1=current_b1,
        current_rmse=rmse_value,
        explored_b0=explored_b0,
        explored_b1=explored_b1,
        explored_rmse=explored_rmse,
        mappable=mappable,
        iteration_count=iteration_count,
    )

    if map_title_override is not None:
        ax_map.set_title(
            map_title_override,
            fontsize=14,
            fontdict={"fontname": FONTNAME},
            y=1.1,
            color="red",
        )

    render_model(
        ax_model=ax_model,
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        annotations=annotations,
        current_b0=current_b0,
        current_b1=current_b1,
        explored_b0=explored_b0,
        explored_b1=explored_b1,
        explored_rmse=explored_rmse,
        mappable=mappable,
    )

    fig.suptitle(
        annotations.get_title(),
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        va="top",
        x=0.5,
        y=1.2,
    )

    return fig, float(rmse_value)


def show_optimal_b_search_gradient(
    mode: str = "rus",
    learning_rate: float = 0.4,
    start_b0: float = 1.2,
    start_b1: float = -0.5,
    max_iterations: int = 10,
    grad_tol: float = 0.02,
    step_tol: float = 1e-4,
    backtracking_max_tries: int = 25,
    backtracking_shrink: float = 0.5,
    pause_frames: int = 3,
    surface_grid_size: int = GRID_SIZE,
):
    annotations = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir = get_tmp_animation_directory()

    dataframe, features_scaled, target_scaled = generate_df_coefficients_vs_error(grid_size=surface_grid_size)

    # Precompute surface once (stable colors)
    errors_surface, intercept_values, slope_values = compute_surface_from_dataframe(dataframe)

    # Build gradient descent trajectory
    path = gradient_descent_path(
        start_b0=float(start_b0),
        start_b1=float(start_b1),
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        learning_rate=float(learning_rate),
        max_iterations=int(max_iterations),
        grad_tol=float(grad_tol),
        step_tol=float(step_tol),
        backtracking_max_tries=int(backtracking_max_tries),
        backtracking_shrink=float(backtracking_shrink),
    )

    image_files: List[Path] = []
    explored_b0: List[float] = []
    explored_b1: List[float] = []
    explored_rmse: List[float] = []

    # Overwrite the same SVG each frame (as in your pipeline)
    raw_svg_file = Path(tmp_dir, f"58_optimization_gradient_{mode}.svg")

    for frame_index, (current_b0, current_b1) in enumerate(path):
        # "iterations done so far" = frame_index
        fig, rmse_value = generate_frame(
            current_b0=float(current_b0),
            current_b1=float(current_b1),
            errors_surface=errors_surface,
            intercept_values=intercept_values,
            slope_values=slope_values,
            features_scaled=features_scaled,
            target_scaled=target_scaled,
            annotations=annotations,
            explored_b0=explored_b0,
            explored_b1=explored_b1,
            explored_rmse=explored_rmse,
            iteration_count=int(frame_index),
            map_title_override=None,
        )

        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close(fig)

        frame_png = Path(tmp_dir, f"58_optimization_gradient_{mode}_{frame_index}.png")
        save_plot_according_to_template(
            raw_svg_file,
            frame_png,
            template_name="template_small.svg",
            dpi=200,
        )
        image_files.append(frame_png)

        # Update explored AFTER saving the frame
        explored_b0.append(float(current_b0))
        explored_b1.append(float(current_b1))
        explored_rmse.append(float(rmse_value))

    # Final frame: mark as best model (last point)
    if len(path) > 0:
        last_b0, last_b1 = path[-1]
        fig, _ = generate_frame(
            current_b0=float(last_b0),
            current_b1=float(last_b1),
            errors_surface=errors_surface,
            intercept_values=intercept_values,
            slope_values=slope_values,
            features_scaled=features_scaled,
            target_scaled=target_scaled,
            annotations=annotations,
            explored_b0=explored_b0,
            explored_b1=explored_b1,
            explored_rmse=explored_rmse,
            iteration_count=int(len(path)),
            map_title_override=annotations.best_model,
        )

        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close(fig)

        final_png = Path(tmp_dir, f"58_optimization_gradient_{mode}_{len(image_files)}.png")
        save_plot_according_to_template(
            raw_svg_file,
            final_png,
            template_name="template_small.svg",
            dpi=200,
        )
        for _ in range(int(pause_frames)):
            image_files.append(final_png)

    gif_path = Path(get_plots_path(), f"58_optimization_gradient_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    show_optimal_b_search_gradient(
        mode="rus",
        learning_rate=0.4,
        start_b0=1.2,
        start_b1=-0.5,
        max_iterations=10,
        grad_tol=0.02,
        step_tol=1e-4,
        pause_frames=3,
    )
