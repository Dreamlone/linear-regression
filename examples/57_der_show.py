import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
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
SLICE_XLIM = (-2.5, 2.5)
SLICE_YLIM = (0.0, 4.5)


@dataclass
class RusAnnotations:
    landscape_x_axis: str = "Сдвиг\nстандартизированный\n($b_0$)"
    landscape_y_axis: str = "Наклон\nстандартизированный\n($b_1$)"
    slice_x_axis: str = "Сдвиг стандартизированный ($b_0$)"
    slice_y_axis: str = "Ошибка модели (MSE)"
    scatter_title: str = "Модель с выбранными коэффициентами"
    scatter_x_axis: str = "Количество комнат стандартизированное\n(x)"
    scatter_y_axis: str = "Стоимость, стандартизированная\n(y)"


@dataclass
class EngAnnotations:
    landscape_x_axis: str = "Intercept, scaled ($b_0$)"
    landscape_y_axis: str = "Slope, scaled ($b_1$)"
    slice_x_axis: str = "Intercept, scaled ($b_0$)"
    slice_y_axis: str = "Model error (MSE)"
    scatter_title: str = "Model with chosen coefficients"
    scatter_x_axis: str = "Number of rooms, scaled (x)"
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


def mse_and_gradients(
    intercept_value: float,
    slope_value: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> Tuple[float, float, float]:
    """
    MSE(b0, b1) = (1/n) sum (y - (b0 + b1*x))^2

    dMSE/db0 = (2/n) sum ( (b0 + b1*x) - y )
    dMSE/db1 = (2/n) sum ( ((b0 + b1*x) - y) * x )
    """
    x_values = np.ravel(features_scaled).astype(float)
    y_values = np.ravel(target_scaled).astype(float)

    predicted = intercept_value + slope_value * x_values
    residual = predicted - y_values

    n = float(len(x_values))
    mse_value = float(np.mean(residual ** 2))

    grad_b0 = float((2.0 / n) * np.sum(residual))
    grad_b1 = float((2.0 / n) * np.sum(residual * x_values))

    return mse_value, grad_b0, grad_b1


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

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(errors_surface)
    return mappable


def create_axes_like_b(fig):
    gs = fig.add_gridspec(1, 3)
    gs.update(wspace=0.4)

    ax_landscape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_slice = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])

    ax_slice.tick_params(axis="both", which="major", labelsize=11)
    ax_model.tick_params(axis="both", which="major", labelsize=11)

    ax_slice.set_ylim(*SLICE_YLIM)
    ax_slice.set_xlim(*SLICE_XLIM)

    return ax_landscape, ax_slice, ax_model


def clip_b0(value: float) -> float:
    return float(np.clip(value, MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER))


def gradient_descent_b0_path(
    start_b0: float,
    slope_fixed: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    grad_tol: float,
    step_tol: float,
    backtracking_max_tries: int = 25,
    backtracking_shrink: float = 0.5,
) -> List[float]:
    """
    True gradient descent on b0 only:
      b0 <- b0 - learning_rate * dMSE/db0
    Includes backtracking to guarantee MSE decreases.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    b0_current = clip_b0(float(start_b0))
    path = [b0_current]

    for _ in range(max_iterations):
        mse_current, grad_b0, _ = mse_and_gradients(b0_current, slope_fixed, features_scaled, target_scaled)

        if abs(grad_b0) < grad_tol:
            break

        lr_try = float(learning_rate)
        accepted = False
        b0_next = b0_current

        for _ in range(backtracking_max_tries):
            candidate = clip_b0(b0_current - lr_try * grad_b0)
            mse_candidate, _, _ = mse_and_gradients(candidate, slope_fixed, features_scaled, target_scaled)

            if mse_candidate <= mse_current:
                b0_next = candidate
                accepted = True
                break

            lr_try *= backtracking_shrink

        if not accepted:
            break

        if abs(b0_next - b0_current) < step_tol:
            b0_current = b0_next
            path.append(b0_current)
            break

        b0_current = b0_next
        path.append(b0_current)

    return path


def render_frame(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    annotations: Union[EngAnnotations, RusAnnotations],
    errors_surface: np.ndarray,
    intercept_surface_values: np.ndarray,
    slope_surface_values: np.ndarray,
    intercept_dense: np.ndarray,
    slice_errors_dense: np.ndarray,
    slope_fixed: float,
    intercept_value: float,
    point_color: str,
    run_learning_rate: float,
    norm,
    cmap,
    max_abs_grad_b0: float,
    max_abs_antigrad_norm_2d: float,
    output_png: Path,
    raw_svg_file: Path,  # overwritten each frame
):
    mse_value, grad_b0, grad_b1 = mse_and_gradients(intercept_value, slope_fixed, features_scaled, target_scaled)

    fig = plt.figure(figsize=(16, 4))
    ax_land, ax_slice, ax_model = create_axes_like_b(fig)
    ax_land.computed_zorder = False

    # --- Landscape + colorbar like B ---
    mappable = plot_mse_surface(
        ax_landscape=ax_land,
        errors_surface=errors_surface,
        intercept_values=intercept_surface_values,
        slope_values=slope_surface_values,
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(mappable, ax=ax_land, shrink=0.8, pad=0.2)
    cbar.ax.set_title("MSE", pad=8, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax_land.view_init(elev=28, azim=125)
    ax_land.invert_xaxis()
    ax_land.set_xlabel(annotations.landscape_x_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_land.set_ylabel(annotations.landscape_y_axis, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax_land.set_zlabel("")  # no Z label
    ax_land.set_title(
        "Ландшафт MSE и антиградиент",
        fontsize=12,
        fontdict={"fontname": FONTNAME},
        x=0.75,
        y=1.35,
    )

    ax_land.scatter(
        float(intercept_value),
        float(slope_fixed),
        float(mse_value),
        c=point_color,
        s=70,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=7,
    )

    # --- Anti-gradient arrow on landscape ---
    anti_b0 = -grad_b0
    anti_b1 = -grad_b1
    anti_norm = float(np.hypot(anti_b0, anti_b1))

    if anti_norm > 0:
        arrow_len_param = 1.1
        scale_den = max(max_abs_antigrad_norm_2d, 1e-12)
        relative = min(1.0, anti_norm / scale_den)
        length = arrow_len_param * relative

        # Rotating arrowhead in 3D is non-trivial; keep quiver with larger head ratio.
        ax_land.quiver(
            float(intercept_value),
            float(slope_fixed),
            float(mse_value),
            float(anti_b0),
            float(anti_b1),
            0.0,
            normalize=True,
            length=float(length),
            color="black",
            linewidth=2.2,
            arrow_length_ratio=0.25,
            pivot="tail",
            zorder=10,
        )

    # --- Slice: dense curve + point + grad/anti-grad arrows ---
    ax_slice.plot(intercept_dense, slice_errors_dense, color="grey", linewidth=2.2, alpha=0.9, zorder=1)
    ax_slice.scatter(
        [float(intercept_value)],
        [float(mse_value)],
        c=point_color,
        s=70,
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )

    ax_slice.grid(color="grey", alpha=0.3, zorder=0)
    ax_slice.set_xlabel(annotations.slice_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_slice.set_ylabel(annotations.slice_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})

    # Updated title: step + current b0 + gradient (no mathtext)
    b0 = "$b_0$"
    ax_slice.set_title(
        f"Шаг спуска = {run_learning_rate:.1f}\n"
        f"{b0} = {intercept_value:.2f},  градиент = {grad_b0:.2f}",
        fontsize=12,
        fontdict={"fontname": FONTNAME},
        y=1.04,
    )

    # Arrow scaling (close to the point)
    dx_max = 0.85
    denom = max(max_abs_grad_b0, 1e-12)
    rel_1d = min(1.0, abs(grad_b0) / denom)
    dx_len = dx_max * rel_1d

    arrow_y = float(mse_value) + 0.04
    arrow_y = min(arrow_y, SLICE_YLIM[1] - 0.25)

    if dx_len > 0:
        sign = 1.0 if grad_b0 >= 0 else -1.0

        # Gradient arrow (red): direction of MSE increase
        ax_slice.annotate(
            "",
            xy=(float(intercept_value) + sign * dx_len, arrow_y),
            xytext=(float(intercept_value), arrow_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
            zorder=6,
        )
        ax_slice.text(
            float(intercept_value) + sign * dx_len,
            arrow_y + 0.05,
            "градиент",
            color="red",
            fontsize=8,
            fontname=FONTNAME,
            ha="center",
            va="bottom",
            zorder=7,
        )

        # Anti-gradient arrow (black): direction of MSE decrease
        ax_slice.annotate(
            "",
            xy=(float(intercept_value) - sign * dx_len, arrow_y),
            xytext=(float(intercept_value), arrow_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
            zorder=6,
        )

        # Anti-gradient label slightly BELOW the arrow to avoid overlap near optimum
        anti_label_y = max(SLICE_YLIM[0] + 0.05, arrow_y - 0.09)
        ax_slice.text(
            float(intercept_value) - sign * dx_len,
            anti_label_y,
            "антиградиент",
            color="black",
            fontsize=8,
            fontname=FONTNAME,
            ha="center",
            va="top",
            zorder=7,
        )

    ax_slice.set_xlim(*SLICE_XLIM)
    ax_slice.set_ylim(*SLICE_YLIM)

    # --- Model: only current line + data; line color synced with point color ---
    ax_model.scatter(features_scaled, target_scaled, s=40, c="white", edgecolor="black", zorder=2)

    x_line = np.array([[-1.5], [1.5]])
    predicted_model = float(intercept_value) + float(slope_fixed) * x_line

    ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c="black", linewidth=2.5, zorder=2)
    ax_model.plot(np.ravel(x_line), np.ravel(predicted_model), c=point_color, linewidth=2.0, zorder=3, alpha=0.85)

    ax_model.set_xlabel(annotations.scatter_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylabel(annotations.scatter_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax_model.set_ylim(-3, 3)
    ax_model.set_xlim(-2, 2)
    ax_model.grid(color="grey", alpha=0.3, zorder=1)
    ax_model.set_title(annotations.scatter_title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)

    fig.suptitle(
        "Градиент и антиградиент MSE (градиентный спуск по $b_0$)",
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        x=0.52,
        y=1.2,
    )

    # Overwrite the same SVG each frame
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close(fig)

    save_plot_according_to_template(raw_svg_file, output_png, template_name="template_small.svg", dpi=DPI)


def show_animation(
    mode: str = "rus",
    surface_grid_size: int = 15,  # fixed as requested
    dense_slice_points: int = 250,
    fixed_slope: Optional[float] = None,
    start_grid_size: int = 15,
    start_indices: Tuple[int, int] = (3, 12),
    colors: Tuple[str, str] = ("gold", "#15D600"),
    learning_rate_fast: float = 0.35,
    learning_rate_slow: float = 0.08,
    max_iterations: int = 80,
    grad_tol: float = 1e-3,
    step_tol: float = 1e-4,
    pause_frames: int = 2,
):
    annotations = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir = get_tmp_animation_directory()

    features_scaled, target_scaled = load_scaled_data()
    slope_value = float(compute_optimal_slope(features_scaled, target_scaled)) if fixed_slope is None else float(fixed_slope)

    # Surface for landscape
    dataframe = generate_df_coefficients_vs_error(features_scaled, target_scaled, grid_size=surface_grid_size)
    errors_surface, intercept_surface_values, slope_surface_values = compute_surface_from_dataframe(dataframe)

    vmin = float(np.nanmin(errors_surface))
    vmax = float(np.nanmax(errors_surface))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    # Dense slice for ax_slice
    intercept_dense = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, dense_slice_points, dtype=float)
    slice_errors_dense = compute_slice_errors(intercept_dense, slope_value, features_scaled, target_scaled)

    # Start points from a coarse grid (like B)
    intercept_starts = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, start_grid_size, dtype=float)
    start_b0_1 = float(intercept_starts[int(start_indices[0])])
    start_b0_2 = float(intercept_starts[int(start_indices[1])])

    path_1 = gradient_descent_b0_path(
        start_b0=start_b0_1,
        slope_fixed=slope_value,
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        learning_rate=learning_rate_fast,
        max_iterations=max_iterations,
        grad_tol=grad_tol,
        step_tol=step_tol,
    )
    path_2 = gradient_descent_b0_path(
        start_b0=start_b0_2,
        slope_fixed=slope_value,
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        learning_rate=learning_rate_slow,
        max_iterations=max_iterations,
        grad_tol=grad_tol,
        step_tol=step_tol,
    )

    # Scaling for arrows across BOTH runs (stable visuals)
    all_b0_values = list(path_1) + list(path_2)
    grad_b0_list: List[float] = []
    antigrad_norm_list: List[float] = []
    for b0 in all_b0_values:
        _, grad_b0, grad_b1 = mse_and_gradients(float(b0), slope_value, features_scaled, target_scaled)
        grad_b0_list.append(float(grad_b0))
        antigrad_norm_list.append(float(np.hypot(-grad_b0, -grad_b1)))

    max_abs_grad_b0 = float(max([abs(v) for v in grad_b0_list] + [1e-12]))
    max_abs_antigrad_norm_2d = float(max(antigrad_norm_list + [1e-12]))

    raw_svg_file = Path(tmp_dir, "grad_descent_frame.svg")  # overwritten each frame

    image_files: List[Path] = []
    frame_idx = 0

    for path, run_color, run_lr in [
        (path_1, colors[0], float(learning_rate_fast)),
        (path_2, colors[1], float(learning_rate_slow)),
    ]:
        for b0 in path:
            frame_png = Path(tmp_dir, f"grad_descent_{frame_idx:04d}.png")
            render_frame(
                features_scaled=features_scaled,
                target_scaled=target_scaled,
                annotations=annotations,
                errors_surface=errors_surface,
                intercept_surface_values=intercept_surface_values,
                slope_surface_values=slope_surface_values,
                intercept_dense=intercept_dense,
                slice_errors_dense=slice_errors_dense,
                slope_fixed=slope_value,
                intercept_value=float(b0),
                point_color=run_color,
                run_learning_rate=run_lr,
                norm=norm,
                cmap=cmap,
                max_abs_grad_b0=max_abs_grad_b0,
                max_abs_antigrad_norm_2d=max_abs_antigrad_norm_2d,
                output_png=frame_png,
                raw_svg_file=raw_svg_file,
            )
            image_files.append(frame_png)
            frame_idx += 1

        for _ in range(pause_frames):
            frame_png = Path(tmp_dir, f"grad_descent_{frame_idx:04d}.png")
            render_frame(
                features_scaled=features_scaled,
                target_scaled=target_scaled,
                annotations=annotations,
                errors_surface=errors_surface,
                intercept_surface_values=intercept_surface_values,
                slope_surface_values=slope_surface_values,
                intercept_dense=intercept_dense,
                slice_errors_dense=slice_errors_dense,
                slope_fixed=slope_value,
                intercept_value=float(path[-1]),
                point_color="grey",
                run_learning_rate=run_lr,
                norm=norm,
                cmap=cmap,
                max_abs_grad_b0=max_abs_grad_b0,
                max_abs_antigrad_norm_2d=max_abs_antigrad_norm_2d,
                output_png=frame_png,
                raw_svg_file=raw_svg_file,
            )
            image_files.append(frame_png)
            frame_idx += 1

    gif_path = Path(get_plots_path(), f"57_grad_antigrad_b0_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    show_animation(
        mode="rus",
        fixed_slope=None,
        start_grid_size=15,
        start_indices=(3, 12),
        learning_rate_fast=0.4,
        learning_rate_slow=0.1,
        max_iterations=80,
        grad_tol=0.02,
        step_tol=0.001,
        pause_frames=2,
    )
