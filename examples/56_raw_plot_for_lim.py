from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset


MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
FONTNAME = "Comic Sans MS"
DPI = 120

# Fixed limits so plots don't "jump"
SLICE_XLIM = (-2.2, 2.2)
SLICE_YLIM = (0.0, 4.5)


@dataclass
class RusAnnotations:
    slice_x_axis: str = "Сдвиг стандартизированный ($b_0$)"
    slice_y_axis: str = "Ошибка модели (MSE)"


@dataclass
class EngAnnotations:
    slice_x_axis: str = "Intercept, scaled ($b_0$)"
    slice_y_axis: str = "Model error (MSE)"


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

    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x.reshape(-1, 1))

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    return features_scaled, target_scaled


def compute_slice_errors(
    intercept_values: np.ndarray,
    slope_fixed: float,
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    intercept_values = np.ravel(intercept_values).astype(float)

    return np.array(
        [
            mean_squared_error(target_scaled, intercept_value + slope_fixed * features_scaled)
            for intercept_value in intercept_values
        ],
        dtype=float,
    )


def create_axes_2x2(fig):
    # sharex/sharey keeps scales identical, but we'll still label all axes explicitly.
    axes = fig.subplots(2, 2)

    for ax in axes.ravel():
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.set_xlim(*SLICE_XLIM)
        ax.set_ylim(*SLICE_YLIM)

    return axes


def render_static_slice_2x2(
    mode: str = "rus",
    grid_left: int = 5,
    grid_right: int = 10,
    dense_grid: int = 100,
    out_name: str = "55_slice_2x2_grid_5_vs_10",
):
    annotations = annotations_by_language(mode)

    features_scaled, target_scaled = load_scaled_data()
    slope_fixed = compute_optimal_slope(features_scaled, target_scaled)

    # Dense "true" curve (grey)
    intercept_dense = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, dense_grid, dtype=float)
    errors_dense = compute_slice_errors(intercept_dense, slope_fixed, features_scaled, target_scaled)

    # Coarse approximations (black with x markers)
    intercept_left = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_left, dtype=float)
    errors_left = compute_slice_errors(intercept_left, slope_fixed, features_scaled, target_scaled)

    intercept_right = np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, grid_right, dtype=float)
    errors_right = compute_slice_errors(intercept_right, slope_fixed, features_scaled, target_scaled)

    fig = plt.figure(figsize=(10, 10))
    axes = create_axes_2x2(fig)

    def draw_slice(ax, intercept_coarse, errors_coarse, title_text: str):
        # True curve (dense) -> grey line
        ax.plot(intercept_dense, errors_dense, color="grey", linewidth=2.0, alpha=0.9, zorder=1)

        # Coarse approximation -> black line + x markers
        ax.plot(
            intercept_coarse,
            errors_coarse,
            color="black",
            linewidth=2.0,
            marker="x",
            markersize=6,
            zorder=2,
        )

        ax.grid(color="grey", alpha=0.3, zorder=0)
        ax.set_title(title_text, fontsize=12, fontdict={"fontname": FONTNAME})

        # IMPORTANT: label EVERY subplot explicitly
        ax.set_xlabel(annotations.slice_x_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax.set_ylabel(annotations.slice_y_axis, fontdict={"fontsize": 10, "fontname": FONTNAME})

    # Row 1
    draw_slice(axes[0, 0], intercept_left, errors_left, f"grid size = {grid_left}")
    draw_slice(axes[0, 1], intercept_right, errors_right, f"grid size = {grid_right}")

    # Row 2 (duplicate of row 1)
    draw_slice(axes[1, 0], intercept_left, errors_left, f"grid size = {grid_left}")
    draw_slice(axes[1, 1], intercept_right, errors_right, f"grid size = {grid_right}")

    fig.suptitle(
        "Срез MSE по $b_0$: истинная кривая (100 точек) vs аппроксимация по сетке",
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        x=0.5,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plots_dir = Path(get_plots_path())
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_svg_file = plots_dir / f"{out_name}_{mode}.svg"
    output_png = plots_dir / f"{out_name}_{mode}.png"

    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close(fig)

    save_plot_according_to_template(raw_svg_file, output_png, template_name="template.svg", dpi=DPI)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    render_static_slice_2x2(mode="rus", grid_left=5, grid_right=9, dense_grid=200)
