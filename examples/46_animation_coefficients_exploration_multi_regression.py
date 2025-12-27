import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import imageio
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 12
ANIMATION_DURATION: int = 150
FONTNAME = "Comic Sans MS"
CMAP_BY_METRIC = {"rmse": 'coolwarm'}

from matplotlib.colors import Normalize

def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
    """Convert regularly spaced centers to bin edges (for imshow3d-style surfaces)."""
    vals = np.asarray(vals, dtype=float)
    if vals.size < 2:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = (vals[:-1] + vals[1:]) / 2.0
    edges[0] = vals[0] - (vals[1] - vals[0]) / 2.0
    edges[-1] = vals[-1] + (vals[-1] - vals[-2]) / 2.0
    return edges


def imshow3d_coeff(
    ax,
    array_2d: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    value_direction: str = "z",
    pos: float = 0.0,
    norm: Normalize | None = None,
    cmap: str = "coolwarm",
    alpha: float = 0.90,
    zorder: int = 0,
):
    if norm is None:
        norm = Normalize(vmin=float(np.nanmin(array_2d)), vmax=float(np.nanmax(array_2d)))

    colors = plt.get_cmap(cmap)(norm(array_2d))
    colors[..., 3] *= alpha

    U, V = np.meshgrid(u_edges, v_edges)

    if value_direction == "z":
        X, Y = U, V
        Z = np.full_like(X, float(pos), dtype=float)
    elif value_direction == "x":
        Y, Z = U, V
        X = np.full_like(Y, float(pos), dtype=float)
    elif value_direction == "y":
        X, Z = U, V
        Y = np.full_like(X, float(pos), dtype=float)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, shade=False, linewidth=0, antialiased=False)
    surf.set_zorder(zorder)
    if hasattr(surf, "set_zsort"):
        surf.set_zsort("min")
    return surf


@dataclass
class Annotations:
    scatter_label_raw = r"$\hat{y} = INTERCEPT + SLOPE\cdot x$"
    scatter_label_scaled = r"$\hat{y}_{scaled} = INTERCEPT + SLOPE \cdot x_{scaled}$"

    def bake_scatter_label_raw(self, intercept: float, slope: float,
                               features_scaler: StandardScaler, target_scaler: StandardScaler):
        mean_x = float(features_scaler.mean_[0])
        scale_x = float(features_scaler.scale_[0])

        mean_y = float(target_scaler.mean_[0])
        scale_y = float(target_scaler.scale_[0])

        slope_raw = (scale_y / scale_x) * slope
        intercept_raw = mean_y + scale_y * intercept - slope_raw * mean_x

        label = self.scatter_label_raw.replace("INTERCEPT", f"{intercept_raw:.1f}")
        label = label.replace("SLOPE", f"{abs(slope_raw):.1f}")
        if slope_raw < 0:
            label = label.replace(" + ", " - ")

        return label

    def bake_scatter_label_scaled(self, intercept: float, slope: float):
        label = self.scatter_label_scaled.replace("INTERCEPT", f"{intercept:.1f}")
        label = label.replace("SLOPE", f"{abs(slope):.1f}")
        if slope < 0:
            label = label.replace(" + ", " - ")

        return label


@dataclass
class RusAnnotations(Annotations):
    map_title: str = "Зависимость METRIC от коэффициентов модели"
    map_x_axis: str = r"Сдвиг стандартизированный ($b_0$)"
    map_y_axis: str = r"Наклон стандартизированный ($b_1$)"
    scatter_x_axis: str = "Количество комнат (x)"
    scatter_y_axis: str = "Стоимость, $ (y)"

    def get_map_title(self, metric: str):
        if metric == "mae":
            map_title = self.map_title.replace("METRIC", "средней абсолютной ошибки")
        elif metric == "mape":
            map_title = self.map_title.replace("METRIC", "средней абсолютной процентной ошибки")
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_title

    def get_map_label(self, metric: str):
        if metric == "mae":
            map_label = "Средняя абсолютная ошибка (MAE), $"
        elif metric == "mape":
            map_label = "Средняя абсолютная процентная ошибка (MAPE), %"
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_label


@dataclass
class EngAnnotations(Annotations):
    map_title: str = "Dependence of METRIC on intercept and slope"
    map_x_axis: str = r"Intercept scaled ($b_0$)"
    map_y_axis: str = r"Slope scaled ($b_1$)"
    scatter_x_axis: str = "Number of the rooms in the apartment (x)"
    scatter_y_axis: str = "Price, $ (y)"

    def get_map_title(self, metric: str):
        if metric == "mae":
            map_title = self.map_title.replace("METRIC", "Mean Absolute Error")
        elif metric == "mape":
            map_title = self.map_title.replace("METRIC", "Mean Absolute Percentage Error")
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_title

    def get_map_label(self, metric: str):
        if metric == "mae":
            map_label = "Mean Absolute Error (MAE), $"
        elif metric == "mape":
            map_label = "Mean Absolute Percentage Error (MAPE), %"
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_label


def annotations_by_language(mode: str):
    if mode == "eng":
        annotations = EngAnnotations()
    elif mode == "rus":
        annotations = RusAnnotations()
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return annotations


def _measure_metric(metric: str, y_true, y_predicted):
    if metric == "rmse":
        metric_value = root_mean_squared_error(np.ravel(y_true), np.ravel(y_predicted))
    elif metric == "mae":
        metric_value = mean_absolute_error(np.ravel(y_true), np.ravel(y_predicted))
    elif metric == "mape":
        metric_value = mean_absolute_percentage_error(np.ravel(y_true), np.ravel(y_predicted))
    else:
        raise NotImplementedError(f"Metric {metric} is not supported")
    return metric_value


def generate_df_coefficients_vs_error(metric: str, features_to_use: list):
    print("Generating the dataset with metrics")
    dataset = get_extended_dataset()
    features = np.array(dataset[features_to_use])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x)
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    dataframe = []
    for coeff_b0 in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
        for coeff_b1 in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
            for coeff_b2 in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
                predicted_scaled = coeff_b0 + coeff_b1 * features_scaled[:, 0] + coeff_b2 * features_scaled[:, 1]
                predicted_raw = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
                metric_value = _measure_metric(metric, y, predicted_raw)
                dataframe.append([coeff_b0, coeff_b1, coeff_b2, metric_value])

    columns = ["coeff_b0", "coeff_b1", "coeff_b2", "metric"]
    dataframe = pd.DataFrame(dataframe, columns=columns)
    print(dataframe.head(5))
    return dataframe, x, y, features_scaled, target_scaled, features_scaler, target_scaler


def compose_cases_to_explore(steps_per_section: int, sections: list):
    cases: list[list[float]] = []

    for section_index, section in enumerate(sections):
        if "start" not in section or "end" not in section:
            raise ValueError("Each section must have 'start' and 'end' keys.")

        start = np.asarray(section["start"], dtype=float)
        end = np.asarray(section["end"], dtype=float)

        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("'start' and 'end' must be 3D points: [b0, b1, b2].")

        t_values = np.linspace(0.0, 1.0, steps_per_section)
        points = (1.0 - t_values)[:, None] * start[None, :] + t_values[:, None] * end[None, :]

        if section_index > 0:
            points = points[1:]

        cases.extend(points.tolist())

    return cases


def explore_coefficients_landscape_3d(mode: str = "eng",
                                      features_to_use: Union[list,None] = None,
                                      metric: str = "rmse"):
    """ Create a 3d plot with multiple coefficients """
    if features_to_use is None:
        features_to_use = ["rooms", "metro_distance"]

    annotations = annotations_by_language(mode)
    sections = [
        {"start": [-1.5, -1.5, -1.5], "end": [-1.5, 0.0, -1.5]},
        {"start": [-1.5, 0.0, -1.5], "end": [-0.3, 1.0, -1.5]},
        {"start": [-0.3, 1.0, -1.5], "end": [-0.3, 1.0, 0.0]},
    ]
    cases_to_visualize = compose_cases_to_explore(steps_per_section=10, sections=sections)
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    (dataframe, features, target, features_scaled, target_scaled,
     features_scaler, target_scaler) = generate_df_coefficients_vs_error(metric, features_to_use)

    coeff_b0 = np.array(dataframe["coeff_b0"])
    coeff_b1 = np.array(dataframe["coeff_b1"])
    coeff_b2 = np.array(dataframe["coeff_b2"])
    metric_values = np.array(dataframe["metric"])

    image_files = []
    explored_path_coeff_b0 = []
    explored_path_coeff_b1 = []
    explored_path_coeff_b2 = []
    for image_index, case in enumerate(cases_to_visualize):
        coeff_b0_i, coeff_b1_i, coeff_b2_i = case
        explored_path_coeff_b0.append(coeff_b0_i)
        explored_path_coeff_b1.append(coeff_b1_i)
        explored_path_coeff_b2.append(coeff_b2_i)

        fig = plt.figure(figsize=(20, 9))
        ax1 = fig.add_axes([0.06, 0.56, 0.40, 0.34], projection="3d")
        ax2 = fig.add_axes([0.06, 0.10, 0.40, 0.34], projection="3d")
        ax3 = fig.add_axes([0.54, 0.33, 0.40, 0.34])
        ax3.axis("off")

        if hasattr(ax1, "computed_zorder"):
            ax1.computed_zorder = False
        if hasattr(ax2, "computed_zorder"):
            ax2.computed_zorder = False

        b0_vals = np.unique(coeff_b0)
        b1_vals = np.unique(coeff_b1)
        b2_vals = np.unique(coeff_b2)
        nx, ny, nz = len(b0_vals), len(b1_vals), len(b2_vals)

        metric_grid = metric_values.reshape(nx, ny, nz)

        ix = int(np.argmin(np.abs(b0_vals - coeff_b0_i)))
        iy = int(np.argmin(np.abs(b1_vals - coeff_b1_i)))
        iz = int(np.argmin(np.abs(b2_vals - coeff_b2_i)))

        slice_xy = metric_grid[:, :, iz]
        slice_yz = metric_grid[ix, :, :]
        slice_xz = metric_grid[:, iy, :]

        b0_edges = _centers_to_edges(b0_vals)
        b1_edges = _centers_to_edges(b1_vals)
        b2_edges = _centers_to_edges(b2_vals)

        cmap_name = CMAP_BY_METRIC.get(metric, "coolwarm")
        norm = Normalize(vmin=float(np.min(metric_values)), vmax=float(np.max(metric_values)))

        # --- AX1: voxels + highlight 3 slices (ix, iy, iz) in red ---
        Xc, Yc, Zc = np.meshgrid(b0_edges, b1_edges, b2_edges, indexing="ij")
        filled = np.ones((nx, ny, nz), dtype=bool)

        facecolors = plt.get_cmap(cmap_name)(norm(metric_grid))
        facecolors[..., 3] *= 0.60

        highlight_mask = np.zeros((nx, ny, nz), dtype=bool)
        highlight_mask[ix, :, :] = True
        highlight_mask[:, iy, :] = True
        highlight_mask[:, :, iz] = True

        facecolors[highlight_mask, 0] = 1.0
        facecolors[highlight_mask, 1] = 0.0
        facecolors[highlight_mask, 2] = 0.0
        facecolors[highlight_mask, 3] = 0.99

        ax1.voxels(Xc, Yc, Zc, filled, facecolors=facecolors, edgecolor="k", linewidth=0.15, shade=False)
        ax1.set_xlabel("coeff_b0", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax1.set_ylabel("coeff_b1", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax1.set_zlabel("coeff_b2", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax1.set_xlim(float(b0_edges[0]), float(b0_edges[-1]))
        ax1.set_ylim(float(b1_edges[0]), float(b1_edges[-1]))
        ax1.set_zlim(float(b2_edges[0]), float(b2_edges[-1]))
        ax1.view_init(elev=22, azim=-55)

        # --- AX2: 3 slice planes + red trajectory ---
        x_wall = float(b0_edges[0])
        y_wall = float(b1_edges[-1])
        z_floor = float(b2_edges[0])

        imshow3d_coeff(ax2, array_2d=slice_xy.T, u_edges=b0_edges, v_edges=b1_edges, value_direction="z",
                       pos=z_floor, norm=norm, cmap=cmap_name, alpha=0.90, zorder=0)
        imshow3d_coeff(ax2, array_2d=slice_yz.T, u_edges=b1_edges, v_edges=b2_edges,
                       value_direction="x", pos=x_wall, norm=norm, cmap=cmap_name, alpha=0.90, zorder=0)
        imshow3d_coeff(ax2, array_2d=slice_xz.T, u_edges=b0_edges, v_edges=b2_edges,
                       value_direction="y", pos=y_wall, norm=norm, cmap=cmap_name, alpha=0.90, zorder=0)

        line = ax2.plot(explored_path_coeff_b0, explored_path_coeff_b1, explored_path_coeff_b2, c="red", linewidth=3)[0]
        line.set_zorder(10_000)
        pt = ax2.scatter(coeff_b0_i, coeff_b1_i, coeff_b2_i, c="red", s=180, marker="D", depthshade=False)
        pt.set_zorder(10_001)

        ax2.set_xlabel("coeff_b0", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax2.set_ylabel("coeff_b1", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax2.set_zlabel("coeff_b2", fontdict={"fontsize": 10, "fontname": FONTNAME})
        ax2.set_xlim(float(b0_edges[0]), float(b0_edges[-1]))
        ax2.set_ylim(float(b1_edges[0]), float(b1_edges[-1]))
        ax2.set_zlim(float(b2_edges[0]), float(b2_edges[-1]))
        ax2.view_init(elev=25, azim=-55)

        # --- One shared colorbar for ax1 + ax2 in the middle ---
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
        mappable.set_array([])
        # To move to the left, decrease the first coordinate
        cax = fig.add_axes([0.44, 0.315, 0.01, 0.39])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(metric.upper(), fontdict={"fontsize": 12, "fontname": FONTNAME})

        raw_svg_file = Path(tmp_dir, f"46_explore_coefficients_3d_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"46_explore_coefficients_3d_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg", dpi=100)
        image_files.append(path_to_final_path)

    features_label = '_'.join(features_to_use)
    gif_path = Path(get_plots_path(), f"46_explore_coefficients_3d_{features_label}_{metric}_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    explore_coefficients_landscape_3d("rus", metric="rmse")
