import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import imageio
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 10
ANIMATION_DURATION: int = 180
FONTNAME = "Comic Sans MS"
CMAP = 'coolwarm'


def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
    """Convert regularly spaced centers to bin edges (for imshow3d-style surfaces)"""
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

    u, v = np.meshgrid(u_edges, v_edges)

    if value_direction == "z":
        x, y = u, v
        z = np.full_like(x, float(pos), dtype=float)
    elif value_direction == "x":
        y, z = u, v
        x = np.full_like(y, float(pos), dtype=float)
    elif value_direction == "y":
        x, z = u, v
        y = np.full_like(x, float(pos), dtype=float)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, shade=False,
                           linewidth=0, antialiased=False)
    surf.set_zorder(zorder)
    if hasattr(surf, "set_zsort"):
        surf.set_zsort("min")
    return surf


@dataclass
class Annotations:
    scatter_label_raw = r"$\hat{y} = INTERCEPT + B1 \cdot x_1 + B2 \cdot x_2$"
    scatter_label_scaled = r"$\hat{y}_{scaled} = INTERCEPT + B1 \cdot x_{1scaled} + B2 \cdot x_{2scaled}$"

    def bake_scatter_label_raw(self, intercept: float, b1: float, b2: float,
                               features_scaler: StandardScaler, target_scaler: StandardScaler):
        mean_x1 = float(features_scaler.mean_[0])
        mean_x2 = float(features_scaler.mean_[1])
        scale_x1 = float(features_scaler.scale_[0])
        scale_x2 = float(features_scaler.scale_[1])

        mean_y = float(target_scaler.mean_[0])
        scale_y = float(target_scaler.scale_[0])

        b1_raw = (scale_y / scale_x1) * b1
        b2_raw = (scale_y / scale_x2) * b2
        intercept_raw = mean_y + scale_y * intercept - b1_raw * mean_x1 - b2_raw * mean_x2

        label = self.scatter_label_raw
        if b1_raw < 0:
            label = label.replace(" + B1", " - B1")
        if b2_raw < 0:
            label = label.replace(" + B2", " - B2")

        label = label.replace("INTERCEPT", f"{intercept_raw:.1f}")
        label = label.replace("B1", f"{abs(b1_raw):.1f}")
        label = label.replace("B2", f"{abs(b2_raw):.1f}")
        return label

    def bake_scatter_label_scaled(self, intercept: float, b1: float, b2: float):
        label = self.scatter_label_scaled
        if b1 < 0:
            label = label.replace(" + B1", " - B1")
        if b2 < 0:
            label = label.replace(" + B2", " - B2")

        label = label.replace("INTERCEPT", f"{intercept:.1f}")
        label = label.replace("B1", f"{abs(b1):.1f}")
        label = label.replace("B2", f"{abs(b2):.1f}")
        return label


@dataclass
class RusAnnotations(Annotations):
    columns_by_name = {"city": "Город",
                       "rooms": "Количество комнат",
                       "area": "Площадь квартиры, м$^2$",
                       "metro_distance": "Расстояние до метро, м",
                       "ac_in_apartment": "Наличие кондиционера",
                       "price": "Стоимость, $"}
    voxels_title: str = "Зависимость RMSE от коэффициентов модели"
    coefficient_0_label: str = "Сдвиг стандартизированный\n($b_0$)"
    colorbar_label: str = "Корень из средней квадратичной ошибки (RMSE)"


@dataclass
class EngAnnotations(Annotations):
    columns_by_name = {"city": "City",
                       "rooms": "Rooms number",
                       "area": "Apartment size, m$^2$",
                       "metro_distance": "Distance to the metro, m",
                       "ac_in_apartment": "Air conditioning available",
                       "price": "Price, $"}
    voxels_title: str = "Dependence of RMSE on coefficients"
    coefficient_0_label: str = "Intercept scaled\n($b_0$)"
    colorbar_label: str = "Root mean square error (RMSE)"


def annotations_by_language(mode: str):
    if mode == "eng":
        annotations = EngAnnotations()
    elif mode == "rus":
        annotations = RusAnnotations()
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return annotations


def generate_df_coefficients_vs_error(features_to_use: list):
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
                metric_value = root_mean_squared_error(np.ravel(y), np.ravel(predicted_raw))
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


def _add_slab(ax, x0, x1, y0, y1, z0, z1, rgba, edgecolor="red", linewidth=2.0, zorder=10_000):
    verts = [
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # x = x0
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # x = x1
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # y = y0
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # y = y1
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # z = z0
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # z = z1
    ]
    poly = Poly3DCollection(verts, facecolors=rgba, edgecolors=edgecolor, linewidths=linewidth)
    poly.set_zorder(zorder)
    if hasattr(poly, "set_zsort"):
        poly.set_zsort("max")
    ax.add_collection3d(poly)


def plot_voxels(ax, b0_edges, b1_edges, b2_edges, nx, ny, nz, ix, iy, iz, norm, metric_grid):
    xc, yc, zc = np.meshgrid(b0_edges, b1_edges, b2_edges, indexing="ij")

    facecolors = plt.get_cmap(CMAP)(norm(metric_grid))
    facecolors[..., 3] = 0.9

    filled_base = np.ones((nx, ny, nz), dtype=bool)
    filled_base[ix, :, :] = False
    filled_base[:, iy, :] = False
    filled_base[:, :, iz] = False

    ax.voxels(xc, yc, zc, filled_base, facecolors=facecolors, edgecolor="k", linewidth=0.06, shade=False)

    x0, x1 = float(b0_edges[ix]), float(b0_edges[ix + 1])
    y0, y1 = float(b1_edges[iy]), float(b1_edges[iy + 1])
    z0, z1 = float(b2_edges[iz]), float(b2_edges[iz + 1])

    X0, X1 = float(b0_edges[0]), float(b0_edges[-1])
    Y0, Y1 = float(b1_edges[0]), float(b1_edges[-1])
    Z0, Z1 = float(b2_edges[0]), float(b2_edges[-1])

    slab_alpha = 0.1
    rgba = (1.0, 0.0, 0.0, slab_alpha)

    # slab for b0 (x-layer)
    _add_slab(ax, x0, x1, Y0, Y1, Z0, Z1, rgba, edgecolor="red", linewidth=1.0)

    # slab for b1 (y-layer)
    _add_slab(ax, X0, X1, y0, y1, Z0, Z1, rgba, edgecolor="red", linewidth=1.0)

    # slab for b2 (z-layer)
    _add_slab(ax, X0, X1, Y0, Y1, z0, z1, rgba, edgecolor="red", linewidth=1.0)

    ax.set_ylabel(r"$b_1$", fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_zlabel(r"$b_2$", fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_xlim(float(b0_edges[0]), float(b0_edges[-1]))
    ax.set_ylim(float(b1_edges[0]), float(b1_edges[-1]))
    ax.set_zlim(float(b2_edges[0]), float(b2_edges[-1]))
    ax.view_init(elev=22, azim=-55)


def plot_projections(ax, b0_edges, b1_edges, b2_edges, slice_xy, slice_yz, slice_xz, norm,
                     explored_path_coeff_b0, explored_path_coeff_b1, explored_path_coeff_b2,
                     coeff_b0_i, coeff_b1_i, coeff_b2_i,
                     features_scaled, target_raw, target_scaler):
    x_wall = float(b0_edges[0])
    y_wall = float(b1_edges[-1])
    z_floor = float(b2_edges[0])

    imshow3d_coeff(ax, array_2d=slice_xy.T, u_edges=b0_edges, v_edges=b1_edges, value_direction="z",
                   pos=z_floor, norm=norm, cmap=CMAP, alpha=0.90, zorder=0)
    imshow3d_coeff(ax, array_2d=slice_yz.T, u_edges=b1_edges, v_edges=b2_edges,
                   value_direction="x", pos=x_wall, norm=norm, cmap=CMAP, alpha=0.90, zorder=0)
    imshow3d_coeff(ax, array_2d=slice_xz.T, u_edges=b0_edges, v_edges=b2_edges,
                   value_direction="y", pos=y_wall, norm=norm, cmap=CMAP, alpha=0.90, zorder=0)

    line = ax.plot(explored_path_coeff_b0, explored_path_coeff_b1, explored_path_coeff_b2, c="red", linewidth=1)[0]
    line.set_zorder(10_000)
    pt = ax.scatter(coeff_b0_i, coeff_b1_i, coeff_b2_i, c="red", s=60, marker="D", depthshade=False)
    pt.set_zorder(10_001)

    g1 = ax.plot([coeff_b0_i, x_wall], [coeff_b1_i, coeff_b1_i], [coeff_b2_i, coeff_b2_i],
                 "--", c="red", linewidth=0.5)[0]
    g2 = ax.plot([coeff_b0_i, coeff_b0_i], [coeff_b1_i, y_wall], [coeff_b2_i, coeff_b2_i],
                 "--", c="red", linewidth=0.5)[0]
    g3 = ax.plot([coeff_b0_i, coeff_b0_i], [coeff_b1_i, coeff_b1_i], [coeff_b2_i, z_floor],
                 "--", c="red", linewidth=0.5)[0]
    g1.set_zorder(10_002)
    g2.set_zorder(10_002)
    g3.set_zorder(10_002)

    predicted_scaled = coeff_b0_i + coeff_b1_i * features_scaled[:, 0] + coeff_b2_i * features_scaled[:, 1]
    predicted_raw = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    rmse_value = root_mean_squared_error(np.ravel(target_raw), np.ravel(predicted_raw))

    dx = float(b0_edges[-1] - b0_edges[0]) * 0.03
    dy = float(b1_edges[-1] - b1_edges[0]) * 0.03
    dz = float(b2_edges[-1] - b2_edges[0]) * 0.03
    text_label = f"{rmse_value:.0f}"
    ax.text(coeff_b0_i + dx, coeff_b1_i + dy, coeff_b2_i + dz, text_label,
            ha="center", va="bottom", fontsize=8, fontname=FONTNAME, color="red")

    ax.set_ylabel(r"$b_1$", fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_zlabel(r"$b_2$", fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_xlim(float(b0_edges[0]), float(b0_edges[-1]))
    ax.set_ylim(float(b1_edges[0]), float(b1_edges[-1]))
    ax.set_zlim(float(b2_edges[0]), float(b2_edges[-1]))
    ax.view_init(elev=25, azim=-55)


def plot_feature_surface(ax, features_raw, target_raw,
                         coeff_b0_i, coeff_b1_i, coeff_b2_i,
                         features_scaler: StandardScaler, target_scaler: StandardScaler,
                         grid_size: int = 30):
    # ax is expected to be 3D: fig.add_axes(..., projection="3d")
    ax.cla()

    x1 = np.asarray(features_raw)[:, 0]
    x2 = np.asarray(features_raw)[:, 1]
    y = np.asarray(target_raw).reshape(-1)

    # scatter of raw data
    ax.scatter(x1, x2, y, s=35, c=y, cmap="cividis", edgecolor='black', vmin=0, vmax=65000)

    # build grid in RAW feature space
    x1_lin = np.linspace(float(x1.min()), float(x1.max()), grid_size)
    x2_lin = np.linspace(float(x2.min()), float(x2.max()), grid_size)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    grid_points_raw = np.c_[X1.ravel(), X2.ravel()]

    # coefficients are in SCALED space -> scale grid first
    grid_points_scaled = features_scaler.transform(grid_points_raw)
    y_pred_scaled = coeff_b0_i + coeff_b1_i * grid_points_scaled[:, 0] + coeff_b2_i * grid_points_scaled[:, 1]

    # convert predictions back to RAW target units
    y_pred_raw = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(X1.shape)

    # model surface in Greys
    ax.plot_surface(X1, X2, y_pred_raw, cmap="cividis", linewidth=0, antialiased=True, alpha=0.75, vmin=0, vmax=65000)

    ax.set_zticks([0, 10000, 20000, 30000, 40000, 50000, 60000, 70000])
    ax.set_zlim(-5000, 75000)


def explore_coefficients_landscape_3d(mode: str = "eng",
                                      features_to_use: Union[list,None] = None):
    """ Create a 3d plot with multiple coefficients """
    if features_to_use is None:
        features_to_use = ["rooms", "metro_distance"]

    annotations = annotations_by_language(mode)
    sections = [
        {"start": [-1.5, -1.5, -1.5], "end": [-1.5, 0.0, -1.5]},
        {"start": [-1.5, 0.0, -1.5], "end": [-0.3, 1.0, -1.5]},
        {"start": [-0.3, 1.0, -1.5], "end": [-0.3, 1.0, 0.3]},
    ]
    cases_to_visualize = compose_cases_to_explore(steps_per_section=30, sections=sections)
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    (dataframe, features, target, features_scaled, target_scaled,
     features_scaler, target_scaler) = generate_df_coefficients_vs_error(features_to_use)

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
        ax1 = fig.add_axes([0.06, 0.51, 0.40, 0.34], projection="3d")
        ax2 = fig.add_axes([0.06, 0.15, 0.40, 0.34], projection="3d")
        ax3 = fig.add_axes([0.395, 0.245, 0.58, 0.51], projection="3d")

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

        norm = Normalize(vmin=float(np.min(metric_values)), vmax=float(np.max(metric_values)))
        plot_voxels(ax1, b0_edges, b1_edges, b2_edges, nx, ny, nz, ix, iy, iz, norm, metric_grid)
        ax1.set_title(annotations.voxels_title, fontdict={'fontsize': 12, 'fontname': FONTNAME}, y=1.03)
        ax1.set_xlabel(annotations.coefficient_0_label, fontdict={"fontsize": 10, "fontname": FONTNAME})

        plot_projections(ax2, b0_edges, b1_edges, b2_edges, slice_xy, slice_yz, slice_xz, norm,
                         explored_path_coeff_b0, explored_path_coeff_b1, explored_path_coeff_b2,
                         coeff_b0_i, coeff_b1_i, coeff_b2_i, features_scaled, target, target_scaler)
        ax2.set_xlabel(annotations.coefficient_0_label, fontdict={"fontsize": 10, "fontname": FONTNAME})

        # One shared colorbar for ax1 + ax2 in the middle
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP))
        mappable.set_array([])
        # To move to the left, - decrease the first coordinate
        cax = fig.add_axes([0.40, 0.315, 0.01, 0.39])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(annotations.colorbar_label, fontdict={"fontsize": 10, "fontname": FONTNAME})

        plot_feature_surface(ax3, features, target,
                             coeff_b0_i, coeff_b1_i, coeff_b2_i,
                             features_scaler, target_scaler,
                             grid_size = 30)
        l = f"{annotations.columns_by_name[features_to_use[0]]} ($x_1$)"
        ax3.set_xlabel(l, fontdict={"fontsize": 10, "fontname": FONTNAME})
        l = f"{annotations.columns_by_name[features_to_use[1]]} ($x_2$)"
        ax3.set_ylabel(l, fontdict={"fontsize": 10, "fontname": FONTNAME})
        l = f"{annotations.columns_by_name['price']} (y)"
        ax3.set_zlabel(l, fontdict={"fontsize": 10, "fontname": FONTNAME})
        if "area" in features_to_use and "rooms" in features_to_use:
            ax3.view_init(elev=25, azim=10)
        else:
            ax3.view_init(elev=25, azim=50)

        raw_label = annotations.bake_scatter_label_raw(coeff_b0_i, coeff_b1_i, coeff_b2_i,
                                                       features_scaler, target_scaler)
        scaled_label = annotations.bake_scatter_label_scaled(coeff_b0_i, coeff_b1_i, coeff_b2_i)
        ax3.set_title(f"{scaled_label}\n\n{raw_label}", fontdict={'fontsize': 12, 'fontname': FONTNAME})

        raw_svg_file = Path(tmp_dir, f"46_explore_coefficients_3d_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"46_explore_coefficients_3d_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg", dpi=100)
        image_files.append(path_to_final_path)

    features_label = '_'.join(features_to_use)
    gif_path = Path(get_plots_path(), f"46_explore_coefficients_3d_{features_label}_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    features_to_use = ["rooms", "metro_distance"]
    explore_coefficients_landscape_3d("rus", features_to_use)
    features_to_use = ["rooms", "area"]
    explore_coefficients_landscape_3d("rus", features_to_use)
