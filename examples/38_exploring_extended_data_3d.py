import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import imageio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_TARGET = 0
MAX_TARGET = 70000
TARGET_TICKS = [0, 25000, 50000]
CMAP = "coolwarm"
ANIM_DURATION = 100
DPI = 80
COLUMN_BORDERS_BY_NAME = {"rooms": {"ticks": [1, 2, 3, 4, 5], "min": 0.5, "max": 5.5},
                          "area": {"ticks": [50, 90, 130], "min": 45, "max": 145},
                          "metro_distance": {"ticks": [600, 1000, 1400, 1800], "min": 500, "max": 2000},
                          "city": {"ticks": None, "min": None, "max": None},
                          "ac_in_apartment": {"ticks": None, "min": None, "max": None}}

def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        feature_label_by_column = {}
        y_label = ""
    elif mode == "rus":
        title = "Визуализация многомерных данных"
        feature_label_by_column = {"rooms": "количество комнат",
                                   "area": "площадь квартиры",
                                   "metro_distance": "расстояние до метро",
                                   "city": "город",
                                   "ac_in_apartment": "есть ли кондиционер"}
        y_label = "стоимость"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, feature_label_by_column, y_label


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        -0.07,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 12, "fontname": FONTNAME},
    )


def encode_feature_for_axis(feature: np.array, mode: str):
    unique_values = np.unique(feature)
    number_unique_values = len(unique_values)

    if number_unique_values == 2 and mode == "rus":
        # Prettify visualization
        unique_values = unique_values[::-1]

    # Categorical or low-cardinality numeric feature
    positions = np.empty_like(feature, dtype=float)
    for idx, val in enumerate(unique_values):
        mask = feature == val
        # + np.random.uniform(-0.05, 0.05, size=mask.sum())
        positions[mask] = idx
    ticks = np.arange(number_unique_values)
    ticklabels = [str(v) for v in unique_values]

    return positions, ticks, ticklabels

def scatter_plot_3d(
    ax,
    first_feature: np.array,
    second_feature: np.array,
    y: np.array,
    first_label: str,
    second_label: str,
    y_label: str,
    first_feature_info: Dict,
    second_feature_info: Dict,
    mode: str,
    hide_colorbar: bool = False,
):
    number_levels = 7
    norm = mcolors.Normalize(vmin=MIN_TARGET, vmax=MAX_TARGET)
    levels = np.linspace(MIN_TARGET, MAX_TARGET, number_levels)

    # Aligning the axis
    x_positions = first_feature.astype(float)
    x_ticks = first_feature_info["ticks"]
    x_ticklabels = first_feature_info["ticks"]
    min_x, max_x = np.min(first_feature), np.max(first_feature)

    cases = [{"min_first": min_x, "max_first": max_x, "first_feature": first_feature}]
    if len(np.unique(second_feature)) <= 3:
        y_positions, y_ticks, y_ticklabels = encode_feature_for_axis(second_feature, mode)
        min_y, max_y = -0.5, len(y_ticks) - 0.5
        method = "categorical"

        values = np.unique(y_positions)
        values.sort()
        updated_cases = []
        for value in values:
            for case in cases:
                case = deepcopy(case)
                case.update({"min_second": value - 0.1, "max_second": value + 0.1, "second_feature": y_positions})
                updated_cases.append(case)
        cases = updated_cases
    else:
        y_positions = second_feature.astype(float)
        y_ticks = second_feature_info["ticks"]
        y_ticklabels = second_feature_info["ticks"]
        min_y, max_y = np.min(second_feature), np.max(second_feature)
        method = "linear"

    fig = ax.figure
    rect = ax.get_position()
    ax.set_axis_off()
    ax3d = fig.add_axes(rect, projection="3d")

    grid_size = 50
    x_lin = np.linspace(min_x, max_x, grid_size)
    y_lin = np.linspace(min_y, max_y, grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

    # Interpolate target onto grid
    if method == "categorical":
        for case in cases:
            # Clip the initial dataframes according to borders
            mask = ((case["first_feature"] >= case["min_first"]) & (case["first_feature"] <= case["max_first"])
                    & (case["second_feature"] >= case["min_second"]) & (case["second_feature"] <= case["max_second"]))
            y_filtered = y[mask]
            first_filtered = case["first_feature"][mask]
            second_filtered = case["second_feature"][mask]

            first_lin = np.linspace(case["min_first"], case["max_first"], grid_size)
            second_lin = np.linspace(case["min_second"], case["max_second"], grid_size)
            first_feature_grid, second_feature_grid = np.meshgrid(first_lin, second_lin)

            scaler_f = StandardScaler()
            scaler_s = StandardScaler()
            first_filtered = scaler_f.fit_transform(first_filtered.reshape(-1, 1))
            second_filtered = scaler_s.fit_transform(second_filtered.reshape(-1, 1))

            # Fit interpolator
            int_model = LinearRegression()
            int_model.fit(np.hstack([first_filtered, second_filtered]), y_filtered.reshape(-1, 1))

            f = scaler_f.transform(first_feature_grid.reshape(-1, 1))
            s = scaler_s.transform(second_feature_grid.reshape(-1, 1))
            z_grid = int_model.predict(np.hstack([f, s]))
            z_grid = z_grid.reshape(first_feature_grid.shape)

            ax3d.contourf(
                first_feature_grid,
                second_feature_grid,
                z_grid,
                zdir="z",
                offset=MIN_TARGET,
                levels=levels,
                cmap=CMAP,
                norm=norm,
                alpha=0.8,
                zorder=1
            )
    else:
        z_grid = griddata(
            points=(np.ravel(x_positions), np.ravel(y_positions)),
            values=np.ravel(y),
            xi=(X_grid, Y_grid),
            method=method,
        )

        # Projection on "floor" (XY plane, z = MIN_TARGET)
        ax3d.contourf(
            X_grid,
            Y_grid,
            z_grid,
            zdir="z",
            offset=MIN_TARGET,
            levels=levels,
            cmap=CMAP,
            norm=norm,
            alpha=0.8,
            zorder=1
        )

    scatter = ax3d.scatter(
        x_positions,
        y_positions,
        y,
        c=y,
        cmap=CMAP,
        vmin=MIN_TARGET,
        vmax=MAX_TARGET,
        s=80,
        edgecolors="black",
        linewidths=1,
        alpha=0.9,
        zorder=3
    )

    cb = fig.colorbar(scatter, ax=ax3d, shrink=0.3, aspect=10, pad=0.25)

    if hide_colorbar:
        cb.ax.set_visible(False)
    else:
        cb.set_label(y_label, fontdict={"fontsize": 10, "fontname": FONTNAME})
        for tick_label in cb.ax.get_yticklabels():
            tick_label.set_fontsize(10)
            tick_label.set_fontname(FONTNAME)

    ax3d.set_xlim(min_x, max_x)
    ax3d.set_ylim(min_y, max_y)
    ax3d.set_zlim(MIN_TARGET, MAX_TARGET)

    ax3d.set_xticks(x_ticks)
    ax3d.set_xticklabels(x_ticklabels)
    ax3d.set_yticks(y_ticks)
    ax3d.set_yticklabels(y_ticklabels)
    ax3d.set_zticks(TARGET_TICKS)
    ax3d.set_zticklabels(TARGET_TICKS)

    # Labels
    ax3d.set_xlabel(first_label, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax3d.set_ylabel(second_label, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax3d.set_zlabel(y_label, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax3d.grid(alpha=0.3)

    # Make tick labels smaller
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        for tick_label in axis.get_ticklabels():
            tick_label.set_fontsize(10)
            tick_label.set_fontname(FONTNAME)

    return ax3d


def plot_new_extended_dataset_as_3d(mode: str = "eng"):
    title, feature_label_by_column, y_label = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists() and len(list(tmp_dir.iterdir())) > 0:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_extended_dataset()
    if mode == "rus":
        dataset["ac_in_apartment"] = dataset["ac_in_apartment"].replace({"no": "нет", "yes": "да"})

    features_names = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
    features = np.array(dataset[features_names])
    target = np.array(dataset["price"])
    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    frames = []
    vertical_view_id = 25
    for view_id in range(-180, 180, 2):
        fig_size = (12, 6)
        fig = plt.figure(figsize=fig_size)
        gs = GridSpec(1, 2, figure=fig)
        gs.update(wspace=0.01)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])

        # 3d scatter plot
        rooms_id = 0
        area_id = 1
        metro_distance_id = 2
        ac_id = 4
        ax_left = scatter_plot_3d(ax_left, x[:, rooms_id], x[:, metro_distance_id], y,
                        first_label=feature_label_by_column["rooms"],
                        second_label=feature_label_by_column["metro_distance"], y_label=y_label,
                        first_feature_info=COLUMN_BORDERS_BY_NAME["rooms"],
                        second_feature_info=COLUMN_BORDERS_BY_NAME["metro_distance"],
                        hide_colorbar=True, mode=mode)
        ax_right = scatter_plot_3d(ax_right, x[:, area_id], x[:, ac_id], y,
                        first_label=feature_label_by_column["area"],
                        second_label=feature_label_by_column["ac_in_apartment"],
                        y_label=y_label,
                        first_feature_info=COLUMN_BORDERS_BY_NAME["area"],
                        second_feature_info=COLUMN_BORDERS_BY_NAME["ac_in_apartment"],
                        hide_colorbar=False, mode=mode)
        ax_left.view_init(vertical_view_id, view_id)
        ax_right.view_init(vertical_view_id, view_id)

        fig.suptitle(title, fontsize=18, fontdict={'fontname': FONTNAME}, va="top", y=0.95)

        raw_svg_file = Path(tmp_dir, f"38_3d_{mode}.svg")
        final_plot = Path(tmp_dir, f"38_3d_{mode}_{vertical_view_id}_{view_id}.png")
        plt.savefig(raw_svg_file)
        plt.close()

        save_plot_according_to_template(raw_svg_file, final_plot,
                                        template_name="template_small.svg",
                                        dpi=DPI)
        frames.append(final_plot)


    gif_path = Path(get_plots_path(), f"38_3d_rotation_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for img in frames:
            writer.append_data(imageio.imread(img))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_new_extended_dataset_as_3d("rus")
