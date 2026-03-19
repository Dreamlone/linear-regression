from copy import deepcopy
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_TARGET = 0
MAX_TARGET = 70000
TARGET_TICKS = [0, 25000, 50000]
CMAP = "coolwarm"
COLUMN_BORDERS_BY_NAME = {"rooms": {"ticks": [1, 2, 3, 4, 5], "min": 0.5, "max": 5.5},
                          "area": {"ticks": [50, 90, 130], "min": 45, "max": 145},
                          "metro_distance": {"ticks": [600, 1000, 1400, 1800], "min": 450, "max": 2000},
                          "city": {"ticks": None, "min": None, "max": None},
                          "ac_in_apartment": {"ticks": None, "min": None, "max": None}}

def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Visualizing multivariate data"
        x_label_by_column = {
            "rooms": "number of rooms",
            "area": "apartment area",
            "metro_distance": "distance to the metro",
            "city": "city",
            "ac_in_apartment": "air conditioning"
        }
        y_label = "price"
        int_method = "Interpolation method"
    elif mode == "rus":
        title = "Визуализация многомерных данных"
        x_label_by_column = {"rooms": "количество комнат",
                             "area": "площадь квартиры",
                             "metro_distance": "расстояние до метро",
                             "city": "город",
                             "ac_in_apartment": "есть ли кондиционер"}
        y_label = "стоимость"
        int_method = "Метод интерполяции"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label_by_column, y_label, int_method


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


def scatter_plot_3d(ax, first_feature: np.array, second_feature: np.array,
                    y: np.array, first_label: str, second_label: str, y_label: str,
                    first_feature_info: Dict, second_feature_info: Dict, mode: str):
    # Aligning the axis
    if len(np.unique(first_feature)) <= 3:
        x_positions, x_ticks, x_ticklabels = encode_feature_for_axis(first_feature, mode)
        min_x, max_x = -0.5, len(x_ticks) - 0.5
    else:
        x_positions = first_feature
        x_ticks = first_feature_info["ticks"]
        x_ticklabels = first_feature_info["ticks"]
        min_x, max_x = first_feature_info["min"], first_feature_info["max"]

    if len(np.unique(second_feature)) <= 3:
        y_positions, y_ticks, y_ticklabels = encode_feature_for_axis(second_feature, mode)
        min_y, max_y = -0.5, len(y_ticks) - 0.5
    else:
        y_positions = second_feature
        y_ticks = second_feature_info["ticks"]
        y_ticklabels = second_feature_info["ticks"]
        min_y, max_y = second_feature_info["min"], second_feature_info["max"]

    fig = ax.figure
    rect = ax.get_position()
    ax.set_axis_off()
    ax3d = fig.add_axes(rect, projection='3d')

    # Scatter in original feature space
    ax3d.scatter(
        x_positions,
        y_positions,
        y,
        c=y,
        cmap=CMAP,
        vmin=MIN_TARGET,
        vmax=MAX_TARGET,
        s=25,
        edgecolors='black',
        linewidths=0.3,
        alpha=0.9,
    )
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
    ax3d.set_xlabel(first_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax3d.set_ylabel(second_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax3d.set_zlabel(y_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})

    ax3d.grid(alpha=0.3)
    ax3d.view_init(20, -60)

    # Make tick labels smaller
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        for tick_label in axis.get_ticklabels():
            tick_label.set_fontsize(5)
            tick_label.set_fontname(FONTNAME)


def scatter_plot_2d(ax, feature: np.array, y: np.array, x_label: str, y_label: str, feature_info: Dict,
                    mode: str):
    number_unique_values = len(np.unique(feature))
    unique_values = np.unique(feature)
    if number_unique_values == 2 and mode == "rus":
        unique_values = unique_values[::-1]

    if number_unique_values <= 3:
        # It is better to use stripplot
        x_center = 0
        for value in unique_values:
            mask = feature == value
            y_filtered = y[mask]
            ax.scatter(np.random.uniform(x_center - 0.1, x_center + 0.1, len(y_filtered)), y_filtered,
                       s=30, c=y_filtered, cmap=CMAP, vmin=MIN_TARGET, vmax=MAX_TARGET, edgecolors='black', linewidths=1.2)
            x_center += 1
        ax.set_xlim(-1, number_unique_values)
        ax.set_xticks(list(range(number_unique_values)))
        ax.xaxis.set_ticklabels(unique_values)
    else:
        # Regular scatter plot
        ax.scatter(feature, y, c=y, cmap=CMAP, vmin=MIN_TARGET, vmax=MAX_TARGET, s=30,
                   edgecolors='black', linewidths=1.2, zorder=2)
        ax.set_xlim(feature_info["min"], feature_info["max"])
        ax.set_xticks(feature_info["ticks"])
        ax.set_xticklabels(feature_info["ticks"])
    ax.set_ylabel(y_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_xlabel(x_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_ylim(MIN_TARGET, MAX_TARGET)
    ax.set_yticks(TARGET_TICKS)
    ax.set_yticklabels(TARGET_TICKS)
    ax.grid(alpha=0.3)

    for axis in [ax.xaxis, ax.yaxis]:
        for tick_label in axis.get_ticklabels():
            tick_label.set_fontsize(6)
            tick_label.set_fontname(FONTNAME)


def _interpolate_within_borders(ax,
                                min_first: float, max_first: float,
                                min_second: float, max_second: float,
                                first_array: np.array, second_array: np.array,
                                y_array: np.array,
                                method: str = "linear"):
    """ Does the interpolation and plot it as contourf """
    grid_size = 100
    levels = 5
    norm = mcolors.Normalize(vmin=MIN_TARGET, vmax=MAX_TARGET)
    model_by_method = {"linear": LinearRegression(),
                       "nearest": KNeighborsRegressor(n_neighbors=1),
                       "gaussian mixture": GaussianProcessRegressor(alpha=0.005),
                       "random forest": RandomForestRegressor()}

    # Clip the initial dataframes according to borders
    mask = ((first_array >= min_first) & (first_array <= max_first)
            & (second_array >= min_second) & (second_array <= max_second))
    y_filtered = y_array[mask]
    first_filtered = first_array[mask]
    second_filtered = second_array[mask]
    if any([len(i) < 1 for i in [y_filtered, first_filtered, second_filtered]]):
        # Nothing to interpolate
        return ax

    # Apply interpolator
    first_lin = np.linspace(min_first, max_first, grid_size)
    second_lin = np.linspace(min_second, max_second, grid_size)
    first_feature_grid, second_feature_grid = np.meshgrid(first_lin, second_lin)

    if len(np.unique(first_filtered)) == 1 and len(np.unique(second_filtered)) == 1:
        # No need to make model - we can take mean value
        z_grid = np.repeat(np.mean(y_filtered), grid_size * grid_size)
        z_grid = z_grid.reshape(first_feature_grid.shape)
    else:
        # Scaling for features
        scaler_f = StandardScaler()
        scaler_s = StandardScaler()
        first_filtered = scaler_f.fit_transform(first_filtered.reshape(-1, 1))
        second_filtered = scaler_s.fit_transform(second_filtered.reshape(-1, 1))

        # Fit interpolator
        int_model = model_by_method[method]
        int_model.fit(np.hstack([first_filtered, second_filtered]), y_filtered.reshape(-1, 1))

        f = scaler_f.transform(first_feature_grid.reshape(-1, 1))
        s = scaler_s.transform(second_feature_grid.reshape(-1, 1))
        z_grid = int_model.predict(np.hstack([f, s]))
        z_grid = z_grid.reshape(first_feature_grid.shape)

    contourf = ax.contourf(first_feature_grid, second_feature_grid, z_grid, norm=norm,
                           levels=np.linspace(MIN_TARGET, MAX_TARGET, levels), cmap=CMAP,
                           vmin=MIN_TARGET, vmax=MAX_TARGET, extend="both")
    ax.contour(first_feature_grid, second_feature_grid, z_grid,
               levels=np.linspace(MIN_TARGET, MAX_TARGET, levels),
               colors="white", linewidths=1.5)

    # Draw borders of the interpolation zones
    ax.plot([min_first, min_first], [min_second, max_second], c='black', linewidth=0.5)
    ax.plot([min_first, max_first], [min_second, min_second], c='black', linewidth=0.5)
    ax.plot([max_first, min_first], [max_second, max_second], c='black', linewidth=0.5)
    ax.plot([max_first, max_first], [max_second, min_second], c='black', linewidth=0.5)
    return contourf


def contour_plot(ax, first_feature: np.array, second_feature: np.array, y: np.array,
                 first_label: str, second_label: str, first_feature_info: Dict, second_feature_info: Dict,
                 int_method: str, mode: str):
    cases = []
    if len(np.unique(first_feature)) <= 3:
        method = "linear"
        first_feature, x_ticks, x_ticklabels = encode_feature_for_axis(first_feature, mode)
        min_first, max_first = -0.5, len(x_ticks) - 0.5

        values = np.unique(first_feature)
        values.sort()
        for value in values:
            cases.append({"min_first": value - 0.5, "max_first": value + 0.5, "first_feature": first_feature})
    else:
        method = "nearest"
        x_ticks = first_feature_info["ticks"]
        x_ticklabels = first_feature_info["ticks"]
        min_first, max_first = first_feature_info["min"], first_feature_info["max"]
        cases.append({"min_first": min_first, "max_first": max_first, "first_feature": first_feature})

    if len(np.unique(second_feature)) <= 3:
        second_feature, y_ticks, y_ticklabels = encode_feature_for_axis(second_feature, mode)
        min_second, max_second = -0.5, len(y_ticks) - 0.5

        values = np.unique(second_feature)
        values.sort()
        updated_cases = []
        for value in values:
            for case in cases:
                case = deepcopy(case)
                case.update({"min_second": value - 0.5, "max_second": value + 0.5, "second_feature": second_feature})
                updated_cases.append(case)
        cases = updated_cases
    else:
        y_ticks = second_feature_info["ticks"]
        y_ticklabels = second_feature_info["ticks"]
        min_second, max_second = second_feature_info["min"], second_feature_info["max"]
        for case in cases:
            case.update({"min_second": min_second, "max_second": max_second, "second_feature": second_feature})

    for case in cases:
        # Each case must contain all necessary information
        contourf = _interpolate_within_borders(ax, case["min_first"], case["max_first"],
                                               case["min_second"], case["max_second"],
                                               case["first_feature"], case["second_feature"],
                                               y, method)

    ax.set_title(f"{int_method}: '{method}'", fontdict={'fontsize': 5, 'fontname': FONTNAME})
    ax.scatter(first_feature, second_feature, marker="x", s=4, c='black', zorder=2)
    cbar = plt.colorbar(
        contourf,
        ax=ax,
        fraction=0.025,
        pad=0.02,
        aspect=30
    )
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontsize(5)
        tick_label.set_fontname(FONTNAME)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=6, fontname=FONTNAME)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=6, fontname=FONTNAME)

    # Labels and grid
    ax.set_ylabel(second_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_xlabel(first_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.grid(alpha=0.3)

    ax.set_xlim(min_first, max_first)
    ax.set_ylim(min_second, max_second)
    return contourf


def plot_new_extended_dataset(mode: str = "eng"):
    title, x_label_by_column, y_label, int_method = annotations_by_language(mode)

    dataset = get_extended_dataset()
    if mode == "rus":
        dataset["ac_in_apartment"] = dataset["ac_in_apartment"].replace({"no": "нет", "yes": "да"})
    features_names = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
    features = np.array(dataset[features_names])
    target = np.array(dataset["price"])
    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    fig_size = (14, 14)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(5, 5, figure=fig, left=0.07, right=1.0)
    gs.update(hspace=0.5, wspace=0.5)

    for row_id, row_feature in zip([0, 1, 2, 3, 4], features_names):
        x_label_row = x_label_by_column[row_feature]
        first_info = COLUMN_BORDERS_BY_NAME[row_feature]

        add_row_label(fig, gs, row_index=row_id, text=x_label_row)

        for column_id, column_feature in zip([0, 1, 2, 3, 4], features_names):
            ax = fig.add_subplot(gs[row_id, column_id])

            x_label_column = x_label_by_column[column_feature]
            second_info = COLUMN_BORDERS_BY_NAME[column_feature]

            if row_id == 0:
                ax.set_title(x_label_column, y =1.2,
                             fontdict={'fontsize': 12, 'fontname': FONTNAME})
            if column_id > row_id:
                # 3d scatter plot
                scatter_plot_3d(ax, x[:, row_id], x[:, column_id], y,
                                first_label=x_label_row, second_label=x_label_column, y_label=y_label,
                                first_feature_info=first_info, second_feature_info=second_info, mode=mode)
            elif column_id == row_id:
                # 2d scatter plot
                scatter_plot_2d(ax, x[:, column_id], y, x_label_row, y_label=y_label,
                                feature_info=COLUMN_BORDERS_BY_NAME[column_feature], mode=mode)
            else:
                # Counter plot
                contour_plot(ax, x[:, row_id], x[:, column_id], y,
                             first_label=x_label_row, second_label=x_label_column,
                             first_feature_info=first_info, second_feature_info=second_info,
                             int_method=int_method, mode=mode)

    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME}, va="top")

    raw_svg_file = Path(get_plots_path(), f"39_eda_extended_dataset_{mode}.svg")
    final_plot = Path(get_plots_path(), f"39_eda_extended_dataset_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template.svg", dpi=300)


if __name__ == '__main__':
    plot_new_extended_dataset("rus")
    plot_new_extended_dataset("eng")
