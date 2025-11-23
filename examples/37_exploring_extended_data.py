from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_TARGET = 0
MAX_TARGET = 70000
TARGET_TICKS = [0, 25000, 50000]
SHIFT = 0.1
COLUMN_BORDERS_BY_NAME = {"rooms": {"ticks": [1, 2, 3, 4, 5], "min": 0.5, "max": 5.5},
                          "area": {"ticks": [50, 90, 130], "min": 45, "max": 145},
                          "metro_distance": {"ticks": [600, 1000, 1400, 1800], "min": 550, "max": 1900},
                          "city": {"ticks": None, "min": None, "max": None},
                          "ac_in_apartment": {"ticks": None, "min": None, "max": None}}

def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        x_label_by_column = {}
        y_label = ""
    elif mode == "rus":
        title = "Визуализация многомерных данных"
        x_label_by_column = {"rooms": "количество комнат",
                             "area": "площадь квартиры",
                             "metro_distance": "расстояние до метро",
                             "city": "город",
                             "ac_in_apartment": "есть ли кондиционер"}
        y_label = "стоимость"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label_by_column, y_label


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        -0.1,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 12, "fontname": FONTNAME},
    )


def encode_feature_for_axis(feature: np.array):
    unique_values = np.unique(feature)
    number_unique_values = len(unique_values)

    # Categorical or low-cardinality numeric feature
    positions = np.empty_like(feature, dtype=float)
    for idx, val in enumerate(unique_values):
        mask = feature == val
        positions[mask] = idx + np.random.uniform(-0.05, 0.05, size=mask.sum())
    ticks = np.arange(number_unique_values)
    ticklabels = [str(v) for v in unique_values]

    return positions, ticks, ticklabels


def scatter_plot_3d(ax, first_feature: np.array, second_feature: np.array,
                    y: np.array, first_label: str, second_label: str, y_label: str,
                    first_feature_info: Dict, second_feature_info: Dict):
    # Aligning the axis
    if len(np.unique(first_feature)) <= 3:
        x_positions, x_ticks, x_ticklabels = encode_feature_for_axis(first_feature)
        min_x, max_x = -0.5, len(x_ticks) - 0.5
    else:
        x_positions = first_feature
        x_ticks = first_feature_info["ticks"]
        x_ticklabels = first_feature_info["ticks"]
        min_x, max_x = first_feature_info["min"], first_feature_info["max"]

    if len(np.unique(second_feature)) <= 3:
        y_positions, y_ticks, y_ticklabels = encode_feature_for_axis(second_feature)
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
        cmap='coolwarm',
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
    ax3d.view_init(20, -50)

    # Make tick labels smaller
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        for tick_label in axis.get_ticklabels():
            tick_label.set_fontsize(5)
            tick_label.set_fontname(FONTNAME)


def scatter_plot_2d(ax, feature: np.array, y: np.array, x_label: str, y_label: str, feature_info: Dict):
    number_unique_values = len(np.unique(feature))
    if number_unique_values <= 3:
        # It is better to use stripplot
        x_center = 0
        for value in np.unique(feature):
            mask = feature == value
            y_filtered = y[mask]
            ax.scatter(np.random.uniform(x_center - 0.1, x_center + 0.1, len(y_filtered)), y_filtered,
                       s=30, c=y_filtered, cmap='coolwarm', vmin=MIN_TARGET, vmax=MAX_TARGET, edgecolors='black', linewidths=1.2)
            x_center += 1
        ax.set_xlim(-1, number_unique_values)
        ax.set_xticks(list(range(number_unique_values)))
        ax.xaxis.set_ticklabels(np.unique(feature))
    else:
        # Regular scatter plot
        ax.scatter(feature, y, c=y, cmap='coolwarm', vmin=MIN_TARGET, vmax=MAX_TARGET, s=30,
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


def contour_plot(ax, first_feature: np.array, second_feature: np.array, y: np.array,
                 first_label: str, second_label: str, first_feature_info: Dict, second_feature_info: Dict):

    if len(np.unique(first_feature)) <= 3 or len(np.unique(second_feature)) <= 3:
        # Disable the categorical plots for now FIXME later
        ax.set_axis_off()
        return None

    # Scaling
    scaler_f = StandardScaler()
    scaler_s = StandardScaler()
    first_feature = scaler_f.fit_transform(first_feature.reshape(-1, 1))
    second_feature = scaler_s.fit_transform(second_feature.reshape(-1, 1))
    int_model = LinearRegression()
    int_model.fit(np.hstack([first_feature, second_feature]), y.reshape(-1, 1))

    # Extent features with border values
    min_f = np.min(first_feature)
    min_s = np.min(second_feature)
    max_f = np.max(first_feature)
    max_s = np.max(second_feature)
    first_ad_on = np.array([min_f, min_f, max_f, max_f]).reshape(-1, 1)
    second_ad_on = np.array([min_s, max_s, max_s, min_s]).reshape(-1, 1)

    first_feature_ext = np.hstack([first_feature.ravel(), first_ad_on.ravel()])
    second_feature_ext = np.hstack([second_feature.ravel(), second_ad_on.ravel()])
    corner_y_values = int_model.predict(np.hstack([first_ad_on, second_ad_on]))
    y = np.hstack([y.ravel(), np.ravel(corner_y_values)])

    # Build a regular grid over the feature space
    grid_size = 50
    first_lin = np.linspace(first_feature_ext.min(), first_feature_ext.max(), grid_size)
    second_lin = np.linspace(second_feature_ext.min(), second_feature_ext.max(), grid_size)
    first_feature_grid, second_feature_grid = np.meshgrid(first_lin, second_lin)
    norm = mcolors.Normalize(vmin=MIN_TARGET, vmax=MAX_TARGET)
    z_grid = griddata(
        points=(np.ravel(first_feature_ext), np.ravel(second_feature_ext)),
        values=np.ravel(y),
        xi=(first_feature_grid, second_feature_grid),
        method="cubic",
    )
    levels = 7
    contourf = ax.contourf(first_feature_grid, second_feature_grid, z_grid, norm=norm,
                           levels=np.linspace(MIN_TARGET, MAX_TARGET, levels), cmap="coolwarm", vmin=MIN_TARGET, vmax=MAX_TARGET,
                           extend="both")
    ax.scatter(first_feature, second_feature, marker="x", s=4, c='black')
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

    contour_lines = ax.contour(first_feature_grid, second_feature_grid, z_grid,
                               levels=np.linspace(MIN_TARGET, MAX_TARGET, levels), colors="black", linewidths=1)

    # ----- Set ticks in scaled space but label them in original coordinates -----
    xticks_orig = np.array(first_feature_info["ticks"])
    xticks_scaled = scaler_f.transform(xticks_orig.reshape(-1, 1)).ravel()
    ax.set_xticks(xticks_scaled)
    ax.set_xticklabels(first_feature_info["ticks"], fontsize=6, fontname=FONTNAME)

    yticks_orig = np.array(second_feature_info["ticks"])
    yticks_scaled = scaler_s.transform(yticks_orig.reshape(-1, 1)).ravel()
    ax.set_yticks(yticks_scaled)
    ax.set_yticklabels(second_feature_info["ticks"], fontsize=6, fontname=FONTNAME)

    # Labels and grid
    ax.set_ylabel(second_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_xlabel(first_label, fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.grid(alpha=0.3)

    min_f = scaler_f.transform(np.array([first_feature_info["min"]]).reshape(-1, 1)).ravel()[0]
    min_s = scaler_s.transform(np.array([second_feature_info["min"]]).reshape(-1, 1)).ravel()[0]
    max_f = scaler_f.transform(np.array([first_feature_info["max"]]).reshape(-1, 1)).ravel()[0]
    max_s = scaler_s.transform(np.array([second_feature_info["max"]]).reshape(-1, 1)).ravel()[0]
    ax.set_xlim(min_f, max_f)
    ax.set_ylim(min_s, max_s)
    return contourf


def plot_new_extended_dataset(mode: str = "eng"):
    title, x_label_by_column, y_label = annotations_by_language(mode)

    dataset = get_extended_dataset()
    features_names = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
    features = np.array(dataset[features_names])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    fig_size = (14, 14)
    # fig, axs = plt.subplots(5, 5, figsize=fig_size)
    # fig.subplots_adjust(left=0.05, right=0.97, hspace=0.5, wspace=0.5)
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97)
    gs = GridSpec(5, 5, figure=fig)
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
                                first_feature_info=first_info, second_feature_info=second_info)
            elif column_id == row_id:
                # 2d scatter plot
                scatter_plot_2d(ax, x[:, column_id], y, x_label_row, y_label=y_label,
                                feature_info=COLUMN_BORDERS_BY_NAME[column_feature])
            else:
                # Counter plot
                contour_plot(ax, x[:, row_id], x[:, column_id], y,
                             first_label=x_label_row, second_label=x_label_column,
                             first_feature_info=first_info, second_feature_info=second_info)

    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME}, va="top")

    raw_svg_file = Path(get_plots_path(), f"37_eda_extended_dataset_{mode}.svg", bbox_inches='tight')
    final_plot = Path(get_plots_path(), f"37_eda_extended_dataset_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template.svg", dpi=300)


if __name__ == '__main__':
    plot_new_extended_dataset("rus")
