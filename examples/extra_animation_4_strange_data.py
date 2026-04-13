import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import imageio.v2 as imageio

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_TARGET = 0
MAX_TARGET = 20000
TEXT_SHIFT = 800
TARGET_TICKS = [0, 5000, 10000, 15000]
CMAP = "coolwarm"
ANIM_DURATION = 130
DPI = 100


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Sometimes one more dimension\nis needed to see the full picture"
        rooms_label = "number of rooms"
        metro_distance_label = "distance to the metro"
        target_label = "price"
        labels = ["apartment 1", "apartment 2", "apartment 3"]
    elif mode == "rus":
        title = "Иногда для полной картины\nне хватает дополнительного измерения"
        rooms_label = "количество комнат"
        metro_distance_label = "расстояние до метро"
        target_label = "стоимость"
        labels = ["квартира 1", "квартира 2", "квартира 3"]
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return rooms_label, metro_distance_label, target_label, title, labels


def scatter_plot_3d(
    ax,
    first_feature: np.array,
    second_feature: np.array,
    target: np.array,
    first_label: str,
    second_label: str,
    target_label: str,
    labels: list
):
    number_levels = 7
    norm = mcolors.Normalize(vmin=MIN_TARGET, vmax=MAX_TARGET)
    levels = np.linspace(MIN_TARGET, MAX_TARGET, number_levels)

    # Aligning the axis
    x_ticks = [1, 2, 3]
    x_ticklabels = x_ticks
    min_x, max_x = 0.5, 3.5

    y_ticks = [0, 500, 1000, 1500]
    y_ticklabels = y_ticks
    min_y, max_y = -200, 1700
    method = "linear"

    fig = ax.figure
    rect = ax.get_position()
    ax.set_axis_off()
    ax3d = fig.add_axes(rect, projection="3d")

    grid_size = 50
    x_lin = np.linspace(min_x, max_x, grid_size)
    y_lin = np.linspace(min_y, max_y, grid_size)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    # Interpolate target onto grid
    z_grid = griddata(
        points=(np.ravel(first_feature), np.ravel(second_feature)),
        values=np.ravel(target),
        xi=(x_grid, y_grid),
        method=method,
    )

    # Projection on "floor" (XY plane, z = MIN_TARGET)
    ax3d.contourf(
        x_grid,
        y_grid,
        z_grid,
        zdir="z",
        offset=MIN_TARGET,
        levels=levels,
        cmap=CMAP,
        norm=norm,
        alpha=1.0,
        zorder=1
    )

    scatter = ax3d.scatter(
        first_feature,
        second_feature,
        target,
        c=target,
        cmap=CMAP,
        vmin=MIN_TARGET,
        vmax=MAX_TARGET,
        s=80,
        edgecolors="black",
        linewidths=1,
        alpha=0.9,
        zorder=3
    )
    for i in range(len(labels)):
        ax3d.text(first_feature[i], second_feature[i], target[i] + TEXT_SHIFT, labels[i], color='black',
                  fontsize=6, va='bottom', ha='left', fontname=FONTNAME)

    cb = fig.colorbar(scatter, ax=ax3d, shrink=0.3, aspect=10, pad=0.25)

    cb.set_label(target_label, fontdict={"fontsize": 10, "fontname": FONTNAME})
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
    ax3d.set_xlabel(first_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax3d.set_ylabel(second_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
    ax3d.set_zlabel(target_label, fontdict={"fontsize": 9, "fontname": FONTNAME})

    ax3d.grid(alpha=0.3)

    # Make tick labels smaller
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        for tick_label in axis.get_ticklabels():
            tick_label.set_fontsize(8)
            tick_label.set_fontname(FONTNAME)

    return ax3d


def plot_strange_data_case(mode: str = "eng"):
    rooms_label, metro_distance_label, target_label, title, labels = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists() and len(list(tmp_dir.iterdir())) > 0:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    prices = [15500, 10500, 5500]
    rooms = [1, 2, 3]
    metro_distance = [70, 460, 1400]

    frames = []
    vertical_view_id = 20
    for view_id in range(-180, 180, 3):
        fig_size = (9, 5)
        fig = plt.figure(figsize=fig_size)
        gs = GridSpec(2, 2, figure=fig)
        gs.update(wspace=0.05, hspace=0.3)
        ax_upper = fig.add_subplot(gs[0, 0])
        ax_big_3d = fig.add_subplot(gs[:, 1])
        ax_lower = fig.add_subplot(gs[1, 0])

        for ax in [ax_upper, ax_lower]:
            box = ax.get_position()
            new_y0 = box.y0 + (box.height - box.height)
            shrink_factor_width = 1.8
            new_width = box.width / shrink_factor_width
            new_x0 = box.x0 + (box.width - new_width) / 2
            ax.set_position([new_x0, new_y0, new_width, box.height])
            ax.spines[['right', 'top']].set_visible(False)

        ax_upper.scatter(rooms, prices, c=prices, cmap=CMAP, s=80, edgecolor="black",
                        vmin=MIN_TARGET, vmax=MAX_TARGET, zorder=2)
        ax_upper.grid(color='grey', alpha=0.5, zorder=1)
        ax_upper.set_xlabel(rooms_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
        ax_upper.set_ylabel(target_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
        ax_upper.set_xticks([1, 2, 3])
        ax_upper.set_xticklabels([1, 2, 3])
        ax_upper.set_xlim(0.5, 3.5)
        ax_upper.set_ylim(MIN_TARGET, MAX_TARGET)
        ax_upper.set_yticks(TARGET_TICKS)
        ax_upper.set_yticklabels(TARGET_TICKS)
        for i in range(len(labels)):
            ax_upper.text(rooms[i] - 0.2, prices[i] + TEXT_SHIFT, labels[i], color='black',
                          fontsize=6, va='bottom', ha='left', fontname=FONTNAME, zorder=3)

        ax_lower.scatter(metro_distance, prices, c=prices, cmap=CMAP, s=80, edgecolor="black",
                        vmin=MIN_TARGET, vmax=MAX_TARGET, zorder=2)
        ax_lower.grid(color='grey', alpha=0.5, zorder=1)
        ax_lower.set_xlabel(metro_distance_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
        ax_lower.set_ylabel(target_label, fontdict={"fontsize": 9, "fontname": FONTNAME})
        ax_lower.set_xticks([0, 500, 1000, 1500])
        ax_lower.set_xticklabels([0, 500, 1000, 1500])
        ax_lower.set_xlim(-200, 1700)
        ax_lower.set_ylim(MIN_TARGET, MAX_TARGET)
        ax_lower.set_yticks(TARGET_TICKS)
        ax_lower.set_yticklabels(TARGET_TICKS)
        for i in range(len(labels)):
            ax_lower.text(metro_distance[i] - 150, prices[i] + TEXT_SHIFT, labels[i], color='black',
                          fontsize=6, va='bottom', ha='left', fontname=FONTNAME, zorder=3)

        ax_big_3d = scatter_plot_3d(ax_big_3d, np.array(rooms), np.array(metro_distance),
                                    target=np.array(prices),
                                    first_label=rooms_label,
                                    second_label=metro_distance_label,
                                    target_label=target_label, labels=labels)
        ax_big_3d.view_init(vertical_view_id, view_id)

        fig.suptitle(title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=1.1, x=0.55)

        raw_svg_file = Path(tmp_dir, f"extra_animation_4_strange_data_{mode}.svg")
        final_plot = Path(tmp_dir, f"extra_animation_4_strange_data_{mode}_{vertical_view_id}_{view_id}.png")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()

        save_plot_according_to_template(raw_svg_file, final_plot,
                                        template_name="template_coolwarm.svg",
                                        dpi=DPI)
        frames.append(final_plot)


    gif_path = Path(get_plots_path(), f"extra_animation_4_strange_data_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for img in frames:
            writer.append_data(imageio.imread(img))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_strange_data_case("rus")
    plot_strange_data_case("eng")
