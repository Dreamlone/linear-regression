import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
ANIM_DURATION = 350
DPI = 90


def draw_arrow(ax, start_point, vec, color, label=None, label_scale=1.02, arrow_length_ratio=0.08):
    """Draw a 3D arrow from start_point with vector vec."""
    ax.quiver(
        start_point[0], start_point[1], start_point[2],
        vec[0], vec[1], vec[2],
        color=color,
        arrow_length_ratio=arrow_length_ratio,
        linewidth=2
    )
    if label is not None:
        tip = start_point + vec
        ax.text(
            tip[0] * label_scale,
            tip[1] * label_scale,
            tip[2] * label_scale,
            label,
            color=color,
            fontname=FONTNAME,
            fontsize=12
        )


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        vector_title = ""
        scatter_title = ""
        x_label = "Object 1"
        y_label = "Object 2"
        z_label = "Object 3"
    elif mode == "rus":
        title = "Выборка с тремя объектами"
        vector_title = "Данные в векторном пространстве"
        scatter_title = "Признак x Vs Отклик y"
        x_label = "Объект 1"
        y_label = "Объект 2"
        z_label = "Объект 3"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, vector_title, scatter_title, x_label, y_label, z_label


def plot_three_observations_vector_animation(mode: str = 'eng'):
    title, vector_title, scatter_title, x_label, y_label, z_label = annotations_by_language(mode)

    x_feature = np.array([1, 2, 3])
    y_vector = np.array([5, 8, 17])

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    image_files = []
    for view_id in range(140, 140 + 360, 5):
        figure = plt.figure(figsize=(16, 7))
        gs = figure.add_gridspec(1, 2, width_ratios=[1, 2])
        ax_vectors = figure.add_subplot(gs[0, 1], projection="3d")
        ax_features = figure.add_subplot(gs[0, 0])
        gs.update(wspace=-0.2)

        # "squash" the right axis by changing its position rectangle
        pos = ax_features.get_position()
        shrink_factor = 0.5
        new_height = pos.height * shrink_factor
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        ax_features.set_position([pos.x0, new_y0, pos.width, new_height])

        ax_features.scatter(x_feature, y_vector, c='black', zorder=2)
        ax_features.set_ylim(0, 20)
        ax_features.set_xlim(0.5, 3.5)
        ax_features.set_ylabel("y", fontname=FONTNAME, fontsize=12)
        ax_features.set_xlabel("x", fontname=FONTNAME, fontsize=12)
        ax_features.set_xticks([1, 2, 3])
        ax_features.set_xticklabels([1, 2, 3])
        ax_features.set_yticks([0, 5, 8, 10, 15, 17, 20])
        ax_features.set_yticklabels([0, 5, 8, 10, 15, 17, 20])
        ax_features.grid(alpha=0.5, zorder=1)
        ax_features.set_title(scatter_title, fontname=FONTNAME, fontsize=12)

        # Limits + equal-ish scale
        axis_min, axis_max = -10, 20
        ax_vectors.set_xlim(axis_min, axis_max)
        ax_vectors.set_ylim(axis_min, axis_max)
        ax_vectors.set_zlim(axis_min, axis_max)
        ax_vectors.set_box_aspect([1, 1, 1])

        # Less dense grid/ticks: every 3 units
        ticks = np.arange(axis_min, axis_max + 1, 2)
        ax_vectors.set_xticks(ticks)
        ax_vectors.set_yticks(ticks)
        ax_vectors.set_zticks(ticks)

        for p, color in zip([x_feature, y_vector], ["green", "orange"]):
            guide_kwargs = dict(linestyle="--", linewidth=1.5, color=color, alpha=0.8)
            ax_vectors.plot([p[0], axis_min], [p[1], p[1]], [p[2], p[2]], **guide_kwargs)
            ax_vectors.plot([p[0], p[0]], [p[1], axis_min], [p[2], p[2]], **guide_kwargs)
            ax_vectors.plot([p[0], p[0]], [p[1], p[1]], [p[2], axis_min], **guide_kwargs)

        ax_vectors.tick_params(axis="both", which="major", length=6, width=1, colors="black")
        ax_vectors.tick_params(axis="z", which="major", length=6, width=1, colors="black")
        for label in ax_vectors.get_xticklabels() + ax_vectors.get_yticklabels() + ax_vectors.get_zticklabels():
            label.set_fontname(FONTNAME)
            label.set_fontsize(10)

        ax_vectors.grid(True)

        # Draw bold coordinate axes through origin
        ax_vectors.plot([axis_min, axis_max], [0, 0], [0, 0], linewidth=2, color="black")
        ax_vectors.plot([0, 0], [axis_min, axis_max], [0, 0], linewidth=2, color="black")
        ax_vectors.plot([0, 0], [0, 0], [axis_min, axis_max], linewidth=2, color="black")

        ax_vectors.set_xlabel(x_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_ylabel(y_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_zlabel(z_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_title(vector_title, fontname=FONTNAME, fontsize=12)

        ax_vectors.view_init(elev=22, azim=view_id)

        origin = np.zeros(3)
        draw_arrow(ax_vectors, origin, y_vector, color="orange", label="y (5, 8, 17)")
        draw_arrow(ax_vectors, origin, x_feature, color="green", label="x (1, 2, 3)", arrow_length_ratio=0.3)

        figure.suptitle(title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", x=0.43, y=0.97)
        raw_svg_file = Path(tmp_dir, f"47_linear_algebra_3d_vector_sum_{mode}_raw.svg")
        if view_id == 140:
            figure.canvas.draw()
            fixed_bbox = figure.get_tightbbox(figure.canvas.get_renderer())
        plt.savefig(raw_svg_file, bbox_inches=fixed_bbox)
        plt.close()

        final_path = Path(tmp_dir, f"47_linear_algebra_3d_vector_sum_{mode}_{view_id}.png")
        save_plot_according_to_template(raw_svg_file, final_path, template_name="template_small.svg", dpi=DPI)
        image_files.append(final_path)

    gif_path = Path(get_plots_path(), f"47_linear_algebra_3d_vector_sum_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    plot_three_observations_vector_animation("rus")
