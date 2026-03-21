import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
ANIM_DURATION = 550
DPI = 140


def draw_arrow(ax, start_point, vec, color, label=None, label_scale=1.02, arrow_length_ratio=0.08, zorder=2):
    """Draw a 3D arrow from start_point with vector vec."""
    ax.quiver(
        start_point[0], start_point[1], start_point[2],
        vec[0], vec[1], vec[2],
        color=color,
        arrow_length_ratio=arrow_length_ratio,
        linewidth=2,
        zorder=zorder
    )
    if label is not None:
        tip = start_point + vec
        ax.text(
            tip[0] * label_scale,
            tip[1] * label_scale,
            tip[2] * label_scale,
            label,
            fontname=FONTNAME,
            color=color,
            fontsize=12
        )


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Different models in observation space"
        vector_title = r"Features vectors $b_0v$, $b_1x$ and prediction $\hat{y}$"
        x_label = "Observation 1"
        y_label = "Observation 2"
        z_label = "Observation 3"
    elif mode == "rus":
        title = "Разные модели в пространстве объектов"
        vector_title = r"Векторы признаков $b_0v$, $b_1x$ и предсказаний $\hat{y}$"
        x_label = "Объект 1"
        y_label = "Объект 2"
        z_label = "Объект 3"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, vector_title, x_label, y_label, z_label


def snake_points(min_x, max_x, min_y, max_y, step_x, step_y=None):
    if step_y is None:
        step_y = step_x

    xs = [min_x + i * step_x for i in range(int((max_x - min_x) / step_x) + 1)]
    ys = [min_y + j * step_y for j in range(int((max_y - min_y) / step_y) + 1)]

    points = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        for x in row:
            points.append([float(x), float(y)])
    return points


def plot_three_observations_vector_surface(mode: str = "eng"):
    title, vector_title, x_label, y_label, z_label = annotations_by_language(mode)

    x_feature = np.array([1, 2, 3], dtype=float)
    y_vector = np.array([5, 8, 17], dtype=float)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    image_files = []
    visited_points = []
    b0_min, b0_max = -6, 6
    b1_min, b1_max = 1, 8
    cases = snake_points(min_x=b0_min, max_x=b0_max, min_y=b1_min, max_y=b1_max, step_x=2, step_y=1)
    border_alpha = 0.05
    for case_index, coefficients in enumerate(cases):
        b0, b1 = coefficients
        beta = np.array(coefficients, dtype=float)

        # Design matrix with intercept: X = [1, x]
        design_matrix = np.column_stack([np.ones_like(x_feature), x_feature])

        # Predictions
        y_hat = design_matrix @ beta

        # Column vectors (basis of the prediction subspace)
        ones_vector = design_matrix[:, 0]  # v = [1, 1, 1]
        x_vector = design_matrix[:, 1]     # x = [1, 2, 3]

        # Scaled vectors for the decomposition: y_hat = b0*v + b1*x
        b0_v = b0 * ones_vector
        b1_x = b1 * x_vector

        figure = plt.figure(figsize=(16, 7))
        gs = figure.add_gridspec(1, 2, width_ratios=[1, 2])
        ax_features = figure.add_subplot(gs[0, 0])
        ax_vectors = figure.add_subplot(gs[0, 1], projection="3d")
        gs.update(wspace=-0.2)

        # "squash" the left axis by changing its position rectangle (keep centered)
        pos = ax_features.get_position()
        shrink_factor = 0.5
        new_height = pos.height * shrink_factor
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        ax_features.set_position([pos.x0, new_y0, pos.width, new_height])

        # Left plot: x vs y
        ax_features.set_xlim(0.5, 3.5)
        ax_features.set_ylim(0, 20)
        ax_features.set_xlabel("x", fontname=FONTNAME, fontsize=12)
        ax_features.set_ylabel("y", fontname=FONTNAME, fontsize=12)
        ax_features.set_xticks([1, 2, 3])
        ax_features.set_yticks([0, 5, 8, 10, 15, 17, 20])
        ax_features.grid(alpha=0.5, zorder=1)

        ax_features.scatter(x_feature, y_vector, c="black", zorder=3)

        x_line = np.array([0.0, 4.0], dtype=float)
        ax_features.plot(x_line, b0 + b1 * x_line, "--", c="black")
        ax_features.set_title(rf'$\hat{{y}} = {b0:.2f} + {b1:.2f}\,x$', fontname=FONTNAME, fontsize=12)

        if len(visited_points) > 0:
            for b0_visited, b1_visited in visited_points:
                ax_features.plot(x_line, b0_visited + b1_visited * x_line, "--", c="grey", alpha=0.2)

        # Right plot: vectors in R^3
        axis_min, axis_max = -5, 20
        ax_vectors.set_xlim(axis_min, 10)
        ax_vectors.set_ylim(axis_min, 10)
        ax_vectors.set_zlim(axis_min, axis_max)
        ax_vectors.set_box_aspect([1, 1, 1])

        ax_vectors.set_xticks(np.arange(axis_min, 10 + 1, 2))
        ax_vectors.set_yticks(np.arange(axis_min, 10 + 1, 2))
        ax_vectors.set_zticks(np.arange(axis_min, axis_max + 1, 2))

        ax_vectors.tick_params(axis="both", which="major", length=6, width=1, colors="black")
        ax_vectors.tick_params(axis="z", which="major", length=6, width=1, colors="black")
        for tick_label in ax_vectors.get_xticklabels() + ax_vectors.get_yticklabels() + ax_vectors.get_zticklabels():
            tick_label.set_fontname(FONTNAME)
            tick_label.set_fontsize(10)

        ax_vectors.grid(True)

        # Bold coordinate axes through origin
        ax_vectors.plot([axis_min, 10], [0, 0], [0, 0], linewidth=2, color="black")
        ax_vectors.plot([0, 0], [axis_min, 10], [0, 0], linewidth=2, color="black")
        ax_vectors.plot([0, 0], [0, 0], [axis_min, axis_max], linewidth=2, color="black")

        ax_vectors.set_xlabel(x_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_ylabel(y_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_zlabel(z_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_title(vector_title, fontname=FONTNAME, fontsize=14)

        ax_vectors.view_init(elev=28, azim=45)

        if len(visited_points) > 0:
            xs, ys, zs = [], [], []
            for b0_visited, b1_visited in visited_points:
                y_hat_visited = b0_visited * ones_vector + b1_visited * x_vector

                xs.append(y_hat_visited[0])
                ys.append(y_hat_visited[1])
                zs.append(y_hat_visited[2])

                ax_vectors.scatter(
                    [y_hat_visited[0]], [y_hat_visited[1]], [y_hat_visited[2]],
                    marker="x",
                    c="grey",
                    s=20,
                    zorder=6
                )

        origin = np.zeros(3, dtype=float)

        draw_arrow(ax_vectors, origin, y_vector, color="orange", label="y", zorder=3)
        draw_arrow(ax_vectors, origin, y_hat, color="black", label=r'$\hat{y}$', zorder=4)

        # Decomposition shown with exactly TWO arrows (works for non-integers):
        # 1) b0 * v (yellow) from origin
        draw_arrow(ax_vectors, origin, b0_v, color="yellow", arrow_length_ratio=0.15)

        # 2) b1 * x (green) starting at the end of b0*v
        draw_arrow(ax_vectors, b0_v, b1_x, color="green")

        if case_index >= len(cases) - 12:
            # Time to show the border
            # Convert (b0, b1) -> predicted vectors -> coordinates in (Object1, Object2, Object3)
            p00 = b0_min * ones_vector + b1_min * x_vector
            p10 = b0_max * ones_vector + b1_min * x_vector
            p11 = b0_max * ones_vector + b1_max * x_vector
            p01 = b0_min * ones_vector + b1_max * x_vector
            perim = np.vstack([p00, p10, p11, p01, p00])
            ax_vectors.plot(perim[:, 0], perim[:, 1], perim[:, 2],
                            color="black", linewidth=1, alpha=border_alpha)
            border_alpha += 0.15
            if border_alpha >= 1:
                border_alpha = 1.0

        figure.suptitle(title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", x=0.43, y=0.97)
        raw_svg_file = Path(tmp_dir, f"animation_15_different_models_vectors_{mode}_raw.svg")
        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close()

        final_path = Path(tmp_dir, f"animation_15_different_models_vectors_{mode}_{case_index}.png")
        save_plot_according_to_template(raw_svg_file, final_path, template_name="template_small.svg", dpi=DPI)
        image_files.append(final_path)
        visited_points.append([b0, b1])

    gif_path = Path(get_plots_path(), f"animation_15_different_models_vectors_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    plot_three_observations_vector_surface("rus")
    plot_three_observations_vector_surface("eng")
