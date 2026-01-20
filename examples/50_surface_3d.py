import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
ANIM_DURATION = 150
DPI = 100


def draw_arrow(
    ax,
    start_point,
    vec,
    color,
    label=None,
    label_scale=1.02,
    arrow_length_ratio=0.08,
    linewidth=2,
    alpha=1.0,
):
    """Draw a 3D arrow from start_point with vector vec."""
    ax.quiver(
        start_point[0], start_point[1], start_point[2],
        vec[0], vec[1], vec[2],
        color=color,
        arrow_length_ratio=arrow_length_ratio,
        linewidth=linewidth,
        alpha=alpha
    )
    if label is not None:
        tip = start_point + vec
        ax.text(
            tip[0] * label_scale,
            tip[1] * label_scale,
            tip[2] * label_scale,
            label,
            c=color,
            fontname=FONTNAME,
            fontsize=12,
            alpha=alpha
        )


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        vector_title = ""
        scatter_title = ""
        obj1_label = "Object 1"
        obj2_label = "Object 2"
        obj3_label = "Object 3"
        x1_label = "Feature x1"
        x2_label = "Feature x2"
        y_label = "Response y"
    elif mode == "rus":
        title = r"Многомерная линейная регрессия когда столбцы $X$ независимы"
        vector_title = r"Тривиальная проекция в $\mathrm{Col}(X)=\mathbb{R}^3$"
        scatter_title = "Исходные данные и модель"
        obj1_label = "Объект 1"
        obj2_label = "Объект 2"
        obj3_label = "Объект 3"
        x1_label = "Признак $x_1$"
        x2_label = "Признак $x_2$"
        y_label = "Отклик $y$"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")

    return (
        title,
        vector_title,
        scatter_title,
        obj1_label,
        obj2_label,
        obj3_label,
        x1_label,
        x2_label,
        y_label,
    )


def plot_three_observations_vector_surface(mode: str = "eng"):
    (
        title,
        vector_title,
        scatter_title,
        obj1_label,
        obj2_label,
        obj3_label,
        x1_label,
        x2_label,
        y_label,
    ) = annotations_by_language(mode)

    # Keep x1 and y unchanged
    x1_feature = np.array([1, 2, 3], dtype=float)
    y_vector = np.array([5, 8, 17], dtype=float)

    # Make x2 linearly independent from span{1, x1}
    # Any vector in span{1, x1} looks like [a+b, a+2b, a+3b] -> equal increments (b, b).
    # Here increments are (8, 2), so x2 is NOT in span{1, x1}.
    x2_feature = np.array([0.0, 8.0, 10.0], dtype=float)

    # Design matrix with intercept: X = [1, x1, x2]
    design_matrix = np.column_stack([np.ones_like(x1_feature), x1_feature, x2_feature])

    # OLS coefficients (full-rank => exact fit for n=p=3)
    beta_ols, *_ = np.linalg.lstsq(design_matrix, y_vector, rcond=None)
    b0_ols = float(beta_ols[0])
    b1_ols = float(beta_ols[1])
    b2_ols = float(beta_ols[2])

    y_hat = design_matrix @ beta_ols

    # Column vectors of X spanning Col(X) in R^3 (n=3 objects)
    ones_vector = design_matrix[:, 0]   # [1,1,1]^T
    x1_vector = design_matrix[:, 1]     # [1,2,3]^T
    x2_vector = design_matrix[:, 2]     # [0,8,10]^T

    print("Design matrix X:\n", design_matrix)
    print("y =", y_vector)
    print("OLS beta (b0, b1, b2) =", beta_ols)
    print("y_hat =", y_hat)
    print("||y - y_hat|| =", float(np.linalg.norm(y_vector - y_hat)))

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    image_files = []
    first_view_id = 240
    frames_one_way = 25
    view_ids = list(range(first_view_id, first_view_id + frames_one_way, 1))
    view_ids = view_ids + list(range(first_view_id + frames_one_way - 2, first_view_id, -1))

    fixed_bbox = None

    # Precompute a "box" inside Col(X) (now it's 3D: all of R^3 for full rank at n=3).
    # We visualize a parallelepiped spanned by ranges of coefficients (b0,b1,b2).
    b0_min, b0_max = -5.0, 5.0
    b1_min, b1_max = -2.0, 8.0
    b2_min, b2_max = -2.0, 2.0

    # corners indexed by bits: i = (b0_is_max<<2) + (b1_is_max<<1) + (b2_is_max<<0)
    corners = []
    for b0_val in (b0_min, b0_max):
        for b1_val in (b1_min, b1_max):
            for b2_val in (b2_min, b2_max):
                corner = b0_val * ones_vector + b1_val * x1_vector + b2_val * x2_vector
                corners.append(corner)
    corners = np.array(corners)  # shape (8,3)

    # Faces by corner indices (using the bit ordering above)
    faces = [
        [0, 1, 3, 2],  # b0 = min
        [4, 5, 7, 6],  # b0 = max
        [0, 1, 5, 4],  # b1 = min
        [2, 3, 7, 6],  # b1 = max
        [0, 2, 6, 4],  # b2 = min
        [1, 3, 7, 5],  # b2 = max
    ]

    for view_id, view in enumerate(view_ids):
        figure = plt.figure(figsize=(16, 7))
        gs = figure.add_gridspec(1, 2, width_ratios=[1.15, 2])
        ax_features = figure.add_subplot(gs[0, 0], projection="3d")
        ax_vectors = figure.add_subplot(gs[0, 1], projection="3d")
        gs.update(wspace=0.02)

        # -------------------------
        # Left plot (features space): 3D scatter + model plane
        # Axes: x1, x2, y (color shows y)
        # -------------------------
        y_min, y_max = 0.0, 20.0
        norm = plt.Normalize(vmin=y_min, vmax=y_max)
        cmap = plt.cm.cividis

        ax_features.scatter(
            x1_feature,
            x2_feature,
            y_vector,
            c=y_vector,
            cmap=cmap,
            norm=norm,
            s=70,
            edgecolors="black",
            linewidths=1.0,
            zorder=3
        )

        x1_grid = np.linspace(0.5, 3.5, 80)
        x2_grid = np.linspace(0.0, 10.0, 80)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

        y_mesh = b0_ols + b1_ols * x1_mesh + b2_ols * x2_mesh

        ax_features.plot_surface(
            x1_mesh, x2_mesh, y_mesh,
            alpha=0.5,
            linewidth=0.0,
            cmap=cmap,
            norm=norm,
            antialiased=True
        )

        ax_features.set_xlim(0.5, 3.5)
        ax_features.set_ylim(0.0, 10.0)
        ax_features.set_zlim(y_min, y_max)

        ax_features.set_xlabel(x1_label, fontname=FONTNAME, fontsize=12, labelpad=8)
        ax_features.set_ylabel(x2_label, fontname=FONTNAME, fontsize=12, labelpad=8)
        ax_features.set_zlabel(y_label, fontname=FONTNAME, fontsize=12, labelpad=8)
        ax_features.set_title(
            f"{scatter_title}\n{b0_ols:.1f} + {b1_ols:.1f}$x_1$ + {b2_ols:.1f}$x_2$",
            fontname=FONTNAME, fontsize=12, pad=10
        )

        ax_features.view_init(elev=18, azim=view - 190)

        for axis in [ax_features.xaxis, ax_features.yaxis, ax_features.zaxis]:
            for tick in axis.get_ticklabels():
                tick.set_fontname(FONTNAME)
                tick.set_fontsize(10)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = figure.colorbar(sm, ax=ax_features, fraction=0.04, pad=0.2, shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname(FONTNAME)

        # -------------------------
        # Right plot (observation space R^3): Col(X) + vectors
        # Here Col(X)=R^3 (full rank with n=3), so we draw a "box" region as a visual cue.
        # -------------------------
        axis_min, axis_max = -20, 20
        ax_vectors.set_xlim(axis_min, 20)
        ax_vectors.set_ylim(axis_min, 20)
        ax_vectors.set_zlim(axis_min, 50)
        ax_vectors.set_box_aspect([1, 1, 1])

        ax_vectors.set_xticks(np.arange(axis_min, 20 + 1, 4))
        ax_vectors.set_yticks(np.arange(axis_min, 20 + 1, 4))
        ax_vectors.set_zticks(np.arange(axis_min, 50 + 1, 4))

        ax_vectors.tick_params(axis="both", which="major", length=6, width=1, colors="black")
        ax_vectors.tick_params(axis="z", which="major", length=6, width=1, colors="black")
        for tick in ax_vectors.get_xticklabels() + ax_vectors.get_yticklabels() + ax_vectors.get_zticklabels():
            tick.set_fontname(FONTNAME)
            tick.set_fontsize(10)

        ax_vectors.grid(True)

        ax_vectors.plot([-5, 10], [0, 0], [0, 0], linewidth=1.5, color="white")
        ax_vectors.plot([0, 0], [-5, 10], [0, 0], linewidth=1.5, color="white")
        ax_vectors.plot([0, 0], [0, 0], [-5, 20], linewidth=1.5, color="white")

        ax_vectors.set_xlabel(obj1_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_ylabel(obj2_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_zlabel(obj3_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_title(vector_title, fontname=FONTNAME, fontsize=12)

        ax_vectors.view_init(elev=28, azim=view)

        # Draw the parallelepiped faces lightly (and edges)
        face_polys = [[corners[idx] for idx in face] for face in faces]
        poly = Poly3DCollection(face_polys, alpha=0.10, facecolor="grey", edgecolor="black", linewidths=0.9)
        ax_vectors.add_collection3d(poly)
        poly = Poly3DCollection(face_polys, facecolor=(1, 0, 0, 0.0), edgecolor="black", linewidths=0.9)
        ax_vectors.add_collection3d(poly)

        origin = np.zeros(3)

        # Decomposition of y_hat as a sum of scaled columns: y_hat = b0*1 + b1*x1 + b2*x2
        v_shift = b0_ols * ones_vector
        v_x1 = b1_ols * x1_vector
        v_x2 = b2_ols * x2_vector

        draw_arrow(
            ax_vectors, origin, v_shift,
            color="#F0C100", label=r"$b_0\cdot\mathbf{1}$",
            arrow_length_ratio=0.3, linewidth=2, alpha=0.9, label_scale=1.01
        )
        p1 = v_shift
        draw_arrow(
            ax_vectors, p1, v_x1,
            color="green", label=r"$b_1\cdot x_1$",
            arrow_length_ratio=0.09, linewidth=2, alpha=0.9, label_scale=1.01
        )
        p2 = v_shift + v_x1
        draw_arrow(
            ax_vectors, p2, v_x2,
            color="tab:blue", label=r"$b_2\cdot x_2$",
            arrow_length_ratio=0.3, linewidth=2, alpha=0.9, label_scale=1.01
        )

        if view_id < frames_one_way:
            # Final prediction vector
            draw_arrow(
                ax_vectors, origin, y_hat, color="black", label=r"$\hat{y}$",
                arrow_length_ratio=0.14, linewidth=2, alpha=1.0, label_scale=1.14
            )
        else:
            # Data vector y
            draw_arrow(ax_vectors, origin, y_vector, color="orange", label="y",
                       arrow_length_ratio=0.14, linewidth=2, alpha=1.0, label_scale=1.14)

        # Residual segment (y_hat -> y) will be (almost) zero for full rank with n=p=3.
        if float(np.linalg.norm(y_hat - y_vector)) > 1e-8:
            ax_vectors.plot(
                [y_hat[0], y_vector[0]],
                [y_hat[1], y_vector[1]],
                [y_hat[2], y_vector[2]],
                linestyle="--",
                color="orange",
                linewidth=2
            )

        figure.suptitle(title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", x=0.5, y=0.97)

        raw_svg_file = Path(tmp_dir, f"50_surface_3d_{mode}_raw.svg")
        if view == first_view_id:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            fixed_bbox = figure.get_tightbbox(renderer)
            fixed_bbox = fixed_bbox.expanded(1.05, 1.08)

        plt.savefig(raw_svg_file, bbox_inches=fixed_bbox)
        plt.close()

        final_path = Path(tmp_dir, f"50_surface_3d_{mode}_{view_id}.png")
        save_plot_according_to_template(raw_svg_file, final_path, template_name="template_small.svg", dpi=DPI)
        image_files.append(final_path)

    gif_path = Path(get_plots_path(), f"50_surface_3d_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    plot_three_observations_vector_surface("rus")
