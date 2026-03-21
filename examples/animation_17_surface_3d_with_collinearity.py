import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
ANIM_DURATION = 150
DPI = 120


def draw_arrow(
    ax,
    start_point,
    vec,
    color,
    label=None,
    label_scale=1.02,
    arrow_length_ratio=0.08,
    linewidth=2.0,
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
        title = r"Multivariate linear regression when the columns of $X$ are linearly dependent"
        vector_title = r"The subspace $\mathrm{Col}(X)$ and two different decompositions of the same $\hat{y}$"
        scatter_title_top = "Model 1"
        scatter_title_bottom = "Model 2"
        obj1_label = "Observation 1"
        obj2_label = "Observation 2"
        obj3_label = "Observation 3"
        x1_label = "Feature $x_1$"
        x2_label = "Feature $x_2$"
        y_label = "Target $y$"
    elif mode == "rus":
        title = r"Многомерная линейная регрессия когда столбцы $X$ линейно зависимы"
        vector_title = r"Подпространство $\mathrm{Col}(X)$ и два разных разложения одного и того же $\hat{y}$"
        scatter_title_top = "Модель 1"
        scatter_title_bottom = "Модель 2"
        obj1_label = "Объект 1"
        obj2_label = "Объект 2"
        obj3_label = "Объект 3"
        x1_label = "Признак $x_1$"
        x2_label = "Признак $x_2$"
        y_label = "Отклик $y$"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")

    return (title, vector_title, scatter_title_top, scatter_title_bottom,
            obj1_label, obj2_label, obj3_label, x1_label, x2_label, y_label)


def plot_three_observations_vector_surface(mode: str = "eng"):
    (
        title,
        vector_title,
        scatter_title_top,
        scatter_title_bottom,
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

    # Second feature is intentionally collinear with (1, x1) so that Col(X) stays a plane in R^3
    # x2 = 5*(x1 - 1) -> [0, 5, 10]
    x2_feature = 5.0 * (x1_feature - 1.0)

    # Design matrix with intercept: X = [1, x1, x2]
    design_matrix = np.column_stack([np.ones_like(x1_feature), x1_feature, x2_feature])

    # Minimum-norm least squares solution (rank-deficient case handled via SVD internally)
    beta_ols, *_ = np.linalg.lstsq(design_matrix, y_vector, rcond=None)
    b0_ols = float(beta_ols[0])
    b1_ols = float(beta_ols[1])
    b2_ols = float(beta_ols[2])

    # Predictions (unique for OLS projection)
    y_hat = design_matrix @ beta_ols

    # Column vectors in observation space R^3 (n=3 objects)
    ones_vector = design_matrix[:, 0]   # [1,1,1]^T
    x1_vector = design_matrix[:, 1]     # [1,2,3]^T
    x2_vector = design_matrix[:, 2]     # [0,5,10]^T

    print("Design matrix X:\n", design_matrix)
    print("y =", y_vector)
    print("OLS beta (b0, b1, b2) =", beta_ols)
    print("y_hat =", y_hat)

    # --- Build an alternative coefficient vector with the SAME predictions ---
    _, _, vt_svd = np.linalg.svd(design_matrix)
    null_dir = vt_svd[-1, :]
    null_dir = null_dir / np.linalg.norm(null_dir)

    def points_same_order(beta_vec: np.ndarray) -> np.ndarray:
        # origin -> b0*1 -> + b1*x1 -> + b2*x2
        vec_shift = float(beta_vec[0]) * ones_vector
        vec_x1 = float(beta_vec[1]) * x1_vector
        vec_x2 = float(beta_vec[2]) * x2_vector
        pt0 = np.zeros(3)
        pt1 = vec_shift
        pt2 = vec_shift + vec_x1
        pt3 = vec_shift + vec_x1 + vec_x2
        return np.vstack([pt0, pt1, pt2, pt3])

    # Pick beta_alt to maximize separation (but keep it inside fixed axis limits)
    safe_x_min, safe_x_max = -5.0, 10.0
    safe_y_min, safe_y_max = -5.0, 10.0
    safe_z_min, safe_z_max = -5.0, 20.0
    margin = 0.6

    main_poly = points_same_order(beta_ols)

    best_score = -np.inf
    best_beta_alt = None

    t_candidates = np.linspace(0.5, 16.0, 65)
    for sign in (-1.0, 1.0):
        for t_val in t_candidates:
            candidate_beta = beta_ols + sign * t_val * null_dir
            alt_poly = points_same_order(candidate_beta)

            all_pts = np.vstack([main_poly, alt_poly, y_vector[None, :], y_hat[None, :]])
            mins = all_pts.min(axis=0)
            maxs = all_pts.max(axis=0)

            fits = (
                (mins[0] >= safe_x_min + margin) and (maxs[0] <= safe_x_max - margin) and
                (mins[1] >= safe_y_min + margin) and (maxs[1] <= safe_y_max - margin) and
                (mins[2] >= safe_z_min + margin) and (maxs[2] <= safe_z_max - margin)
            )
            if not fits:
                continue

            # Separation score: spread intermediate vertices (p1, p2) with the SAME order
            score = float(
                np.linalg.norm(main_poly[1] - alt_poly[1]) +
                np.linalg.norm(main_poly[2] - alt_poly[2])
            )
            if score > best_score:
                best_score = score
                best_beta_alt = candidate_beta

    beta_alt = best_beta_alt if best_beta_alt is not None else (beta_ols + 6.0 * null_dir)
    y_hat_alt = design_matrix @ beta_alt

    print("Alternative beta (same y_hat) =", beta_alt)
    print("||y_hat - y_hat_alt|| =", float(np.linalg.norm(y_hat - y_hat_alt)))
    print("Separation score =", best_score)

    # --- Color palettes (soft -> saturated) ---
    blue_soft = "#C7D7FF"    # for b0*1
    blue_mid = "#5B86FF"     # for b1*x1
    blue_strong = "#1D4ED8"  # for b2*x2

    red_soft = "#FFD0D0"     # for b0*1
    red_mid = "#FF6B6B"      # for b1*x1
    red_strong = "#DC2626"   # for b2*x2

    # Animation temp folder
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    image_files = []

    first_view_id = 39
    frames_one_way = 25
    view_ids = list(range(first_view_id, first_view_id + frames_one_way, 1))
    view_ids = view_ids + list(range(first_view_id + frames_one_way - 2, first_view_id, -1))

    fixed_bbox = None
    for view_id, view in enumerate(view_ids):
        figure = plt.figure(figsize=(16, 7))

        # Left column: 2 feature plots (top/bottom). Right column: vectors plot spanning both rows.
        gs = figure.add_gridspec(
            2, 2,
            width_ratios=[1.15, 2],
            height_ratios=[1, 1]
        )
        ax_features_top = figure.add_subplot(gs[0, 0], projection="3d")
        ax_features_bottom = figure.add_subplot(gs[1, 0], projection="3d")
        ax_vectors = figure.add_subplot(gs[:, 1], projection="3d")
        gs.update(wspace=0.02, hspace=0.20)

        # Feature plots (both): 3D scatter + plane
        y_min, y_max = 0.0, 20.0
        norm = plt.Normalize(vmin=y_min, vmax=y_max)
        cmap = plt.cm.cividis

        x1_grid = np.linspace(0.5, 3.5, 80)
        x2_grid = np.linspace(0.0, 10.0, 80)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

        def setup_feature_axis(ax, beta_vec, title_text, text_color: str):
            ax.scatter(
                x1_feature, x2_feature, y_vector,
                c=y_vector, cmap=cmap, norm=norm,
                s=70, edgecolors="black", linewidths=1.0, zorder=3
            )

            y_mesh = float(beta_vec[0]) + float(beta_vec[1]) * x1_mesh + float(beta_vec[2]) * x2_mesh
            ax.plot_surface(
                x1_mesh, x2_mesh, y_mesh,
                alpha=0.50, linewidth=0.0,
                cmap=cmap, norm=norm,
                antialiased=True
            )

            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(0.0, 10.0)
            ax.set_zlim(y_min, y_max)

            ax.set_xlabel(x1_label, fontname=FONTNAME, fontsize=12, labelpad=8)
            ax.set_ylabel(x2_label, fontname=FONTNAME, fontsize=12, labelpad=8)
            ax.set_zlabel(y_label, fontname=FONTNAME, fontsize=12, labelpad=8)

            ax.set_title(
                f"{title_text}\n{float(beta_vec[0]):.1f} + {float(beta_vec[1]):.1f}$x_1$ + {float(beta_vec[2]):.1f}$x_2$",
                fontname=FONTNAME, fontsize=10, pad=None, c=text_color, y=0.92
            )

            ax.view_init(elev=18, azim=view - 170)

            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                for tick in axis.get_ticklabels():
                    tick.set_fontname(FONTNAME)
                    tick.set_fontsize(9)

        setup_feature_axis(ax_features_top, beta_ols, scatter_title_top, text_color="tab:blue")
        setup_feature_axis(ax_features_bottom, beta_alt, scatter_title_bottom, text_color="tab:red")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = figure.colorbar(sm, ax=[ax_features_top, ax_features_bottom], fraction=0.04, pad=0.18, shrink=0.90)
        cbar.ax.tick_params(labelsize=9)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname(FONTNAME)

        # Right plot (observation space R^3): Col(X) plane + two decompositions
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
        for tick in ax_vectors.get_xticklabels() + ax_vectors.get_yticklabels() + ax_vectors.get_zticklabels():
            tick.set_fontname(FONTNAME)
            tick.set_fontsize(10)

        ax_vectors.grid(True)

        ax_vectors.plot([axis_min, 10], [0, 0], [0, 0], linewidth=1.5, color="grey")
        ax_vectors.plot([0, 0], [axis_min, 10], [0, 0], linewidth=1.5, color="grey")
        ax_vectors.plot([0, 0], [0, 0], [axis_min, axis_max], linewidth=1.5, color="grey")

        ax_vectors.set_xlabel(obj1_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_ylabel(obj2_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_zlabel(obj3_label, fontname=FONTNAME, fontsize=12)
        ax_vectors.set_title(vector_title, fontname=FONTNAME, fontsize=12)

        ax_vectors.view_init(elev=28, azim=view)

        # Col(X) surface in R^3 (parametrized by two coefficients)
        b0_min, b0_max = -11, 8
        b1_min, b1_max = 2, 10
        b0_grid = np.linspace(b0_min, b0_max, 30)
        b1_grid = np.linspace(b1_min, b1_max, 30)
        b0_mesh, b1_mesh = np.meshgrid(b0_grid, b1_grid)

        pred_mesh = (
            b0_mesh[..., None] * ones_vector[None, None, :]
            + b1_mesh[..., None] * x1_vector[None, None, :]
        )

        surface_x = pred_mesh[..., 0]
        surface_y = pred_mesh[..., 1]
        surface_z = pred_mesh[..., 2]

        # ax_vectors.plot_surface(
        #     surface_x, surface_y, surface_z,
        #     alpha=0.28, linewidth=0.0,
        #     cmap="Greys", edgecolor="white",
        #     antialiased=True
        # )
        ax_vectors.plot_surface(
                surface_x, surface_y, surface_z,
                alpha=0.2, linewidth=0.0,
                color="grey",
                antialiased=True
            )

        p00 = b0_min * ones_vector + b1_min * x1_vector
        p10 = b0_max * ones_vector + b1_min * x1_vector
        p11 = b0_max * ones_vector + b1_max * x1_vector
        p01 = b0_min * ones_vector + b1_max * x1_vector
        perim = np.vstack([p00, p10, p11, p01, p00])
        ax_vectors.plot(perim[:, 0], perim[:, 1], perim[:, 2], color="black", linewidth=1)

        origin = np.zeros(3)

        # Data vector y
        draw_arrow(ax_vectors, origin, y_vector, color="orange", label="y")

        # Model 1 (blue), same order: shift -> x1 -> x2
        v_shift = b0_ols * ones_vector
        v_x1 = b1_ols * x1_vector
        v_x2 = b2_ols * x2_vector

        draw_arrow(ax_vectors, origin, v_shift, color=blue_soft, label=None,
                   arrow_length_ratio=0.30, linewidth=2.2, alpha=1.0)
        p1 = v_shift
        draw_arrow(ax_vectors, p1, v_x1, color=blue_mid, label=None,
                   arrow_length_ratio=0.15, linewidth=2.2, alpha=1.0)
        p2 = v_shift + v_x1
        draw_arrow(ax_vectors, p2, v_x2, color=blue_strong, label=None,
                   arrow_length_ratio=0.06, linewidth=2.2, alpha=1.0)

        # Model 2 (red), SAME order: shift -> x1 -> x2
        v_shift_alt = float(beta_alt[0]) * ones_vector
        v_x1_alt = float(beta_alt[1]) * x1_vector
        v_x2_alt = float(beta_alt[2]) * x2_vector

        draw_arrow(ax_vectors, origin, v_shift_alt, color=red_soft, label=None,
                   arrow_length_ratio=0.30, linewidth=2.2, alpha=0.95)
        q1 = v_shift_alt
        draw_arrow(ax_vectors, q1, v_x1_alt, color=red_mid, label=None,
                   arrow_length_ratio=0.05, linewidth=2.2, alpha=0.95)
        q2 = v_shift_alt + v_x1_alt
        draw_arrow(ax_vectors, q2, v_x2_alt, color=red_strong, label=None,
                   arrow_length_ratio=0.6, linewidth=2.2, alpha=0.95)

        # Final prediction vector (unique)
        draw_arrow(ax_vectors, origin, y_hat, color="black", label=r"$\hat{y}$",
                   arrow_length_ratio=0.08, linewidth=2.2, alpha=1.0, label_scale=1.14)

        # Residuals
        ax_vectors.plot([y_hat[0], y_vector[0]],[y_hat[1], y_vector[1]],[y_hat[2], y_vector[2]],
                        linestyle="--", color="orange", linewidth=2)

        figure.suptitle(title, fontsize=14, fontdict={"fontname": FONTNAME}, va="top", x=0.50, y=0.97)

        raw_svg_file = Path(tmp_dir, f"animation_17_surface_3d_with_collinearity_{mode}_raw.svg")
        if view == first_view_id:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            fixed_bbox = figure.get_tightbbox(renderer)
            fixed_bbox = fixed_bbox.expanded(1.05, 1.08)

        plt.savefig(raw_svg_file, bbox_inches=fixed_bbox)
        plt.close()

        final_path = Path(tmp_dir, f"animation_17_surface_3d_with_collinearity_{mode}_{view_id}.png")
        save_plot_according_to_template(raw_svg_file, final_path, template_name="template_small.svg", dpi=DPI)
        image_files.append(final_path)

    gif_path = Path(get_plots_path(), f"animation_17_surface_3d_with_collinearity_{mode}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=ANIM_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    # plot_three_observations_vector_surface("rus")
    plot_three_observations_vector_surface("eng")
