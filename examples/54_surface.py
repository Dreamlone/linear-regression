from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"


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
            c=color,
            fontname=FONTNAME,
            fontsize=12
        )


def annotations_by_language(mode: str):
    if mode == "eng":
        title = r"Optimal model in the prediction subspace $\mathrm{Col}(X)$"
        vector_title = r"$\hat{y}_C$ as the projection of the target onto the plane $\mathrm{Col}(X)$"
        x_label = "Observation 1"
        y_label = "Observation 2"
        z_label = "Observation 3"
        scatter_title = "Three different models, A, B, and C"
    elif mode == "rus":
        title = r"Оптимальная модель в подпространстве предсказаний $\mathrm{Col}(X)$"
        vector_title = "$\hat{y}_С$ как проекция отклика на плоскость $\mathrm{Col}(X)$"
        x_label = "Объект 1"
        y_label = "Объект 2"
        z_label = "Объект 3"
        scatter_title = "Три разных модели - A, B, C"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, vector_title, scatter_title, x_label, y_label, z_label


def plot_three_observations_vector_surface(mode: str = "eng"):
    title, vector_title, scatter_title, x_label, y_label, z_label = annotations_by_language(mode)

    x_feature = np.array([1, 2, 3], dtype=float)
    y_vector = np.array([5, 8, 17], dtype=float)

    # Design matrix with intercept: X = [1, x]
    design_matrix = np.column_stack([np.ones_like(x_feature), x_feature])

    # OLS coefficients: beta = [b0, b1] (this is the "correct" solution we use only as a reference)
    beta_ols, *_ = np.linalg.lstsq(design_matrix, y_vector, rcond=None)
    b0_ols = float(beta_ols[0])
    b1_ols = float(beta_ols[1])

    # Two alternative solutions (symmetric around OLS so that OLS lies "in the middle")
    # Mean of these two is exactly the OLS solution [-2, 6] for this dataset.
    beta_a = np.array([2.0, 4.5])
    beta_b = np.array([-6.0, 7.5])
    beta_c = np.array([b0_ols, b1_ols])

    # Predictions for the two alternative solutions
    y_hat_a = design_matrix @ beta_a
    y_hat_b = design_matrix @ beta_b
    y_hat_c = design_matrix @ beta_c

    # Column vectors spanning the prediction subspace (hyperplane through origin in R^3)
    ones_vector = design_matrix[:, 0]   # [1,1,1]^T
    x_vector = design_matrix[:, 1]      # [1,2,3]^T

    print("Design matrix X:\n", design_matrix)
    print("y =", y_vector)
    print("OLS beta (b0, b1) =", beta_ols)
    print("beta_a =", beta_a, "=> y_hat_a =", y_hat_a)
    print("beta_b =", beta_b, "=> y_hat_b =", y_hat_b)

    figure = plt.figure(figsize=(16, 7))
    gs = figure.add_gridspec(1, 2, width_ratios=[1, 2])
    ax_features = figure.add_subplot(gs[0, 0])
    ax_vectors = figure.add_subplot(gs[0, 1], projection="3d")
    gs.update(wspace=-0.2)

    # "Squash" the left axis vertically (keep centered)
    pos = ax_features.get_position()
    shrink_factor = 0.5
    new_height = pos.height * shrink_factor
    new_y0 = pos.y0 + (pos.height - new_height) / 2
    ax_features.set_position([pos.x0, new_y0, pos.width, new_height])

    # Left plot: x vs y with two lines (two different solutions)
    ax_features.scatter(x_feature, y_vector, c="black", zorder=3)

    x_line = np.linspace(0.5, 3.5, 200)
    y_line_a = beta_a[0] + beta_a[1] * x_line
    y_line_b = beta_b[0] + beta_b[1] * x_line

    ax_features.plot(x_line, y_line_a, c='purple',
                     linewidth=1, label=f"A: $b_0$={beta_a[0]:.1f}, $b_1$={beta_a[1]:.1f}")
    ax_features.plot(x_line, y_line_b, c='blue',
                     linewidth=1, label=f"B: $b_0$={beta_b[0]:.1f}, $b_1$={beta_b[1]:.1f}")

    # Optional (very faint) OLS line just as a visual "middle" reference
    y_line_ols = b0_ols + b1_ols * x_line
    ax_features.plot(x_line, y_line_ols, '--', linewidth=2, c='black',
                     label=f"C: $b_0$={b0_ols:.1f}, $b_1$={b1_ols:.1f}")

    ax_features.set_xlim(0.5, 3.5)
    ax_features.set_ylim(0, 20)
    ax_features.set_xlabel("x", fontname=FONTNAME, fontsize=12)
    ax_features.set_ylabel("y", fontname=FONTNAME, fontsize=12)
    ax_features.set_xticks([1, 2, 3])
    ax_features.set_yticks([0, 5, 8, 10, 15, 17, 20])
    ax_features.grid(alpha=0.5, zorder=1)
    ax_features.set_title(scatter_title, fontname=FONTNAME, fontsize=12)
    ax_features.legend(fontsize=10)

    # Right plot: y, hyperplane (Col(X)), and two prediction vectors on it
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
    for label in ax_vectors.get_xticklabels() + ax_vectors.get_yticklabels() + ax_vectors.get_zticklabels():
        label.set_fontname(FONTNAME)
        label.set_fontsize(10)

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

    # Hyperplane (prediction subspace) in R^3:
    # any y_hat = b0 * ones_vector + b1 * x_vector
    b0_min, b0_max = -11, 8
    b1_min, b1_max = 2, 10
    b0_grid = np.linspace(b0_min, b0_max, 30)
    b1_grid = np.linspace(b1_min, b1_max, 30)
    b0_mesh, b1_mesh = np.meshgrid(b0_grid, b1_grid)

    # Convert (b0, b1) -> predicted vectors -> coordinates in (Object1, Object2, Object3)
    pred_mesh = (
        b0_mesh[..., None] * ones_vector[None, None, :]
        + b1_mesh[..., None] * x_vector[None, None, :]
    )
    surface_x = pred_mesh[..., 0]
    surface_y = pred_mesh[..., 1]
    surface_z = pred_mesh[..., 2]

    ax_vectors.plot_surface(
        surface_x, surface_y, surface_z,
        alpha=0.3,
        linewidth=0.0,
        cmap="BuPu",
        edgecolor="white",
        antialiased=True
    )

    # Map 4 corners to 3D: p = b0*ones_vector + b1*x_vector
    p00 = b0_min * ones_vector + b1_min * x_vector
    p10 = b0_max * ones_vector + b1_min * x_vector
    p11 = b0_max * ones_vector + b1_max * x_vector
    p01 = b0_min * ones_vector + b1_max * x_vector

    # Close the loop
    perim = np.vstack([p00, p10, p11, p01, p00])

    ax_vectors.plot(
        perim[:, 0], perim[:, 1], perim[:, 2],
        color="black", linewidth=1
    )
    origin = np.zeros(3)

    # y vector (data)
    draw_arrow(ax_vectors, origin, y_vector, color="orange", label="y")

    # Two prediction vectors (must lie on the hyperplane)
    draw_arrow(ax_vectors, origin, y_hat_a, color="purple", label="$\hat{y}_A$")
    draw_arrow(ax_vectors, origin, y_hat_b, color="blue", label="$\hat{y}_B$")
    draw_arrow(ax_vectors, origin, y_hat_c, color="black", label="$\hat{y}_C$")

    ax_vectors.plot([y_hat_c[0], y_vector[0]],[y_hat_c[1], y_vector[1]],[y_hat_c[2], y_vector[2]],
                    linestyle="--",  color="orange", linewidth=2)

    figure.suptitle(title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", x=0.43, y=0.97)
    raw_svg_file = Path(get_plots_path(), f"54_surface_{mode}_raw.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    final_path = Path(get_plots_path(), f"54_surface_{mode}.png")
    save_plot_according_to_template(raw_svg_file, final_path, template_name="template_small.svg", dpi=100)


if __name__ == "__main__":
    plot_three_observations_vector_surface("rus")
    plot_three_observations_vector_surface("eng")
