from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import stats

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, COLOR_BY_DATASET


def generate_synthetic_residual_sets(n_samples: int = 500, random_state: int = 2025):
    """Generate synthetic predicted values and residuals for three datasets A, B, C"""
    rng = np.random.default_rng(random_state)

    # A: almost normal, homoscedastic
    predicted_a = np.linspace(5000, 60000, n_samples)
    residuals_a = rng.normal(loc=0.0, scale=1200.0, size=n_samples)

    # B: variance grows with prediction
    predicted_b = np.linspace(5000, 60000, n_samples)
    scale_b = 600.0 + 0.07 * predicted_b
    residuals_b = rng.normal(loc=0.0, scale=scale_b)
    residuals_b = np.clip(residuals_b, -7700, 7700)

    # C: crescent-like
    predicted_c = np.linspace(5000, 60000, n_samples)
    t = (predicted_c - predicted_c.min()) / (predicted_c.max() - predicted_c.min())

    residuals_c = np.empty_like(t)
    mask_left = t < 0.15
    residuals_c[mask_left] = rng.normal(loc=-3500.0, scale=500.0, size=mask_left.sum())

    mask_mid = (t >= 0.15) & (t <= 0.7)
    norm_mid = (t[mask_mid] - 0.15) / (0.7 - 0.15)
    hump = (np.sin(np.pi * norm_mid) ** 0.8)
    residuals_c[mask_mid] = -3500.0 + 7500.0 * hump + rng.normal(
        loc=0.0, scale=450.0, size=mask_mid.sum()
    )

    mask_right = t > 0.7
    norm_right = (t[mask_right] - 0.7) / (1.0 - 0.7)
    start_val = -1500.0
    end_val = -3500.0
    residuals_c[mask_right] = (
        start_val + (end_val - start_val) * norm_right
        + rng.normal(loc=0.0, scale=400.0, size=mask_right.sum())
    )

    residuals_c = np.clip(residuals_c, -7700, 7700)

    return (
        (predicted_a, residuals_a),
        (predicted_b, residuals_b),
        (predicted_c, residuals_c),
    )


def _plot_residuals(ax, actual_v, predicted_v, x_label, fontdict, color,
                    hide_y_ticks: bool = False):
    actual_v = np.ravel(actual_v)
    predicted_v = np.ravel(predicted_v)
    residuals = actual_v - predicted_v
    ax.scatter(
        predicted_v,
        residuals,
        s=40,
        c=color,
        edgecolor="black",
        alpha=0.4,
    )
    ax.plot([0, 65000], [0, 0], '--', color="black", alpha=0.3)

    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, 65000)
    ax.set_xlabel(x_label, fontdict=fontdict)

    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])

    return residuals


def _plot_hist(
    ax,
    residuals,
    y_label,
    fontdict,
    color,
    hide_y_ticks: bool = False,
    residuals_distribution_label: str = "",
    normal_distribution_label: str = "",
):
    kde = stats.gaussian_kde(residuals)
    xx = np.linspace(-8000, 8000, 1000)
    kde_vals = kde(xx)

    mu = residuals.mean()
    sigma = residuals.std(ddof=1)
    if sigma < 1e-6:
        sigma = 1e-6
    normal_vals = stats.norm.pdf(xx, loc=mu, scale=sigma)

    max_density = 0.00034
    kde_max = max(kde_vals.max(), 1e-9)
    kde_scaled = kde_vals * (max_density / kde_max)

    normal_max = max(normal_vals.max(), 1e-9)
    normal_scaled = normal_vals * (max_density / normal_max)

    line_res_kde, = ax.plot(
        kde_scaled,
        xx,
        color=color,
        linewidth=2,
        label=residuals_distribution_label,
    )

    line_norm, = ax.plot(
        normal_scaled,
        xx,
        color="grey",
        linewidth=2.0,
        alpha=0.9,
        label=normal_distribution_label,
    )

    ax.plot([0, max_density], [0, 0], '--', color="black", alpha=0.3)
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, max_density)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])

    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_label, fontdict=fontdict)

    ax.legend(
        handles=[line_res_kde, line_norm],
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="white",
    )

    return max_density


def _plot_qq(
    ax,
    residuals,
    x_label,
    y_label,
    fontdict,
    color,
    hide_y_ticks: bool = False,
):
    residuals = np.ravel(residuals)

    mu = residuals.mean()
    sigma = residuals.std(ddof=1)
    if sigma < 1e-6:
        sigma = 1e-6

    percs = np.linspace(1, 99, 99)
    sample_q = np.percentile(residuals, percs)
    theo_q = stats.norm.ppf(percs / 100.0, loc=mu, scale=sigma)

    q_min = min(sample_q.min(), theo_q.min())
    q_max = max(sample_q.max(), theo_q.max())

    ax.scatter(
        theo_q,
        sample_q,
        s=22,
        c=color,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
    )
    ax.plot([q_min, q_max], [q_min, q_max], '--', color="black", alpha=0.3)

    ax.set_xlabel(x_label, fontdict=fontdict)
    if hide_y_ticks is False:
        ax.set_ylabel(y_label, fontdict=fontdict)

    ax.grid(color='grey', alpha=0.1)


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Predicted"
        y_label = "Residuals (actual - predicted)"
        qq_x_label = "Theoretical quantiles (normal, estimated)"
        qq_y_label = "Sample residual quantiles"
        title = "Visual assessment of regression quality\nResidual plots, histograms and QQ-plots"
        residuals_distribution_label = "Residuals KDE"
        normal_distribution_label = "Normal (estimated from residuals)"
    elif mode == "rus":
        x_label = "Предсказания"
        y_label = "Остатки (реальные - предсказанные)"
        qq_x_label = "Теоретические квантили\n(нормальное распределение,\nоценка по остаткам)"
        qq_y_label = "Выборочные квантили остатков"
        title = (
            "Визуальная оценка качества моделирования\n"
            "Квантильные биплоты: нормальное распределение (оценка по остаткам) vs остатки"
        )
        residuals_distribution_label = "Плотность остатков"
        normal_distribution_label = "Нормальное (оценка по остаткам)"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (
        x_label,
        y_label,
        title,
        qq_x_label,
        qq_y_label,
        residuals_distribution_label,
        normal_distribution_label,
    )


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {"fontsize": 14, "fontname": fontname}
    (
        x_label,
        y_label,
        title,
        qq_x_label,
        qq_y_label,
        residuals_distribution_label,
        normal_distribution_label,
    ) = annotations_by_language(mode)

    (pred_a, res_a), (pred_b, res_b), (pred_c, res_c) = generate_synthetic_residual_sets(
        n_samples=500,
        random_state=2025,
    )

    fig_size = (16, 14)
    fig, axs = plt.subplots(3, 3, figsize=fig_size)
    fig.suptitle(
        title,
        fontsize=20,
        fontdict={"fontname": fontname},
        va="top",
        y=0.995,
    )

    # A
    actual_a = pred_a + res_a
    _plot_residuals(
        axs[0, 0],
        actual_a,
        pred_a,
        x_label,
        fontdict,
        COLOR_BY_DATASET["A"],
        hide_y_ticks=False,
    )
    axs[0, 0].set_ylabel(y_label, fontdict=fontdict)
    axs[0, 0].set_title("A*", fontsize=20, fontdict={"fontname": fontname})

    _plot_hist(
        axs[1, 0],
        res_a,
        y_label,
        fontdict,
        COLOR_BY_DATASET["A"],
        hide_y_ticks=False,
        residuals_distribution_label=residuals_distribution_label,
        normal_distribution_label=normal_distribution_label,
    )

    _plot_qq(
        axs[2, 0],
        res_a,
        qq_x_label,
        qq_y_label,
        fontdict,
        COLOR_BY_DATASET["A"],
        hide_y_ticks=False,
    )

    # B
    actual_b = pred_b + res_b
    _plot_residuals(
        axs[0, 1],
        actual_b,
        pred_b,
        x_label,
        fontdict,
        COLOR_BY_DATASET["B"],
        hide_y_ticks=True,
    )
    axs[0, 1].set_title("B*", fontsize=20, fontdict={"fontname": fontname})

    _plot_hist(
        axs[1, 1],
        res_b,
        y_label,
        fontdict,
        COLOR_BY_DATASET["B"],
        hide_y_ticks=True,
        residuals_distribution_label=residuals_distribution_label,
        normal_distribution_label=normal_distribution_label,
    )

    _plot_qq(
        axs[2, 1],
        res_b,
        qq_x_label,
        qq_y_label,
        fontdict,
        COLOR_BY_DATASET["B"],
        hide_y_ticks=True,
    )

    # C
    actual_c = pred_c + res_c
    _plot_residuals(
        axs[0, 2],
        actual_c,
        pred_c,
        x_label,
        fontdict,
        COLOR_BY_DATASET["C"],
        hide_y_ticks=True,
    )
    axs[0, 2].set_title("C*", fontsize=20, fontdict={"fontname": fontname})

    _plot_hist(
        axs[1, 2],
        res_c,
        y_label,
        fontdict,
        COLOR_BY_DATASET["C"],
        hide_y_ticks=True,
        residuals_distribution_label=residuals_distribution_label,
        normal_distribution_label=normal_distribution_label,
    )

    _plot_qq(
        axs[2, 2],
        res_c,
        qq_x_label,
        qq_y_label,
        fontdict,
        COLOR_BY_DATASET["C"],
        hide_y_ticks=True,
    )

    # highlight bottom row (QQ plots) with a red rectangle
    bottom_axes = [axs[2, 0], axs[2, 1], axs[2, 2]]
    bboxes = [ax.get_position(fig) for ax in bottom_axes]
    x0 = min(b.x0 for b in bboxes) - 0.05
    y0 = min(b.y0 for b in bboxes) - 0.07
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)

    pad_x = 0.01
    pad_y = 0.015

    rect = patches.Rectangle(
        (x0 - pad_x, y0 - pad_y),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) + 2 * pad_y,
        transform=fig.transFigure,
        fill=False,
        edgecolor="red",
        linewidth=2.2,
        zorder=200,
    )
    fig.add_artist(rect)

    raw_svg_file = Path(get_plots_path(), f"11_7_datasets_residuals_qq_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(
        raw_svg_file,
        Path(get_plots_path(), f"11_7_datasets_residuals_qq_{mode}.png"),
    )


if __name__ == "__main__":
    plot_errors_plots("rus")
