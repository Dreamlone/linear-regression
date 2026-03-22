from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {"fontsize": 14, "fontname": FONTNAME}


def prepare_empty_axis(ax, text: str | None = None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")
    ax.set_aspect("equal")
    if text is not None:
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontdict={"fontsize": 12, "fontname": FONTNAME},
            color="#777777",
        )


def prepare_kde_axis(ax, samples, color_hex: str, y_label: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")
    ax.set_aspect("equal")

    x_data_min, x_data_max = -4.0, 4.0
    x_grid = np.linspace(x_data_min, x_data_max, 400)
    kde = stats.gaussian_kde(samples)
    y_vals = kde(x_grid)

    x_norm = (x_grid - x_data_min) / (x_data_max - x_data_min)
    y_norm = y_vals / y_vals.max()
    y_norm = 0.8 * y_norm + 0.1

    ax.plot(x_norm, y_norm, color=color_hex, lw=2)

    ax.set_xlabel("V", fontdict={"fontsize": 8, "fontname": FONTNAME})
    ax.set_ylabel(y_label, fontdict={"fontsize": 8, "fontname": FONTNAME})


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        left_distribution = "Normal\ndistribution"
        right_distribution = "Bimodal\ndistribution"
        first_row = "Sample size"
        second_row = "Sampled data"
        third_row = "Frequency\nhistogram\nfixed bins"
        fourth_row = "Frequency\nhistogram\nRice rule"
        fifth_row = "Kernel density\nestimation"
        sixth_row = "Boxplot"
    elif mode == "rus":
        title = ""
        left_distribution = "Нормальное\nраспределение"
        right_distribution = "Бимодальное\nраспределение"
        first_row = "Размер семпла"
        second_row = "Извлеченная выборка"
        third_row = "Частотная\nгистограмма\nфиксированное кол-во бинов"
        fourth_row = "Частотная\nгистограмма\nПравило Райса"
        fifth_row = "Ядерная\nоценка\nплотности"
        sixth_row = "Ящик с усами"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (
        title,
        left_distribution,
        right_distribution,
        first_row,
        second_row,
        third_row,
        fourth_row,
        fifth_row,
        sixth_row,
    )


def prepare_scatter_axis(ax, x_values: np.ndarray, rng: np.random.Generator, color_hex: str):
    x_data_min, x_data_max = -4.0, 4.0
    x_norm = (x_values - x_data_min) / (x_data_max - x_data_min)
    y_values = rng.uniform(0.1, 0.9, size=x_values.shape[0])

    ax.scatter(
        x_norm,
        y_values,
        s=5,
        c=color_hex,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.set_yticks([])
    ax.set_xlabel("V", fontdict={"fontsize": 6, "fontname": FONTNAME})
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")


def prepare_hist_axis(ax, x_values: np.ndarray, color_hex: str, bins: int = 10):
    x_data_min, x_data_max = -4.0, 4.0
    counts, bin_edges = np.histogram(x_values, bins=bins, range=(x_data_min, x_data_max))

    if counts.max() == 0:
        counts_norm = counts
    else:
        counts_norm = counts / counts.max()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x_norm = (bin_centers - x_data_min) / (x_data_max - x_data_min)
    y_norm = 0.8 * counts_norm + 0.1

    width = 1.0 / bins * 0.9

    ax.bar(
        x_norm,
        y_norm,
        width=width,
        color=color_hex,
        edgecolor="none",
        align="center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.set_yticks([])
    ax.set_xlabel("V", fontdict={"fontsize": 6, "fontname": FONTNAME})
    ax.set_ylabel(f"bins = {bins}", fontdict={"fontsize": 6, "fontname": FONTNAME})
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")


def prepare_sample_kde_axis(ax, x_values: np.ndarray, color_hex: str, line_width: float = 1.5):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")
    ax.set_aspect("equal")

    x_data_min, x_data_max = -4.0, 4.0
    x_grid = np.linspace(x_data_min, x_data_max, 300)

    if x_values.size < 3:
        ax.set_xlabel("V", fontdict={"fontsize": 6, "fontname": FONTNAME})
        return

    kde = stats.gaussian_kde(x_values)
    y_vals = kde(x_grid)

    x_norm = (x_grid - x_data_min) / (x_data_max - x_data_min)
    y_norm = y_vals / y_vals.max()
    y_norm = 0.8 * y_norm + 0.1

    ax.plot(x_norm, y_norm, color=color_hex, lw=line_width)
    ax.set_xlabel("V", fontdict={"fontsize": 6, "fontname": FONTNAME})


def prepare_boxplot_axis(ax, x_values: np.ndarray, color_hex: str):
    # square axes
    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b3b3b3")
    ax.set_aspect("equal")

    x_data_min, x_data_max = -4.0, 4.0
    x_norm = (x_values - x_data_min) / (x_data_max - x_data_min)

    ax.boxplot(
        x_norm,
        vert=True,
        positions=[0.65],
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor=color_hex, edgecolor="#777777"),
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(color="#777777", linewidth=0.8),
        capprops=dict(color="#777777", linewidth=0.8),
        flierprops=dict(
            markerfacecolor=color_hex,
            marker="o",
            markersize=3,
            linestyle="none",
            alpha=0.6,
        ),
    )
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        -0.05,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 6, "fontname": FONTNAME},
    )


def plot_layout(mode: str = "eng"):
    (
        title,
        left_distribution,
        right_distribution,
        first_row,
        second_row,
        third_row,
        fourth_row,
        fifth_row,
        sixth_row,
    ) = annotations_by_language(mode)

    fig = plt.figure(figsize=(8, 7.0))
    gs = fig.add_gridspec(
        nrows=8,
        ncols=4,
        height_ratios=[0.15, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        width_ratios=[1, 1, 1, 1],
        left=0.055,
        right=0.97,
        top=0.96,
        bottom=0.04,
        wspace=0.35,
        hspace=0.35,
    )

    fig.suptitle(title, fontdict={"fontsize": 14, "fontname": FONTNAME})

    ax_top_left = fig.add_subplot(gs[1, 0:2])
    ax_top_right = fig.add_subplot(gs[1, 2:4])

    rng = np.random.default_rng(2025)

    normal_samples = rng.normal(loc=0.0, scale=1.0, size=2000)
    prepare_kde_axis(
        ax_top_left,
        normal_samples,
        color_hex="#70B7CC",
        y_label=left_distribution,
    )

    bimodal_left = rng.normal(loc=-1.0, scale=0.7, size=1000)
    bimodal_right = rng.normal(loc=2.0, scale=0.5, size=1000)
    bimodal_samples = np.concatenate([bimodal_left, bimodal_right])
    prepare_kde_axis(
        ax_top_right,
        bimodal_samples,
        color_hex="orange",
        y_label=right_distribution,
    )

    small_axes = [[None for _ in range(4)] for _ in range(6)]
    for row in range(6):
        for col in range(4):
            ax = fig.add_subplot(gs[row + 2, col])
            prepare_empty_axis(ax)
            small_axes[row][col] = ax

    labels = ["30", "500", "30", "500"]
    for col, text in enumerate(labels):
        small_axes[0][col].text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontdict={"fontsize": 12, "fontname": FONTNAME},
            color="#777777",
        )
    add_row_label(fig, gs, row_index=2, text=first_row)

    # row 2: scatters
    idx_30_normal = rng.choice(normal_samples.shape[0], size=30, replace=False)
    x_30_normal = normal_samples[idx_30_normal]
    prepare_scatter_axis(small_axes[1][0], x_30_normal, rng, color_hex="#70B7CC")

    idx_500_normal = rng.choice(normal_samples.shape[0], size=500, replace=False)
    x_500_normal = normal_samples[idx_500_normal]
    prepare_scatter_axis(small_axes[1][1], x_500_normal, rng, color_hex="#70B7CC")

    idx_30_bimodal = rng.choice(bimodal_samples.shape[0], size=30, replace=False)
    x_30_bimodal = bimodal_samples[idx_30_bimodal]
    prepare_scatter_axis(small_axes[1][2], x_30_bimodal, rng, color_hex="orange")

    idx_500_bimodal = rng.choice(bimodal_samples.shape[0], size=500, replace=False)
    x_500_bimodal = bimodal_samples[idx_500_bimodal]
    prepare_scatter_axis(small_axes[1][3], x_500_bimodal, rng, color_hex="orange")
    add_row_label(fig, gs, row_index=3, text=second_row)

    # row 3: fixed hist
    n_bins = 5
    prepare_hist_axis(small_axes[2][0], x_30_normal, color_hex="#70B7CC", bins=n_bins)
    prepare_hist_axis(small_axes[2][1], x_500_normal, color_hex="#70B7CC", bins=n_bins)
    prepare_hist_axis(small_axes[2][2], x_30_bimodal, color_hex="orange", bins=n_bins)
    prepare_hist_axis(small_axes[2][3], x_500_bimodal, color_hex="orange", bins=n_bins)
    add_row_label(fig, gs, row_index=4, text=third_row)

    # row 4: Rice hist
    x_30_normal_k = int(np.ceil(2 * len(x_30_normal) ** (1 / 3)))
    x_500_normal_k = int(np.ceil(2 * len(x_500_normal) ** (1 / 3)))
    x_30_bimodal_k = int(np.ceil(2 * len(x_30_bimodal) ** (1 / 3)))
    x_500_bimodal_k = int(np.ceil(2 * len(x_500_bimodal) ** (1 / 3)))
    prepare_hist_axis(small_axes[3][0], x_30_normal, color_hex="#70B7CC", bins=x_30_normal_k)
    prepare_hist_axis(small_axes[3][1], x_500_normal, color_hex="#70B7CC", bins=x_500_normal_k)
    prepare_hist_axis(small_axes[3][2], x_30_bimodal, color_hex="orange", bins=x_30_bimodal_k)
    prepare_hist_axis(small_axes[3][3], x_500_bimodal, color_hex="orange", bins=x_500_bimodal_k)
    add_row_label(fig, gs, row_index=5, text=fourth_row)

    # row 5: sample KDE
    prepare_sample_kde_axis(small_axes[4][0], x_30_normal, color_hex="#70B7CC", line_width=1)
    prepare_sample_kde_axis(small_axes[4][1], x_500_normal, color_hex="#70B7CC", line_width=1)
    prepare_sample_kde_axis(small_axes[4][2], x_30_bimodal, color_hex="orange", line_width=1)
    prepare_sample_kde_axis(small_axes[4][3], x_500_bimodal, color_hex="orange", line_width=1)
    add_row_label(fig, gs, row_index=6, text=fifth_row)

    # row 6: boxplots (vertical, square, with colors)
    prepare_boxplot_axis(small_axes[5][0], x_30_normal, color_hex="#70B7CC")
    prepare_boxplot_axis(small_axes[5][1], x_500_normal, color_hex="#70B7CC")
    prepare_boxplot_axis(small_axes[5][2], x_30_bimodal, color_hex="orange")
    prepare_boxplot_axis(small_axes[5][3], x_500_bimodal, color_hex="orange")
    add_row_label(fig, gs, row_index=7, text=sixth_row)

    raw_svg_file = Path(get_plots_path(), f"extra_8_distributions_{mode}.svg")
    final_plot = Path(get_plots_path(), f"extra_8_distributions_{mode}.png")

    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(
        raw_svg_file,
        final_plot,
        template_name="template_orange.svg",
    )


if __name__ == "__main__":
    plot_layout("rus")
    plot_layout("eng")
