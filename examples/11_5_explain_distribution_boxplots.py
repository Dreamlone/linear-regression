from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 62
N_BINS = 15

def generate_synthetic_data(n_samples=100, noise_std=1.0, outlier_fraction=0.1, random_state=2025):
    """ Generate dataset """
    rng = np.random.default_rng(random_state)
    x = np.linspace(0, 10, n_samples)
    true_slope = 2.0
    true_intercept = 5.0
    y = true_slope * x + true_intercept + rng.normal(0, noise_std, size=n_samples)
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = rng.choice(n_samples, size=n_outliers, replace=False)
    y[outlier_indices] += rng.normal(0, 20 * noise_std, size=n_outliers)

    x = x + 17
    y = y + 17
    print(f"Min x: {min(x)}, Max x: {max(x)}")
    print(f"Min y: {min(y)}, Max x: {max(y)}")
    return x, y


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        first_plot_title = ""
        second_plot_title = ""
        third_plot_title = ""
        median_label = "median"
    elif mode == "rus":
        title = "Ящик с усами"
        first_plot_title = "Исходный датасет"
        second_plot_title = "Ядерная оценка плотности\nи частотная гистограмма (k = 10)"
        third_plot_title = 'Диаграмма "ящик с усами"\nв вертикальной ориентации'
        median_label = "медиана"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, first_plot_title, second_plot_title, third_plot_title, median_label


def plot_distribution_boxplot(mode: str = "eng"):
    title, first_plot_title, second_plot_title, third_plot_title, median_label = annotations_by_language(mode)

    x, y = generate_synthetic_data()

    fig_size = (14, 5)
    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3)

    axs[0].scatter(y, x, s=40, c="grey", alpha=0.8)
    axs[0].set_xlabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0].set_xlim(MIN_Y, MAX_Y)
    axs[0].set_title(first_plot_title, fontdict=FONTDICT)
    axs[0].yaxis.set_ticklabels([])
    axs[0].yaxis.set_ticks([])

    kde = stats.gaussian_kde(y)
    y_grid = np.linspace(MIN_Y, MAX_Y, 1000)
    kde_values = kde(y_grid)
    axs[1].plot(y_grid, kde_values, color='black', lw=2, zorder=2)
    axs[1].set_ylim(0, 0.10)
    axs[1].set_title(second_plot_title, fontdict=FONTDICT)
    axs[1].set_xlabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})

    k = 10
    axs[1].hist(y, density=True, range=(MIN_Y, MAX_Y),
                   alpha=0.5, rwidth=0.9, bins=k,
                   color="grey", orientation='vertical', zorder=1)
    median_value = np.median(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    axs[1].plot([median_value, median_value], [0.07, 0.048], '--', c='grey')
    axs[1].text(
        median_value * 0.95, 0.088, median_label,
        ha="center", va="top",
        rotation="vertical", rotation_mode="anchor",
        fontdict={'fontname': FONTNAME}
    )
    axs[1].plot([q1, q1], [0.07, 0.048], '--', c='grey')
    axs[1].plot([q3, q3], [0.07, 0.048], '--', c='grey')
    for q_coordinate, q_label in zip([q1, q3], ["Q1", "Q3"]):
        axs[1].text(
            q_coordinate * 0.95, 0.082, q_label,
            ha="center", va="top",
            rotation="vertical", rotation_mode="anchor",
            fontdict={'fontname': FONTNAME}
        )

    ax_boxplot = axs[1].twinx()
    ax_boxplot.boxplot(
        y,
        positions=[1.5],
        orientation="horizontal",
        widths=0.1,
        patch_artist=True,
        boxprops=dict(facecolor='orange', color='grey'),
        medianprops=dict(color='white'),
        whiskerprops=dict(color='grey'),
        capprops=dict(color='grey'),
        flierprops=dict(markerfacecolor='orange', marker='o', markersize=4, linestyle='none', alpha=0.5),
        zorder=2
    )
    ax_boxplot.set_ylim(0, 2)
    ax_boxplot.set_yticks([])

    axs[2].boxplot(
        y,
        orientation="vertical",
        widths=0.1,
        patch_artist=True,
        boxprops=dict(facecolor='orange', color='grey'),
        medianprops=dict(color='white'),
        whiskerprops=dict(color='grey'),
        capprops=dict(color='grey'),
        flierprops=dict(markerfacecolor='orange', marker='o', markersize=4, linestyle='none', alpha=0.5),
        zorder=2
    )
    axs[2].xaxis.set_ticklabels([])
    axs[2].xaxis.set_ticks([])
    axs[2].set_ylabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[2].set_title(third_plot_title, fontdict=FONTDICT)

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, va="top", y=1.2)

    raw_svg_file = Path(get_plots_path(), f"11_5_boxplot_explanation_{mode}.svg")
    final_plot = Path(get_plots_path(), f"11_5_boxplot_explanation_{mode}.png")
    plt.savefig(raw_svg_file,  bbox_inches='tight')
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_orange_small.svg")


if __name__ == '__main__':
    plot_distribution_boxplot("rus")
