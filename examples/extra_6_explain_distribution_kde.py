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
        title = "Another way to visualize a distribution"
        right_plot_title = "Density"
        freq = "Kernel density estimation"
    elif mode == "rus":
        title = "Еще один способ визуализации распределения"
        right_plot_title = "Плотность распределения"
        freq = "Ядерная оценка плотности"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, right_plot_title, freq


def plot_kde_explanation(mode: str = "eng"):
    """ Plot the diagram with kernel density estimation """
    title, right_plot_title, freq = annotations_by_language(mode)

    x, y = generate_synthetic_data()

    fig_size = (10, 5)
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)

    axs[0].scatter(y, x, s=40, c="grey", alpha=0.8)
    axs[0].set_xlabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0].set_xlim(MIN_Y, MAX_Y)
    axs[0].yaxis.set_ticklabels([])
    axs[0].yaxis.set_ticks([])

    kde = stats.gaussian_kde(y)
    y_grid = np.linspace(MIN_Y, MAX_Y, 1000)
    kde_values = kde(y_grid)
    axs[1].plot(y_grid, kde_values, color='red', lw=2, alpha=0.5, zorder=1)
    axs[1].set_ylim(0, 0.10)

    bandwidth = kde.factor * np.std(y)

    # Show some individual distributions
    for idx, color in zip([9, 22], ["#70B7CC", "orange"]):
        y0 = y[idx]
        kernel_x = np.linspace(y0 - 3 * bandwidth, y0 + 3 * bandwidth, 300)
        kernel_y = 16 + (stats.norm.pdf(kernel_x, loc=y0, scale=bandwidth) * 410 / len(y))
        axs[0].plot(kernel_x, kernel_y, color=color, lw=2)
        axs[0].scatter([y0], [x[idx]], color=color, s=50, alpha=1, zorder=2)
        axs[0].plot([y0, y0], [max(kernel_y), x[idx]], '--', color=color, lw=1)
        axs[1].plot(kernel_x, stats.norm.pdf(kernel_x, loc=y0, scale=bandwidth) * 2 / len(y), color=color, lw=2)

    axs[1].set_xlabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_ylabel(right_plot_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_title(freq, fontdict=FONTDICT)
    axs[1].set_xlim(MIN_Y, MAX_Y)

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, va="top", y=1)

    raw_svg_file = Path(get_plots_path(), f"extra_6_kde_{mode}.svg")
    final_plot = Path(get_plots_path(), f"extra_6_kde_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_orange_small.svg")


if __name__ == '__main__':
    plot_kde_explanation("rus")
    plot_kde_explanation("eng")
