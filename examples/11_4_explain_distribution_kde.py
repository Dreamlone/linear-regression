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
        title = ""
        first_plot_title = ""
        freq = ""
    elif mode == "rus":
        title = "Еще один способ визуализации распределения"
        first_plot_title = "Датасет с двумя переменными (X и Y)"
        freq = "Ядерная оценка плотности"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, first_plot_title, freq


def plot_kde_explanation(mode: str = "eng"):
    """ Plot the diagram with kernel density estimation """
    title, first_plot_title, freq = annotations_by_language(mode)

    x, y = generate_synthetic_data()

    fig_size = (11, 6)
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)

    axs[0].scatter(y, x, s=40, c="grey", alpha=0.8)
    axs[0].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0].set_xlim(MIN_Y, MAX_Y)

    kde = stats.gaussian_kde(y)
    y_grid = np.linspace(MIN_Y, MAX_Y, 1000)
    kde_values = kde(y_grid)
    axs[1].plot(y_grid, kde_values, color='red', lw=2, alpha=0.5, label="Ядерная оценка")
    axs[1].set_ylim(0, 0.10)

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(y), size=5, replace=False)
    bandwidth = kde.factor * np.std(y)  # оценка ширины ядра

    for idx in sample_indices:
        y0 = y[idx]
        kernel_x = np.linspace(y0 - 3 * bandwidth, y0 + 3 * bandwidth, 300)
        kernel_y = stats.norm.pdf(kernel_x, loc=y0, scale=bandwidth) * 1 / len(y)

    axs[1].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_ylabel("Плотность распределения", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_title(freq, fontdict=FONTDICT)
    axs[1].set_xlim(MIN_Y, MAX_Y)

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, va="top", y=1.05)

    raw_svg_file = Path(get_plots_path(), f"11_4_kde_{mode}.svg", bbox_inches='tight')
    final_plot = Path(get_plots_path(), f"11_4_kde_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_orange.svg")


if __name__ == '__main__':
    plot_kde_explanation("rus")
