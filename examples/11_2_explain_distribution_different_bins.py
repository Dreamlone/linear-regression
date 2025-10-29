from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

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
        title = "Частотные гистограммы с разным количеством интервалов k"
        first_plot_title = "Исходный датасет"
        freq = "Относительная частота"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, first_plot_title, freq


def plot_hist_explanation(mode: str = "eng"):
    title, first_plot_title, freq = annotations_by_language(mode)

    x, y = generate_synthetic_data()

    fig_size = (12, 9)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)
    axs[0, 0].set_visible(False)
    axs[0, 2].set_visible(False)

    axs[0, 1].scatter(y, x, s=40, c="grey", alpha=0.8)
    axs[0, 1].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0, 1].set_xlim(MIN_Y, MAX_Y)
    axs[0, 1].set_title(first_plot_title, fontdict=FONTDICT)
    axs[0, 1].yaxis.set_ticklabels([])
    axs[0, 1].yaxis.set_ticks([])

    axs[1, 0].set_xlim(MIN_Y, MAX_Y)
    axs[1, 0].set_ylim(0, 0.1)
    axs[1, 0].hist(y, density=True, range=(MIN_Y, MAX_Y),
                alpha=0.9, rwidth=0.9, bins=5,
                color="#C7B875", orientation='vertical')
    axs[1, 0].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 0].set_ylabel(freq, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 0].grid(alpha=0.2)
    axs[1, 0].set_title("k = 5", fontdict={'fontsize': 16, 'fontname': FONTNAME})

    axs[1, 1].set_xlim(MIN_Y, MAX_Y)
    axs[1, 1].set_ylim(0, 0.1)
    axs[1, 1].hist(y, density=True, range=(MIN_Y, MAX_Y),
                   alpha=0.9, rwidth=0.9, bins=10,
                   color="#C7B875", orientation='vertical')
    axs[1, 1].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 1].grid(alpha=0.2)
    axs[1, 1].set_title("k = 10", fontdict={'fontsize': 16, 'fontname': FONTNAME})
    # axs[1, 1].yaxis.set_ticklabels([])

    axs[1, 2].set_xlim(MIN_Y, MAX_Y)
    axs[1, 2].set_ylim(0, 0.1)
    axs[1, 2].grid(alpha=0.2)
    axs[1, 2].hist(y, density=True, range=(MIN_Y, MAX_Y),
                   alpha=0.9, rwidth=0.9, bins=20,
                   color="#C7B875", orientation='vertical')
    axs[1, 2].set_xlabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 2].set_title("k = 20", fontdict={'fontsize': 16, 'fontname': FONTNAME})
    # axs[1, 2].yaxis.set_ticklabels([])

    # Vertical offset (5% upwards) for some axes
    shift = 0.1
    for ax_to_move in [axs[1, 0], axs[1, 2]]:
        pos = ax_to_move.get_position()
        ax_to_move.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, va="top", y=0.98)

    raw_svg_file = Path(get_plots_path(), f"11_2_hist_bins_{mode}.svg", bbox_inches='tight')
    final_plot = Path(get_plots_path(), f"11_2_hist_bins_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_orange.svg")


if __name__ == '__main__':
    plot_hist_explanation("rus")