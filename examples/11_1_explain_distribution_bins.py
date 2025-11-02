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
        third_x = ""
        fourth_plot_title = ""
        fourth_x = ""
    elif mode == "rus":
        title = "Визуализация выборки в виде частотной гистограммы (для переменной V)"
        first_plot_title = "Датасет с двумя переменными (U и V)"
        second_plot_title = "Наложение k интервалов\nи подсчет наблюдений"
        third_plot_title = "Частотная гистограмма"
        third_x = "Абсолютная частота"
        fourth_plot_title = "Частотная гистограмма\nв вертикальной ориентации"
        fourth_x = "Относительная частота"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, first_plot_title, second_plot_title, third_plot_title, third_x, fourth_plot_title, fourth_x


def plot_hist_explanation(mode: str = "eng"):
    title, first_plot_title, second_plot_title, third_plot_title, third_x, fourth_plot_title, fourth_x = annotations_by_language(mode)

    x, y = generate_synthetic_data()

    fig_size = (17, 4)
    fig, axs = plt.subplots(1, 4, figsize=fig_size)
    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.3)

    axs[0].scatter(x, y, s=40, c="grey", alpha=0.8)
    axs[0].set_ylabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0].set_xlabel("U", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0].set_ylim(MIN_Y, MAX_Y)
    axs[0].set_title(first_plot_title, fontdict=FONTDICT)

    bin_edges = np.linspace(MIN_Y, MAX_Y, N_BINS + 1)
    bin_indices = np.digitize(y, bins=bin_edges, right=False) - 1
    cmap = colormaps.get_cmap('tab20c').resampled(N_BINS)
    colors = [cmap(i) for i in bin_indices]
    axs[1].scatter(x, y, c=colors, s=40, alpha=1)
    axs[1].set_ylabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_xlabel("U", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1].set_title(second_plot_title, fontdict=FONTDICT)
    axs[1].set_ylim(MIN_Y, MAX_Y)
    axs[1].set_title(second_plot_title, fontdict=FONTDICT)
    for edge in bin_edges:
        axs[1].axhline(y=edge, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # Manual histogram plotting
    hist_counts, _ = np.histogram(y, bins=bin_edges)
    for i in range(N_BINS):
        bin_bottom = bin_edges[i]
        bin_top = bin_edges[i + 1]
        bin_height = bin_top - bin_bottom
        bin_count = hist_counts[i]
        color = cmap(i)

        axs[2].barh(
            y=bin_bottom + bin_height / 2,
            width=bin_count,
            height=bin_height,
            color=color,
            edgecolor='white',
            align='center'
        )
        axs[2].text(
            x=bin_count + 0.5,
            y=bin_bottom + bin_height / 2,
            s=str(bin_count),
            va='center',
            ha='left',
            fontsize=11,
            fontname=FONTNAME,
            color=color
        )
    axs[2].set_xlim(0, 25)
    axs[2].set_ylim(MIN_Y, MAX_Y)
    axs[2].set_ylabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[2].set_title(third_plot_title, fontdict=FONTDICT)
    axs[2].set_xlabel(third_x, fontdict={'fontsize': 12, 'fontname': FONTNAME})

    axs[3].set_xlim(MIN_Y, MAX_Y)
    axs[3].set_ylim(0, 0.06)
    axs[3].hist(y, density=True, range=(MIN_Y, MAX_Y),
                alpha=0.8, rwidth=0.9, bins=N_BINS,
                color="grey", orientation='vertical')
    axs[3].set_xlabel("V", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[3].set_title(fourth_plot_title, fontdict=FONTDICT)
    axs[3].set_ylabel(fourth_x, fontdict={'fontsize': 12, 'fontname': FONTNAME})

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, va="top", y=1.2)

    raw_svg_file = Path(get_plots_path(), f"11_1_hist_explanation_{mode}.svg")
    final_plot = Path(get_plots_path(), f"11_1_hist_explanation_{mode}.png")
    plt.savefig(raw_svg_file,  bbox_inches='tight')
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_orange_small_small.svg")


if __name__ == '__main__':
    plot_hist_explanation("rus")