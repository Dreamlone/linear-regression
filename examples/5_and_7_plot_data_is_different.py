from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

from examples.paths import get_plots_path

COLORS = ['red', 'orange', 'green', 'blue']


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of rooms"
        y_label = "Price, $"
        text_label = "Wow, why is it decreasing?"
        x_coord_for_text = 1.7
        clean_title = "Such different data (and coefficients for approximation)"
        dirty_title = "Sometimes it is impossible to approximate data perfectly"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        text_label = "Что за хрень, почему цена уменьшается?"
        x_coord_for_text = 1.0
        clean_title = "Такие разные данные (и коэффициенты для аппроксимации)"
        dirty_title = "Иногда точно провести прямую через все точки невозможно"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, text_label, x_coord_for_text, clean_title, dirty_title


def get_data():
    all_prices = [[11000, 21000, 31000],
                  [19000, 29000, 39000],
                  [20000, 40000, 60000],
                  [15500, 10500, 5500]]
    all_prices_with_noise = [[11000 - 5000, 21000 + 10000, 31000 - 5000],
                             [19000 + 3000, 29000 - 6000, 39000 + 3000],
                             [20000 - 5000, 40000 + 10000, 60000 - 5000],
                             [15500 - 3000, 10500 + 6000, 5500 - 3000]]
    titles = [r"$b_0 = 1000, b_1 = 10000$",
              r"$b_0 = 9000, b_1 = 10000$",
              r"$b_0 = 0, b_1 = 20000$",
              r"$b_0 = 20500, b_1 = -5000$"]
    rooms = [1, 2, 3]

    i = 0
    for rooms, p, p_with_noise in zip([rooms] * 4, all_prices, all_prices_with_noise):
        yield i, rooms, p, p_with_noise, COLORS[i], titles[i]
        i += 1


def plot_data_can_be_different(mode: str = "eng"):
    """ Generate several different datasets """
    # Font and figure settings
    fontname = "Comic Sans MS"
    x_label, y_label, text_label, x_coord_for_text, clean_title, dirty_title = annotations_by_language(mode)
    for with_noise in [True, False]:
        print("Starting generation of the different data plot...")
        fig_size = (21, 5)
        fig, axs = plt.subplots(1, 4, figsize=fig_size)

        for (plot_index, rooms, prices, prices_with_noise, color, title) in get_data():
            ax = axs[plot_index]
            if with_noise:
                ax.scatter(rooms, prices_with_noise, s=80, c=color, edgecolor="black")
            else:
                ax.scatter(rooms, prices, s=80, c=color, edgecolor="black")

            if plot_index == 0:
                ax.set_ylabel(y_label,
                              fontdict={'fontsize': 14, 'fontname': fontname})
            else:
                ax.yaxis.set_ticklabels([])
            ax.set_xlabel(x_label,
                          fontdict={'fontsize': 14, 'fontname': fontname})

            if plot_index == 3:
                ax.text(x_coord_for_text, 60000, text_label,
                        fontsize=12, color='grey',
                        fontname=fontname)

            if with_noise:
                # Trick - find the solution - fit regression model
                regression = LinearRegression()
                regression.fit(np.array(rooms).reshape(-1, 1),
                               np.array(prices_with_noise).reshape(-1, 1))
                prices = regression.predict(np.array(rooms).reshape(-1, 1))

            ax.plot(rooms, prices, '--', c=color, alpha=0.5)
            if with_noise:
                for sample_index in range(0, len(prices)):
                    if prices_with_noise[sample_index] != prices[sample_index, 0]:
                        ax.plot([rooms[sample_index], rooms[sample_index]],
                                [prices_with_noise[sample_index], prices[sample_index, 0]],
                                '-', c="grey", alpha=0.5)

            ax.grid(color='grey', alpha=0.1)
            if with_noise is False:
                ax.set_title(title, color=color,
                             fontdict={'fontsize': 16, 'fontname': fontname})
            ax.set_ylim(0, 65000)

        if with_noise:
            title = dirty_title
        else:
            title = clean_title
        plt.suptitle(title, fontdict={'fontsize': 14, 'fontname': fontname})

        # Show plot
        if with_noise:
            name = f"7_errors_{mode}.svg"
        else:
            name = f"5_data_is_different_clean_{mode}.svg"
        plt.savefig(Path(get_plots_path(), name))
        plt.close()
        print("Different data plot was successfully generated")


if __name__ == '__main__':
    plot_data_can_be_different("rus")
    plot_data_can_be_different("eng")
