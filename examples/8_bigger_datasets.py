from pathlib import Path

import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import get_datasets


def annotations_by_language(mode: str):
    if mode == "eng":
        table_data = [["Number of rooms", "Price, $"]]
        x_label = "Number of rooms"
        y_label = "Price, $"
        title = "Apartment price vs. number of rooms"
    elif mode == "rus":
        table_data = [["Количество комнат в квартире", "Стоимость, $"]]
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Зависимость стоимости квартиры от количества комнат"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return table_data, x_label, y_label, title


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {'fontsize': 14, 'fontname': fontname}
    table_data, x_label, y_label, title = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig_size = (16, 5)
    fig, axs = plt.subplots(1, 3, figsize=fig_size)

    axs[0].scatter(rooms, good_prices, s=80, c="green", edgecolor="black")
    axs[0].grid(color='grey', alpha=0.1)
    axs[0].set_ylim(0, 65000)
    axs[0].set_ylabel(y_label, fontdict=fontdict)
    axs[0].set_xlabel(x_label, fontdict=fontdict)
    axs[0].set_title("A", fontdict=fontdict)

    axs[1].scatter(rooms, bad_prices_first, s=80, c="orange", edgecolor="black")
    axs[1].grid(color='grey', alpha=0.1)
    axs[1].set_ylim(0, 65000)
    axs[1].yaxis.set_ticklabels([])
    axs[1].set_xlabel(x_label, fontdict=fontdict)
    axs[1].set_title("B", fontdict=fontdict)

    axs[2].scatter(rooms, bad_prices_second, s=80, c="#ff8484",
                   edgecolor="black")
    axs[2].grid(color='grey', alpha=0.1)
    axs[2].set_ylim(0, 65000)
    axs[2].yaxis.set_ticklabels([])
    axs[2].set_xlabel(x_label, fontdict=fontdict)
    axs[2].set_title("C", fontdict=fontdict)

    plt.savefig(Path(get_plots_path(), f"8_datasets_{mode}.svg"))
    plt.close()


if __name__ == '__main__':
    plot_errors_plots("rus")
    plot_errors_plots("eng")
