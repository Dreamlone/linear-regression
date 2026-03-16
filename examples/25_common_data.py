from pathlib import Path

import matplotlib.pyplot as plt

from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of rooms"
        y_label = "Price, $"
        title = "All available data D. The statistical population including datasets A, B, and C"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Генеральная совокупность состоящая из датасетов A, B и C"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, title


def plot_all_datasets_in_one(mode: str = "eng"):
    """
    Generate the plot with linear regression line and equation

    To generate english plot choose mode "eng"
    To generate russian - "rus"
    """
    x_label, y_label, title = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    # Font and figure settings
    fontname = "Comic Sans MS"
    fig_size = (10, 6)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(rooms, good_prices, color='#AA8C00', edgecolor="black", alpha=0.5, s=50)
    ax.scatter(rooms, bad_prices_first, color='#AA8C00', edgecolor="black", alpha=0.5, s=50)
    ax.scatter(rooms, bad_prices_second, color='#AA8C00', edgecolor="black", alpha=0.5, s=50)
    ax.set_xlabel(x_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax.set_ylabel(y_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([10000, 20000, 30000, 40000, 50000])
    ax.grid(color='grey', alpha=0.1)
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(get_plots_path(), f"25_common_data_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"25_common_data_{mode}.png"))


if __name__ == '__main__':
    plot_all_datasets_in_one("rus")
    plot_all_datasets_in_one("eng")
