from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import get_datasets


def _get_predicted(rooms: np.array, actual_prices: np.array):
    """ Build the model using analytical solution for one feature model """
    mean_x = np.mean(rooms)  # Average number of rooms
    mean_y = np.mean(actual_prices)  # Average price

    numerator = np.sum((rooms - mean_x) * (actual_prices - mean_y))
    denominator = np.sum((rooms - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    print(f"Model b0 + b1 * x: {intercept} + {slope} * x")
    predicted_prices = [(intercept + slope * room) for room in rooms]
    return np.array(predicted_prices)


def annotations_by_language(mode: str):
    if mode == "eng":
        table_data = [["Rooms in the apartment", "Price, $"]]
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        title = "Dependence of apartment price on room number"
        actual = "Actual"
        predicted = "Predicted"
    elif mode == "rus":
        table_data = [["Количество комнат в квартире", "Стоимость, $"]]
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Зависимость стоимости квартиры от количества комнат"
        actual = "Реальные значения"
        predicted = "Предсказанные"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return table_data, x_label, y_label, title, actual, predicted


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {'fontsize': 14, 'fontname': fontname}
    table_data, x_label, y_label, title, actual, predicted = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig_size = (16, 5)
    fig, axs = plt.subplots(1, 3, figsize=fig_size)

    axs[0].scatter(rooms, good_prices, s=80, c="green",
                   edgecolor="black", alpha=0.4, label=actual)
    axs[0].grid(color='grey', alpha=0.1)
    axs[0].set_ylim(0, 65000)
    predicted_good = _get_predicted(rooms, good_prices)
    axs[0].plot(rooms, predicted_good, '--', c="black")
    axs[0].scatter(rooms, predicted_good, c="black", s=75,
                   marker='x', label=predicted)
    axs[0].set_ylabel(y_label, fontdict=fontdict)
    axs[0].set_xlabel(x_label, fontdict=fontdict)
    axs[0].set_title("A", fontdict=fontdict)
    axs[0].legend(loc='lower right')

    axs[1].scatter(rooms, bad_prices_first, s=80, c="orange",
                   edgecolor="black", alpha=0.4, label=actual)
    axs[1].grid(color='grey', alpha=0.1)
    axs[1].set_ylim(0, 65000)
    axs[1].yaxis.set_ticklabels([])
    predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    axs[1].plot(rooms, predicted_bad_first, '--', c="black")
    axs[1].set_xlabel(x_label, fontdict=fontdict)
    axs[1].scatter(rooms, predicted_bad_first, s=75, c="black",
                   marker='x', label=predicted)
    axs[1].set_title("B", fontdict=fontdict)
    axs[1].legend(loc='lower right')

    axs[2].scatter(rooms, bad_prices_second, s=80, c="#ff8484",
                   edgecolor="black", alpha=0.4, label=actual)
    axs[2].grid(color='grey', alpha=0.1)
    axs[2].set_ylim(0, 65000)
    axs[2].yaxis.set_ticklabels([])
    predicted_bad_second = _get_predicted(rooms, bad_prices_second)
    axs[2].plot(rooms, predicted_bad_second, '--', c="black")
    axs[2].set_xlabel(x_label, fontdict=fontdict)
    axs[2].scatter(rooms, predicted_bad_second, s=75, c="black",
                   marker='x', label=predicted)
    axs[2].set_title("C", fontdict=fontdict)
    axs[2].legend(loc='lower right')

    plt.savefig(Path(get_plots_path(), f"7_datasets_predicted_{mode}.svg"))
    plt.close()


if __name__ == '__main__':
    plot_errors_plots("rus")
