from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.paths import get_plots_path
from examples.utils import get_datasets, symmetric_mean_absolute_percentage_error, save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


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


def _plot_predicted_with_actual(ax, rooms, actual, predicted, y_label, x_label, dataset_name):
    """ Draw a simple predicted and actual values plot """
    ax.scatter(rooms, actual, s=80, c="grey", alpha=0.4, edgecolor="black")
    ax.plot(rooms, predicted, '--', c="black")
    ax.scatter(rooms, predicted, c="black", s=75, marker='x')
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(0, 65000)
    if y_label is None:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_label, fontdict=FONTDICT)
    ax.set_xlabel(x_label, fontdict=FONTDICT)
    ax.set_title(dataset_name, fontsize=20, fontdict={'fontname': FONTNAME})


def _plot_metrics(ax, actual: np.array, predicted: np.array, dataset_name: str, good_model: str, bad_model: str):
    """ Metrics visualization """
    mape_metric = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"{dataset_name}. MAPE (actual vs predicted): {mape_metric:.3f}")

    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    if mape_metric < 5:
        label = good_model
        color = "green"
    else:
        label = bad_model
        color = "red"
    ax.text(0.5, 0.5, f'MAPE: {round(mape_metric, 1)}\n{label}',
            transform=ax.transAxes,
            ha='center', va='center', color=color,
            fontsize=20, fontdict=FONTDICT)


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Metrics are cool!"
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        y_label_metric = "Is model good or not?"
        good_model = "Good!"
        bad_model = "Bad"
    elif mode == "rus":
        title = "Метрики - это удобно!"
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        y_label_metric = "Хорошая модель или нет?"
        good_model = "Хорошо!"
        bad_model = "Не очень"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label, y_label, y_label_metric, good_model, bad_model


def plot_why_metrics_are_good(mode: str = "eng"):
    title, x_label, y_label, y_label_metric, good_model, bad_model = annotations_by_language(mode)

    # Get datasets and build models
    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    predicted_good = _get_predicted(rooms, good_prices)
    predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    predicted_bad_second = _get_predicted(rooms, bad_prices_second)

    fig_size = (15, 10)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)

    _plot_predicted_with_actual(axs[0, 0], rooms, good_prices, predicted_good, y_label, x_label, "A")
    # Add name for this group of subplots
    axs[1, 0].set_ylabel(y_label_metric, fontdict=FONTDICT)
    _plot_metrics(axs[1, 0], good_prices, predicted_good, "A", good_model, bad_model)
    _plot_predicted_with_actual(axs[0, 1], rooms, bad_prices_first, predicted_bad_first,
                                None, x_label, "B")
    _plot_metrics(axs[1, 1], bad_prices_first, predicted_bad_first, "B", good_model, bad_model)
    _plot_predicted_with_actual(axs[0, 2], rooms, bad_prices_second, predicted_bad_second,
                                None, x_label, "C")
    _plot_metrics(axs[1, 2], bad_prices_second, predicted_bad_second, "C", good_model, bad_model)

    raw_svg_file = Path(get_plots_path(), f"12_raw_why_metrics_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"12_why_metrics_{mode}.png"))


if __name__ == '__main__':
    plot_why_metrics_are_good("rus")
