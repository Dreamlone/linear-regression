from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from examples.paths import get_plots_path
from examples.utils import get_datasets


def _get_predicted(rooms: np.array, actual_prices: np.array):
    regression = LinearRegression()
    regression.fit(np.array(rooms).reshape(-1, 1),
                   np.array(actual_prices).reshape(-1, 1))
    predicted_prices = regression.predict(np.array(rooms).reshape(-1, 1))

    b0 = regression.intercept_[0]
    b1 = np.ravel(regression.coef_)[0]
    print(f"Model b0 + b1 * x: {b0} + {b1} * x")
    return predicted_prices


def _apply_common_style(ax, x_label, fontdict, color, hide_y_ticks: bool) -> None:
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(0, 65000)
    ax.set_xlim(0, 65000)
    ax.set_xlabel(x_label, fontdict=fontdict)

    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])


def _plot_qq_order_stats(ax, array_x, array_y, x_label, fontdict, color,
                        hide_y_ticks: bool = False) -> None:
    """
    Q–Q plot using order statistics (sorted values):
    points are (sorted_x[i], sorted_y[i]).
    """
    x_sorted = np.sort(np.ravel(array_x))
    y_sorted = np.sort(np.ravel(array_y))

    ax.scatter(x_sorted, y_sorted, s=25, c=color, alpha=0.6)
    ax.plot([0, 60000], [0, 60000], color="black", alpha=0.5)
    _apply_common_style(ax, x_label, fontdict, color, hide_y_ticks)


def _plot_qq(ax, array_x, array_y, x_label, fontdict, color,
             hide_y_ticks: bool = False) -> None:
    """
    Q–Q plot using sample quantiles on a fixed grid (0..99 percentiles).
    """
    percs = np.arange(1, 100)
    qn_x = np.percentile(array_x, percs)
    qn_y = np.percentile(array_y, percs)

    ax.scatter(qn_x, qn_y, s=25, c=color, alpha=0.6)
    ax.plot([0, 60000], [0, 60000], color="black", alpha=0.5)

    # Put labels for 25, 50 and 75 percentiles
    for i in [25, 50, 75]:
        ax.text(qn_x[i - 1], qn_y[i - 1] - 5000, i)
        ax.scatter([qn_x[i - 1]], [qn_y[i - 1]], s=70, c=color, edgecolor="black")

    _apply_common_style(ax, x_label, fontdict, color, hide_y_ticks)


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label_sorted = "Actual prices (sorted)"
        y_label_sorted = "Predicted prices (sorted)"
        x_label_quant = "Actual price quantiles"
        y_label_quant = "Predicted price quantiles"
    elif mode == "rus":
        x_label_sorted = "Реальная стоимость,\nупорядоченные значения"
        y_label_sorted = "Предсказанная стоимость,\nупорядоченные значения"
        x_label_quant = "Квантили,\nреальная стоимость"
        y_label_quant = "Квантили,\nпредсказанная стоимость"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")

    return x_label_sorted, y_label_sorted, x_label_quant, y_label_quant


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {'fontsize': 14, 'fontname': fontname}

    x_label_sorted, y_label_sorted, x_label_quant, y_label_quant = annotations_by_language(mode)
    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig_size = (16, 10)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)

    # Column 1: good dataset
    predicted_good = _get_predicted(rooms, good_prices)
    _plot_qq_order_stats(axs[0, 0], good_prices, predicted_good, x_label_sorted, fontdict, "green", False)
    axs[0, 0].set_ylabel(y_label_sorted, fontdict=fontdict)
    axs[0, 0].set_title("A", fontdict=fontdict)

    _plot_qq(axs[1, 0], good_prices, predicted_good, x_label_quant, fontdict, "green", False)
    axs[1, 0].set_ylabel(y_label_quant, fontdict=fontdict)

    # Column 2: bad dataset 1
    predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    _plot_qq_order_stats(axs[0, 1], bad_prices_first, predicted_bad_first, x_label_sorted, fontdict, "orange", True)
    axs[0, 1].set_title("B", fontdict=fontdict)

    _plot_qq(axs[1, 1], bad_prices_first, predicted_bad_first, x_label_quant, fontdict, "orange", True)

    # Column 3: bad dataset 2
    predicted_bad_second = _get_predicted(rooms, bad_prices_second)
    _plot_qq_order_stats(axs[0, 2], bad_prices_second, predicted_bad_second, x_label_sorted, fontdict, "#ff8484", True)
    axs[0, 2].set_title("C", fontdict=fontdict)

    _plot_qq(axs[1, 2], bad_prices_second, predicted_bad_second, x_label_quant, fontdict, "#ff8484", True)

    plt.savefig(Path(get_plots_path(), f"11_datasets_qq_plot_{mode}.svg"))
    plt.close()


if __name__ == '__main__':
    plot_errors_plots("rus")
    plot_errors_plots("eng")