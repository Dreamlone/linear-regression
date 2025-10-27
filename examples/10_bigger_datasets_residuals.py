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


def _plot_residuals(ax, actual_v, predicted_v, x_label, fontdict, color,
                    hide_y_ticks: bool = False) -> None:
    actual_v = np.ravel(actual_v)
    predicted_v = np.ravel(predicted_v)
    residuals = actual_v - predicted_v
    ax.scatter(predicted_v, residuals, s=80, c=color, edgecolor="black",
               alpha=0.4)
    ax.plot([0, 65000], [0, 0], '--', color="black", alpha=0.3)

    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, 65000)
    ax.set_xlabel(x_label, fontdict=fontdict)

    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Predicted"
        y_label = "Residuals (actual - predicted)"
    elif mode == "rus":
        x_label = "Предсказания"
        y_label = "Остатки (реальные - предсказанные)"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {'fontsize': 14, 'fontname': fontname}
    x_label, y_label = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig_size = (16, 5)
    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    predicted_good = _get_predicted(rooms, good_prices)
    _plot_residuals(axs[0], good_prices, predicted_good, x_label, fontdict, "green")
    axs[0].set_ylabel(y_label, fontdict=fontdict)
    axs[0].set_title("A", fontdict=fontdict)

    predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    _plot_residuals(axs[1], bad_prices_first, predicted_bad_first, x_label,
                    fontdict, "orange", True)
    axs[1].set_title("B", fontdict=fontdict)

    predicted_bad_second = _get_predicted(rooms, bad_prices_second)
    _plot_residuals(axs[2], bad_prices_second, predicted_bad_second, x_label,
                    fontdict, "#ff8484", True)
    axs[2].set_title("C", fontdict=fontdict)

    plt.savefig(Path(get_plots_path(), f"10_datasets_residuals_{mode}.svg"))
    plt.close()


if __name__ == '__main__':
    plot_errors_plots("rus")
