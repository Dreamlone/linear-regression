from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template


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
                    hide_y_ticks: bool = False):
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

    return residuals


def _plot_hist(ax, residuals, y_label, fontdict, color, hide_y_ticks: bool = False):
    kde = stats.gaussian_kde(residuals)

    n_bins = int(np.ceil(2 * len(residuals) ** (1 / 3)))
    ax.hist(residuals, density=True, range=(-8000, 8000),
            alpha=0.5, rwidth=0.9, bins=n_bins,
            color=color, orientation='horizontal')
    xx = np.linspace(-8000, 8000, 1000)
    ax.plot([0, 0.0008], [0, 0], '--', color="black", alpha=0.3)
    ax.plot(kde(xx), xx, color=color, linewidth=1.5)
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, 0.0008)

    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_label, fontdict=fontdict)

    ax.set_xticks([0, 0.0004, 0.0008])


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Predicted"
        y_label = "Residuals (actual - predicted)"
        hist_label = "Frequency"
        title = "Visual evaluation of model quality\nResidual plot and residuals distribution"
    elif mode == "rus":
        x_label = "Предсказания"
        y_label = "Остатки (реальные - предсказанные)"
        hist_label = "Частота"
        title = "Визуальная оценка качества моделирования\nГрафик остатков вместе с частотным распределением"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, hist_label, title


def plot_errors_plots(mode: str = "eng"):
    fontname = "Comic Sans MS"
    fontdict = {'fontsize': 14, 'fontname': fontname}
    x_label, y_label, hist_label, title = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig_size = (16, 10)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)
    fig.suptitle(title, fontsize=20, fontdict={'fontname': fontname}, va="top", y=1.02)

    predicted_good = _get_predicted(rooms, good_prices)
    residuals = _plot_residuals(axs[0, 0], good_prices, predicted_good, x_label,
                                fontdict, "green")
    axs[0, 0].set_ylabel(y_label, fontdict=fontdict)
    axs[0, 0].set_title("A", fontsize=20, fontdict={'fontname': fontname})
    _plot_hist(axs[1, 0], residuals, y_label, fontdict, "green")
    axs[1, 0].set_xlabel(hist_label, fontdict=fontdict)

    predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    residuals = _plot_residuals(axs[0, 1], bad_prices_first, predicted_bad_first, x_label,
                    fontdict, "orange", True)
    axs[0, 1].set_title("B", fontsize=20, fontdict={'fontname': fontname})
    _plot_hist(axs[1, 1], residuals, y_label, fontdict, "orange", True)
    axs[1, 1].set_xlabel(hist_label, fontdict=fontdict)

    predicted_bad_second = _get_predicted(rooms, bad_prices_second)
    residuals = _plot_residuals(axs[0, 2], bad_prices_second, predicted_bad_second, x_label,
                    fontdict, "#ff8484", True)
    axs[0, 2].set_title("C", fontsize=20, fontdict={'fontname': fontname})
    _plot_hist(axs[1, 2], residuals, y_label, fontdict, "#ff8484", True)
    axs[1, 2].set_xlabel(hist_label, fontdict=fontdict)

    raw_svg_file = Path(get_plots_path(), f"13_raw_datasets_residuals_kde_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"13_datasets_residuals_kde_{mode}.png"))


if __name__ == '__main__':
    plot_errors_plots("rus")
    plot_errors_plots("eng")