from pathlib import Path

from matplotlib.gridspec import GridSpec
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template, COLOR_BY_DATASET

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
    return np.array(predicted_prices), intercept, slope


def _plot_predicted_with_actual(ax, rooms, actual, predicted, y_label, x_label, dataset_name,
                                prediction_interval_lower: np.array, prediction_interval_upper: np.array,
                                show_name: bool = True, show_x_label: bool = True):
    """ Draw a simple predicted and actual values plot """
    ax.scatter(rooms, actual, s=70, c=COLOR_BY_DATASET[dataset_name], alpha=1.0, edgecolor="black", zorder=3)
    ax.plot(rooms, predicted, '--', c="black", zorder=3)
    ax.grid(color='grey', alpha=0.1, zorder=2)
    ax.set_ylim(0, 65000)

    ax.fill_between(rooms, prediction_interval_lower, prediction_interval_upper, color="red", alpha=0.1, zorder=1)
    ax.plot(rooms, prediction_interval_lower, color="red", linewidth=1, alpha=0.5, zorder=1)
    ax.plot(rooms, prediction_interval_upper, color="red", linewidth=1, alpha=0.5, zorder=1)

    if y_label is None:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_label, fontsize=12, fontdict=FONTDICT)

    if show_x_label:
        ax.set_xlabel(x_label, fontsize=12, fontdict=FONTDICT)
    else:
        ax.xaxis.set_ticklabels([])

    if show_name:
        ax.set_title(dataset_name, fontsize=18, fontdict={'fontname': FONTNAME})


def _compute_prediction_interval(
    x_train: np.array,
    y_train: np.array,
    intercept: float,
    slope: float,
    confidence_level: float,
):
    """
    Compute prediction interval for simple linear regression (one feature).
    Interval is based on training data only.
    """
    x_train = np.ravel(x_train)
    y_train = np.ravel(y_train)

    predicted_train = intercept + slope * x_train
    residuals = y_train - predicted_train
    sample_size = len(x_train)
    degrees_of_freedom = sample_size - 2

    # Residual standard error
    residual_variance = np.sum(residuals ** 2) / degrees_of_freedom
    residual_std = np.sqrt(residual_variance)

    # Geometry in x
    mean_x_train = np.mean(x_train)
    sum_squares_x = np.sum((x_train - mean_x_train) ** 2)

    # t critical value for two-sided interval
    alpha_level = 1.0 - confidence_level
    t_critical = stats.t.ppf(1.0 - alpha_level / 2.0, df=degrees_of_freedom)

    predicted_all = intercept + slope * x_train

    standard_error_prediction = residual_std * np.sqrt(
        1.0
        + 1.0 / sample_size
        + (x_train - mean_x_train) ** 2 / sum_squares_x
    )

    margin = t_critical * standard_error_prediction

    lower_bound = predicted_all - margin
    upper_bound = predicted_all + margin

    return lower_bound, upper_bound


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of rooms"
        y_label = "Price, $"
        title = "Prediction intervals for models A, B & C"
        confidence_label = "Confidence levels "
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Предсказательные интервалы для разных моделей"
        confidence_label = "Уровень доверия"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (
        x_label,
        y_label,
        title, confidence_label
    )


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        0.0,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 12, "fontname": FONTNAME},
    )


def plot_prediction_intervals(mode: str = "eng"):
    """ Shows prediction intervals for A, B and C models with different confidence levels """
    x_label, y_label, title, confidence_label = annotations_by_language(mode)

    # Get datasets and build models
    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    predicted_good, good_intercept, good_slope = _get_predicted(rooms, good_prices)
    predicted_bad_first, bad_first_intercept, bad_first_slope = _get_predicted(rooms, bad_prices_first)
    predicted_bad_second, bad_second_intercept, bad_second_slope  = _get_predicted(rooms, bad_prices_second)

    fig_size = (16, 14)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(3, 3, figure=fig)
    gs.update(hspace=0.1)

    show_name = True
    show_x_label = False
    for row_id, confidence_level in zip([0, 1, 2], [0.90, 0.95, 0.99]):
        if row_id != 0:
            show_name = False
        if row_id == 2:
            show_x_label = True

        add_row_label(fig, gs, row_index=row_id, text=f"{confidence_label}: {confidence_level}")

        predicted_good_lower, predicted_good_upper = _compute_prediction_interval(
            x_train=rooms,
            y_train=good_prices,
            intercept=good_intercept,
            slope=good_slope,
            confidence_level=confidence_level,
        )

        _plot_predicted_with_actual(fig.add_subplot(gs[row_id, 0]), rooms, good_prices, predicted_good,
                                    y_label, x_label, "A", predicted_good_lower,
                                    predicted_good_upper, show_name, show_x_label)

        predicted_bad_first_lower, predicted_bad_first_upper = _compute_prediction_interval(
            x_train=rooms,
            y_train=bad_prices_first,
            intercept=bad_first_intercept,
            slope=bad_first_slope,
            confidence_level=confidence_level,
        )

        _plot_predicted_with_actual(fig.add_subplot(gs[row_id, 1]), rooms, bad_prices_first, predicted_bad_first,
                                    None, x_label, "B", predicted_bad_first_lower,
                                    predicted_bad_first_upper, show_name, show_x_label)

        predicted_bad_second_lower, predicted_bad_second_upper = _compute_prediction_interval(
            x_train=rooms,
            y_train=bad_prices_second,
            intercept=bad_second_intercept,
            slope=bad_second_slope,
            confidence_level=confidence_level,
        )

        _plot_predicted_with_actual(fig.add_subplot(gs[row_id, 2]), rooms, bad_prices_second, predicted_bad_second,
                                    None, x_label, "C", predicted_bad_second_lower,
                                    predicted_bad_second_upper, show_name, show_x_label)
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME}, y=0.96)

    raw_svg_file = Path(get_plots_path(), f"22_prediction_intervals_datasets_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(
        raw_svg_file,
        Path(get_plots_path(), f"22_prediction_intervals_datasets_{mode}.png"),
    )


if __name__ == "__main__":
    plot_prediction_intervals("rus")
    plot_prediction_intervals("eng")
