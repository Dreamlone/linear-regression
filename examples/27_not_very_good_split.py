from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import legend
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template, take_sample_manual

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def annotations_by_language(mode: str, apply_distortion: bool):
    if mode == "eng":
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        if apply_distortion:
            title = ""
            distorted_label = ""
        else:
            title = ""
            distorted_label = None
        all_label = ""
        sample_label = ""
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        if apply_distortion:
            title = "А еще мы можем получить искаженные данные,\nнапример из-за технических ошибок или человеческого фактора"
            distorted_label = "Искаженные значения"
        else:
            title = "Когда с семплом не повезло (несмотря на его размер)"
            distorted_label = None
        all_label = "Все данные"
        sample_label = "Семпл"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, title, all_label, sample_label, distorted_label


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


def _split_train_test(features: np.array, target: np.array, test_size: float = 0.4):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=14 * 20)

    return x_train, x_test, y_train, y_test


def plot_all_datasets_in_one(mode: str = "eng", apply_distortion: bool = False):
    """
    Generate the plot with linear regression line and equation

    To generate english plot choose mode "eng"
    To generate russian - "rus"
    """
    x_label, y_label, title, all_label, sample_label, distorted_label = annotations_by_language(mode, apply_distortion)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concat([rooms, rooms, rooms])
    common_target = np.concat([good_prices, bad_prices_first, bad_prices_second])

    x_sampled, y_sampled, distorted_x, distorted_y = take_sample_manual(common_features, common_target, apply_distortion)
    predicted, intercept, slope = _get_predicted(x_sampled, y_sampled)

    print("--- METRICS ---")
    rmse_metric = root_mean_squared_error(y_sampled, predicted)
    print(f"RMSE (actual vs predicted): {rmse_metric:.3f}")
    # Font and figure settings
    fontname = "Comic Sans MS"
    fig_size = (10, 6)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(common_features, common_target, color='white', edgecolor="black", alpha=1, s=50, zorder=1,
               label=all_label)
    ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black', zorder=1, alpha=0.3)
    ax.scatter(x_sampled, y_sampled, color='#ff8787', edgecolor="black", s=50, zorder=2, label=sample_label)
    ax.plot([min(x_sampled), max(x_sampled)], [min(predicted), max(predicted)], '--',
            color='red', alpha=0.4, zorder=2)
    if apply_distortion:
        ax.scatter(distorted_x, distorted_y, color='white', edgecolor="red", s=200, zorder=1, label=distorted_label)

    ax.set_xlabel(x_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax.set_ylabel(y_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    legend = ax.legend(fontsize=16)
    legend.get_title().set_fontname(FONTNAME)
    for text in legend.get_texts():
        text.set_fontname(FONTNAME)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([10000, 20000, 30000, 40000, 50000])
    ax.set_ylim(0, 70000)
    ax.grid(color='grey', alpha=0.1)
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})

    if apply_distortion:
        raw_svg_file = Path(get_plots_path(), f"27_not_good_sample_distorted_{mode}.svg")
        final_plot = Path(get_plots_path(), f"27_not_good_sample_distorted_{mode}.png")
    else:
        raw_svg_file = Path(get_plots_path(), f"27_not_good_sample_{mode}.svg")
        final_plot = Path(get_plots_path(), f"27_not_good_sample_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot)


if __name__ == '__main__':
    plot_all_datasets_in_one("rus", True)
    plot_all_datasets_in_one("rus")
