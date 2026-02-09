from pathlib import Path
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, get_extended_dataset, take_sample_manual

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def show_normalization():
    np.set_printoptions(precision=2, suppress=True)

    features = np.array([[1, 10],
                         [2, 20],
                         [3, 505]])
    min_max = MinMaxScaler()
    transformed = min_max.fit_transform(np.copy(features))
    standard_scaler = StandardScaler()
    transformed_sc = standard_scaler.fit_transform(np.copy(features))

    print("Training sample. MinMaxScaler")
    print(transformed.round(2))
    print("Training sample. StandardScaler")
    print(transformed_sc.round(2))

    new_features = np.array([[0, 5],
                             [2, 20],
                             [100, 30]])
    print("New data. MinMaxScaler")
    new_transformed = min_max.transform(np.copy(new_features))
    print(new_transformed.round(2))

    print("New data. StandardScaler")
    new_transformed = standard_scaler.transform(np.copy(new_features))
    print(new_transformed.round(2))


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        data_title = ""
        x_label = ""
        y_label = ""
        rooms_label = ""
        metro_distance = ""
        min_max_title = ""
        standard_scaling_title = ""
    elif mode == "rus":
        title = "Результаты нормализации и стандартизации"
        data_title = "Исходные признаки"
        x_label = "Индекс наблюдения в датасете"
        y_label = "Значение признака"
        rooms_label = "Количество комнат"
        metro_distance = "Расстояние до станции метро"
        min_max_title = "Min-Max нормализация"
        standard_scaling_title = "Стандартизация"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, data_title, x_label, y_label, rooms_label, metro_distance, min_max_title, standard_scaling_title



def plot_normalization_effect(mode: str = "eng"):
    (title, data_title, x_label, y_label, rooms_label, metro_distance,
     min_max_title, standard_scaling_title) = annotations_by_language(mode)
    x_ticks = [1, 5, 10, 15]

    dataset = get_extended_dataset()
    features_names = ["rooms", "metro_distance"]
    features = np.array(dataset[features_names])
    target = np.array(dataset["price"])
    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    fig_size = (18, 9)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)
    for ax in [axs[0, 0], axs[0, 2]]:
        ax.cla()  # Clear axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    axs[1, 1].set_visible(False)

    axs[0, 1].plot(np.arange(len(x)), x[:, 0], c='red', zorder=2, label=rooms_label)
    axs[0, 1].plot(np.arange(len(x)), x[:, 1], c='blue', zorder=3, label=metro_distance)
    axs[0, 1].set_title(data_title, fontdict=FONTDICT, y=1.05)
    axs[0, 1].set_xlabel(x_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0, 1].set_ylabel(y_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 1].grid(alpha=0.5, zorder=1)
    axs[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), prop={'family': FONTNAME, 'size': 14})

    min_max = MinMaxScaler()
    transformed = min_max.fit_transform(np.copy(x))
    axs[1, 0].plot(np.arange(len(x)), transformed[:, 0], c='red', zorder=2)
    axs[1, 0].plot(np.arange(len(x)), transformed[:, 1], c='blue', zorder=3)
    axs[1, 0].set_xlabel(x_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 0].set_ylabel(y_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 0].set_xticks(x_ticks)
    axs[1, 0].set_title(min_max_title, fontdict=FONTDICT, y=1.05)
    axs[1, 0].grid(alpha=0.5, zorder=1)

    standard_scaler = StandardScaler()
    transformed_sc = standard_scaler.fit_transform(np.copy(x))
    axs[1, 2].plot(np.arange(len(x)), transformed_sc[:, 0], c='red', zorder=2)
    axs[1, 2].plot(np.arange(len(x)), transformed_sc[:, 1], c='blue', zorder=3)
    axs[1, 2].set_xlabel(x_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 2].set_ylabel(y_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    axs[1, 2].set_xticks(x_ticks)
    axs[1, 2].set_title(standard_scaling_title, fontdict=FONTDICT, y=1.05)
    axs[1, 2].grid(alpha=0.5, zorder=1)

    # Vertical offset (5% upwards) for some axes
    shift = 0.05
    for ax_to_move in [axs[1, 0], axs[1, 2]]:
        pos = ax_to_move.get_position()
        ax_to_move.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

    # Vertical offset (5% downwards) for some axes
    shift = 0.1
    for ax_to_move in [axs[0, 1]]:
        pos = ax_to_move.get_position()
        ax_to_move.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    fig.suptitle(title, fontsize=18, fontdict={'fontname': FONTNAME}, va="top", y=0.92)

    raw_svg_file = Path(get_plots_path(), f"44_normalization_{mode}.svg")
    final_plot = Path(get_plots_path(), f"44_normalization_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_small.svg")


if __name__ == "__main__":
    show_normalization()
    plot_normalization_effect("rus")
