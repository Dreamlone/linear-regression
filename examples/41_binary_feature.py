from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from examples.paths import get_plots_path

FONTNAME = "Comic Sans MS"


def draw_binary_regression(features: np.array, target: np.array, file_name: str):
    model = LinearRegression()
    model.fit(features, target)

    print('Binary regression')
    print(model.intercept_)
    print(model.coef_)

    fig_size = (12, 6)
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    axs[0].scatter(features[:,1], target, c='grey', alpha=0.3, s=150, zorder=2)
    axs[0].scatter(features[:2, 1], target[:2, :], c='black', s=150, zorder=3)
    axs[0].plot(features[:2, 1], np.ravel(model.predict(features[:2, :])), '--', c='black', zorder=2)
    axs[0].grid(alpha=0.3, zorder=1)
    axs[0].set_xlim(0, 4.5)
    axs[0].set_ylim(0, 4.5)
    axs[0].set_xticks([0, 1, 2, 3, 4])
    axs[0].set_xticklabels([0, 1, 2, 3, 4])
    axs[0].set_yticks([0, 1, 2, 3, 4])
    axs[0].set_yticklabels([0, 1, 2, 3, 4])

    axs[1].scatter(features[:, 1], target, c='grey', alpha=0.3, s=150, zorder=2)
    axs[1].scatter(features[2:, 1], target[2:, :], c='black', s=150, zorder=3)
    axs[1].plot(features[2:, 1], np.ravel(model.predict(features[2:, :])), '--', c='black', zorder=2)
    axs[1].grid(alpha=0.3, zorder=1)
    axs[1].set_xlim(0, 4.5)
    axs[1].set_ylim(0, 4.5)
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].set_xticklabels([0, 1, 2, 3, 4])
    axs[1].set_yticks([0, 1, 2, 3, 4])
    axs[1].set_yticklabels([0, 1, 2, 3, 4])

    for ax in [axs[0], axs[1]]:
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)

    raw_svg_file = Path(get_plots_path(), file_name)
    plt.savefig(raw_svg_file)
    plt.close()


def draw_pair_regression(features: np.array, target: np.array, file_name: str):
    # Take only one column as a feature
    features = features[:, 1]
    model = LinearRegression()
    model.fit(features.reshape(-1, 1), target)

    print('Pair regression')
    print(model.intercept_)
    print(model.coef_)

    fig_size = (6, 6)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    ax.scatter(features, target, c='black', s=150, zorder=3)
    ax.plot(features, np.ravel(model.predict(features.reshape(-1, 1))), '--', c='black', zorder=2)
    ax.grid(alpha=0.3, zorder=1)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels([0, 1, 2, 3, 4])

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(16)
        tick_label.set_fontname(FONTNAME)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(16)
        tick_label.set_fontname(FONTNAME)

    raw_svg_file = Path(get_plots_path(), file_name)
    plt.savefig(raw_svg_file)
    plt.close()


def draw_binary_regression_models():
    features_simple = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_simple = np.array([[1], [3], [2], [4]])
    features_harder = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_harder = np.array([[1], [3], [2], [3]])
    features_hard = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_hard = np.array([[1], [3], [3], [1]])

    all_features = [features_simple, features_harder, features_hard]
    all_targets = [target_simple, target_harder, target_hard]
    for features, target, file_name in zip(all_features, all_targets,
                                           ["41_binary_simple.svg", "41_binary_harder.svg", "41_binary_hard.svg"]):
        print(file_name)
        draw_binary_regression(features, target, file_name)
        draw_pair_regression(features, target, file_name.replace(".svg", "_one_feature.svg"))


if __name__ == '__main__':
    draw_binary_regression_models()