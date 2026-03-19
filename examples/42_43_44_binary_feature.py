from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from examples.paths import get_plots_path

FONTNAME = "Comic Sans MS"


def add_interaction_term(f: np.array):
    """ Add new feature (column) to the source table """
    new_column = f[:, 0] * f[:, 1]
    extended_features = np.hstack([f, new_column.reshape(-1, 1)])
    return extended_features


def draw_binary_regression(features: np.array, target: np.array, ax_1, ax_2):
    model = LinearRegression()
    model.fit(features, target)

    print('Binary regression')
    print(model.intercept_)
    print(model.coef_)

    ax_1.scatter(features[:,1], target, c='grey', alpha=0.3, s=150, zorder=2)
    ax_1.scatter(features[:2, 1], target[:2, :], c='red', s=150, zorder=3)
    ax_1.plot(features[:2, 1], np.ravel(model.predict(features[:2, :])), '--', c='red', zorder=2)
    ax_1.grid(alpha=0.3, zorder=1)
    ax_1.set_xlim(0, 4.5)
    ax_1.set_ylim(0, 4.5)
    ax_1.set_xticks([0, 1, 2, 3, 4])
    ax_1.set_xticklabels([0, 1, 2, 3, 4])
    ax_1.set_yticks([0, 1, 2, 3, 4])
    ax_1.set_yticklabels([0, 1, 2, 3, 4])

    ax_2.scatter(features[:, 1], target, c='grey', alpha=0.3, s=150, zorder=2)
    ax_2.scatter(features[2:, 1], target[2:, :], c='blue', s=150, zorder=3)
    ax_2.plot(features[2:, 1], np.ravel(model.predict(features[2:, :])), '--', c='blue', zorder=2)
    ax_2.grid(alpha=0.3, zorder=1)
    ax_2.set_xlim(0, 4.5)
    ax_2.set_ylim(0, 4.5)
    ax_2.set_xticks([0, 1, 2, 3, 4])
    ax_2.set_xticklabels([0, 1, 2, 3, 4])
    ax_2.set_yticks([0, 1, 2, 3, 4])
    ax_2.set_yticklabels([0, 1, 2, 3, 4])

    for ax in [ax_1, ax_2]:
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)


def draw_pair_regression(features: np.array, target: np.array, ax):
    # Take only one column as a feature (the second one)
    column_features = features[:, 1]
    model = LinearRegression()
    model.fit(column_features.reshape(-1, 1), target)

    print('Pair regression')
    print(model.intercept_)
    print(model.coef_)

    ax.scatter(column_features, target, c='black', s=150, zorder=3)
    ax.plot(column_features, np.ravel(model.predict(column_features.reshape(-1, 1))), '--', c='black', zorder=2)
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


def draw_regression_with_interaction(features: np.array, target: np.array, ax_4, ax_5):
    model = LinearRegression()
    model.fit(features, target)

    print('Regression model with interaction term')
    print(f"y = {model.intercept_[0]:.1f}, {model.coef_}")

    # Let's build 2 separate models for comparison
    first_model = LinearRegression()
    first_model.fit(features[:2, 1].reshape(-1, 1), target[:2, :])
    print("Separate model for x1=1")
    print(f"y = {first_model.intercept_[0]:.1f}, {first_model.coef_}")

    second_model = LinearRegression()
    second_model.fit(features[2:, 1].reshape(-1, 1), target[2:, :])
    print("Separate model for x1=0")
    print(f"y = {second_model.intercept_[0]:.1f}, {second_model.coef_}")

    ax_4.scatter(features[:, 1], target, c='grey', alpha=0.3, s=150, zorder=2)
    ax_4.scatter(features[:2, 1], target[:2, :], c='red', s=150, zorder=3)
    ax_4.plot(features[:2, 1], np.ravel(model.predict(features[:2, :])), '--', c='red', zorder=2)
    ax_4.grid(alpha=0.3, zorder=1)
    ax_4.set_xlim(0, 4.5)
    ax_4.set_ylim(0, 4.5)
    ax_4.set_xticks([0, 1, 2, 3, 4])
    ax_4.set_xticklabels([0, 1, 2, 3, 4])
    ax_4.set_yticks([0, 1, 2, 3, 4])
    ax_4.set_yticklabels([0, 1, 2, 3, 4])

    ax_5.scatter(features[:, 1], target, c='grey', alpha=0.3, s=150, zorder=2)
    ax_5.scatter(features[2:, 1], target[2:, :], c='blue', s=150, zorder=3)
    ax_5.plot(features[2:, 1], np.ravel(model.predict(features[2:, :])), '--', c='blue', zorder=2)
    ax_5.grid(alpha=0.3, zorder=1)
    ax_5.set_xlim(0, 4.5)
    ax_5.set_ylim(0, 4.5)
    ax_5.set_xticks([0, 1, 2, 3, 4])
    ax_5.set_xticklabels([0, 1, 2, 3, 4])
    ax_5.set_yticks([0, 1, 2, 3, 4])
    ax_5.set_yticklabels([0, 1, 2, 3, 4])

    for ax in [ax_4, ax_5]:
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontsize(16)
            tick_label.set_fontname(FONTNAME)


def draw_binary_regression_models():
    # Prepare 3 datasets
    features_simple = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_simple = np.array([[1], [3], [2], [4]])
    features_harder = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_harder = np.array([[1], [3], [2], [3]])
    features_hard = np.array([[1, 1], [1, 3], [0, 1], [0, 3]])
    target_hard = np.array([[1], [3], [3], [1]])

    fig_size = (20, 12)
    fig, axs = plt.subplots(3, 5, figsize=fig_size)

    all_features = [features_simple, features_harder, features_hard]
    all_targets = [target_simple, target_harder, target_hard]
    row_index = 0
    for features, target, case_name in zip(all_features, all_targets, ["simple", "advanced", "super advanced"]):
        print(f"=== {case_name} ===")
        ax_1 = axs[row_index, 0]
        ax_2 = axs[row_index, 1]
        ax_3 = axs[row_index, 2]
        draw_binary_regression(features, target, ax_1, ax_2)
        draw_pair_regression(features, target, ax_3)

        ax_4 = axs[row_index, 3]
        ax_5 = axs[row_index, 4]
        features = add_interaction_term(features)
        draw_regression_with_interaction(features, target, ax_4, ax_5)

        if row_index == 0:
            ax_1.set_title(r'Model with 2 features $x_1$=1', fontname=FONTNAME)
            ax_2.set_title(r'Model with 2 features $x_1$=0', fontname=FONTNAME)
            ax_3.set_title(r"Model with one feature $x_2$", fontname=FONTNAME)
            ax_4.set_title(r"Model with interaction term $x_1$=1", fontname=FONTNAME)
            ax_5.set_title(r"Model with interaction term $x_1$=0", fontname=FONTNAME)
        row_index += 1

    raw_svg_file = Path(get_plots_path(), "42_43_44_binary_features.svg")
    plt.savefig(raw_svg_file)
    plt.close()


if __name__ == '__main__':
    draw_binary_regression_models()