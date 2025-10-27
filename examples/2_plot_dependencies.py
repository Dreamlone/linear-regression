from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.exceptions import RankWarning

from examples.paths import get_plots_path
warnings.simplefilter('ignore', RankWarning)


def annotations_by_language(mode: str):
    if mode == "eng":
        feature_name = "Feature"
        target_name = "Target"
    elif mode == "rus":
        feature_name = "Признак"
        target_name = "Целевая переменная"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return feature_name, target_name


def plot_different_functional_dependencies(mode: str = "eng"):
    """
    Generate the plot with example how different features - target pairs
    can be

    To generate english plot choose mode "eng"
    To generate russian - "rus"
    """
    feature_name, target_name = annotations_by_language(mode)

    # Font and figure settings
    fontname = "Comic Sans MS"
    print("Starting generation of the different data dependencies plot...")
    fig_size = (20, 6)

    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    axs[0].plot([1, 2, 3], [1, 2, 3], '--', c='orange')
    axs[0].scatter([1, 2, 3], [1, 2, 3], s=80, c='black')
    axs[0].grid(color='grey', alpha=0.1)
    axs[0].set_ylim(0, 10)
    axs[0].set_xticks([1, 2, 3])

    z = np.polyfit(np.array([1, 2, 3]), np.array([1, 4, 9]), 3)
    p = np.poly1d(z)
    axs[1].scatter([1, 2, 3], [1, 4, 9], s=80, c='black')
    x_array = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
               2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    axs[1].plot(x_array, [p(i) for i in x_array],
                '--', c='orange')
    axs[1].grid(color='grey', alpha=0.1)
    axs[1].set_ylim(0, 10)
    axs[1].set_xticks([1, 2, 3])

    axs[2].plot([1, 1.5, 1.5, 2.5, 2.5, 3], [1, 1, 7, 7, 1, 1], '--', c='orange')
    axs[2].scatter([1, 2, 3], [1, 7, 1], s=80, c='black')
    axs[2].grid(color='grey', alpha=0.1)
    axs[2].set_ylim(0, 10)
    axs[2].set_xticks([1, 2, 3])

    for i in [0, 1, 2]:
        axs[i].set_ylabel(target_name,
                          fontdict={'fontsize': 14, 'fontname': fontname})
        axs[i].set_xlabel(feature_name,
                          fontdict={'fontsize': 14, 'fontname': fontname})

    # Show plot
    plt.savefig(Path(get_plots_path(), f"2_dependencies_{mode}.svg"))
    plt.close()
    print("Different data dependencies plot was successfully generated")


if __name__ == '__main__':
    plot_different_functional_dependencies("rus")
    plot_different_functional_dependencies("eng")
