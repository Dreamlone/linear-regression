from pathlib import Path

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Pairwise correlations matrix"
        field_names = {
            "rooms": "Rooms",
            "area": "Area, m²",
            "metro_distance": "Distance\nto the metro, m",
            "price": "Price, $"
        }
        cbar_label = "Correlation coefficient"
    elif mode == "rus":
        title = "Матрица попарных корреляций"
        field_names = {
            "rooms": "Количество комнат",
            "area": "Площадь, м²",
            "metro_distance": "Расстояние\nдо метро, м",
            "price": "Цена, $"
        }
        cbar_label = "Коэффициент корреляции"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, field_names, cbar_label


def plot_correlation_matrix(mode: str = "eng"):
    """
    Generate the plot with correlation matrix
    Example: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """
    title, field_names, cbar_label = annotations_by_language(mode)

    # Read the data
    dataset = get_extended_dataset()
    features_names = ["rooms", "area", "metro_distance"]
    features = np.array(dataset[features_names])
    target = np.array(dataset["price"])
    x, y, _, _ = take_sample_manual(features, target, apply_distortion=True)

    df = pd.DataFrame(x, columns=features_names)
    df["price"] = y

    # Compute the correlation matrix
    corr = df.corr()

    # Rename rows/columns for pretty labels on axes
    corr_display = corr.rename(index=field_names, columns=field_names)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_display, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(
        corr_display,
        mask=mask,
        cmap='bwr',
        vmax=1.0,
        center=0,
        vmin=-1.0,
        square=True,
        annot=True,
        annot_kws={"fontsize": 11, "fontname": FONTNAME},
        linewidths=.5,
        ax=ax, alpha=0.7,
        cbar_kws={"shrink": .5}
    )

    # Set tick label font and orientation
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=12,
        fontname=FONTNAME
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=12,
        fontname=FONTNAME
    )

    # Colorbar label and font
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=12, fontname=FONTNAME)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontname(FONTNAME)
        lbl.set_fontsize(10)

    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME}, x=0.45, y=0.89)

    raw_svg_file = Path(get_plots_path(), f"41_corr_matrix_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()
    save_plot_according_to_template(
        raw_svg_file,
        Path(get_plots_path(), f"41_corr_matrix_{mode}.png")
    )


if __name__ == '__main__':
    plot_correlation_matrix("rus")
    plot_correlation_matrix("eng")
