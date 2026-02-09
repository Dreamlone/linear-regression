import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, symmetric_mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def produce_x_and_y_cases():
    cases = [
        {"x": np.array([1, 2, 3]), "y": np.array([1, 2, 3])},
        {"x": np.array([1, 2, 3]), "y": np.array([1, 2, 2])},
        {"x": np.array([1, 2, 3]), "y": np.array([1, 2, 1])},
        {"x": np.array([1, 2, 3]), "y": np.array([2, 2, 1])},
        {"x": np.array([1, 2, 3]), "y": np.array([3, 2, 1])},
        {"x": np.array([1, 2, 3]), "y": np.array([1, 3, 1])},
        {"x": np.array([1, 2, 3]), "y": np.array([3, 1, 3])},
    ]
    for item in cases:
        yield item["x"], item["y"]


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        y_label_metric = "Metrics"
        title = "How to calculate MAPE and SMAPE"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        y_label_metric = "Метрики"
        title = "Как рассчитывать MAPE и SMAPE"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, y_label_metric, title


def plot_animation_about_mape_smape(mode: str = "eng", animation_duration: float = 2000):
    x_label, y_label, y_label_metric, title = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    image_files = []
    image_index = 0
    for x, y in produce_x_and_y_cases():
        print(f"Generating the plots for case number {image_index}")

        # Create figure and subplots
        fig = plt.figure(figsize=(10, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 2])

        # Shift it to the center
        ax_top = fig.add_axes([0.35, 0.5, 0.3, 0.4])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        fig.suptitle(title, fontsize=16, fontdict=FONTDICT)

        # Scatter plot (3 points) + square axes
        ax_top.scatter(x, y, color='black')
        ax_top.set_xlim(0, 4)
        ax_top.set_ylim(0, 4)
        ax_top.set_aspect('equal', adjustable='box')

        ax_top.set_xlabel("X", fontdict=FONTDICT, loc='right')
        ax_top.set_ylabel("Y", fontdict=FONTDICT, loc='top')
        ax_top.set_xticks([1, 2, 3])
        ax_top.set_yticks([1, 2, 3])
        ax_top.grid(color='grey', alpha=0.5)

        # Add X/Y table (axes coordinates) — your final values
        col_x_x = -0.52
        col_x_y = -0.34
        header_y = 0.95
        row_start_y = 0.8
        row_step = 0.15

        ax_top.text(
            col_x_x, header_y, " X ",
            transform=ax_top.transAxes,
            ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
            clip_on=False
        )
        ax_top.text(
            col_x_y, header_y, " Y ",
            transform=ax_top.transAxes,
            ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
            clip_on=False
        )

        for row_idx, (x_val, y_val) in enumerate(zip(x, y)):
            y_pos = row_start_y - row_idx * row_step

            ax_top.text(
                col_x_x, y_pos, f" {x_val} ",
                transform=ax_top.transAxes,
                ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"),
                clip_on=False
            )
            ax_top.text(
                col_x_y, y_pos, f" {y_val} ",
                transform=ax_top.transAxes,
                ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"),
                clip_on=False
            )

        # MAPE box
        ax1.axis('off')
        formula_mape = r"$MAPE = \frac{100\%}{n} \sum \left| \frac{y_i - x_i}{y_i} \right|$"

        terms_mape = " + ".join([
            fr"\left|\frac{{{yi:.0f}-{xi:.0f}}}{{{yi:.0f}}}\right|"
            for xi, yi in zip(x, y)
        ])
        mape_value = np.mean([abs((yi - xi) / yi) * 100 for xi, yi in zip(x, y)])
        sub_calc_mape = r"$MAPE = \frac{100\%}{" + f"{len(x)}" + r"}\left(" + terms_mape + r"\right)$"
        result_mape = f"MAPE = {mape_value:.1f}%"
        # verify:
        # mape_value_new = mean_absolute_percentage_error(y, x) * 100

        ax1.text(0.5, 1.45, "Mean Absolute Percentage Error", fontsize=16, ha='center',
                 fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 1.05, formula_mape, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.55, sub_calc_mape, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.0, result_mape, fontsize=18, ha='center',
                 fontdict={'fontname': FONTNAME}, color='#6aa4f9')

        # SMAPE box
        ax2.axis('off')
        formula_smape = r"$SMAPE = \frac{100\%}{n} \sum \frac{|y_i - x_i|}{(|x_i| + |y_i|)/2}$"

        terms_smape = " + ".join([
            fr"\frac{{|{yi:.0f}-{xi:.0f}|}}{{({abs(xi):.0f}+{abs(yi):.0f})/2}}"
            for xi, yi in zip(x, y)
        ])
        smape_value = np.mean([abs(yi - xi) / ((abs(xi) + abs(yi)) / 2) * 100 for xi, yi in zip(x, y)])
        sub_calc_smape = r"$SMAPE = \frac{100\%}{" + f"{len(x)}" + r"}\left(" + terms_smape + r"\right)$"
        result_smape = f"SMAPE = {smape_value:.1f}%"
        # verify:
        # smape_value_new = symmetric_mean_absolute_percentage_error(y, x)

        ax2.text(0.5, 1.45, "Symmetric MAPE", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 1.05, formula_smape, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.55, sub_calc_smape, fontsize=9.5, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.0, result_smape, fontsize=18, ha='center', fontdict={'fontname': FONTNAME}, color='blue')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        raw_svg_file = Path(tmp_dir, f"16_raw_how_to_mape_smape_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file)
        plt.close()

        path_to_final_path = Path(tmp_dir, f"16_how_to_calculate_mape_smape_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path)

        image_files.append(path_to_final_path)
        image_index += 1

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"16_how_to_calculate_mape_smape_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_animation_about_mape_smape("rus")
