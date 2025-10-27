import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, symmetric_mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def produce_x_and_y_cases():
    cases = [{"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 4, 5])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 4, 4])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 4, 3])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 3, 3])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 3, 2])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 2, 2])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 2, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([2, 2, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([3, 2, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([3, 3, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([4, 3, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([4, 4, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([5, 4, 3, 2, 1])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([5, 1, 5, 1, 5])},
             {"x": np.array([1, 2, 3, 4, 5]), "y": np.array([1, 4, 1, 4, 1])}]
    for i in cases:
        yield i["x"], i["y"]


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


def plot_animation_about_mape_smape(mode: str = "eng", animation_duration: float = 1600):
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

        # Merge top two subplots for the graph
        ax_top.scatter(x, y, color='black')
        ax_top.set_xlim(0, 6)
        ax_top.set_ylim(0, 6)
        ax_top.set_xlabel("X", fontdict=FONTDICT, loc='right')
        ax_top.set_ylabel("Y", fontdict=FONTDICT, loc='top')
        ax_top.set_xticks([1, 2, 3, 4, 5])
        ax_top.set_yticks([1, 2, 3, 4, 5])
        ax_top.grid(color='grey', alpha=0.5)

        # Add textboxes for x and y (some kind of table)
        ax_top.text(-3.5, 6, " X ", ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax_top.text(-2.5, 6, " Y ", ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        row_index = 5
        for x_val, y_val in zip(x, y):
            ax_top.text(-3.5, row_index, f" {x_val} ", ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                        bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"))
            ax_top.text(-2.5, row_index, f" {y_val} ", ha='right', va='center', fontsize=14, fontdict={'fontname': FONTNAME},
                        bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"))
            row_index -= 1

        # Correlation coefficient box
        ax1.axis('off')
        formula_mape = r"$MAPE = \frac{100\%}{n} \sum \left| \frac{y_i - x_i}{y_i} \right|$"

        terms_mape = " + ".join([
            fr"\left|\frac{{{yi:.0f}-{xi:.0f}}}{{{yi:.0f}}}\right|"
            for xi, yi in zip(x, y)
        ])
        mape_value = np.mean([abs((yi - xi) / yi) * 100 for xi, yi in zip(x, y)])
        sub_calc = r"$MAPE = \frac{100\%}{" + f"{len(x)}" + r"}\left(" + terms_mape + r"\right)$"
        result_mape = f"MAPE = {mape_value:.1f}%"
        # To verify how it works - please uncomment code below
        # mape_value_new = mean_absolute_percentage_error(y, x) * 100
        # result_mape = f"MAPE = {mape_value:.1f}% = {mape_value_new:.1f}%"

        ax1.text(0.5, 1.45, "Mean Absolute Percentage Error", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 1.05, formula_mape, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.55, sub_calc, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.0, result_mape, fontsize=18, ha='center', fontdict={'fontname': FONTNAME}, color='#6aa4f9')

        # R² box
        ax2.axis('off')
        formula_smape = r"$SMAPE = \frac{100\%}{n} \sum \frac{|y_i - x_i|}{(|x_i| + |y_i|)/2}$"

        terms_smape = " + ".join([
            fr"\frac{{|{yi:.0f}-{xi:.0f}|}}{{({abs(xi):.0f}+{abs(yi):.0f})/2}}"
            for xi, yi in zip(x, y)
        ])
        smape_value = np.mean([abs(yi - xi) / ((abs(xi) + abs(yi)) / 2) * 100 for xi, yi in zip(x, y)])
        sub_calc = r"$SMAPE = \frac{100\%}{" + f"{len(x)}" + r"}\left(" + terms_smape + r"\right)$"
        result_smape = f"SMAPE = {smape_value:.1f}%"
        # To verify how it works - please uncomment code below
        # smape_value_new = symmetric_mean_absolute_percentage_error(y, x)
        # result_smape = f"SMAPE = {smape_value:.1f}% = {smape_value_new:.1f}%"

        ax2.text(0.5, 1.45, "Symmetric MAPE", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 1.05, formula_smape, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.55, sub_calc, fontsize=9.5, ha='center', fontdict={'fontname': FONTNAME})
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
