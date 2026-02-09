import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

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
    for case in cases:
        yield case["x"], case["y"]


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        y_label_metric = "Metrics"
        title = "How to calculate correlation coefficient and R²"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        y_label_metric = "Метрики"
        title = "Как рассчитывать коэффициент корреляции и R²"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, y_label_metric, title


def calculate_metrics(x: np.array, y: np.array):
    """Calculate correlation coefficient and R²."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))
    r = numerator / denominator
    r_squared = r ** 2
    return x_mean, y_mean, r, r_squared


def plot_animation_about_correlation_and_r2(mode: str = "eng", animation_duration: float = 2000):
    x_label, y_label, y_label_metric, title = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)

    image_files = []
    image_index = 0

    for x, y in produce_x_and_y_cases():
        print(f"Generating the plots for case number {image_index}")
        x_mean, y_mean, r, r_squared = calculate_metrics(x, y)

        # Create figure and subplots
        fig = plt.figure(figsize=(10, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 2])

        # Shift it to the center
        ax_top = fig.add_axes([0.35, 0.5, 0.3, 0.4])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        fig.suptitle(title, fontsize=16, fontdict=FONTDICT)

        # Scatter plot
        ax_top.scatter(x, y, color='black')

        # Make plot square: same ranges + equal aspect
        ax_top.set_xlim(0, 4)
        ax_top.set_ylim(0, 4)
        ax_top.set_aspect('equal', adjustable='box')

        ax_top.set_xlabel("X", fontdict=FONTDICT, loc='right')
        ax_top.set_ylabel("Y", fontdict=FONTDICT, loc='top')
        ax_top.set_xticks([1, 2, 3])
        ax_top.set_yticks([1, 2, 3])
        ax_top.grid(color='grey', alpha=0.5)

        # Add X/Y table (shift left + increase vertical spacing)
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

        # Correlation coefficient box
        ax1.axis('off')
        formula_r = r"$r = \frac{{\sum(x_i - \bar{{x}})(y_i - \bar{{y}})}}{{\sqrt{{\sum(x_i - \bar{{x}})^2}} \cdot \sqrt{{\sum(y_i - \bar{{y}})^2}}}}$"

        terms_num = " + ".join([f"({xi:.0f}-{x_mean:.0f})({yi:.0f}-{y_mean:.0f})" for xi, yi in zip(x, y)])
        terms_den_x = " + ".join([f"({xi:.0f}-{x_mean:.0f})²" for xi in x])
        terms_den_y = " + ".join([f"({yi:.0f}-{y_mean:.0f})²" for yi in y])

        sub_calc_r = (
            r"$r = \frac{" + terms_num + r"}"
            r"{\sqrt{" + terms_den_x + r"} \cdot \sqrt{" + terms_den_y + r"}}$"
        )
        result_r = f"r = {r:.2f}"

        ax1.text(0.5, 1.45, "Correlation coefficient", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 1.05, formula_r, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.55, sub_calc_r, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.0, result_r, fontsize=18, ha='center', fontdict={'fontname': FONTNAME}, color='red')

        # R² box
        ax2.axis('off')

        formula_r2 = r"$R^2 = 1 - \frac{\sum (y_i - x_i)^2}{\sum (y_i - \bar{y})^2}$"

        terms_res = " + ".join([f"({yi:.0f}-{yhat:.0f})²" for yi, yhat in zip(y, x)])
        terms_tot = " + ".join([f"({yi:.0f}-{y_mean:.0f})²" for yi in y])

        sub_calc_r2 = (
            r"$R^2 = 1 - \frac{" + terms_res + r"}{" + terms_tot + r"}$"
        )

        ss_res = sum((yi - xi) ** 2 for yi, xi in zip(y, x))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r2_value = 1 - (ss_res / ss_tot)
        result_r2 = f"$R^2$ = {r2_value:.2f}"

        ax2.text(0.5, 1.45, "R²", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 1.05, formula_r2, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.55, sub_calc_r2, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.0, result_r2, fontsize=18, ha='center', fontdict={'fontname': FONTNAME}, color='red')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        raw_svg_file = Path(tmp_dir, f"14_raw_how_to_calculate_correlation_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file)
        plt.close()

        path_to_final_path = Path(tmp_dir, f"14_how_to_calculate_correlation_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path)

        image_files.append(path_to_final_path)
        image_index += 1

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"14_how_to_calculate_correlation_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_animation_about_correlation_and_r2("rus")
