import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.metrics import r2_score

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

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
    """ Calculate correlation coefficient and R² """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))
    r = numerator / denominator
    r_squared = r ** 2

    return x_mean, y_mean, r, r_squared


def plot_animation_about_correlation_and_r2(mode: str = "eng", animation_duration: float = 1600):
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
        formula_r = r"$r = \frac{{\sum(x_i - \bar{{x}})(y_i - \bar{{y}})}}{{\sqrt{{\sum(x_i - \bar{{x}})^2}} \cdot \sqrt{{\sum(y_i - \bar{{y}})^2}}}}$"
        # Dynamic generation of the sub-calculation using actual data
        terms_num = " + ".join([f"({xi:.0f}-{x_mean:.0f})({yi:.0f}-{y_mean:.0f})" for xi, yi in zip(x, y)])
        terms_den_x = " + ".join([f"({xi:.0f}-{x_mean:.0f})²" for xi in x])
        terms_den_y = " + ".join([f"({yi:.0f}-{y_mean:.0f})²" for yi in y])

        # Full dynamic sub-calculation in LaTeX
        sub_calc = (
                r"$r = \frac{" + terms_num + r"}"
                                             r"{\sqrt{" + terms_den_x + r"} \cdot \sqrt{" + terms_den_y + r"}}$"
        )
        result_r = f"r = {r:.2f}"
        # To verify how it works - please uncomment code below
        # corr = stats.pearsonr(x, y)
        # result_r = f"r = {corr.correlation:.2f} = {r:.2f}"

        ax1.text(0.5, 1.45, "Correlation coefficient", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 1.05, formula_r, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.55, sub_calc, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax1.text(0.5, 0.0, result_r, fontsize=18, ha='center', fontdict={'fontname': FONTNAME}, color='red')

        # R² box
        ax2.axis('off')

        # LaTeX-style full formula (detailed, no SS_ terms)
        formula_r2 = r"$R^2 = 1 - \frac{\sum (y_i - x_i)^2}{\sum (y_i - \bar{y})^2}$"

        # Dynamic term breakdown
        terms_res = " + ".join([f"({yi:.0f}-{yhat:.0f})²" for yi, yhat in zip(y, x)])
        terms_tot = " + ".join([f"({yi:.0f}-{y_mean:.0f})²" for yi in y])

        # Dynamic substitution in LaTeX
        sub_calc = (
                r"$R^2 = 1 - \frac{" + terms_res + r"}{" + terms_tot + r"}$"
        )

        # Optionally, you can compute the numeric result as well
        ss_res = sum((yi - xi) ** 2 for yi, xi in zip(y, x))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r2_value = 1 - (ss_res / ss_tot)
        result_r2 = f"$R^2$ = {r2_value:.2f}"

        # To verify how it works - please uncomment code below
        # r_squared = r2_score(y, x)
        # result_r2 = f"$R^2$ = {r_squared:.2f} = {r2_value:.2f}"

        ax2.text(0.5, 1.45, "R²", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 1.05, formula_r2, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax2.text(0.5, 0.55, sub_calc, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
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
