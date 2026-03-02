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
    for item in cases:
        yield item["x"], item["y"]


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of rooms"
        y_label = "Price, $"
        y_label_metric = "Metrics"
        title = "How to calculate Bias, MAE and RMSE"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        y_label_metric = "Метрики"
        title = "Как рассчитывать смещение, MAE и RMSE"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, y_label_metric, title


def plot_animation_about_mae_rmse(mode: str = "eng", animation_duration: float = 2000):
    x_label, y_label, y_label_metric, title = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)

    image_files = []
    image_index = 0
    for x, y in produce_x_and_y_cases():
        print(f"Generating the plots for case number {image_index}")

        bias = np.mean(y - x)
        mae = np.mean(np.abs(y - x))
        rmse = np.sqrt(np.mean((y - x) ** 2))

        # Create figure and subplots
        fig = plt.figure(figsize=(11, 7))
        gs = gridspec.GridSpec(
            2, 4,
            height_ratios=[2, 1],
            width_ratios=[0.01, 1.1, 1.1, 1.1]
        )

        # Top scatter + table
        ax_top = fig.add_axes([0.4, 0.5, 0.3, 0.4])
        ax_top.scatter(x, y, color='black')

        ax_top.set_xlim(0, 4)
        ax_top.set_ylim(0, 4)
        ax_top.set_aspect('equal', adjustable='box')

        ax_top.set_xlabel("X", fontdict=FONTDICT, loc='right')
        ax_top.set_ylabel("Y", fontdict=FONTDICT, loc='top')
        ax_top.set_xticks([1, 2, 3])
        ax_top.set_yticks([1, 2, 3])
        ax_top.grid(color='grey', alpha=0.5)

        # Add X/Y table (shift left + increase vertical spacing) — values you set
        col_x_x = -0.52
        col_x_y = -0.34
        header_y = 0.95
        row_start_y = 0.8
        row_step = 0.15

        ax_top.text(
            col_x_x, header_y, " X ",
            transform=ax_top.transAxes,
            ha='right', va='center',
            fontsize=14, fontdict={'fontname': FONTNAME},
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
            clip_on=False
        )
        ax_top.text(
            col_x_y, header_y, " Y ",
            transform=ax_top.transAxes,
            ha='right', va='center',
            fontsize=14, fontdict={'fontname': FONTNAME},
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
            clip_on=False
        )

        for row_idx, (x_val, y_val) in enumerate(zip(x, y)):
            y_pos = row_start_y - row_idx * row_step

            ax_top.text(
                col_x_x, y_pos, f" {x_val} ",
                transform=ax_top.transAxes,
                ha='right', va='center',
                fontsize=14, fontdict={'fontname': FONTNAME},
                bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"),
                clip_on=False
            )
            ax_top.text(
                col_x_y, y_pos, f" {y_val} ",
                transform=ax_top.transAxes,
                ha='right', va='center',
                fontsize=14, fontdict={'fontname': FONTNAME},
                bbox=dict(boxstyle="round", facecolor="orange", edgecolor="gray"),
                clip_on=False
            )

        fig.suptitle(title, fontsize=16, fontdict=FONTDICT)

        # Bottom plots: Bias, MAE, RMSE
        ax_bias = fig.add_subplot(gs[1, 1])
        ax_bias.axis('off')
        bias_expr = r"$\text{Bias} = \frac{1}{n} \sum (y_i - x_i)$"
        bias_terms = " + ".join([f"({yi}-{xi})" for xi, yi in zip(x, y)])
        bias_sub = rf"$\text{{Bias}} = \frac{{{bias_terms}}}{{{len(x)}}}$"
        bias_result = f"Bias = {bias:.2f}"

        ax_bias.text(0.5, 1.4, "Bias", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax_bias.text(0.5, 1.0, bias_expr, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax_bias.text(0.5, 0.55, bias_sub, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax_bias.text(0.5, 0.0, bias_result, fontsize=18, ha='center', color='green',
                     fontdict={'fontname': FONTNAME})

        ax_mae = fig.add_subplot(gs[1, 2])
        ax_mae.axis('off')
        mae_expr = r"$\text{MAE} = \frac{1}{n} \sum |y_i - x_i|$"
        mae_terms = " + ".join([f"|{yi}-{xi}|" for xi, yi in zip(x, y)])
        mae_sub = rf"$\text{{MAE}} = \frac{{{mae_terms}}}{{{len(x)}}}$"
        mae_result = f"MAE = {mae:.2f}"

        ax_mae.text(0.5, 1.4, "Mean Absolute Error", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax_mae.text(0.5, 1.0, mae_expr, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax_mae.text(0.5, 0.55, mae_sub, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax_mae.text(0.5, 0.0, mae_result, fontsize=18, ha='center', color='green',
                    fontdict={'fontname': FONTNAME})

        ax_rmse = fig.add_subplot(gs[1, 3])
        ax_rmse.axis('off')
        rmse_expr = r"$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - x_i)^2}$"
        rmse_terms = " + ".join([f"({yi}-{xi})²" for xi, yi in zip(x, y)])
        rmse_sub = rf"$\text{{RMSE}} = \sqrt{{\frac{{{rmse_terms}}}{{{len(x)}}}}}$"
        rmse_result = f"RMSE = {rmse:.2f}"

        ax_rmse.text(0.5, 1.4, "Root Mean Square Error", fontsize=16, ha='center', fontdict={'fontname': FONTNAME})
        ax_rmse.text(0.5, 1.0, rmse_expr, fontsize=14, ha='center', fontdict={'fontname': FONTNAME})
        ax_rmse.text(0.5, 0.55, rmse_sub, fontsize=10, ha='center', fontdict={'fontname': FONTNAME})
        ax_rmse.text(0.5, 0.0, rmse_result, fontsize=18, ha='center', color='green',
                     fontdict={'fontname': FONTNAME})

        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        raw_svg_file = Path(tmp_dir, f"animation_2_how_to_mae_rmse_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file)
        plt.close()

        path_to_final_path = Path(tmp_dir, f"animation_2_how_to_mae_rmse_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path)

        image_files.append(path_to_final_path)
        image_index += 1

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"animation_2_how_to_calculate_mae_rmse_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_animation_about_mae_rmse("rus")
    plot_animation_about_mae_rmse("eng")
