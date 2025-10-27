import shutil
from pathlib import Path
import numpy as np
import imageio
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
DPI = 150
COLORS = ['#4961d2', '#5875e1', '#6788ee', '#779af7', '#88abfd',
          '#9abbff', '#aac7fd', '#bad0f8', '#c9d7f0', '#d6dce4',
          '#e3d9d3', '#edd1c2', '#f4c6af', '#f7b89c', '#f7a889',
          '#f39475', '#ec7f63', '#e26952', '#d55042', '#c53334']
np.random.seed(2025)


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "The significance level meaning in statistical testing"
        stat_prefix = "Percentage of experiments with false results"
        experiment_label = "Experiment"
    elif mode == "rus":
        title = "Смысл уровня значимости при статистическом тестировании"
        stat_prefix = "Доля экспериментов с ложным результатом"
        experiment_label = "Эксперимент"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, stat_prefix, experiment_label


def _get_basic_plot(p_value_data_for_vis: dict):
    fig = plt.figure(figsize=(12, 8))  # Больше, но сохраняем пропорции

    # Параметры главного графика и сетки
    main_width = 0.30
    main_height = 0.48
    main_bottom = 0.26

    cols = 4
    rows = 5
    box_w, box_h = 0.095, 0.1
    dx = 0.105
    dy = 0.12

    # Вычисление общей ширины (main_plot + отступ + 4 блока p-value)
    gap = 0.07  # расстояние между основным графиком и сеткой
    total_width = main_width + gap + (cols - 1) * dx + box_w

    # Центрируем по ширине: левый край всего блока
    total_left = 0.5 - total_width / 2

    # Координаты основного графика
    main_left = total_left
    main_center_y = main_bottom + main_height / 2

    # Координаты начала сетки
    grid_left = main_left + main_width + gap
    total_grid_height = (rows - 1) * dy
    grid_bottom = main_center_y - total_grid_height / 1.75

    # === MAIN PLOT ===
    main_plot = fig.add_axes([main_left, main_bottom, main_width, main_height])
    main_plot.set_xlim(0, 101)
    main_plot.set_ylim(0, 101)
    main_plot.set_xlabel('x', labelpad=10, fontdict=FONTDICT)
    main_plot.set_ylabel('y', labelpad=10, fontdict=FONTDICT)

    p_value_plots = []
    for i in range(20):
        col = i % cols
        row = i // cols
        x = grid_left + col * dx
        y = grid_bottom + (rows - 1 - row) * dy
        ax_box = fig.add_axes([x, y, box_w, box_h])
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.set_frame_on(True)
        ax_box.tick_params(bottom=False, left=False)
        ax_box.set_facecolor('white')
        ax_box.set_xlim(0, 101)
        ax_box.set_ylim(0, 101)

        p_value_plots.append(ax_box)

        if p_value_data_for_vis.get(i) is not None:
            previous_experiment = p_value_data_for_vis.get(i)

            # Have something to visualize
            ax_box.scatter(previous_experiment["x"], previous_experiment["y"],
                           s=10, c=previous_experiment["color"], alpha=0.8, edgecolor="black")
            ax_box.plot(previous_experiment["x"], previous_experiment["predicted"],
                        c=previous_experiment["color"])
            p_value_label = previous_experiment["p-value label"]
            ax_box.set_title(p_value_label, fontsize=8, pad=2, fontdict={'fontname': FONTNAME},
                             c=previous_experiment["text_color"])

    return fig, main_plot, p_value_plots


def generate_initial_plot(mode, case_name, tmp_dir, x, y, experiment_id, false_results_number,
                          p_value_data_for_vis):
    title, stat_prefix, experiment_label = annotations_by_language(mode)
    if experiment_id == 0:
        ratio = 0
    else:
        ratio = round((false_results_number / experiment_id) * 100)
    statistics_message = f"{stat_prefix} = {false_results_number} / {experiment_id} = {ratio} %"

    fig, main_plot, p_value_plots = _get_basic_plot(p_value_data_for_vis)
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    fig.text(0.5, 0.1, statistics_message, fontsize=20, fontdict={'fontname': FONTNAME},
             ha='center', va='center')

    main_plot.scatter(x, y, c='grey', s=12, alpha=0.8)
    raw_svg_file = Path(tmp_dir, f"20_raw_alpha_animation_{mode}_{case_name}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    postfix_in_file_name = "initial"
    path_to_file = Path(tmp_dir,
                        f"20_train_test_animation_{mode}_{case_name}_{experiment_id}_{postfix_in_file_name}.png")
    save_plot_according_to_template(raw_svg_file, path_to_file, dpi=DPI, template_name="template_coolwarm.svg")

    return path_to_file


def generate_sampled(mode, case_name, tmp_dir, x, y, experiment_id, false_results_number, p_value_data_for_vis):
    title, stat_prefix, experiment_label = annotations_by_language(mode)
    ratio = round((false_results_number / (experiment_id + 1)) * 100)
    statistics_message = f"{stat_prefix} = {false_results_number} / {experiment_id + 1} = {ratio} %"

    fig, main_plot, p_value_plots = _get_basic_plot(p_value_data_for_vis)
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    fig.text(0.5, 0.1, statistics_message, fontsize=20, fontdict={'fontname': FONTNAME},
             ha='center', va='center')

    x_sampled, _, y_sampled, _ = train_test_split(x, y, test_size=0.7, random_state=experiment_id * 10)

    main_plot.scatter(x, y, c='grey', s=12, alpha=0.7)
    main_plot.scatter(x_sampled, y_sampled, c=COLORS[experiment_id], s=40, edgecolor='black')
    main_plot.set_title(f"{experiment_label}: {experiment_id + 1}", fontsize=12, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(tmp_dir, f"20_raw_alpha_animation_{mode}_{case_name}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    postfix_in_file_name = "sampling"
    path_to_file = Path(tmp_dir,
                        f"20_train_test_animation_{mode}_{case_name}_{experiment_id}_{postfix_in_file_name}.png")
    save_plot_according_to_template(raw_svg_file, path_to_file, dpi=DPI, template_name="template_coolwarm.svg")

    return path_to_file


def generate_plot_with_model(mode, case_name, tmp_dir, x, y, experiment_id,
                             false_results_number,
                             p_value_data_for_vis):
    title, stat_prefix, experiment_label = annotations_by_language(mode)
    fig, main_plot, p_value_plots = _get_basic_plot(p_value_data_for_vis)
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})

    x_sampled, _, y_sampled, _ = train_test_split(x, y, test_size=0.7, random_state=experiment_id * 10)

    main_plot.scatter(x, y, c='grey', s=12, alpha=0.7)
    main_plot.scatter(x_sampled, y_sampled, c=COLORS[experiment_id], s=40, edgecolor='black')
    main_plot.set_title(f"{experiment_label}: {experiment_id + 1}", fontsize=12, fontdict={'fontname': FONTNAME})

    # Fit the model
    feature_in_model = sm.add_constant(x_sampled)
    model = sm.OLS(y_sampled, feature_in_model).fit()
    predicted = model.predict(feature_in_model)

    main_plot.plot(x_sampled, predicted, c=COLORS[experiment_id], linewidth=10, alpha=0.1)
    main_plot.plot(x_sampled, predicted, c=COLORS[experiment_id], linewidth=7, alpha=0.2)
    main_plot.plot(x_sampled, predicted, c=COLORS[experiment_id], linewidth=4, alpha=0.4)
    main_plot.plot(x_sampled, predicted, c=COLORS[experiment_id], linewidth=1)
    main_plot.plot(x_sampled, predicted, c="grey", linewidth=0.5)

    p_value_plots[experiment_id].scatter(x_sampled, y_sampled,
                                         s=10, alpha=0.8, color=COLORS[experiment_id], edgecolor="black")
    p_value_plots[experiment_id].plot(x_sampled, predicted, color=COLORS[experiment_id])

    if model.f_pvalue < 0.05:
        if case_name == "random":
            false_results_number += 1
            text_color = "red"
            label_to_show = f'p-value {model.f_pvalue:.2f}'
        else:
            # For linear case the value is tiny
            label_to_show = r'$p\text{-value} \ll 0.01$'
            text_color = "black"
    else:
        text_color = "black"
        label_to_show = f'p-value {model.f_pvalue:.2f}'

    p_value_plots[experiment_id].set_title(label_to_show,
                                           fontsize=8, pad=2, fontdict={'fontname': FONTNAME}, c=text_color)
    p_value_data_for_vis[experiment_id] = {"x": x_sampled, "y": y_sampled,
                                           "predicted": predicted, "color": COLORS[experiment_id],
                                           "p-value label": label_to_show, "text_color": text_color}

    # Add updated label
    ratio = round((false_results_number / (experiment_id + 1)) * 100)
    statistics_message = f"{stat_prefix} = {false_results_number} / {experiment_id + 1} = {ratio} %"
    fig.text(0.5, 0.1, statistics_message, fontsize=20, fontdict={'fontname': FONTNAME},
             ha='center', va='center')

    raw_svg_file = Path(tmp_dir, f"20_raw_alpha_animation_{mode}_{case_name}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    postfix_in_file_name = "model"
    path_to_file = Path(tmp_dir,
                        f"20_train_test_animation_{mode}_{case_name}_{experiment_id}_{postfix_in_file_name}.png")
    save_plot_according_to_template(raw_svg_file, path_to_file, dpi=DPI, template_name="template_coolwarm.svg")

    return path_to_file, false_results_number


def generate_dataset_by_name(case_name: str):
    # Generate features + target
    if case_name == "random":
        x = np.random.uniform(1, 100, size=100)
        y = np.random.uniform(1, 100, size=100)
    else:
        # Generate linear dependency
        x = np.arange(0, 100)
        y = x + np.random.normal(scale=15, size=100)
        y[y < 0] = 0
        y[y > 100] = 100

    return x, y


def create_animation_alpha(mode: str = "eng", case_name: str = 'random', animation_duration: float = 500):

    # Create directory to store tmp files for animation
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    x, y = generate_dataset_by_name(case_name)

    number_of_iterations = 20
    false_results_number = 0
    image_files = []
    p_value_data_for_vis = {}
    for experiment_id in range(0, number_of_iterations):
        print(f"Generating frames for experiment {experiment_id}. Case {case_name}")
        # Generate the plot with initial data
        path_to_file = generate_initial_plot(mode, case_name, tmp_dir, x, y, experiment_id,
                                             false_results_number, p_value_data_for_vis)
        image_files.append(path_to_file)

        # Now take random sample!
        path_to_file = generate_sampled(mode, case_name, tmp_dir, x, y, experiment_id,
                                        false_results_number, p_value_data_for_vis)
        image_files.append(path_to_file)

        path_to_file, false_results_number = generate_plot_with_model(mode, case_name, tmp_dir, x, y,
                                                                      experiment_id, false_results_number,
                                                                      p_value_data_for_vis)
        image_files.append(path_to_file)

    gif_path = Path(get_plots_path(), f"20_alpha_explanation_{mode}_{case_name}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    create_animation_alpha("rus", "random")
    create_animation_alpha("rus", "linear")