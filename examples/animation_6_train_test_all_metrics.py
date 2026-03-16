import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path, get_tmp_animation_directory, get_results_path
from examples.utils import save_plot_according_to_template, get_datasets, symmetric_mean_absolute_percentage_error

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 12, 'fontname': FONTNAME}
# Took this value from previous step when the model was identified on all data
TEST_COLOR = "#c2c2c2"
BEST_EXPECTED_RMSE_ALL = 3873.270


@dataclass
class ExperimentData:
    x: np.array
    y: np.array
    sampled_x: np.array
    sampled_y: np.array
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array
    number_of_iterations_per_setup: int
    number_of_iterations_per_setup_for_visualization: int


@dataclass
class ExperimentResults:
    predict_train: np.array
    predict_test: np.array
    predict_all: np.array
    rmse_metric_train: float
    rmse_metric_test: float
    rmse_metric_all: float


def _produce_computation(experiment_setup: str, experiment_id: int,
                         number_of_iterations_per_setup: int,
                         number_of_iterations_per_setup_for_visualization: int):
    if experiment_setup == "small":
        # 10 objects will remain in "train" part
        test_size_ratio = 0.77
    else:
        # 20 objects will remain in "train" part
        test_size_ratio = 0.55

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concat([rooms, rooms, rooms])
    common_target = np.concat([good_prices, bad_prices_first, bad_prices_second])

    # Imitate that we take the sample
    # For 10 objects - test ratio must be 0.77
    sampled_features, _, sampled_target, _ = _split_train_test(experiment_id, common_features,
                                                               common_target, test_size_ratio)
    x_train, x_test, y_train, y_test = _split_train_test(experiment_id, sampled_features, sampled_target)
    exp_data = ExperimentData(x=common_features, y=common_target, sampled_x=sampled_features,
                              sampled_y=sampled_target, x_train=x_train,
                              y_train=y_train, x_test=x_test, y_test=y_test,
                              number_of_iterations_per_setup=number_of_iterations_per_setup,
                              number_of_iterations_per_setup_for_visualization=number_of_iterations_per_setup_for_visualization)

    model_predicted_train, intercept, slope = _fit_the_model(exp_data.x_train, exp_data.y_train)
    model_predicted_test = np.array([(intercept + slope * room) for room in exp_data.x_test])
    model_predicted_all = np.array([(intercept + slope * room) for room in exp_data.x])

    rmse_metric_train = root_mean_squared_error(exp_data.y_train, model_predicted_train)
    rmse_metric_test = root_mean_squared_error(exp_data.y_test, model_predicted_test)
    rmse_metric_all = root_mean_squared_error(exp_data.y, model_predicted_all)

    exp_results = ExperimentResults(predict_train=model_predicted_train, predict_test=model_predicted_test,
                                    predict_all=model_predicted_all, rmse_metric_train=rmse_metric_train,
                                    rmse_metric_test=rmse_metric_test, rmse_metric_all=rmse_metric_all)

    return exp_data, exp_results


def _fit_the_model(rooms: np.array, actual_prices: np.array):
    """ Build the model using analytical solution for one feature model """
    mean_x = np.mean(rooms)  # Average number of rooms
    mean_y = np.mean(actual_prices)  # Average price

    numerator = np.sum((rooms - mean_x) * (actual_prices - mean_y))
    denominator = np.sum((rooms - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    print(f"Model b0 + b1 * x: {intercept} + {slope} * x")
    predicted_prices = [(intercept + slope * room) for room in rooms]
    return np.array(predicted_prices), intercept, slope


def _split_train_test(random_state: int, features: np.array, target: np.array, test_size: float = 0.4):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state * 20)

    return x_train, x_test, y_train, y_test


def annotations_by_language(mode: str, sample_size: int):
    if mode == "eng":
        x_label = "Number of rooms"
        y_label = "Price, $"
        title = "Relationship between sample size and metrics on the full dataset"
        columns = ["Sample size", "RMSE train", "RMSE test", "RMSE full data"]
        model_title = "Fitted models"
        train_label = "Train"
        test_label = "Test"
        sampling_title = f"Sampling {sample_size} observations"
        all_data_title = "Full dataset"
        train_test_title = "Train-test split"
        small_title = "RMSE on the full dataset\n(model fitted on a sample of 10 observations)"
        big_title = "RMSE on the full dataset\n(model fitted on a sample of 20 observations)"
        experiment_x_label = "Experiment number"
        great_model = "Reference model"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Взаимосвязь между размером выборок и метриками на генеральной совокупности"
        columns = ["Размер семпла", "RMSE обучение", "RMSE тест", "RMSE все данные"]
        model_title = "Обученные модели"
        train_label = "Обучение"
        test_label = "Тест"
        sampling_title = f"Семплирование {sample_size} объектов"
        all_data_title = "Все данные"
        train_test_title = "Разбиение на обучение и тест"
        small_title = "RMSE по генеральной совокупности\n(модель обучена на семпле из 10 объектов)"
        big_title = "RMSE по генеральной совокупности\n(модель обучена на семпле из 20 объектов)"
        experiment_x_label = "Номер эксперимента"
        great_model = "Эталонная модель"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (title, columns, model_title, x_label, y_label, train_label, test_label, sampling_title,
            all_data_title, train_test_title, small_title, big_title, experiment_x_label, great_model)


def generate_plot(mode: str, tmp_dir: Path,
                  table_to_show: list,
                  lines_for_visualization: list,
                  experiment_number: int,
                  postfix_in_file_name: str,
                  exp_data: ExperimentData,
                  exp_results: ExperimentResults,
                  results: list, rows_to_color: dict):
    (title, columns, model_title, x_label,
     y_label, train_label, test_label, sampling_title, all_data_title,
     train_test_title, small_title, big_title, experiment_x_label, great_model) = annotations_by_language(mode, len(exp_data.sampled_x))

    if experiment_number < exp_data.number_of_iterations_per_setup_for_visualization:
        current_color = "#fc5e54"
    else:
        current_color = "#5499fc"

    # Create figure and axes
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax2 = plt.subplot2grid((2, 4), (1, 0))
    ax3 = plt.subplot2grid((2, 4), (0, 1), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax5 = plt.subplot2grid((2, 4), (1, 3))

    # Best model which were identified previously
    ax2.plot([1, 5], [11133.333333333334, 46600], c='black', zorder=1)

    # --- Ax1: Initial dataset ---
    if postfix_in_file_name == "show_data":
        # Show all data at once
        ax1.scatter(exp_data.x, exp_data.y, color='white', edgecolor="grey")
        ax1.set_title(all_data_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    elif postfix_in_file_name == "sampling":
        # Show the sample
        ax1.set_title(sampling_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
        ax1.scatter(exp_data.x, exp_data.y, color="#00000000", edgecolor="grey")
        ax1.scatter(exp_data.sampled_x, exp_data.sampled_y, color="white", edgecolor="black")
    else:
        # For all other cases we need to show split
        ax1.scatter(exp_data.x_train, exp_data.y_train, color=current_color, label=train_label, edgecolor="black")
        ax1.scatter(exp_data.x_test, exp_data.y_test, color=TEST_COLOR, label=test_label, edgecolor="black")
        ax1.legend(loc='upper left', prop={'family': FONTNAME})

    ax1.set_ylabel(y_label, fontdict=FONTDICT)
    ax1.set_ylim(0, 65000)
    ax1.set_xlim(0, 6)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.grid(color='grey', alpha=0.5)

    bar_width = 0.6
    ax4.set_ylim(0, 7000)
    ax4.set_xlim(0, 6)
    ax4.spines[['right', 'top']].set_visible(False)
    ax4.set_xticks([1, 2, 3, 4, 5])
    ax4.set_xlabel(experiment_x_label, fontdict={'fontsize': 10, 'fontname': FONTNAME})

    ax5.set_ylim(0, 7000)
    ax5.set_xlim(0, 6)
    ax5.spines[['right', 'top']].set_visible(False)
    ax5.set_xticks([1, 2, 3, 4, 5])
    ax5.set_xlabel(experiment_x_label, fontdict={'fontsize': 10, 'fontname': FONTNAME})

    # --- Ax2: Model plot ---
    if len(lines_for_visualization) > 1:
        for line_vis_params in lines_for_visualization:
            line = line_vis_params["predicted"]
            ax2.plot([min(exp_data.x), max(exp_data.x)], [min(line), max(line)], '--',
                     color=line_vis_params["color"], alpha=0.3)

    if postfix_in_file_name in ["show_data", "sampling"]:
        pass
    elif postfix_in_file_name == "split":
        ax1.set_title(train_test_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    else:
        ax1.set_title(train_test_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
        ax2.plot([min(exp_data.x), max(exp_data.x)],
                 [min(exp_results.predict_all), max(exp_results.predict_all)],
                 '--', color=current_color, zorder=2)
        ax2.scatter(exp_data.x_train, exp_data.y_train, color=current_color, edgecolor="black", zorder=3)
        lines_for_visualization.append({"predicted": exp_results.predict_all, "color": current_color})

    ax2.set_title(model_title, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax2.set_xlabel(x_label, fontdict=FONTDICT)
    ax2.set_ylabel(y_label, fontdict=FONTDICT)
    ax2.set_ylim(0, 65000)
    ax2.set_xlim(0, 6)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.grid(color='grey', alpha=0.5)

    # --- Ax3: Table with metrics ---
    ax3.axis("off")

    # Create the table
    if postfix_in_file_name == "score_metrics":
        new_row = [str(len(exp_data.sampled_x)), f"{exp_results.rmse_metric_train:.0f}",
                   f"{exp_results.rmse_metric_test:.0f}", f"{exp_results.rmse_metric_all:.0f}"]

        table_to_show[experiment_number] = new_row

    table = ax3.table(cellText=table_to_show,
                      colLabels=columns,
                      loc='center',
                      cellLoc='center',
                      colLoc='center', bbox=[-0.05, 0.0, 1.00, 1.00])
    for (row, col), cell in table.get_celld().items():
        if row in rows_to_color:
            cell.get_text().set_color(rows_to_color[row])

    table.scale(1, 2.1)  # Adjust for better vertical spacing
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.get_text().set_fontname("Comic Sans MS")

    ax4.plot([0, 6], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', linewidth=1, color='black')
    ax4.text(7.3, BEST_EXPECTED_RMSE_ALL, f"{great_model}\n{BEST_EXPECTED_RMSE_ALL:.0f}",
             fontname=FONTNAME, fontsize=9, va='center', ha='center')
    ax4.set_title(small_title, fontdict={'fontsize': 10, 'fontname': FONTNAME}, pad=10)

    ax5.plot([0, 6], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', linewidth=1, color='black')
    ax5.text(7.3, BEST_EXPECTED_RMSE_ALL, f"{great_model}\n{BEST_EXPECTED_RMSE_ALL:.0f}",
             fontname=FONTNAME, fontsize=9, va='center', ha='center')
    ax5.set_title(big_title, fontdict={'fontsize': 10, 'fontname': FONTNAME}, pad=10)

    if postfix_in_file_name == "bar_metrics":
        if len(results) <= exp_data.number_of_iterations_per_setup:
            ax4.bar(experiment_number + 1, exp_results.rmse_metric_all, color="red", alpha=0.5, width=bar_width)
        else:
            ax5.bar(experiment_number - exp_data.number_of_iterations_per_setup_for_visualization + 1,
                    exp_results.rmse_metric_all, color="blue", alpha=0.5, width=bar_width)

    # For all other stages: show_data, sampling, split, model_fit, and score_metrics
    for i in range(0, exp_data.number_of_iterations_per_setup_for_visualization):
        if i < len(results) - 1:
            ax4.bar(i + 1, results[i]["rmse all"], color="red", alpha=0.5, width=bar_width)

    start_id = exp_data.number_of_iterations_per_setup
    for i in range(start_id, start_id + exp_data.number_of_iterations_per_setup_for_visualization):
        if i < len(results) - 1:
            ax5.bar(i - exp_data.number_of_iterations_per_setup + 1, results[i]["rmse all"], color="blue",
                    alpha=0.5, width=bar_width)

    # plt.tight_layout()
    fig.subplots_adjust(top=0.8, hspace=0.5)

    # Overall title
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})
    raw_svg_file = Path(tmp_dir, f"animation_6_train_test_all_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    plot_path = Path(tmp_dir, f"animation_6_train_test_all_{mode}_{experiment_number}_{postfix_in_file_name}.png")
    save_plot_according_to_template(raw_svg_file, plot_path, template_name="template_small.svg", dpi=140)

    return plot_path, table_to_show, lines_for_visualization


def measure_metrics_for_train_test_all(vis: bool = True, mode: str = "eng", animation_duration: float = 1000):
    """
    Launch two sets of experiments and produce one csv file with results
    If vis is True, will generate animation for first 5 cases per experiment setup
    """
    number_of_iterations_per_setup = 30
    number_of_iterations_per_setup_for_visualization = 5
    total_number_of_iterations_for_visualization = 2 * number_of_iterations_per_setup_for_visualization

    # Create directory to store tmp files for animation
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    table = []
    for _ in range(0, total_number_of_iterations_for_visualization):
        table.append(["", "", "", ""])

    rows_to_color = {}
    for row_id in range(1, total_number_of_iterations_for_visualization + 1):
        if row_id <= number_of_iterations_per_setup_for_visualization:
            rows_to_color.update({row_id: "#fc5e54"})
        else:
            rows_to_color.update({row_id: "#5499fc"})

    lines_for_visualization = []
    image_files = []
    results = []
    total_plot_id = 0
    for experiment_setup in ["small", "big"]:
        for experiment_id in range(0, number_of_iterations_per_setup):
            print(f"Generating the data for experiment setup {experiment_setup}. Experiment {experiment_id}")
            exp_data, exp_results = _produce_computation(experiment_setup, experiment_id,
                                                         number_of_iterations_per_setup,
                                                         number_of_iterations_per_setup_for_visualization)
            results.append({"experiment_setup": experiment_setup, "id": experiment_id,
                            "rmse train": exp_results.rmse_metric_train,
                            "rmse test": exp_results.rmse_metric_test,
                            "rmse all": exp_results.rmse_metric_all})

            if vis is False:
                # No need to generate plots
                continue
            elif vis is True and experiment_id >= number_of_iterations_per_setup_for_visualization:
                # No need to make the visualization for the rest of runs
                continue

            print(f"Generate plot for model {experiment_id}")
            path_to_file, table, lines_for_visualization  = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                          total_plot_id, postfix_in_file_name="show_data",
                                                                          exp_data=exp_data, exp_results=exp_results,
                                                                          results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)

            path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                         total_plot_id,
                                                                         postfix_in_file_name="sampling",
                                                                         exp_data=exp_data, exp_results=exp_results,
                                                                         results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)

            # Perform splitting into train and test
            path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                         total_plot_id, postfix_in_file_name="split",
                                                                         exp_data=exp_data, exp_results=exp_results,
                                                                         results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)

            # Fit the model
            path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                         total_plot_id, postfix_in_file_name="model_fit",
                                                                         exp_data=exp_data, exp_results=exp_results,
                                                                         results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)

            # Store the results into table
            path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                         total_plot_id, postfix_in_file_name="score_metrics",
                                                                         exp_data=exp_data, exp_results=exp_results,
                                                                         results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)

            # Store the results into table
            path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                         total_plot_id,
                                                                         postfix_in_file_name="bar_metrics",
                                                                         exp_data=exp_data, exp_results=exp_results,
                                                                         results=results, rows_to_color=rows_to_color)
            image_files.append(path_to_file)
            total_plot_id += 1

    # Generate animation from the files
    if vis:
        gif_path = Path(get_plots_path(), f"animation_6_train_test_all_{mode}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
            for image_file in image_files:
                writer.append_data(imageio.imread(image_file))
        print(f"GIF saved at {gif_path}")
        shutil.rmtree(tmp_dir)

    # Save results as csv file
    results = pd.DataFrame(results)
    results.to_csv(Path(get_results_path(), "train_test_all_metrics.csv"), index=False)


if __name__ == '__main__':
    measure_metrics_for_train_test_all(vis=True, mode="rus")
    measure_metrics_for_train_test_all(vis=True, mode="eng")
