import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, get_datasets

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
TRAIN_COLOR = "orange"
TEST_COLOR = "grey"


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

def _split_train_test(random_state: int, features: np.array, target: np.array):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.4, random_state=random_state * 10)

    return x_train, x_test, y_train, y_test


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        title = "Models under different train and test splits"
        columns = ["Model", "MAPE train", "MAPE test"]
        model_title = "Model fitting"
        metrics_table_title = "Metrics per model"
        train_label = "Train"
        test_label = "Test"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Модели при разных разбиениях на обучение и тест"
        columns = ["Модель", "MAPE обучение", "MAPE тест"]
        model_title = "Обучение модели"
        metrics_table_title = "Метрики"
        train_label = "Обучение"
        test_label = "Тест"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, columns, model_title, metrics_table_title, x_label, y_label, train_label, test_label


def generate_plot(mode: str, tmp_dir: Path,
                  table_to_show: list,
                  lines_for_visualization: list,
                  experiment_number: int,
                  postfix_in_file_name: str):
    title, columns, model_title, metrics_table_title, x_label, y_label, train_label, test_label = annotations_by_language(mode)
    rooms, _, dataset_b_prices, _ = get_datasets()

    x_train, x_test, y_train, y_test = _split_train_test(experiment_number, rooms, dataset_b_prices)

    # Create figure and axes
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)

    # --- Ax1: Initial dataset ---
    if postfix_in_file_name == "show_data":
        # Show all data at once
        ax1.scatter(rooms, dataset_b_prices, color='white', edgecolor="black")
    else:
        # For all other cases we need to show split
        ax1.scatter(x_train, y_train, color=TRAIN_COLOR, label=train_label, edgecolor="black", alpha=0.6)
        ax1.scatter(x_test, y_test, color=TEST_COLOR, label=test_label, edgecolor="black", alpha=0.6)
        ax1.legend(loc='upper left', prop={'family': FONTNAME})

    ax1.set_title("B", fontdict=FONTDICT)
    ax1.set_ylabel(y_label, fontdict=FONTDICT)
    ax1.set_ylim(0, 65000)
    ax1.set_xlim(0, 6)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.grid(color='grey', alpha=0.5)

    # --- Ax2: Model plot ---
    if len(lines_for_visualization) > 1:
        for line in lines_for_visualization:
            ax2.plot(rooms, line, '--', color='grey', alpha=0.6)

    if postfix_in_file_name in ["show_data", "split"]:
        pass
    else:
        model_predicted_train, intercept, slope = _fit_the_model(x_train, y_train)
        model_predicted_test = [(intercept + slope * room) for room in x_test]
        model_predicted_all = [(intercept + slope * room) for room in rooms]

        ax2.scatter(x_train, y_train, color=TRAIN_COLOR, edgecolor="black", alpha=0.6)
        ax2.plot(rooms, model_predicted_all, color=TRAIN_COLOR)
        lines_for_visualization.append(model_predicted_all)

    ax2.set_title(model_title, fontdict=FONTDICT)
    ax2.set_xlabel(x_label, fontdict=FONTDICT)
    ax2.set_ylabel(y_label, fontdict=FONTDICT)
    ax2.set_ylim(0, 65000)
    ax2.set_xlim(0, 6)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.grid(color='grey', alpha=0.5)

    # --- Ax3: Table with metrics ---
    ax3.axis("off")
    ax3.set_title(metrics_table_title, fontdict=FONTDICT)

    # Create the table
    if postfix_in_file_name == "score_metrics":
        mape_metric_train = mean_absolute_percentage_error(y_train, model_predicted_train) * 100
        mape_metric_test = mean_absolute_percentage_error(y_test, model_predicted_test) * 100
        new_row = [rf"${intercept:.0f} + {slope:.0f} \cdot x$", f"{mape_metric_train:.1f} %", f"{mape_metric_test:.1f} %"]

        table_to_show[experiment_number] = new_row

    table = ax3.table(cellText=table_to_show,
                      colLabels=columns,
                      loc='center',
                      cellLoc='center',
                      colLoc='center')

    table.scale(1, 2)  # Adjust for better vertical spacing
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.get_text().set_fontname("Comic Sans MS")

    # plt.tight_layout()
    fig.subplots_adjust(top=0.8, hspace=0.5)

    # Overall title
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    raw_svg_file = Path(tmp_dir, f"22_raw_train_test_animation_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    plot_path = Path(tmp_dir, f"20_train_test_animation_{mode}_{experiment_number}_{postfix_in_file_name}.png")
    save_plot_according_to_template(raw_svg_file, plot_path)

    return plot_path, table_to_show, lines_for_visualization


def create_animation_with_train_test(mode: str = "eng", animation_duration: float = 1500):
    """ Generate animation for train test splitting """
    number_of_iterations = 5

    # Create directory to store tmp files for animation
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    table = []
    for _ in range(0, number_of_iterations):
        table.append(["", "", ""])

    lines_for_visualization = []

    image_files = []
    for experiment_id in range(0, number_of_iterations):
        print(f"Generate plot for model {experiment_id}")
        path_to_file, table, lines_for_visualization  = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                      experiment_id, postfix_in_file_name="show_data")
        image_files.append(path_to_file)

        # Perform splitting into train and test
        path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                     experiment_id, postfix_in_file_name="split")
        image_files.append(path_to_file)

        # Fit the model
        path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                     experiment_id, postfix_in_file_name="model_fit")
        image_files.append(path_to_file)

        # Store the results into table
        path_to_file, table, lines_for_visualization = generate_plot(mode, tmp_dir, table, lines_for_visualization,
                                                                     experiment_id, postfix_in_file_name="score_metrics")
        image_files.append(path_to_file)

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"22_train_test_animation_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    create_animation_with_train_test("rus")