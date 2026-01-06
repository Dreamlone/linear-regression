import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import imageio
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

MIN_COEFFICIENT_BORDER = -2
MAX_COEFFICIENT_BORDER = 2
GRID_SIZE = 45
ANIMATION_DURATION: int = 150
FONTNAME = "Comic Sans MS"
CMAP_BY_METRIC = {"mae": 'coolwarm', "mape": "coolwarm"}


@dataclass
class Annotations:
    scatter_label_raw = r"$\hat{y} = INTERCEPT + SLOPE\cdot x$"
    scatter_label_scaled = r"$\hat{y}_{scaled} = INTERCEPT + SLOPE \cdot x_{scaled}$"

    def bake_scatter_label_raw(self, intercept: float, slope: float,
                               features_scaler: StandardScaler, target_scaler: StandardScaler):
        # StandardScaler: x_scaled = (x - mean_x) / scale_x
        mean_x = float(features_scaler.mean_[0])
        scale_x = float(features_scaler.scale_[0])

        mean_y = float(target_scaler.mean_[0])
        scale_y = float(target_scaler.scale_[0])

        # Convert coefficients from scaled space to raw space:
        # y = intercept_raw + slope_raw * x
        slope_raw = (scale_y / scale_x) * slope
        intercept_raw = mean_y + scale_y * intercept - slope_raw * mean_x

        label = self.scatter_label_raw.replace("INTERCEPT", f"{intercept_raw:.1f}")
        label = label.replace("SLOPE", f"{abs(slope_raw):.1f}")
        if slope_raw < 0:
            label = label.replace(" + ", " - ")

        return label

    def bake_scatter_label_scaled(self, intercept: float, slope: float):
        label = self.scatter_label_scaled.replace("INTERCEPT", f"{intercept:.1f}")
        label = label.replace("SLOPE", f"{abs(slope):.1f}")
        if slope < 0:
            label = label.replace(" + ", " - ")

        return label


@dataclass
class RusAnnotations(Annotations):
    map_title: str = "Зависимость METRIC от коэффициентов модели"
    map_x_axis: str = r"Сдвиг стандартизированный ($b_0$)"
    map_y_axis: str = r"Наклон стандартизированный ($b_1$)"
    scatter_x_axis: str = "Количество комнат (x)"
    scatter_y_axis: str = "Стоимость, $ (y)"

    def get_map_title(self, metric: str):
        if metric == "mae":
            map_title = self.map_title.replace("METRIC", "средней абсолютной ошибки")
        elif metric == "mape":
            map_title = self.map_title.replace("METRIC", "средней абсолютной процентной ошибки")
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_title

    def get_map_label(self, metric: str):
        if metric == "mae":
            map_label = "Средняя абсолютная ошибка (MAE), $"
        elif metric == "mape":
            map_label = "Средняя абсолютная процентная ошибка (MAPE), %"
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_label


@dataclass
class EngAnnotations(Annotations):
    map_title: str = "Dependence of METRIC on intercept and slope"
    map_x_axis: str = r"Intercept scaled ($b_0$)"
    map_y_axis: str = r"Slope scaled ($b_1$)"
    scatter_x_axis: str = "Number of the rooms in the apartment (x)"
    scatter_y_axis: str = "Price, $ (y)"

    def get_map_title(self, metric: str):
        if metric == "mae":
            map_title = self.map_title.replace("METRIC", "Mean Absolute Error")
        elif metric == "mape":
            map_title = self.map_title.replace("METRIC", "Mean Absolute Percentage Error")
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_title

    def get_map_label(self, metric: str):
        if metric == "mae":
            map_label = "Mean Absolute Error (MAE), $"
        elif metric == "mape":
            map_label = "Mean Absolute Percentage Error (MAPE), %"
        else:
            raise NotImplementedError(f"Metric {metric} is not supported")
        return map_label


def annotations_by_language(mode: str):
    if mode == "eng":
        annotations = EngAnnotations()
    elif mode == "rus":
        annotations = RusAnnotations()
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return annotations


def _measure_metric(metric: str, y_true, y_predicted):
    if metric == "mae":
        metric_value = mean_absolute_error(np.ravel(y_true), np.ravel(y_predicted))
    elif metric == "mape":
        metric_value = mean_absolute_percentage_error(np.ravel(y_true), np.ravel(y_predicted))
    else:
        raise NotImplementedError(f"Metric {metric} is not supported")
    return metric_value


def generate_df_coefficients_vs_error(metric: str):
    print("Generating the dataset with metrics")
    dataset = get_extended_dataset()
    features = np.array(dataset["rooms"])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    # Apply scaling on both features and target
    features_scaler = StandardScaler()
    features_scaled = features_scaler.fit_transform(x.reshape(-1, 1))
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y.reshape(-1, 1))

    # Generate dataframe with errors
    dataframe = []
    for intercept in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
        for slope in np.linspace(MIN_COEFFICIENT_BORDER, MAX_COEFFICIENT_BORDER, GRID_SIZE):
            predicted_scaled = intercept + slope * features_scaled
            predicted_raw = target_scaler.inverse_transform(predicted_scaled)

            metric_value = _measure_metric(metric, y, predicted_raw)

            dataframe.append([intercept, slope, metric_value])

    columns = ["intercept", "slope", "metric"]
    dataframe = pd.DataFrame(dataframe, columns=columns)
    print(dataframe.head(5))
    return dataframe, x, y, features_scaled, target_scaled, features_scaler, target_scaler


def compose_cases_to_explore(steps_per_section: int, sections: list):
    cases: list[list[float]] = []

    for section_index, section in enumerate(sections):
        if "start" not in section or "end" not in section:
            raise ValueError("Each section must have 'start' and 'end' keys.")

        start = np.asarray(section["start"], dtype=float)
        end = np.asarray(section["end"], dtype=float)

        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("'start' and 'end' must be 2D points: [b0, b1].")

        t_values = np.linspace(0.0, 1.0, steps_per_section)
        points = (1.0 - t_values)[:, None] * start[None, :] + t_values[:, None] * end[None, :]

        # Avoid duplicating the joint point between sections
        if section_index > 0:
            points = points[1:]

        cases.extend(points.tolist())

    return cases


def explore_coefficients_landscape(mode: str = "eng", metric: str = "mae"):
    """ Create a 2d map with coefficients vs MAE """
    annotations = annotations_by_language(mode)
    sections = [
        {"start": [-1.5, -1.5], "end": [-1.5, 0.0]},
        {"start": [-1.5, 0.0], "end": [-0.3, 1.0]},
        {"start": [-0.3, 1.0], "end": [1.2, 1.0]},
    ]
    cases_to_visualize = compose_cases_to_explore(steps_per_section=30, sections=sections)
    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)
        tmp_dir = get_tmp_animation_directory()

    (dataframe, features, target, features_scaled, target_scaled,
     features_scaler, target_scaler) = generate_df_coefficients_vs_error(metric)

    # Create grid for visualization
    coeff_b0 = np.array(dataframe["intercept"])
    coeff_b1 = np.array(dataframe["slope"])
    metric_values = np.array(dataframe["metric"])
    intercept_range, slope_range = np.meshgrid(np.unique(coeff_b0), np.unique(coeff_b1))

    image_files = []
    explored_path_coeff_b0 = []
    explored_path_coeff_b1 = []
    for image_index, case in enumerate(cases_to_visualize):
        intercept_i, slope_i = case
        explored_path_coeff_b0.append(intercept_i)
        explored_path_coeff_b1.append(slope_i)

        # First plot
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(121)

        # Interpolate values over grid
        errors = griddata((coeff_b0, coeff_b1), metric_values, (intercept_range, slope_range), method='cubic')
        cs = ax.contourf(intercept_range, slope_range, errors, levels=20, cmap=CMAP_BY_METRIC[metric])

        ax.scatter(intercept_i, slope_i, c='red', marker="D", s=100, zorder=2)
        predicted_scaled = intercept_i + slope_i * features_scaled
        predicted_raw = target_scaler.inverse_transform(predicted_scaled)
        # Calculate metric
        metric_value = _measure_metric(metric, target, predicted_raw)
        if metric == "mae":
            text_label = f"{metric_value:.0f}"
        else:
            text_label = f"{metric_value:.1f}"
        ax.text(intercept_i + 0.1, slope_i + 0.07,text_label, ha="center", va="bottom",
                fontsize=9, fontname=FONTNAME, color="red")

        ax.set_title(annotations.get_map_title(metric), fontdict={'fontsize': 14, 'fontname': FONTNAME}, y=1.03)
        ax.set_xlabel(annotations.map_x_axis, fontdict={'fontsize': 14, 'fontname': FONTNAME})
        ax.set_ylabel(annotations.map_y_axis, fontdict={'fontsize': 14, 'fontname': FONTNAME})
        ax.plot(explored_path_coeff_b0, explored_path_coeff_b1, c='red', zorder=1)

        cbar = fig.colorbar(cs)
        cbar.set_label(annotations.get_map_label(metric), fontdict={'fontsize': 12, 'fontname': FONTNAME})

        ax = fig.add_subplot(122)
        ax.scatter(features, target, s=60, c='black')

        features_for_model = features_scaler.transform(np.array([[1], [5]]))
        predicted_scaled = intercept_i + slope_i * features_for_model
        predicted_raw = target_scaler.inverse_transform(predicted_scaled)
        ax.plot([1, 5], np.ravel(predicted_raw), c='red')
        ax.set_xlabel(annotations.scatter_x_axis, fontdict={'fontsize': 14, 'fontname': FONTNAME})
        ax.set_ylabel(annotations.scatter_y_axis, fontdict={'fontsize': 14, 'fontname': FONTNAME})
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([0, 10000, 20000, 30000, 40000, 50000, 60000, 70000])
        ax.set_ylim(-5000, 75000)
        ax.set_xlim(0.5, 5.5)
        ax.grid(color='grey', alpha=0.7)
        raw_label = annotations.bake_scatter_label_raw(intercept_i, slope_i, features_scaler, target_scaler)

        scaled_label = annotations.bake_scatter_label_scaled(intercept_i, slope_i)
        ax.set_title(f"{scaled_label}\n\n{raw_label}", fontdict={'fontsize': 12, 'fontname': FONTNAME})

        raw_svg_file = Path(tmp_dir, f"45_explore_coefficients_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"45_explore_coefficients_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg", dpi=100)
        image_files.append(path_to_final_path)

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"45_explore_coefficients_{metric}_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    explore_coefficients_landscape("rus", metric="mae")
