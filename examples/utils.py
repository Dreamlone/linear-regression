import os
from pathlib import Path

import numpy as np
import pandas as pd

from examples.paths import get_plots_path, get_plots_templates_path

COLOR_BY_DATASET = {"A": "green",
                    "B": "orange",
                    "C": "#ff8484"}

def get_datasets():
    rooms = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])

    good_prices = np.array([9000, 10000, 11000,
                            18500, 20000, 21500,
                            29000, 30000, 31000,
                            38500, 40000, 41500,
                            49000, 50000, 51000])
    bad_prices_first = np.array([9000, 10000, 11000,
                                 18500, 20000, 21500,
                                 27000, 30000, 33000,
                                 36000, 40000, 44000,
                                 43000, 50000, 57000])
    bad_prices_second = np.array([9000, 10000, 11000,
                                  20000, 21000, 22000,
                                  30000, 31000, 32000,
                                  34000, 35000, 36000,
                                  35000, 36000, 37000])

    return rooms, good_prices, bad_prices_first, bad_prices_second


def get_extended_dataset() -> pd.DataFrame:
    prices = np.array([9000, 10000, 11000, 18500, 20000, 21500, 29000, 30000, 31000,
                       38500, 40000, 41500, 49000, 50000, 51000, 9000, 10000, 11000,
                       18500, 20000, 21500, 27000, 30000, 33000, 36000, 40000, 44000,
                       43000, 50000, 57000, 9000, 10000, 11000, 20000, 21000, 22000,
                       30000, 31000, 32000, 34000, 35000, 36000, 35000, 36000, 37000])
    rooms = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    city_name = np.array(["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
                          "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
                          "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"])
    # Area depends on rooms number
    area = 28 + 22 * rooms

    base = np.where(city_name == "A", 400, np.where(city_name == "B", 800, 1200))
    metro_distance = (base + (6 - rooms) * 120).astype(int)

    ac_bool = np.zeros_like(rooms, dtype=bool)
    for city in ["A", "B", "C"]:
        m = (city_name == city)
        med_price = np.median(prices[m])
        ac_bool[m] = (rooms[m] >= 3) | (prices[m] >= med_price)
    ac_in_apartment = np.where(ac_bool, "yes", "no")

    return pd.DataFrame({
        "city": city_name,
        "rooms": rooms,
        "area": area,
        "metro_distance": metro_distance,
        "ac_in_apartment": ac_in_apartment,
        "price": prices
    })


def extract_row(array: np.array, i_to_pick: int):
    if len(array.shape) == 2:
        return array[i_to_pick, :]
    else:
        return array[i_to_pick]


def split_train_test_manual(features: np.array, target: np.array, apply_distortion: bool):
    """ Function for 'bad' sampling with not vary good data """
    distorted_x, distorted_y = None, None
    ids_to_pick = [0, 1, 3, 7, 8, 11, 15, 18, 21, 22, 23, 26, 27, 28, 29, 32, 35, 38, 41, 44]
    print(f"Sample size: {len(ids_to_pick)}")
    x_sample = []
    y_sample = []
    for i in ids_to_pick:
        x_sample.append(extract_row(features, i))
        y_sample.append(extract_row(target, i))
    x_sampled, y_sampled = np.array(x_sample), np.array(y_sample)

    if apply_distortion:
        # Apply changes in-place (affecting y_sampled)
        distorted_x = []
        distorted_y = []
        for i in [2]:
            y_sampled[i] = y_sampled[i] - 10000
            distorted_x.append(extract_row(x_sampled, i))
            distorted_y.append(extract_row(y_sampled, i))
        for i in [9, 10]:
            y_sampled[i] = y_sampled[i] + 10000
            distorted_x.append(extract_row(x_sampled, i))
            distorted_y.append(extract_row(y_sampled, i))
        for i in [11]:
            y_sampled[i] = y_sampled[i] + 20000
            distorted_x.append(extract_row(x_sampled, i))
            distorted_y.append(extract_row(y_sampled, i))
        distorted_x, distorted_y = np.array(distorted_x), np.array(distorted_y)

    return x_sampled, y_sampled, distorted_x, distorted_y


def symmetric_mean_absolute_percentage_error(actual, predicted):
    """ Function to calculate SMAPE metric """

    actual = np.array(actual)
    predicted = np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    # Avoid division by zero
    non_zero = denominator != 0

    smape = np.zeros_like(actual, dtype=float)
    smape[non_zero] = np.abs(predicted[non_zero] - actual[non_zero]) / denominator[non_zero]
    return 100 * np.mean(smape)


def mm_to_pt(mm):
    """ Convert mm into pt units """
    return mm * 2.83465


def save_plot_according_to_template(path_to_svg_file: Path,
                                    final_path_to_png_file: Path,
                                    dpi: int = 200,
                                    template_name: str = 'template.svg'):
    """ Combine raw svg graph and template """
    try:
        import cairosvg
        from svgutils.transform import fromfile

        template_svg = fromfile(str(Path(get_plots_templates_path(), template_name)))
        plot_svg = fromfile(str(path_to_svg_file))

        # Combine SVGs
        # Get dimensions (strings like '800pt')
        template_width = mm_to_pt(float(template_svg.width.replace("mm", "")))
        template_height = mm_to_pt(float(template_svg.height.replace("mm", "")))
        plot_width = float(plot_svg.width.replace("pt", ""))
        plot_height = float(plot_svg.height.replace("pt", ""))

        # Calculate scale factor to fit plot inside template
        scale_x = (template_width - 40) / plot_width
        scale_y = (template_height - 10) / plot_height
        scale = min(scale_x, scale_y)  # keep aspect ratio

        plot_root = plot_svg.getroot()
        plot_root.scale(scale)

        # Get new plot size
        new_plot_width = plot_width * scale
        new_plot_height = plot_height * scale

        # Center the scaled plot
        x_center = (template_width - new_plot_width) / 2
        y_center = (template_height - new_plot_height) / 2
        plot_root.moveto(x_center, y_center)

        template_svg.append([plot_root])
        result_svg_path = Path(get_plots_path(), 'applied_template.svg')
        template_svg.save(result_svg_path)

        # Convert into png and delete the tmp file
        if final_path_to_png_file.name.endswith("png"):
            cairosvg.svg2png(url=str(result_svg_path), write_to=str(final_path_to_png_file), dpi=dpi)
        elif final_path_to_png_file.name.endswith("svg"):
            cairosvg.svg2svg(url=str(result_svg_path), write_to=str(final_path_to_png_file))
        else:
            raise NotImplementedError(f"Format for final file is not supported. Please choose .png or .svg")

        os.remove(str(result_svg_path))

    except Exception as ex:
        print(f"Saving picture with style from template failed because of {ex}")
