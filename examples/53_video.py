import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from scipy.interpolate import griddata
from examples.paths import get_plots_path, get_tmp_animation_directory


def explore_coefficients_landscape(number_of_values: int = 50,
                                   step_of_animation: int = 5):
    """ Create a map with metric value and save as video """
    dpi = 120
    video_path = Path(get_plots_path(), "coefficients.mp4")
    actual_prices = {1: 10000, 2: 20000, 3: 30000, 4: 40000}
    intercept_min = -10000
    intercept_max = 10000
    slope_min = -3500
    slope_max = 13000

    dataframe = []
    for intercept in np.linspace(intercept_min, intercept_max, number_of_values):
        for slope in np.linspace(slope_min, slope_max, number_of_values):
            predicted = []
            errors = []
            for rooms in [1, 2, 3, 4]:
                predicted_value = intercept + slope * rooms
                predicted.append(predicted_value)

                error = abs(actual_prices[rooms] - predicted_value)
                errors.append(error)

            experiment_details = [intercept, slope]
            experiment_details.extend(predicted)
            experiment_details.append(sum(errors) / 4)
            dataframe.append(experiment_details)

    columns = ["intercept", "slope", "predicted_1", "predicted_2",
               "predicted_3", "predicted_4", "MAE"]
    dataframe = pd.DataFrame(dataframe, columns=columns)

    print(dataframe.head(10))
    fontname = "Comic Sans MS"

    # Create grid
    x = dataframe["intercept"]
    y = dataframe["slope"]
    z = dataframe["MAE"]
    xi = np.linspace(intercept_min, intercept_max, number_of_values)
    yi = np.linspace(slope_min, slope_max, number_of_values)
    intercept_range, slope_range = np.meshgrid(xi, yi)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        shutil.rmtree(tmp_dir)

    image_files = []
    max_n = 0
    old_predictions = []
    checked = []

    for row_id, row in dataframe.iterrows():
        if max_n > 1000:
            break
        if int(row_id) % step_of_animation != 0:
            continue

        # Generate plot for every row
        current_intercept = row["intercept"]
        current_slope = row["slope"]
        if current_intercept == intercept_min or current_intercept == intercept_max:
            continue
        if current_slope == slope_min or current_slope == slope_max:
            continue

        max_n += 1
        predicted_prices = [row["predicted_1"], row["predicted_2"],
                            row["predicted_3"], row["predicted_4"]]

        # First plot
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(121)

        # Interpolate Z values over grid
        errors = griddata((x, y), z, (intercept_range, slope_range), method='cubic')
        cs = ax.contourf(intercept_range, slope_range, errors, levels=20, cmap='coolwarm')
        ax.scatter(current_intercept, current_slope, c='red', marker='x', s=100)
        ax.set_title("Dependence of Mean Absolute Error on intercept and slope",
                     fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_xlabel(r"Intercept ($B_0$)", fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_ylabel(r"Slope ($B1$)", fontdict={'fontsize': 14, 'fontname': fontname})

        if len(checked) > 0:
            for checked_points in checked:
                ax.scatter(checked_points[0], checked_points[-1], c='grey', alpha=0.8,
                           marker='x', s=100)
        cbar = fig.colorbar(cs)
        cbar.set_label("Mean Absolute Error (MAE)",
                       fontdict={'fontsize': 12, 'fontname': fontname})

        ax = fig.add_subplot(122)
        ax.scatter(list(actual_prices.keys()), list(actual_prices.values()),
                   s=80, c='black', label="Actual data")
        ax.plot(list(actual_prices.keys()), predicted_prices,
                c='red', label="Predicted")
        if len(old_predictions) > 0:
            for old_price in old_predictions:
                ax.plot(list(actual_prices.keys()), old_price, '--',
                        c='grey', alpha=0.2)
        ax.set_xlabel("Number of the rooms in the apartment",
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_ylabel("Price, USD",
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.legend(fontsize=14, loc='upper left')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_yticks([10000, 20000, 30000, 40000, 50000, 60000])
        ax.set_ylim(0, 65000)
        ax.set_xlim(0.5, 4.5)
        ax.grid(color='grey', alpha=0.7)
        ax.set_title(f"predicted price = {current_intercept:.1f} + {current_slope:.1f} x rooms number",
                     fontdict={'fontsize': 14, 'fontname': fontname})

        image_name = f"{row_id}.png"
        image_path = Path(tmp_dir, image_name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, dpi=dpi)
        plt.close()
        print(f"Finished generating image {image_name}")

        image_files.append(image_path)
        old_predictions.append(predicted_prices)
        checked.append([current_intercept, current_slope])

    # Create video
    fps = 2  # Standard FPS for smoother playback
    frame_duration = int(700 / (1000 / fps))  # Convert 700ms to frame count
    frame_size = (dpi * 20, dpi * 9)  # Size should match image resolution
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        for _ in range(frame_duration):  # Repeat the frame for ~700ms duration
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at {video_path}")


if __name__ == '__main__':
    explore_coefficients_landscape(50, 80)
