from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

import matplotlib.pyplot as plt

from examples.paths import get_plots_path
import numpy as np


def prepare_simulated_table(rooms, metro_dist, reg, scaler,
                            points_density: int = 150) -> pd.DataFrame:
    first_feature_simulated = np.linspace(min(rooms) - 1,
                                          max(rooms) + 1, points_density)
    second_feature_simulated = np.linspace(min(metro_dist) - 1,
                                           max(metro_dist) + 1, points_density)
    features_ = []
    for metro_distance_feature in second_feature_simulated:
        constant_dist = [metro_distance_feature] * len(first_feature_simulated)
        features_.append(pd.DataFrame({'rooms': first_feature_simulated,
                                       'metro_dist': constant_dist}))
    features_ = pd.concat(features_)
    cols = ["rooms", "metro_dist"]
    features_['predicted'] = reg.predict(
        scaler.transform(np.array(features_[cols])))

    return features_


def three_d_plot_with_distance_feature():
    """ Generate 3d plot for multi dimension data """
    i = 0
    for dataset in [{"prices": [15500, 10500, 5500],
                     "rooms": [1, 2, 3],
                     "metro_dist": [150, 520, 2500]},
                    {"prices": [15500, 10500, 5500, 6100, 7500, 14400],
                     "rooms": [1, 2, 3, 1, 2, 3],
                     "metro_dist": [150, 520, 2500, 1350, 690, 380]}]:
        prices = np.array(dataset["prices"], dtype=np.float64)
        rooms = np.array(dataset["rooms"], dtype=np.float64)

        # Additional feature - distance to the nearest metro station, meters
        metro_dist = np.array(dataset["metro_dist"], dtype=np.float64)

        # Fit linear regression model on the data
        reg = LinearRegression()
        scaler = StandardScaler()
        features = np.hstack([rooms.reshape(-1, 1), metro_dist.reshape(-1, 1)])
        scaled_features = scaler.fit_transform(features)
        reg.fit(scaled_features, prices.reshape(-1, 1))

        features_ = prepare_simulated_table(rooms, metro_dist, reg, scaler, 90)
        fig = plt.figure(figsize=(20, 9))
        fontname = "Comic Sans MS"
        # First plot
        ax = fig.add_subplot(121, projection='3d')
        surf = ax.scatter(np.array(features_['rooms']),
                          np.array(features_['metro_dist']),
                          np.array(features_["predicted"]),
                          c=np.array(features_["predicted"]), cmap='coolwarm',
                          s=10, linewidth=0.0, alpha=0.2,
                          vmin=min(prices), vmax=max(prices), zorder=1)
        cb = fig.colorbar(surf, shrink=0.3, aspect=10)
        cb.set_label(f'Predicted price', fontsize=12)
        ax.view_init(10, -90)
        ax.set_xlabel('Rooms number',
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_zlabel("Price, USD",
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.scatter(np.array(rooms), np.array(metro_dist), np.array(prices),
                   c=np.array(prices),
                   cmap='coolwarm', s=150, linewidth=1.5, alpha=1.0,
                   edgecolor="black",
                   vmin=min(prices),
                   vmax=max(prices), zorder=3)
        ax.set_zlim(0, 20000)

        # Second plot
        features_ = prepare_simulated_table(rooms, metro_dist, reg, scaler, 200)
        ax = fig.add_subplot(122, projection='3d')
        surf = ax.scatter(np.array(features_['rooms']),
                          np.array(features_['metro_dist']),
                          np.array(features_["predicted"]),
                          c=np.array(features_["predicted"]), cmap='coolwarm',
                          s=10, linewidth=0.0, alpha=0.2,
                          vmin=min(prices), vmax=max(prices))
        ax.scatter(np.array(rooms), np.array(metro_dist), np.array(prices),
                   c=np.array(prices),
                   cmap='coolwarm', s=150, linewidth=1.5, alpha=1.0,
                   edgecolor="black",
                   vmin=min(prices),
                   vmax=max(prices))
        ax.view_init(40, -50)
        ax.set_xlabel('Rooms number',
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_ylabel('Distance to the nearest metro st., m',
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_zlabel("Price, USD",
                      fontdict={'fontsize': 14, 'fontname': fontname})
        ax.set_zlim(0, 20000)
        if i == 0:
            fig.suptitle("New features can help to clarify the picture",
                         fontsize=20,
                         fontdict={'fontname': fontname})
        else:
            fig.suptitle("And new data samples might help to make approximation better",
                         fontsize=20,
                         fontdict={'fontname': fontname})
        fig.savefig(Path(get_plots_path(), f"4_multi_dimension_data_{i}.png"),
                    dpi=350, bbox_inches='tight')
        plt.close()
        print(f"Finished plot generation for dataset {i}")
        i += 1


if __name__ == '__main__':
    three_d_plot_with_distance_feature()
