from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 70000


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
    elif mode == "rus":
        title = "Пример многомерной регрессии"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title


def plot_new_extended_dataset(mode: str = "eng"):
    title = annotations_by_language(mode)

    dataset = get_extended_dataset()
    features = np.array(dataset[["rooms", "area", "metro_distance", "city", "ac_in_apartment"]])
    target = np.array(dataset["price"])
    x, y, _, _ = split_train_test_manual(features, target, apply_distortion=True)

    cmap = 'coolwarm'
    x_vals = x[:, 2]  # feature 2
    y_vals = x[:, 0]  # feature 1
    z_vals = y        # target

    # Fit linear regression plane: z = a * x_vals + b * y_vals + c
    regression_features = np.column_stack([x_vals, y_vals])
    linear_model = LinearRegression()
    linear_model.fit(regression_features, z_vals)

    # Create grid for plotting regression plane
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_vals.min(), x_vals.max(), 30),
        np.linspace(y_vals.min(), y_vals.max(), 30)
    )
    grid_features = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    z_grid = linear_model.predict(grid_features).reshape(x_grid.shape)

    fig = plt.figure(figsize=(16, 7))

    # First plot
    ax = fig.add_subplot(121, projection='3d')
    points = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=np.ravel(z_vals),
        cmap=cmap,
        s=100,
        alpha=0.8, edgecolors='black', linewidth=0.3,
        vmin=min(z_vals),
        vmax=max(z_vals),
        zorder=2
    )

    regression_surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=cmap,
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        vmin=min(z_vals),
        vmax=max(z_vals),
        zorder=1
    )

    cb = fig.colorbar(regression_surface, shrink=0.3, aspect=10)
    cb.set_label('Отклик', fontsize=12)

    ax.view_init(0, 0)
    ax.set_xlabel("Признак 2", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("Признак 1", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("Отклик", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.zaxis.set_ticklabels([])

    # Second plot
    ax = fig.add_subplot(122, projection='3d')
    points = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=np.ravel(z_vals),
        cmap=cmap,
        s=100,
        alpha=0.8, edgecolors='black', linewidth=0.3,
        vmin=min(z_vals),
        vmax=max(z_vals),
        zorder=2
    )

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=cmap,
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        vmin=min(z_vals),
        vmax=max(z_vals),
        zorder=1
    )

    ax.view_init(30, 35)
    ax.set_xlabel("Признак 2", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("Признак 1", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("Отклик", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.zaxis.set_ticklabels([])

    if title is not None:
        plt.suptitle(title, fontsize=15)

    raw_svg_file = Path(get_plots_path(), f"36_{mode}.svg")
    final_plot = Path(get_plots_path(), f"36_{mode}.png")
    plt.savefig(raw_svg_file, bbox_inches='tight')
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template.svg")


if __name__ == '__main__':
    plot_new_extended_dataset("rus")
