from pathlib import Path

import matplotlib.pyplot as plt

from examples.paths import get_plots_path


def annotations_by_language(mode: str):
    if mode == "eng":
        table_data = [["Rooms in the apartment", "Price, $"]]
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        title = "Linear regression allows predict price based on the room number"
        predicted_text = "so we can predict this\nprice now"
        equation_text = r"$\text{price} = b_0 + b_1 \cdot \text{rooms}$"
        equation_text_optim = r"$\text{price} = 0 + 10000 \cdot \text{rooms}$"
        predicted_text_ext = "and this one\nas well"
    elif mode == "rus":
        table_data = [["Количество комнат в квартире", "Стоимость, $"]]
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Линейная регрессия позволяет предсказать стоимость на основе количества комнат"
        predicted_text = "мы можем рассчитать\nэту стоимость"
        equation_text = r"$\text{цена} = b_0 + b_1 \cdot \text{комнаты}$"
        equation_text_optim = r"$\text{цена} = 0 + 10000 \cdot \text{комнаты}$"
        predicted_text_ext = "и эту\nтоже"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (table_data, x_label, y_label, title, predicted_text,
            equation_text, equation_text_optim, predicted_text_ext)


def plot_linear_regression(mode: str = "eng"):
    """
    Generate the plot with linear regression line and equation

    To generate english plot choose mode "eng"
    To generate russian - "rus"
    """
    (table_data, x_label, y_label,
     title, predicted_text, equation_text,
     equation_text_optim, predicted_text_ext) = annotations_by_language(mode)
    prices = [10000, 20000, 40000]
    rooms = [1, 2, 4]

    # Actual calculation is happening here
    linear_regression_values = []
    for room_number in [1, 2, 3, 4, 5]:
        # B0 is 0 and B1 is 10000
        estimated_value = 0 + 10000 * room_number
        linear_regression_values.append(estimated_value)

    # Create table data
    table_data += list(zip(rooms, prices))

    # Font and figure settings
    fontname = "Comic Sans MS"
    print("Starting generation of the linear regression plot...")
    fig_size = (10, 6)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(rooms, prices, s=80, c='black')
    ax.plot([1, 2, 3, 4, 5], linear_regression_values, '--', c='blue', alpha=0.9)
    ax.text(1.6, 18000, equation_text, fontsize=14, color='blue',
            fontname=fontname, rotation=30)
    ax.text(1.3, 11000, equation_text_optim, fontsize=14, color='black',
            fontname=fontname, rotation=30)

    # So we can predict something
    ax.scatter(3, 30000, color='blue', s=150, alpha=0.5)
    ax.text(3.15, 27000, predicted_text,
            fontsize=14, color='blue',
            fontname=fontname)
    ax.scatter(5, 50000, color='blue', s=150, alpha=0.5)
    ax.text(4.8, 44000, predicted_text_ext,
            fontsize=14, color='blue',
            fontname=fontname)
    ax.set_xlabel(x_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax.set_ylabel(y_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([10000, 20000, 30000, 40000, 50000])
    ax.grid(color='grey', alpha=0.1)
    plt.title(title,
              fontdict={'fontsize': 14, 'fontname': fontname})

    # Show plot
    plt.savefig(Path(get_plots_path(), f"3_plot_linear_regression_{mode}.svg"))
    plt.close()
    print("Linear regression plot was successfully generated")


if __name__ == '__main__':
    plot_linear_regression("rus")
    plot_linear_regression("eng")
