from pathlib import Path

import matplotlib.pyplot as plt

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template


def annotations_by_language(mode: str):
    if mode == "eng":
        table_data = [["Rooms in the apartment", "Price, $"]]
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        title = "Dependence of apartment price on room number"
    elif mode == "rus":
        table_data = [["Количество комнат в квартире", "Стоимость, $"]]
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Зависимость стоимости квартиры от количества комнат"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return table_data, x_label, y_label, title


def plot_apartment_data(mode: str = "eng"):
    """
    Initial plot with tabular view and simple scatter plot
    Function will produce 1 svg file

    To generate english plot choose mode "eng" - 1_plot_initial_data_eng.svg
    To generate russian - "rus" - 1_plot_initial_data_rus.svg
    """

    table_data, x_label, y_label, title = annotations_by_language(mode)
    prices = [10000, 20000, 40000]
    rooms = [1, 2, 4]

    # Create table data
    table_data += list(zip(rooms, prices))

    # Font and figure settings
    fontname = "Comic Sans MS"
    print("Starting generation of the initial plot...")
    fig_size = (10, 6)

    fig, axs = plt.subplots(1, 2, figsize=fig_size,
                            gridspec_kw={'width_ratios': [1, 3], 'wspace': 0.3})

    # Table subplot
    axs[0].axis('tight')
    axs[0].axis('off')
    table = axs[0].table(cellText=table_data, colLabels=None, loc='center',
                         cellLoc='center')
    for key, cell in table.get_celld().items():
        cell.get_text().set_fontname(fontname)

    # Scatter plot subplot
    axs[1].scatter(rooms, prices, s=80, c='black')
    axs[1].set_xlabel(x_label,
                      fontdict={'fontsize': 14, 'fontname': fontname})
    axs[1].set_ylabel(y_label,
                      fontdict={'fontsize': 14, 'fontname': fontname})
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_yticks([10000, 20000, 30000, 40000])
    axs[1].grid(color='grey', alpha=0.1)
    plt.suptitle(title, fontdict={'fontsize': 14, 'fontname': fontname})

    # Save plot as svg file
    raw_svg_file = Path(get_plots_path(), f"1_plot_initial_data_{mode}_raw.svg")
    final_plot = Path(get_plots_path(), f"1_plot_initial_data_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches='tight')

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template.svg")
    # Delete intermediate svg file
    raw_svg_file.unlink()
    print("Initial plot was successfully generated")


if __name__ == '__main__':
    plot_apartment_data("rus")
    plot_apartment_data("eng")
