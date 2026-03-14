from pathlib import Path
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template, COLOR_BY_DATASET

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def _get_predicted(rooms: np.array, actual_prices: np.array):
    feature_in_model = sm.add_constant(rooms)

    # Build the model
    model = sm.OLS(actual_prices, feature_in_model).fit()
    predicted = model.predict(feature_in_model)
    return model, predicted


def _plot_predicted_with_actual(ax, rooms, actual, predicted, y_label, x_label,
                                dataset_name, naive_model, main_model):
    """ Draw a simple predicted and actual values plot """
    mean_actual = np.mean(actual)
    ax.plot(rooms, [float(mean_actual)] * len(rooms), '--', c="black", label=naive_model)

    ax.scatter(rooms, actual, s=80, c="grey", alpha=0.4, edgecolor="black")
    ax.plot(rooms, predicted, c=COLOR_BY_DATASET[dataset_name], label=main_model)
    ax.scatter(rooms, predicted, c=COLOR_BY_DATASET[dataset_name], s=75, marker='x')
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(0, 65000)
    if y_label is None:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_label, fontdict=FONTDICT)
    ax.set_xlabel(x_label, fontdict=FONTDICT)
    ax.set_title(dataset_name, fontsize=20, fontdict={'fontname': FONTNAME})
    ax.legend(loc='upper left', prop={'family': FONTNAME, 'size': 14})


def _plot_model_equation(ax, dataset_name: str, model, avg_target):
    intercept = round(model.params[0], 1)
    slope = round(model.params[-1], 1)

    ax.text(0.5, 0.75, rf"$H_0: \hat{{y}} = {avg_target} + 0 \cdot x$",
            transform=ax.transAxes,
            ha='center', va='center',
            color="black",
            fontsize=18, fontdict=FONTDICT)
    ax.text(0.5, 0.5, "VS",
            transform=ax.transAxes,
            ha='center', va='center',
            color="grey",
            fontsize=18, fontdict=FONTDICT)
    equation_text = rf"$H_1: \hat{{y}} = {intercept} + {slope} \cdot x$"
    ax.text(0.5, 0.25, equation_text,
            transform=ax.transAxes,
            ha='center', va='center',
            color=COLOR_BY_DATASET[dataset_name],
            fontsize=18, fontdict=FONTDICT)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])


def _plot_final_result(ax, model, significance_level, good_message, bad_message):
    summary = model.summary()

    # Need to exclude Notes section
    summary_lines = summary.as_text().splitlines()
    cutoff_index = None
    for i, line in enumerate(summary_lines):
        if line.strip().startswith("Notes:"):
            cutoff_index = i
            break
    if cutoff_index is not None:
        summary_lines = summary_lines[:cutoff_index]
    summary_text = "\n".join(summary_lines)

    ax.bar([2, 4], [model.f_pvalue, significance_level], width=0.3, color="grey", alpha=0.8)
    ax.set_xlim(0, 10)
    ax.set_xticks([2, 4])
    ax.set_xticklabels(["p-value", r"$\alpha$"])

    ax.set_ylim(0, 0.2)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])

    # Plot text labels
    text_buffer = 0.003
    ax.text(2, model.f_pvalue + text_buffer, f"{model.f_pvalue:.2f}", fontsize=12,
            color="grey", alpha=0.8, va='bottom', ha='center')
    ax.text(4, significance_level + text_buffer, f"{significance_level:.2f}",
            color="grey", alpha=0.8, fontsize=12, va='bottom', ha='center')

    ax.text(0.2, 0.195, summary_text,
            fontsize=5, color="grey", alpha=0.8,
            va='top', ha='left',
            family='monospace')

    # Print the final verdict
    if model.f_pvalue < significance_level:
        final_message = good_message
        color = "black"
    else:
        final_message = bad_message
        color = "red"
    ax.text(7.5, 0.065, final_message,
            va='top', ha='center', color=color,
            fontsize=12, fontdict=FONTDICT)


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Results of statistical testing using the F-test"
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        y_label_test = "Alternatives\nunder consideration"
        y_label_result = "Test results"
        naive_model = "Naive model"
        main_model = "Linear regression"
        good_message = "".join((r"p-value < $\alpha$", "\nThe model is\nstatistically\nsignificant"))
        bad_message = "".join((r"p-value ≥ $\alpha$", "\nThe model is not\nstatistically\nsignificant"))
    elif mode == "rus":
        title = "Результаты статистического тестирования по F критерию"
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        y_label_test = "Рассматриваемые\nальтернативы"
        y_label_result = "Результаты тестирования"
        naive_model = "Наивная модель"
        main_model = "Лин. регрессия"
        good_message = "".join((r"p-value < $\alpha$", "\nМодель\nстатистически\nзначима"))
        bad_message = "".join((r"p-value ≥ $\alpha$", "\nМодель\nне является\nстатистически значимой"))
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label, y_label, y_label_test, y_label_result, naive_model, main_model, good_message, bad_message


def plot_models_and_statistical_tests(mode: str = "eng", significance_level: float = 0.05):
    (title, x_label, y_label, y_label_test, y_label_result,
     naive_model, main_model, good_message, bad_message) = annotations_by_language(mode)

    # Get datasets and build models
    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    good_model, predicted_good = _get_predicted(rooms, good_prices)
    first_bad_model, predicted_bad_first = _get_predicted(rooms, bad_prices_first)
    second_bad_model, predicted_bad_second = _get_predicted(rooms, bad_prices_second)

    fig_size = (15, 12)
    fig, axs = plt.subplots(3, 3, figsize=fig_size)

    _plot_predicted_with_actual(axs[0, 0], rooms, good_prices, predicted_good,
                                y_label, x_label, "A", naive_model, main_model)
    _plot_model_equation(axs[1, 0], "A", good_model, np.mean(good_prices))
    _plot_final_result(axs[2, 0], good_model, significance_level, good_message, bad_message)

    # Add name for this group of subplots
    axs[1, 0].set_ylabel(y_label_test, fontdict=FONTDICT)
    axs[2, 0].set_ylabel(y_label_result, fontdict=FONTDICT)

    _plot_predicted_with_actual(axs[0, 1], rooms, bad_prices_first, predicted_bad_first,
                                None, x_label, "B", naive_model, main_model)
    _plot_model_equation(axs[1, 1], "B", first_bad_model, np.mean(bad_prices_first))
    _plot_final_result(axs[2, 1], first_bad_model, significance_level, good_message, bad_message)

    _plot_predicted_with_actual(axs[0, 2], rooms, bad_prices_second, predicted_bad_second,
                                None, x_label, "C", naive_model, main_model)
    _plot_model_equation(axs[1, 2], "C", second_bad_model, np.mean(bad_prices_second))
    _plot_final_result(axs[2, 2], second_bad_model, significance_level, good_message, bad_message)

    raw_svg_file = Path(get_plots_path(), f"19_raw_tests_for_datasets_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"19_tests_for_datasets_{mode}.png"))


if __name__ == '__main__':
    plot_models_and_statistical_tests("rus")
    plot_models_and_statistical_tests("eng")
