from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import symmetric_mean_absolute_percentage_error, save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Metrics values at increasing ground truth and predicted values"
        x_label = "Offset relative to the initial dataset"
        y_label_left = "MAPE and SMAPE, %"
        y_label_right = "MAE"
        actual_label = "Actual values"
        predicted_label = "Predicted"
    elif mode == "rus":
        title = "Значения метрик при возрастании значений наблюдений и предсказаний"
        x_label = "Сдвиг относительно начального датасета"
        y_label_left = "MAPE и SMAPE, %"
        y_label_right = "MAE"
        actual_label = "Наблюдения"
        predicted_label = "Предсказания"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label, y_label_left, y_label_right, actual_label, predicted_label


def _calculate_metrics(ground_truth: np.array, predicted: np.array):
    mape_metric = mean_absolute_percentage_error(ground_truth, predicted) * 100
    smape_metric = symmetric_mean_absolute_percentage_error(ground_truth, predicted)

    mae_metric = mean_absolute_error(y_pred=predicted, y_true=ground_truth)

    return {"mape": mape_metric, "smape": smape_metric, "mae": mae_metric}


def difference_between_mape_and_smape(mode: str = "eng"):
    title, x_label, y_label_left, y_label_right, actual_label, predicted_label = annotations_by_language(mode)

    ground_truth_values = np.array([1, 2, 3])
    predicted_values = np.array([1, 10, 50])

    initial_result = _calculate_metrics(ground_truth_values, predicted_values)
    incremented = [0]
    mape_values = [initial_result["mape"]]
    smape_values = [initial_result["smape"]]
    mae_values = [initial_result["mae"]]

    for i in np.arange(10, 110, 10):
        result = _calculate_metrics(ground_truth_values + i, predicted_values + i)

        incremented.append(i)
        mape_values.append(result["mape"])
        smape_values.append(result["smape"])
        mae_values.append(result["mae"])

    fig_size = (11, 7)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    ax.plot(incremented, mape_values, c='#6aa4f9', alpha=0.6, label="MAPE")
    ax.scatter(incremented, mape_values, c='#6aa4f9')

    ax.plot(incremented, smape_values, c='blue', alpha=0.6, label="SMAPE")
    ax.scatter(incremented, smape_values, c='blue')
    ax.grid()
    ax.tick_params(axis='y', colors='blue')
    ax.yaxis.label.set_color('blue')
    ax.set_xlabel(x_label, fontdict=FONTDICT)
    ax.set_ylabel(y_label_left, fontdict=FONTDICT, color="blue")

    # Add text labels
    for i in range(2):
        ax.text(incremented[i] + 2, mape_values[i] + 5, f"{mape_values[i]:.0f}",
                color='#6aa4f9', fontsize=12, va='bottom', ha='center')
        ax.text(incremented[i] + 2, smape_values[i] + 5, f"{smape_values[i]:.0f}", color='blue', fontsize=12,
                va='bottom', ha='center')
    ax.text(incremented[-1] + 2, mape_values[-1] + 5, f"{mape_values[-1]:.0f}",
            color='#6aa4f9', fontsize=12, va='bottom', ha='center')
    ax.text(incremented[-1] - 2, smape_values[-1] + 5, f"{smape_values[-1]:.0f}", color='blue', fontsize=12,
            va='bottom', ha='center')

    ax2 = ax.twinx()
    mae_line, = ax2.plot(incremented, mae_values, c='green', alpha=0.6, label="MAE")
    ax2.scatter(incremented, mae_values, c='green')
    ax2.spines['right'].set_color('green')
    ax2.spines['right'].set_linewidth(2.5)

    ax2.spines['left'].set_color('blue')
    ax2.spines['left'].set_linewidth(2.5)

    ax2.tick_params(axis='y', colors='green')
    ax2.yaxis.label.set_color('green')
    ax2.set_ylim(0, 50)
    ax2.set_ylabel(y_label_right, fontdict=FONTDICT, color="green")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mae_line)
    labels.append('MAE')
    ax.legend(handles=handles, labels=labels, loc='upper center', prop={'family': FONTNAME, 'size': 16})

    ax.text(-0.05, -0.12, f"{actual_label}: {ground_truth_values}", transform=ax.transAxes,
            fontsize=12, fontname=FONTNAME, bbox=dict(boxstyle="round", facecolor="#dbdbdb", edgecolor="gray"))

    ax.text(-0.05, -0.18, f"{predicted_label}: {predicted_values}", transform=ax.transAxes,
            fontsize=12, fontname=FONTNAME, bbox=dict(boxstyle="round", facecolor="#dbdbdb", edgecolor="gray"))

    ax.text(0.8, -0.12, f"{actual_label}: {ground_truth_values + 100}", transform=ax.transAxes,
            fontsize=12, fontname=FONTNAME, bbox=dict(boxstyle="round", facecolor="#dbdbdb", edgecolor="gray"))

    ax.text(0.8, -0.18, f"{predicted_label}: {predicted_values + 100}", transform=ax.transAxes,
            fontsize=12, fontname=FONTNAME, bbox=dict(boxstyle="round", facecolor="#dbdbdb", edgecolor="gray"))

    raw_svg_file = Path(get_plots_path(), f"17_difference_mape_smape_metrics_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"17_difference_mape_smape_metrics_{mode}.png"),
                                    template_name="template_green.svg")


if __name__ == '__main__':
    difference_between_mape_and_smape("rus")
