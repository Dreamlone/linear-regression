from pathlib import Path

from matplotlib import gridspec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.paths import get_plots_path
from examples.utils import get_datasets, symmetric_mean_absolute_percentage_error, save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def _get_predicted(rooms: np.array, actual_prices: np.array):
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


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label = "Number of the rooms in the apartment"
        y_label = "Price, $"
        title = "Linear regression fitted to the full population"
        price = "price"
        room_label = "rooms number"
    elif mode == "rus":
        x_label = "Количество комнат в квартире"
        y_label = "Стоимость, $"
        title = "Линейная регрессия построенная по генеральной совокупности"
        price = "цена"
        room_label = "кол-во комнат"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label, y_label, title, price, room_label


def _plot_metrics(ax, rooms: np.array, actual: np.array, predicted: np.array, dataset_name: str):
    """ Metrics visualization """
    # First - correlation coefficient
    ax2 = ax.twinx() # R2
    ax3 = ax2.twinx() # Bias
    ax4 = ax3.twinx() # MAE
    ax5 = ax4.twinx() # RMSE
    ax6 = ax5.twinx() # MAPE
    ax7 = ax6.twinx() # SMAPE
    bar_width = 0.6

    feature_corr = stats.pearsonr(rooms, actual)
    target_corr = stats.pearsonr(actual, predicted)

    print(f"--- --- {dataset_name} --- ---")
    print(f"{dataset_name}. Pearson correlation coefficient (feature vs actual): {feature_corr.correlation:.3f}")
    print(f"{dataset_name}. Pearson correlation coefficient (actual vs predicted): {target_corr.correlation:.3f}")

    corr_bars = ax.bar('Correlation\ncoefficient', target_corr.correlation, width=bar_width, color='#ffa5a5')
    ax.set_ylim(0, 1.1)
    for bar in corr_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
                f'{round(feature_corr.correlation, 3)}', ha='center')
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=6)

    ######
    # R2 #
    ######
    r2 = r2_score(actual, predicted)
    print(f"{dataset_name}. R2 (actual vs predicted): {r2:.3f}")

    r2_bars = ax2.bar(r'$R^2$', r2, width=bar_width, color='#ff5454')
    ax2.set_ylim(0, 1.1)
    for bar in r2_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
                 f'{round(r2, 3)}', ha='center')
    ax2.yaxis.set_ticklabels([])
    ax2.axis('off')

    # Draw bounding box for correlation coefficients
    x_start = corr_bars[0].get_x() - 0.1
    x_end = r2_bars[0].get_x() + r2_bars[0].get_width()
    width = x_end - x_start + 0.1
    height = 1.1

    # Add rectangle
    rect = patches.Rectangle(
        (x_start, 0),  # (x, y) starting from base of bars
        width,  # total width covering bar1 and bar2
        height,  # height enough to cover both bars
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    x0, y0 = rect.get_xy()  # lower-left corner
    x1, y1 = x0, y0 + rect.get_height()  # upper-left corner
    ax.text(x0, y0, '- 0', color='red', fontsize=7, va='bottom', ha='left', weight='bold')
    ax.text(x1, y1, f'- {height},', color='red', fontsize=7, va='top', ha='left', weight='bold')

    ########
    # Bias #
    ########
    bias = np.mean(np.ravel(predicted) - np.ravel(actual))
    print(f"{dataset_name}. Bias (actual vs predicted): {bias:.3f}")
    bias_bars = ax3.bar('Bias', bias, width=bar_width, color='#9beb8d')
    ax3.set_ylim(0, 5000)
    for bar in bias_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 10,
                 f'{round(bias)}', ha='center')
    ax3.yaxis.set_ticklabels([])
    ax3.axis('off')

    #######
    # MAE #
    #######
    mae_metric = mean_absolute_error(y_pred=predicted, y_true=actual)
    print(f"{dataset_name}. MAE (actual vs predicted): {mae_metric:.3f}")
    mae_bars = ax4.bar('MAE', mae_metric, width=bar_width, color='#76df63')
    ax4.set_ylim(0, 5000)
    for bar in mae_bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(mae_metric)}', ha='center')
    ax4.yaxis.set_ticklabels([])
    ax4.axis('off')

    ########
    # RMSE #
    ########
    rmse_metric = root_mean_squared_error(actual, predicted)
    print(f"{dataset_name}. RMSE (actual vs predicted): {rmse_metric:.3f}")
    rmse_bars = ax5.bar('RMSE', rmse_metric, width=bar_width, color='#20b206')
    ax5.set_ylim(0, 5000)
    for bar in rmse_bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(rmse_metric)}', ha='center')
    ax5.yaxis.set_ticklabels([])
    ax5.axis('off')

    x_start = bias_bars[0].get_x() - 0.1
    x_end = rmse_bars[0].get_x() + rmse_bars[0].get_width()
    width = x_end - x_start + 0.1
    height = 5000

    # Add rectangle
    rect = patches.Rectangle(
        (x_start, 0),  # (x, y) starting from base of bars
        width,  # total width covering bar1 and bar2
        height,  # height enough to cover both bars
        linewidth=1,
        edgecolor='green',
        facecolor='none'
    )
    ax5.add_patch(rect)
    x0, y0 = rect.get_xy()  # lower-left corner
    x1, y1 = x0, y0 + rect.get_height()  # upper-left corner
    ax5.text(x0, y0, '- 0', color='green', fontsize=7, va='bottom', ha='left', weight='bold')
    ax5.text(x1, y1, f'- {height}, $', color='green', fontsize=7, va='top', ha='left', weight='bold')

    ########
    # MAPE #
    ########
    mape_metric = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"{dataset_name}. MAPE (actual vs predicted): {mape_metric:.3f}")
    mape_bars = ax6.bar('MAPE', mape_metric, width=bar_width, color='#89b3f1')
    ax6.set_ylim(0, 100)
    for bar in mape_bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(mape_metric, 1)}', ha='center')
    ax6.yaxis.set_ticklabels([])
    ax6.axis('off')

    smape_metric = symmetric_mean_absolute_percentage_error(actual, predicted)
    print(f"{dataset_name}. SMAPE (actual vs predicted): {smape_metric:.3f}")
    smape_bars = ax7.bar('SMAPE', smape_metric, width=bar_width, color='#1360d1')
    ax7.set_ylim(0, 100)
    for bar in smape_bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(smape_metric, 1)}', ha='center')
    ax7.yaxis.set_ticklabels([])
    ax7.axis('off')

    x_start = mape_bars[0].get_x() - 0.1
    x_end = smape_bars[0].get_x() + smape_bars[0].get_width()
    width = x_end - x_start + 0.1
    height = 100

    rect = patches.Rectangle(
        (x_start, 0),  # (x, y) starting from base of bars
        width,  # total width covering bar1 and bar2
        height,  # height enough to cover both bars
        linewidth=1,
        edgecolor='blue',
        facecolor='none'
    )
    ax7.add_patch(rect)
    x0, y0 = rect.get_xy()  # lower-left corner
    x1, y1 = x0, y0 + rect.get_height()  # upper-left corner
    ax7.text(x0, y0, '- 0', color='blue', fontsize=7, va='bottom', ha='left', weight='bold')
    ax7.text(x1, y1, f'- {height}, %', color='blue', fontsize=7, va='top', ha='left', weight='bold')


def plot_all_datasets_in_one(mode: str = "eng"):
    x_label, y_label, title, price, room_label = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concat([rooms, rooms, rooms])
    common_target = np.concat([good_prices, bad_prices_first, bad_prices_second])
    predicted, intercept, slope = _get_predicted(common_features, common_target)

    # Font and figure settings
    equation = rf"{price} = ${intercept:.1f} + {slope:.1f} \cdot${room_label}"

    fontname = "Comic Sans MS"
    fig_size = (12, 5)
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.scatter(common_features, common_target, color='#AA8C00', edgecolor="black", alpha=0.5, s=50)
    ax1.plot([min(common_features), max(common_features)], [min(predicted), max(predicted)], '--', c="black")
    print(f"Model line x: {[min(common_features), max(common_features)]}")
    print(f"Model line y: {[min(predicted), max(predicted)]}")

    ax1.set_xlabel(x_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax1.set_ylabel(y_label,
                  fontdict={'fontsize': 14, 'fontname': fontname})
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_yticks([10000, 20000, 30000, 40000, 50000])
    ax1.grid(color='grey', alpha=0.1)
    ax1.text(0.37, 0.1, equation, fontsize=12, fontdict={'fontname': FONTNAME},
             ha='center', va='top', transform=ax1.transAxes)

    _plot_metrics(ax2, common_features, common_target,
                  predicted, "D")
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(get_plots_path(), f"26_common_data_model_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"26_common_data_model_{mode}.png"),
                                    template_name="template_small.svg")


if __name__ == '__main__':
    plot_all_datasets_in_one("rus")
    plot_all_datasets_in_one("eng")
