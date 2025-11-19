from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, get_datasets, split_train_test_manual, \
    symmetric_mean_absolute_percentage_error

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 65000
BEST_MODEL_RMSE = 3873


def _plot_metrics(ax, rooms: np.array, actual: np.array, predicted: np.array):
    """ Metrics visualization """
    # First - correlation coefficient
    ax2 = ax.twinx() # R2
    ax3 = ax2.twinx() # Bias
    ax4 = ax3.twinx() # MAE
    ax5 = ax4.twinx() # RMSE
    ax6 = ax5.twinx() # MAPE
    ax7 = ax6.twinx() # SMAPE
    bar_width = 0.6
    bars_alpha = 0.3

    feature_corr = stats.pearsonr(rooms, actual)
    target_corr = stats.pearsonr(actual, predicted)

    corr_bars = ax.bar('Correlation\ncoefficient', target_corr.correlation, width=bar_width, color='#ffa5a5',
                       alpha=bars_alpha)
    ax.set_ylim(0, 1.1)
    for bar in corr_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
                f'{round(feature_corr.correlation, 3)}', ha='center', fontsize=7)
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=5)

    ######
    # R2 #
    ######
    r2 = r2_score(actual, predicted)

    r2_bars = ax2.bar(r'$R^2$', r2, width=bar_width, color='#ff5454', alpha=bars_alpha)
    ax2.set_ylim(0, 1.1)
    for bar in r2_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
                 f'{round(r2, 3)}', ha='center', fontsize=7)
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
        facecolor='none',
        alpha=bars_alpha
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
    bias_bars = ax3.bar('Bias', bias, width=bar_width, color='#9beb8d', alpha=bars_alpha)
    ax3.set_ylim(0, 6000)
    for bar in bias_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 10,
                 f'{round(bias)}', ha='center', fontsize=7)
    ax3.yaxis.set_ticklabels([])
    ax3.axis('off')

    #######
    # MAE #
    #######
    mae_metric = mean_absolute_error(y_pred=predicted, y_true=actual)
    mae_bars = ax4.bar('MAE', mae_metric, width=bar_width, color='#76df63', alpha=bars_alpha)
    ax4.set_ylim(0, 6000)
    for bar in mae_bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(mae_metric)}', ha='center', fontsize=7)
    ax4.yaxis.set_ticklabels([])
    ax4.axis('off')

    ########
    # RMSE #
    ########
    rmse_metric = root_mean_squared_error(actual, predicted)
    rmse_bars = ax5.bar('RMSE', rmse_metric, width=bar_width, color='#20b206')
    ax5.set_ylim(0, 6000)
    for bar in rmse_bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(rmse_metric)}', ha='center', fontsize=7)
        ax5.plot([bar.get_x() - bar.get_width() / 3, bar.get_x() + bar.get_width()], [BEST_MODEL_RMSE, BEST_MODEL_RMSE],
                 linestyle='--', linewidth=1, c='black')
        ax5.text(bar.get_x() - bar.get_width(), BEST_MODEL_RMSE,
                 f'{BEST_MODEL_RMSE}', ha='center', fontsize=7)
    ax5.yaxis.set_ticklabels([])
    ax5.axis('off')

    x_start = bias_bars[0].get_x() - 0.1
    x_end = rmse_bars[0].get_x() + rmse_bars[0].get_width()
    width = x_end - x_start + 0.1
    height = 6000

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
    mape_bars = ax6.bar('MAPE', mape_metric, width=bar_width, color='#89b3f1', alpha=bars_alpha)
    ax6.set_ylim(0, 100)
    for bar in mape_bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(mape_metric, 1)}', ha='center', fontsize=7)
    ax6.yaxis.set_ticklabels([])
    ax6.axis('off')

    smape_metric = symmetric_mean_absolute_percentage_error(actual, predicted)
    smape_bars = ax7.bar('SMAPE', smape_metric, width=bar_width, color='#1360d1', alpha=bars_alpha)
    ax7.set_ylim(0, 100)
    for bar in smape_bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(smape_metric, 1)}', ha='center', fontsize=7)
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
        facecolor='none',
        alpha=bars_alpha
    )
    ax7.add_patch(rect)
    x0, y0 = rect.get_xy()  # lower-left corner
    x1, y1 = x0, y0 + rect.get_height()  # upper-left corner
    ax7.text(x0, y0, '- 0', color='blue', fontsize=7, va='bottom', ha='left', weight='bold')
    ax7.text(x1, y1, f'- {height}, %', color='blue', fontsize=7, va='top', ha='left', weight='bold')

    return rmse_metric


def plot_main_graph(ax, x, y, best_model_label, ransac_model, cooks_model, first_plot_title):
    ax.scatter(x, y, s=40, facecolors='white', edgecolors='black', linewidths=0.6, zorder=2)
    ax.set_ylabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.set_xlabel("X", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.set_xlim(0, 6)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black', zorder=3, label=best_model_label)
    cooks_b0_base, cooks_b1_base = 3720.4, 8709.3
    ax.plot([1, 5], [cooks_b0_base + cooks_b1_base * i for i in [1, 5]], c='orange', zorder=3, label=cooks_model)
    ransac_b0_base, ransac_b1_base = 1090, 9335
    ax.plot([1, 5], [ransac_b0_base + ransac_b1_base*i for i in [1, 5]], c='red', zorder=3, label=ransac_model)
    ax.legend(loc='upper left', prop={'family': FONTNAME, 'size': 8})
    ax.set_title(first_plot_title, fontdict={'fontsize': 16, 'fontname': FONTNAME})


def annotations_by_language(mode: str):
    if mode == "eng":
        title = ""
        first_plot_title = ""
        best_model_label = ""
        ransac_model = "RANSAC"
        cooks_model = ""
    elif mode == "rus":
        title = "Метрики моделей на генеральной совокупности"
        first_plot_title = "Модели построенные на\nочищенных данных"
        best_model_label = "Эталонная модель"
        ransac_model = "Консенсус случайной выборки"
        cooks_model = "Расстояние Кука"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, first_plot_title, best_model_label, ransac_model, cooks_model


def plot_ransac_and_cooks_models(mode: str = "eng"):
    title, first_plot_title, best_model_label, ransac_model, cooks_model = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concatenate([rooms, rooms, rooms])
    common_target = np.concatenate([good_prices, bad_prices_first, bad_prices_second])
    x, y, _, _ = split_train_test_manual(common_features, common_target, apply_distortion=True)

    fig_size = (12, 9)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)
    for ax in [axs[0, 0], axs[0, 2]]:
        ax.cla()  # Clear axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    axs[1, 1].set_visible(False)

    axs[0, 0].text(0.5, 0.3, cooks_model, transform=axs[0, 0].transAxes,
                   ha='center', va='center', color="orange", fontsize=18, fontdict=FONTDICT)
    axs[0, 0].text(0.5, 0.15, r"$\hat{y} = 3720 + 8709 \cdot x$", transform=axs[0, 0].transAxes,
                   ha='center', va='center', color="orange", fontsize=18, fontdict=FONTDICT)

    axs[0, 2].text(0.5, 0.35, f"{ransac_model}\n(RANSAC)", transform=axs[0, 2].transAxes,
                   ha='center', va='center', color="red", fontsize=18, fontdict=FONTDICT)
    axs[0, 2].text(0.5, 0.15, r"$\hat{y} = 1090 + 9335 \cdot x$", transform=axs[0, 2].transAxes,
                   ha='center', va='center', color="red", fontsize=18, fontdict=FONTDICT)

    plot_main_graph(axs[0, 1], x, y, best_model_label, ransac_model, cooks_model, first_plot_title)

    cooks_b0_base, cooks_b1_base = 3720.4, 8709.3
    cooks_predicted = [cooks_b0_base + cooks_b1_base * i for i in common_features]
    _plot_metrics(axs[1, 0], common_features, common_target, cooks_predicted)

    ransac_b0_base, ransac_b1_base = 1090, 9335
    ransac_predicted = [ransac_b0_base + ransac_b1_base * i for i in common_features]
    _plot_metrics(axs[1, 2], common_features, common_target, ransac_predicted)

    # Vertical offset (5% upwards) for some axes
    shift = 0.05
    for ax_to_move in [axs[1, 0], axs[1, 2]]:
        pos = ax_to_move.get_position()
        ax_to_move.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

    # Vertical offset (5% downwards) for some axes
    shift = 0.1
    for ax_to_move in [axs[0, 1]]:
        pos = ax_to_move.get_position()
        ax_to_move.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME}, va="top", y=0.98)

    raw_svg_file = Path(get_plots_path(), f"34_ransac_cook_models_{mode}.svg", bbox_inches='tight')
    final_plot = Path(get_plots_path(), f"34_ransac_cook_models_{mode}.png")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template.svg")


if __name__ == '__main__':
    plot_ransac_and_cooks_models("rus")
