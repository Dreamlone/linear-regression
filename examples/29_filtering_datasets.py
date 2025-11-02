from pathlib import Path
from typing import Union

from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, get_datasets, split_train_test_manual, \
    symmetric_mean_absolute_percentage_error

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = -25000
MAX_Y = 85000
BEST_MODEL_RMSE = 3873
N_BINS = 12


def _prettify_plot(ax, x: Union[np.array, None] = None,
                   lower_bound: Union[float, None] = None,
                   upper_bound: Union[float, None] = None):
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.grid(alpha=0.3)
    if lower_bound is not None and x is not None:
        ax.plot([min(x), max(x)], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    if upper_bound is not None and x is not None:
        ax.plot([min(x), max(x)], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])


def _draw_model_plot(ax, x_in: np.array, y_in: np.array, new_model_label: str, best_model_label: str,
                     common_features: np.array):
    predicted, intercept, slope = _build_model(x_in, y_in)
    eq = rf"$y = {intercept:.0f} {'+' if slope >= 0 else '-'} {abs(slope):.0f}x$"
    ax.set_title(eq, fontsize=10, fontdict={'fontname': FONTNAME})
    ax.scatter(x_in, y_in, color='#5b94e5', edgecolor="black", s=10, alpha=0.2)
    ax.plot(x_in, predicted, zorder=1, label=new_model_label)
    # Best model which were identified previously
    ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black', zorder=2, label=best_model_label)
    ax.legend(loc='upper left', prop={'family': FONTNAME, 'size': 5})
    ax.set_ylim(0, 70000)
    ax.grid(alpha=0.8)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])

    # Make predictions on all data
    predicted = [(intercept + slope * room) for room in common_features]
    return np.array(predicted)


def _build_model(rooms: np.array, actual_prices: np.array):
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
                f'{round(feature_corr.correlation, 3)}', ha='center', fontsize=5)
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
                 f'{round(r2, 3)}', ha='center', fontsize=5)
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
    ax.text(x0, y0, '- 0', color='red', fontsize=5, va='bottom', ha='left', weight='bold')
    ax.text(x1, y1, f'- {height},', color='red', fontsize=5, va='top', ha='left', weight='bold')

    ########
    # Bias #
    ########
    bias = np.mean(np.ravel(predicted) - np.ravel(actual))
    bias_bars = ax3.bar('Bias', bias, width=bar_width, color='#9beb8d', alpha=bars_alpha)
    ax3.set_ylim(0, 6000)
    for bar in bias_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 10,
                 f'{round(bias)}', ha='center', fontsize=5)
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
                 f'{round(mae_metric)}', ha='center', fontsize=5)
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
                 f'{round(rmse_metric)}', ha='center', fontsize=5)
        ax5.plot([bar.get_x() - bar.get_width() / 3, bar.get_x() + bar.get_width()], [BEST_MODEL_RMSE, BEST_MODEL_RMSE],
                 linestyle='--', linewidth=1, c='black')
        ax5.text(bar.get_x() - bar.get_width(), BEST_MODEL_RMSE,
                 f'{BEST_MODEL_RMSE}', ha='center', fontsize=5)
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
    ax5.text(x0, y0, '- 0', color='green', fontsize=5, va='bottom', ha='left', weight='bold')
    ax5.text(x1, y1, f'- {height}, $', color='green', fontsize=5, va='top', ha='left', weight='bold')

    ########
    # MAPE #
    ########
    mape_metric = mean_absolute_percentage_error(actual, predicted) * 100
    mape_bars = ax6.bar('MAPE', mape_metric, width=bar_width, color='#89b3f1', alpha=bars_alpha)
    ax6.set_ylim(0, 100)
    for bar in mape_bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(mape_metric, 1)}', ha='center', fontsize=5)
    ax6.yaxis.set_ticklabels([])
    ax6.axis('off')

    smape_metric = symmetric_mean_absolute_percentage_error(actual, predicted)
    smape_bars = ax7.bar('SMAPE', smape_metric, width=bar_width, color='#1360d1', alpha=bars_alpha)
    ax7.set_ylim(0, 100)
    for bar in smape_bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(smape_metric, 1)}', ha='center', fontsize=5)
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
    ax7.text(x0, y0, '- 0', color='blue', fontsize=5, va='bottom', ha='left', weight='bold')
    ax7.text(x1, y1, f'- {height}, %', color='blue', fontsize=5, va='top', ha='left', weight='bold')

    return rmse_metric


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        0.00,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 11, "fontname": FONTNAME},
    )


def filter_iqr(x, y, whisker=1.5):
    y = np.asarray(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    lower_bound = q1 - whisker * iqr
    upper_bound = q3 + whisker * iqr
    print("IQR", lower_bound, upper_bound)
    mask_inliers = (y >= lower_bound) & (y <= upper_bound)

    x_in = x[mask_inliers]
    y_in = y[mask_inliers]
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]

    return lower_bound, upper_bound, x_in, y_in, x_out, y_out


def filter_3sigma(x, y, threshold: float = 1.0):
    y = np.asarray(y)
    mean_y = np.mean(y)
    std_y = np.std(y)

    lower_bound = mean_y - threshold * std_y
    upper_bound = mean_y + threshold * std_y

    mask_inliers = (y >= lower_bound) & (y <= upper_bound)

    x_in = x[mask_inliers]
    y_in = y[mask_inliers]
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]

    return lower_bound, upper_bound, x_in, y_in, x_out, y_out


def filter_zscore(x, y, threshold):
    y = np.asarray(y)
    mean_y = np.mean(y)
    std_y = np.std(y)

    z_scores = (y - mean_y) / std_y
    mask_inliers = np.abs(z_scores) <= threshold

    x_in = x[mask_inliers]
    y_in = y[mask_inliers]
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]

    lower_bound = mean_y - threshold * std_y
    upper_bound = mean_y + threshold * std_y

    return lower_bound, upper_bound, x_in, y_in, x_out, y_out


def filter_modified_zscore(x, y, threshold):
    y = np.asarray(y)
    median_y = np.median(y)
    mad_y = np.median(np.abs(y - median_y))

    if mad_y == 0:
        raise ValueError("MAD is zero - modified Z-score is not applicable")

    modified_z_scores = 0.6745 * (y - median_y) / mad_y
    mask_inliers = np.abs(modified_z_scores) <= threshold

    x_in = x[mask_inliers]
    y_in = y[mask_inliers]
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]
    lower_bound = median_y - (threshold * mad_y) / 0.6745
    upper_bound = median_y + (threshold * mad_y) / 0.6745

    return lower_bound, upper_bound, x_in, y_in, x_out, y_out


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Metrics are cool!"
        iqr_plot = ""
        sigma_plot = ""
        z_score_plot = ""
        z_score_modified_plot = ""
        main_plot = "Is model good or not?"
        new_model_label = ""
        best_model_label = ""
        second_row = ""
        third_row = ""
        fourth_row = ""
        final_row_plot = ""
    elif mode == "rus":
        title = "Инициализация моделей на очищенных от выбросов данных"
        iqr_plot = "Межквартильный размах\n(1IQR)"
        sigma_plot = "Метод 3х сигм\n(2σ)"
        z_score_plot = "Z-оценка\n(порог=1.5)"
        z_score_modified_plot = "Модифицированная Z-оценка\n(порог=1.5)"
        main_plot = "Данные с выбросами"
        new_model_label = "Новая модель по данным"
        best_model_label = "Эталонная модель"
        second_row = "Фильтрация выбросов\nразными способами"
        third_row = "Инициализация модели"
        fourth_row = (
            "Метрики\n"
            "на генеральной совокупности\n"
            r"$\mathbf{RMSE\ эталонной\ модели}$"
            "\n"
            r"$\mathbf{3873}$"
        )
        final_row_plot = "RMSE до эталонной модели"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (title, iqr_plot, sigma_plot, z_score_plot,
            z_score_modified_plot, main_plot, new_model_label,
            best_model_label, second_row, third_row, fourth_row, final_row_plot)


def generate_synthetic_data(n_samples=100, noise_std=1.0, outlier_fraction=0.1, random_state=2025):
    rng = np.random.default_rng(random_state)
    x = np.linspace(0, 10, n_samples)
    true_slope = 2.0
    true_intercept = 5.0
    y = true_slope * x + true_intercept + rng.normal(0, noise_std, size=n_samples)
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = rng.choice(n_samples, size=n_outliers, replace=False)
    y[outlier_indices] += rng.normal(0, 20 * noise_std, size=n_outliers)

    x = x + 17
    y = y + 17
    print(f"Min x: {min(x)}, Max x: {max(x)}")
    print(f"Min y: {min(y)}, Max x: {max(y)}")
    return x, y


def plot_filtering_sample_and_init_model(mode: str = "eng"):
    (title, iqr_plot, sigma_plot, z_score_plot, z_score_modified_plot,
     main_plot, new_model_label, best_model_label, second_row, third_row,
     fourth_row, final_row_plot) = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concat([rooms, rooms, rooms])
    common_target = np.concat([good_prices, bad_prices_first, bad_prices_second])

    x, y, distorted_x, distorted_y = split_train_test_manual(common_features, common_target, apply_distortion=True)

    fig_size = (11, 10)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(5, 4, figure=fig)
    gs.update(hspace=0.5)
    ax_main = fig.add_subplot(gs[0:2, 1:3])
    add_row_label(fig, gs, row_index=2, text=second_row)
    add_row_label(fig, gs, row_index=3, text=third_row)
    add_row_label(fig, gs, row_index=4, text=fourth_row)

    ax_main.set_title(main_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_main.scatter(x, y, color='#5b94e5', edgecolor="black", s=50, alpha=0.8)
    ax_main.set_ylim(MIN_Y, MAX_Y)
    ax_main.grid(alpha=0.3)
    ax_main.xaxis.set_ticks([1, 2, 3, 4, 5])
    ax_main.set_ylabel("y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_main.set_xlabel("x", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    # Change the vertical size of the plot so it won't overlap
    pos = ax_main.get_position()
    new_height = pos.height * 0.8
    ax_main.set_position([pos.x0, pos.y0 + (pos.height - new_height), pos.width, new_height])

    # Find outliers according IQR
    lower_bound, upper_bound, x_in, y_in, x_out, y_out = filter_iqr(x, y, whisker=1.0)
    ax_iqr = fig.add_subplot(gs[2, 0])
    ax_iqr.set_title(iqr_plot, fontsize=10, fontdict={'fontname': FONTNAME})
    ax_iqr.scatter(x_in, y_in, color='#5b94e5', edgecolor="black", s=10, alpha=0.8)
    ax_iqr.scatter(x_out, y_out, color='red', edgecolor="black", s=40, alpha=0.8)
    _prettify_plot(ax_iqr, x, lower_bound, upper_bound)

    ax_iqr_model = fig.add_subplot(gs[3, 0])
    predicted = _draw_model_plot(ax_iqr_model, x_in, y_in, new_model_label, best_model_label,
                                 common_features)

    ax_iqr_metrics = fig.add_subplot(gs[4, 0])
    rmse_metric = _plot_metrics(ax_iqr_metrics, common_features, common_target, predicted)
    rmse_delta = rmse_metric - BEST_MODEL_RMSE
    ax_iqr_metrics.set_title(f"{final_row_plot}\n{rmse_delta:.0f}",
                             fontsize=10, fontdict={'fontname': FONTNAME})

    # 3 sigma
    lower_bound, upper_bound, x_in, y_in, x_out, y_out = filter_3sigma(x, y, threshold=2)
    ax_3sigma = fig.add_subplot(gs[2, 1])
    ax_3sigma.set_title(sigma_plot, fontsize=10, fontdict={'fontname': FONTNAME})
    ax_3sigma.scatter(x_in, y_in, color='#5b94e5', edgecolor="black", s=10, alpha=0.8)
    ax_3sigma.scatter(x_out, y_out, color='red', edgecolor="black", s=40, alpha=0.8)
    _prettify_plot(ax_3sigma, x, lower_bound, upper_bound)

    ax_3sigma_model = fig.add_subplot(gs[3, 1])
    predicted = _draw_model_plot(ax_3sigma_model, x_in, y_in, new_model_label, best_model_label,
                                 common_features)
    ax_3sigma_metrics = fig.add_subplot(gs[4, 1])
    rmse_metric = _plot_metrics(ax_3sigma_metrics, common_features, common_target, predicted)
    rmse_delta = rmse_metric - BEST_MODEL_RMSE
    ax_3sigma_metrics.set_title(f"{final_row_plot}\n{rmse_delta:.0f}",
                                fontsize=10, fontdict={'fontname': FONTNAME})

    # Z score
    lower_bound, upper_bound, x_in, y_in, x_out, y_out = filter_zscore(x, y, threshold=1.5)
    ax_z_score = fig.add_subplot(gs[2, 2])
    ax_z_score.set_title(z_score_plot, fontsize=10, fontdict={'fontname': FONTNAME})
    ax_z_score.scatter(x_in, y_in, color='#5b94e5', edgecolor="black", s=10, alpha=0.8)
    ax_z_score.scatter(x_out, y_out, color='red', edgecolor="black", s=40, alpha=0.8)
    _prettify_plot(ax_z_score, x, lower_bound, upper_bound)

    ax_z_score_model = fig.add_subplot(gs[3, 2])
    predicted = _draw_model_plot(ax_z_score_model, x_in, y_in, new_model_label, best_model_label,
                                 common_features)
    ax_z_score_metrics = fig.add_subplot(gs[4, 2])
    rmse_metric = _plot_metrics(ax_z_score_metrics, common_features, common_target, predicted)
    rmse_delta = rmse_metric - BEST_MODEL_RMSE
    ax_z_score_metrics.set_title(f"{final_row_plot}\n{rmse_delta:.0f}",
                                 fontsize=10, fontdict={'fontname': FONTNAME})

    # modified Z score
    lower_bound, upper_bound, x_in, y_in, x_out, y_out = filter_modified_zscore(x, y, threshold=1.5)
    ax_mod_z_score = fig.add_subplot(gs[2, 3])
    ax_mod_z_score.set_title(z_score_modified_plot, fontsize=10, fontdict={'fontname': FONTNAME})
    ax_mod_z_score.scatter(x_in, y_in, color='#5b94e5', edgecolor="black", s=10, alpha=0.8)
    ax_mod_z_score.scatter(x_out, y_out, color='red', edgecolor="black", s=40, alpha=0.8)
    _prettify_plot(ax_mod_z_score, x, lower_bound, upper_bound)

    ax_mod_z_score_model = fig.add_subplot(gs[3, 3])
    predicted = _draw_model_plot(ax_mod_z_score_model, x_in, y_in, new_model_label, best_model_label,
                                 common_features)
    ax_mod_z_score_metrics = fig.add_subplot(gs[4, 3])
    rmse_metric = _plot_metrics(ax_mod_z_score_metrics, common_features, common_target, predicted)
    rmse_delta = rmse_metric - BEST_MODEL_RMSE
    ax_mod_z_score_metrics.set_title(f"{final_row_plot}\n{rmse_delta:.0f}",
                                     fontsize=10, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(get_plots_path(), f"29_filtering_and_model_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()
    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"29_filtering_and_model_{mode}.png"))


if __name__ == '__main__':
    plot_filtering_sample_and_init_model("rus")
