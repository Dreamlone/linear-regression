from pathlib import Path

from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import matplotlib.patches as patches

from examples.utils import save_plot_according_to_template, get_datasets, symmetric_mean_absolute_percentage_error

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
TRAIN_COLOR = "orange"
TEST_COLOR = "grey"

PREDICTION_INTERVAL_CONFIDENCE = 0.95


def _plot_metrics(ax, rooms: np.array, actual: np.array, predicted: np.array, dataset_name: str):
    """ Metrics visualization """
    # First - correlation coefficient
    ax2 = ax.twinx()  # R2
    ax3 = ax2.twinx()  # Bias
    ax4 = ax3.twinx()  # MAE
    ax5 = ax4.twinx()  # RMSE
    ax6 = ax5.twinx()  # MAPE
    ax7 = ax6.twinx()  # SMAPE
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
                f'{round(feature_corr.correlation, 2)}', ha='center', fontsize=6)
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=4)

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
                 f'{round(r2, 2)}', ha='center', fontsize=6)
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
    ax.text(x0, y0, '- 0', color='red', fontsize=6, va='bottom', ha='left', weight='bold')
    ax.text(x1, y1, f'- {height},', color='red', fontsize=6, va='top', ha='left', weight='bold')

    ########
    # Bias #
    ########
    bias = np.mean(np.ravel(predicted) - np.ravel(actual))
    print(f"{dataset_name}. Bias (actual vs predicted): {bias:.3f}")
    bias_bars = ax3.bar('Bias', bias, width=bar_width, color='#9beb8d')
    ax3.set_ylim(-5000, 5000)
    for bar in bias_bars:
        height = bar.get_height()

        if height < 0:
            # Need to shift a bit further
            shifted_height = height + (height * 0.3)
        else:
            shifted_height = height + 10

        ax3.text(bar.get_x() + bar.get_width() / 2, shifted_height,
                 f'{round(bias)}', ha='center', fontsize=6)
    ax3.yaxis.set_ticklabels([])
    ax3.axis('off')

    #######
    # MAE #
    #######
    mae_metric = mean_absolute_error(y_pred=predicted, y_true=actual)
    print(f"{dataset_name}. MAE (actual vs predicted): {mae_metric:.3f}")
    mae_bars = ax4.bar('MAE', mae_metric, width=bar_width, color='#76df63')
    ax4.set_ylim(-5000, 5000)
    for bar in mae_bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(mae_metric)}', ha='center', fontsize=6)
    ax4.yaxis.set_ticklabels([])
    ax4.axis('off')

    ########
    # RMSE #
    ########
    rmse_metric = root_mean_squared_error(actual, predicted)
    print(f"{dataset_name}. RMSE (actual vs predicted): {rmse_metric:.3f}")
    rmse_bars = ax5.bar('RMSE', rmse_metric, width=bar_width, color='#20b206')
    ax5.set_ylim(-5000, 5000)
    for bar in rmse_bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 70,
                 f'{round(rmse_metric)}', ha='center', fontsize=6)
    ax5.yaxis.set_ticklabels([])
    ax5.axis('off')

    x_start = bias_bars[0].get_x() - 0.1
    x_end = rmse_bars[0].get_x() + rmse_bars[0].get_width()
    width = x_end - x_start + 0.1
    height = 10000

    # Add rectangle
    rect = patches.Rectangle(
        (x_start, -5000),  # (x, y) starting from base of bars
        width,  # total width covering bar1 and bar2
        height,  # height enough to cover both bars
        linewidth=1,
        edgecolor='green',
        facecolor='none'
    )
    green_line = patches.Rectangle((x_start, 0), width, 0, linewidth=1, edgecolor='green',
                                   facecolor='none', alpha=0.5)
    ax5.add_patch(rect)
    ax5.add_patch(green_line)
    x0, y0 = rect.get_xy()  # lower-left corner
    x1, y1 = x0, y0 + rect.get_height()  # upper-left corner
    ax5.text(x0, y0, '- -5000', color='green', fontsize=6, va='bottom', ha='left', weight='bold')
    ax5.text(x1, y1, f'- 5000, $', color='green', fontsize=6, va='top', ha='left', weight='bold')

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
                 f'{round(mape_metric, 1)}', ha='center', fontsize=6)
    ax6.yaxis.set_ticklabels([])
    ax6.axis('off')

    smape_metric = symmetric_mean_absolute_percentage_error(actual, predicted)
    print(f"{dataset_name}. SMAPE (actual vs predicted): {smape_metric:.3f}")
    smape_bars = ax7.bar('SMAPE', smape_metric, width=bar_width, color='#1360d1')
    ax7.set_ylim(0, 100)
    for bar in smape_bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{round(smape_metric, 1)}', ha='center', fontsize=6)
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
    ax7.text(x0, y0, '- 0', color='blue', fontsize=6, va='bottom', ha='left', weight='bold')
    ax7.text(x1, y1, f'- {height}, %', color='blue', fontsize=6, va='top', ha='left', weight='bold')


def _fit_the_model(rooms: np.array, actual_prices: np.array):
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

def _compute_prediction_interval(
    x_all: np.array,
    x_train: np.array,
    y_train: np.array,
    intercept: float,
    slope: float,
    confidence_level: float = PREDICTION_INTERVAL_CONFIDENCE,
):
    """
    Compute prediction interval for simple linear regression (one feature).
    Interval is based on training data only.
    """
    x_train = np.ravel(x_train)
    y_train = np.ravel(y_train)

    predicted_train = intercept + slope * x_train
    residuals = y_train - predicted_train
    sample_size = len(x_train)
    degrees_of_freedom = sample_size - 2

    # Residual standard error
    residual_variance = np.sum(residuals ** 2) / degrees_of_freedom
    residual_std = np.sqrt(residual_variance)

    # Geometry in x
    mean_x_train = np.mean(x_train)
    sum_squares_x = np.sum((x_train - mean_x_train) ** 2)

    # t critical value for two-sided interval
    alpha_level = 1.0 - confidence_level
    t_critical = stats.t.ppf(1.0 - alpha_level / 2.0, df=degrees_of_freedom)

    x_all = np.ravel(x_all)
    predicted_all = intercept + slope * x_all

    standard_error_prediction = residual_std * np.sqrt(
        1.0
        + 1.0 / sample_size
        + (x_all - mean_x_train) ** 2 / sum_squares_x
    )

    margin = t_critical * standard_error_prediction

    lower_bound = predicted_all - margin
    upper_bound = predicted_all + margin

    return lower_bound, upper_bound



def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Model diagnostics on training and test samples"
        x_label = "Number of rooms"
        x_label_residuals = "Predicted"
        y_label_residuals = "Residuals (actual - predicted)"
        y_label = "Price, $"
        train_label = "Train"
        test_label = "Test"
        metrics = "Metrics"
    elif mode == "rus":
        title = "Диагностика моделей на обучающей и тестовой выборках"
        x_label = "Количество комнат в квартире"
        x_label_residuals = "Предсказания"
        y_label_residuals = "Остатки (реальные - предсказанные)"
        y_label = "Стоимость, $"
        train_label = "Обучение"
        test_label = "Тест"
        metrics = "Метрики"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, x_label, y_label, train_label, test_label, x_label_residuals, y_label_residuals, metrics


def _plot_predicted_with_actual(
    ax,
    x_train,
    x_test,
    y_train,
    y_test,
    rooms,
    predicted,
    prediction_interval_lower,
    prediction_interval_upper,
    y_label,
    x_label,
    dataset_name,
    train_label: str,
    test_label: str
):
    """ Draw a simple predicted and actual values plot with prediction interval """
    # Train and test points
    ax.scatter(
        x_train,
        y_train,
        s=30,
        color=TRAIN_COLOR,
        label=train_label,
        edgecolor="black",
        alpha=0.6,
        zorder=3,
    )
    ax.scatter(
        x_test,
        y_test,
        s=30,
        color=TEST_COLOR,
        label=test_label,
        edgecolor="black",
        alpha=0.6,
        zorder=3,
    )
    ax.legend(loc='upper left', prop={'family': FONTNAME})

    # Prediction interval band
    ax.fill_between(
        rooms,
        prediction_interval_lower,
        prediction_interval_upper,
        color="red",
        alpha=0.1,
        zorder=1,
    )
    ax.plot(
        rooms,
        prediction_interval_lower,
        color="red",
        linewidth=1,
        alpha=0.5,
        zorder=1,
    )
    ax.plot(
        rooms,
        prediction_interval_upper,
        color="red",
        linewidth=1,
        alpha=0.5,
        zorder=1,
    )

    # Regression line
    ax.plot(rooms, predicted, '--', c="black", zorder=2)

    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(0, 65000)
    if dataset_name == "A":
        ax.set_ylabel(y_label, fontsize=10, fontdict={'fontname': FONTNAME})
    else:
        ax.yaxis.set_ticklabels([])
    ax.set_title(dataset_name, fontsize=12, fontdict={'fontname': FONTNAME})
    ax.text(0.5, 0.1, x_label, fontsize=10, fontdict={'fontname': FONTNAME},
            ha='center', va='top', transform=ax.transAxes)


def _plot_residuals(ax, actual_v, predicted_v, x_label, color):
    actual_v = np.ravel(actual_v)
    predicted_v = np.ravel(predicted_v)
    residuals = actual_v - predicted_v
    ax.scatter(predicted_v, residuals, s=20, c=color, edgecolor="black", alpha=0.7)
    ax.plot([0, 65000], [0, 0], '--', color="black", alpha=0.3)

    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, 65000)
    ax.text(0.5, 0.1, x_label, fontsize=6, fontdict={'fontname': FONTNAME},
            ha='center', va='top', transform=ax.transAxes)

    ax.set_xticks([10000, 30000, 50000])
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    return residuals


def _plot_hist(ax, residuals, color):
    kde = stats.gaussian_kde(residuals)

    rice_k = int(np.ceil(2 * len(residuals) ** (1 / 3)))
    ax.hist(residuals, density=True, range=(-8000, 8000),
            alpha=0.5, rwidth=0.9, bins=rice_k,
            color=color, orientation='horizontal')
    xx = np.linspace(-8000, 8000, 1000)
    ax.plot([0, 0.0008], [0, 0], '--', color="black", alpha=0.3)
    ax.plot(kde(xx), xx, color=color, linewidth=2)
    ax.grid(color='grey', alpha=0.1)
    ax.set_ylim(-8000, 8000)
    ax.set_xlim(0, 0.0008)
    ax.set_xticks([0, 0.0004])
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def plot_metrics_per_train_test_datasets(mode: str = "eng"):
    title, x_label, y_label, train_label, test_label, x_label_residuals, y_label_residuals, metrics = annotations_by_language(mode)
    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(nrows=4, ncols=6, figure=fig, hspace=0.22, wspace=0.1)

    # First row (wide) - (A, B, C)
    titles = ['A', 'B', 'C']
    all_datasets = [good_prices, bad_prices_first, bad_prices_second]
    current_col = 0
    for column_id in range(len(titles)):
        first_row_ax = fig.add_subplot(gs[0, 2 * column_id:2 * column_id + 2])

        x_train, x_test, y_train, y_test = train_test_split(rooms, all_datasets[column_id],
                                                            test_size=0.4, random_state=10)
        model_predicted_train, intercept, slope = _fit_the_model(x_train, y_train)
        model_predicted_test = [(intercept + slope * room) for room in x_test]
        model_predicted_all = np.array([(intercept + slope * room) for room in rooms])

        prediction_interval_lower, prediction_interval_upper = _compute_prediction_interval(
            x_all=rooms,
            x_train=x_train,
            y_train=y_train,
            intercept=intercept,
            slope=slope,
            confidence_level=PREDICTION_INTERVAL_CONFIDENCE,
        )

        _plot_predicted_with_actual(
            first_row_ax,
            x_train,
            x_test,
            y_train,
            y_test,
            rooms,
            model_predicted_all,
            prediction_interval_lower,
            prediction_interval_upper,
            y_label,
            x_label,
            titles[column_id],
            train_label,
            test_label,
        )
        first_row_ax.set_xticks([1, 2, 3, 4, 5])
        first_row_ax.tick_params(axis='x', labelsize=6)
        first_row_ax.tick_params(axis='y', labelsize=6)

        ###########
        # Metrics #
        ###########
        train_ax = fig.add_subplot(gs[1, current_col])
        train_ax.set_title(train_label, fontsize=8, fontdict={'fontname': FONTNAME})
        _plot_metrics(train_ax, x_train, y_train, model_predicted_train, titles[column_id])
        if column_id == 0:
            train_ax.set_ylabel(metrics, fontsize=10, fontdict={'fontname': FONTNAME})

        test_ax = fig.add_subplot(gs[1, current_col + 1])
        test_ax.set_title(test_label, fontsize=8, fontdict={'fontname': FONTNAME})
        _plot_metrics(test_ax, x_test, y_test, model_predicted_test, titles[column_id])

        #############
        # Residuals #
        #############
        train_ax = fig.add_subplot(gs[2, current_col])
        train_ax.set_title(train_label, fontsize=8, fontdict={'fontname': FONTNAME})
        train_residuals = _plot_residuals(train_ax, y_train, model_predicted_train, x_label_residuals, TRAIN_COLOR)
        if column_id != 0:
            train_ax.yaxis.set_ticklabels([])
        else:
            train_ax.set_ylabel(y_label_residuals, fontsize=6, fontdict={'fontname': FONTNAME})

        test_ax = fig.add_subplot(gs[2, current_col + 1])
        test_ax.set_title(test_label, fontsize=8, fontdict={'fontname': FONTNAME})
        test_residuals = _plot_residuals(test_ax, y_test, model_predicted_test, x_label_residuals, TEST_COLOR)
        test_ax.yaxis.set_ticklabels([])

        train_ax = fig.add_subplot(gs[3, current_col])
        train_ax.set_title(train_label, fontsize=8, fontdict={'fontname': FONTNAME})
        _plot_hist(train_ax, train_residuals, TRAIN_COLOR)
        if column_id != 0:
            train_ax.yaxis.set_ticklabels([])
        else:
            train_ax.set_ylabel(y_label_residuals, fontsize=6, fontdict={'fontname': FONTNAME})

        test_ax = fig.add_subplot(gs[3, current_col + 1])
        test_ax.set_title(test_label, fontsize=8, fontdict={'fontname': FONTNAME})
        _plot_hist(test_ax, test_residuals, TEST_COLOR)
        test_ax.yaxis.set_ticklabels([])

        current_col += 2

    # Minimum external indents
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.03, right=0.98)
    raw_svg_file = Path(get_plots_path(), f"24_datasets_metric_on_train_test_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches='tight')
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"24_datasets_metric_on_train_test_{mode}.png"))


if __name__ == '__main__':
    plot_metrics_per_train_test_datasets("rus")
    plot_metrics_per_train_test_datasets("eng")
