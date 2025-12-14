from pathlib import Path

from matplotlib import patches
from matplotlib.gridspec import GridSpec
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path
from examples.utils import get_datasets, save_plot_according_to_template, COLOR_BY_DATASET

np.random.seed(1999)

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


def draw_parameter_sliders(ax, n_value=30, n_min=30, n_max=100, n_label: str = "",
                           noise_value=0.1, noise_min=0.0, noise_max=0.5, noise_label: str = "",
                           confidence_value=0.90, confidence_options=(0.90, 0.95, 0.99), confidence_label: str = ""):
    """Draw three pseudo-slider controls for n, noise and confidence on the given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Common layout for sliders
    slider_left = 0.25
    slider_right = 0.95
    slider_track_height = 0.02
    handle_radius = 0.015

    # Vertical positions for three sliders (from top to bottom)
    y_positions = {
        "n": 0.78,
        "noise": 0.50,
        "confidence": 0.22,
    }

    label_font = {"fontname": FONTNAME, "fontsize": 10}
    value_font = {"fontname": FONTNAME, "fontsize": 9}

    def draw_continuous_slider(y_center, label, value, vmin, vmax, fmt="{:.0f}"):
        """Draw a single continuous slider."""
        # Background label
        ax.text(-0.4, y_center, label, va="center", ha="left", **label_font)

        # Slider track (light grey rectangle)
        track_width = slider_right - slider_left
        track_bottom = y_center - slider_track_height / 2.0
        track_rect = patches.FancyBboxPatch(
            (slider_left, track_bottom),
            track_width,
            slider_track_height,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            edgecolor="0.7",
            facecolor="0.95",
            linewidth=1,
            zorder=1,
        )
        ax.add_patch(track_rect)

        # Normalized position of the handle
        if vmax > vmin:
            norm_value = (value - vmin) / (vmax - vmin)
        else:
            norm_value = 0.0
        norm_value = max(0.0, min(1.0, norm_value))  # clip to [0, 1]

        handle_x = slider_left + norm_value * track_width
        handle_y = y_center

        # Active track (filled from left to handle)
        active_rect = patches.FancyBboxPatch(
            (slider_left, track_bottom),
            (handle_x - slider_left),
            slider_track_height,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            edgecolor="none",
            facecolor="tab:red",
            alpha=0.3,
            zorder=2,
        )
        ax.add_patch(active_rect)

        # Handle
        handle = patches.Circle(
            (handle_x, handle_y),
            radius=handle_radius,
            facecolor="tab:red",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(handle)

        # Current value text on the right
        ax.text(
            1.2,
            y_center,
            fmt.format(value),
            va="center",
            ha="center",
            **value_font,
        )

    def draw_discrete_slider(y_center, label, value, options):
        """Draw a discrete slider with several fixed positions."""
        ax.text(-0.4, y_center, label, va="center", ha="left", **label_font)

        # Slider line
        ax.plot(
            [slider_left, slider_right],
            [y_center, y_center],
            color="0.7",
            linewidth=2,
            zorder=1,
        )

        num_options = len(options)
        if num_options < 2:
            return

        track_width = slider_right - slider_left

        # Draw discrete positions
        for option in options:
            index = options.index(option)
            if num_options == 1:
                norm_pos = 0.5
            else:
                norm_pos = index / (num_options - 1)
            x_pos = slider_left + norm_pos * track_width

            is_selected = (abs(option - value) < 1e-6)

            marker_facecolor = "tab:red" if is_selected else "white"
            marker_edgecolor = "tab:red" if is_selected else "0.5"
            marker_size = 50 if is_selected else 40

            ax.scatter(
                x_pos,
                y_center,
                s=marker_size,
                facecolor=marker_facecolor,
                edgecolor=marker_edgecolor,
                zorder=2 if is_selected else 1.5,
            )

            # Option label below the marker
            ax.text(
                x_pos,
                y_center - 0.07,
                f"{option:.2f}",
                va="top",
                ha="center",
                fontsize=8,
                fontname=FONTNAME,
                color="black" if is_selected else "0.4",
            )

        # Current value text on the right (duplicates highlight)
        ax.text(
            1.2,
            y_center,
            f"{value:.2f}",
            va="center",
            ha="center",
            **value_font,
        )

    # Draw individual sliders
    draw_continuous_slider(
        y_center=y_positions["n"],
        label=n_label,
        value=n_value,
        vmin=n_min,
        vmax=n_max,
        fmt="{:.0f}",
    )

    draw_continuous_slider(
        y_center=y_positions["noise"],
        label=noise_label,
        value=noise_value,
        vmin=noise_min,
        vmax=noise_max,
        fmt="{:.2f}",
    )

    draw_discrete_slider(
        y_center=y_positions["confidence"],
        label=confidence_label,
        value=confidence_value,
        options=list(confidence_options),
    )


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


def _compute_prediction_interval(
    x_train: np.array,
    y_train: np.array,
    intercept: float,
    slope: float,
    confidence_level: float,
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

    predicted_all = intercept + slope * x_train

    standard_error_prediction = residual_std * np.sqrt(
        1.0
        + 1.0 / sample_size
        + (x_train - mean_x_train) ** 2 / sum_squares_x
    )

    margin = t_critical * standard_error_prediction

    lower_bound = predicted_all - margin
    upper_bound = predicted_all + margin

    return lower_bound, upper_bound


def _plot_residuals(ax, actual_v, predicted_v, x_label, y_label):
    actual_v = np.ravel(actual_v)
    predicted_v = np.ravel(predicted_v)
    residuals = actual_v - predicted_v
    ax.scatter(predicted_v, residuals, s=30, c="red", edgecolor="black",
               alpha=0.4, zorder=2)
    ax.plot([-5, 105], [0, 0], '--', color="black", alpha=0.3, zorder=2)

    ax.grid(color='grey', alpha=0.1, zorder=1)
    ax.set_ylim(-75, 75)
    ax.set_xlim(-5, 105)
    ax.set_xlabel(x_label, fontdict=FONTDICT)
    ax.set_ylabel(y_label, fontdict=FONTDICT)

    return residuals


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label_res = "Number of the rooms in the apartment"
        y_label_res = "Price, $"
        title = ""
        interval_label = "Уровень доверия"
        model_label = "Model"
        n_component = "n (sample size)"
        noise_component = "noise"
        conf_component = "confidence level"
    elif mode == "rus":
        x_label_res = "Предсказания"
        y_label_res = "Остатки (реальные - предсказанные)"
        title = "Данные - модель - предсказательные интервалы"
        interval_label = "Предсказательный интервал"
        model_label = "Модель"
        n_component = "n (размер выборки)"
        noise_component = "шум"
        conf_component = "уровень доверия"

    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return x_label_res, y_label_res, title, interval_label, model_label, n_component, noise_component, conf_component


def add_row_label(fig: plt.Figure, gs: plt.GridSpec, row_index: int, text: str):
    row_box = gs[row_index, :].get_position(fig)
    y_center = (row_box.y0 + row_box.y1) / 2.0
    fig.text(
        0.02,
        y_center,
        text,
        va="center",
        ha="center",
        fontdict={"fontsize": 10, "fontname": FONTNAME},
    )


def generate_dataset(n: int, noise_component: float = None):
    x = np.linspace(0, 100, n)
    if noise_component is None:
        y = np.copy(x)
    else:
        # Use 50 (centroid) for proper scaling
        y = x + (50 * np.random.normal(0, noise_component, n))
    return x, y


def plot_prediction_intervals(mode: str = "eng"):
    """ Shows prediction intervals for A, B and C models with different confidence levels """
    (x_label_res, y_label_res, title, interval_label, model_label, n_component_label,
     noise_component_label, conf_component_label) = annotations_by_language(mode)

    # Get datasets and build models
    n_value = 80
    noise_component = 0.3
    x, y = generate_dataset(n=n_value, noise_component=noise_component)
    confidence_level = 0.99
    x_point_to_show = 70

    predicted, intercept, slope = _get_predicted(x, y)
    predicted_lower, predicted_upper = _compute_prediction_interval(
        x_train=x,
        y_train=y,
        intercept=intercept,
        slope=slope,
        confidence_level=confidence_level,
    )
    prediction_point_to_show = intercept + slope * x_point_to_show

    ###############################
    # Fit models from statsmodels #
    ###############################
    import statsmodels.api as sm
    alpha = 1 - confidence_level

    # x_train, y_train — твои данные (1D или 2D для X)
    x_train = x.reshape(-1, 1)  # для парной регрессии
    x_train = sm.add_constant(x_train)
    model = sm.OLS(y, x_train).fit()
    pred = model.get_prediction(x_train)
    pred_df = pred.summary_frame(alpha=alpha)
    predicted_lower_c, predicted_upper_c = np.array(pred_df["obs_ci_lower"]), np.array(pred_df["obs_ci_upper"])

    fig_size = (16, 14)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(4, 3, figure=fig)
    gs.update(hspace=0.7, wspace=0.7)

    ax_intervals = fig.add_subplot(gs[0:2, 0:2])
    ax_residuals = fig.add_subplot(gs[2:4, 0:2])
    ax_parameters = fig.add_subplot(gs[0, 2])
    ax_calculations = fig.add_subplot(gs[1:4, 2])

    fig.suptitle(title, fontsize=18, fontdict={'fontname': FONTNAME}, y=0.97)

    ax_intervals.scatter(x, y, s=30, c="grey", alpha=0.5, edgecolor="black", zorder=3)
    ax_intervals.grid(color='grey', alpha=0.1, zorder=2)
    ax_intervals.plot(x, predicted, '--', c="black", zorder=3, label=model_label)
    ax_intervals.fill_between(x, predicted_lower, predicted_upper, color="red", alpha=0.1, zorder=1,
                              label=interval_label)
    ax_intervals.plot(x, predicted_lower, color="red", linewidth=1, alpha=0.5, zorder=1)
    ax_intervals.plot(x, predicted_upper, color="red", linewidth=1, alpha=0.5, zorder=1)
    ax_intervals.set_ylim(-75, 175)
    ax_intervals.set_xlim(-5, 105)
    ax_intervals.legend(loc='upper left', prop={'family': FONTNAME, 'size': 14})
    ax_intervals.set_ylabel("y", fontsize=14, fontdict=FONTDICT)
    ax_intervals.set_xlabel("x", fontsize=14, fontdict=FONTDICT)
    ax_intervals.plot([-5, x_point_to_show],
                      [prediction_point_to_show, prediction_point_to_show], c='grey', alpha=0.8, zorder=2)
    ax_intervals.plot([x_point_to_show, x_point_to_show],
                      [-75, prediction_point_to_show], c='grey', alpha=0.8, zorder=2)
    # Draw x_0 label
    ax_intervals.scatter(x_point_to_show, -50, s=500, c="white", alpha=1.0, zorder=3)
    ax_intervals.text(x_point_to_show, -50, r"$x_0$",
                      va="center", ha="center", fontsize=14, fontname=FONTNAME, zorder=4)

    # Draw a prediction for this label
    ax_intervals.scatter(10, prediction_point_to_show, s=1500, c="white", alpha=1.0, zorder=3)
    ax_intervals.text(10, prediction_point_to_show, r"$\hat y(x_0)$",
                      va="center", ha="center", fontsize=14, fontname=FONTNAME, zorder=4)

    # TODO delete this part - I made just to verify that calculations are correct
    ax_intervals.plot(x, predicted_lower_c, color="purple", linewidth=1, alpha=0.5, zorder=1)
    ax_intervals.plot(x, predicted_upper_c, color="purple", linewidth=1, alpha=0.5, zorder=1)

    # Draw sliders with default values:
    draw_parameter_sliders(
        ax_parameters,
        n_value=n_value,
        n_min=10,
        n_max=100,
        n_label=n_component_label,
        noise_value=noise_component,
        noise_min=0.0,
        noise_max=0.5,
        noise_label=noise_component_label,
        confidence_value=confidence_level,
        confidence_options=(0.90, 0.95, 0.99),
        confidence_label=conf_component_label
    )

    residuals = _plot_residuals(ax_residuals, y, predicted, x_label_res, y_label_res)
    ax_residuals.plot([prediction_point_to_show, prediction_point_to_show], [-75, 0], c='grey', alpha=0.8)
    ax_residuals.scatter(prediction_point_to_show, -60, s=500, c="white", alpha=1.0, zorder=3)
    ax_residuals.text(prediction_point_to_show, -60, r"$\hat y(x_0)$",
                      va="center", ha="center", fontsize=14, fontname=FONTNAME, zorder=4)

    ax_calculations.axis("off")
    lines = []

    # Заголовок
    lines.append(r"Расчёт предсказательного интервала")
    lines.append("")  # пустая строка

    lines.append(r"1. Модель:")
    lines.append(r"   $\hat y = b_0 + b_1 x$")
    sign = "+" if slope >= 0 else "-"
    lines.append(
        rf"   $\hat y = {intercept:.2f} {sign} {abs(slope):.2f} \cdot x$"
    )
    lines.append("")

    lines.append(r"2. Остатки:")
    lines.append(r"   $e = y - \hat y$")
    lines.append("")

    lines.append(r"3. Оценка шума (остаточное стандартное отклонение):")
    s = np.sqrt(np.sum(residuals ** 2) / (n_value - 2))
    lines.append(
        rf"   $s = \sqrt{{\dfrac{{\sum e^2}}{{n - 2}}}} = {s:.2f}$"
    )
    lines.append("")

    s = np.sqrt(np.sum(residuals ** 2) / (n_value - 2))
    mean_x = np.mean(x_train)
    sum_squares_x = np.sum((x_train - mean_x) ** 2)
    se_pred_x0 = s * np.sqrt(
        1.0
        + 1.0 / n_value
        + (x_point_to_show - mean_x) ** 2 / sum_squares_x
    )
    lines.append(
        rf"4. Стандартная ошибка предсказания в точке $x_0$ "
        rf"($x_0 = {x_point_to_show:.0f}$):"
    )
    lines.append(
        rf"   $se_{{\mathrm{{predicted}}}}(x_0) = "
        rf"{s:.2f} \cdot \sqrt{{1 + \dfrac{{1}}{{{n_value}}} + "
        rf"\dfrac{{({x_point_to_show:.2f} - {mean_x:.2f})^2}}{{{sum_squares_x:.2f}}}}}"
        rf" = {se_pred_x0:.2f}$"
    )
    lines.append("")

    dof = n_value - 2
    alpha_level = 1.0 - confidence_level
    t_crit = stats.t.ppf(1.0 - alpha_level / 2.0, df=dof)
    y_hat_0 = intercept + slope * x_point_to_show
    margin_x0 = t_crit * se_pred_x0

    lower_x0 = y_hat_0 - margin_x0
    upper_x0 = y_hat_0 + margin_x0
    ax_intervals.scatter([x_point_to_show, x_point_to_show], [lower_x0, upper_x0],
                         marker="x", s=50, c="black", alpha=1.0, zorder=3)
    lines.append(
        rf"5. Границы предсказательного интервала ({int(confidence_level * 100)}\%):"
    )

    lines.append(
        rf"   $\hat y_0 \pm t_{{\alpha/2,\,{dof}}} \cdot se_{{\mathrm{{pred}}}}(x_0)"
        rf" = {y_hat_0:.2f} \pm {t_crit:.2f} \cdot {se_pred_x0:.2f}"
        rf" = [{lower_x0:.2f};\, {upper_x0:.2f}]$"
    )
    lines.append("")

    lines.append(r"Где:")
    lines.append(r"   $x$ — значения признака;")
    lines.append(r"   $y$ — реальные значения отклика;")
    lines.append(r"   $\hat y$ (predicted) — прогноз модели;")
    lines.append(r"   $b_0$ (intercept) и $b_1$ (slope) — коэффициенты модели;")
    lines.append(r"   $n$ — размер выборки.")

    text = "\n".join(lines)

    ax_calculations.text(
        -0.2,
        1.0,
        text,
        transform=ax_calculations.transAxes,
        va="top",
        fontsize=13,
        fontname=FONTNAME,
    )

    raw_svg_file = Path(get_plots_path(), f"20_1_prediction_intervals_{mode}.svg")
    plt.savefig(raw_svg_file, bbox_inches="tight")
    plt.close()

    save_plot_according_to_template(
        raw_svg_file,
        Path(get_plots_path(), f"20_1_prediction_intervals_{mode}.png"),
    )


if __name__ == "__main__":
    plot_prediction_intervals("rus")
