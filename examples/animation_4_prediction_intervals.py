import shutil
from pathlib import Path

import imageio
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template

np.random.seed(1999)

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
ANIM_DURATION = 1800
DPI = 150

FIXED_INTERCEPT = 0.0
FIXED_SLOPE = 1.0
RNG_SEED = 1999


def draw_parameter_sliders(ax, n_value=30, n_min=30, n_max=100, n_label: str = "",
                           noise_value=0.1, noise_min=0.0, noise_max=0.5, noise_label: str = "",
                           confidence_value=0.90, confidence_options=(0.90, 0.95, 0.99), confidence_label: str = ""):
    """Draw three pseudo-slider controls for n, noise and confidence on the given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    slider_left = 0.25
    slider_right = 0.95
    slider_track_height = 0.02
    handle_radius = 0.015

    y_positions = {
        "n": 0.78,
        "noise": 0.50,
        "confidence": 0.22,
    }

    label_font = {"fontname": FONTNAME, "fontsize": 10}
    value_font = {"fontname": FONTNAME, "fontsize": 9}

    def draw_continuous_slider(y_center, label, value, vmin, vmax, fmt="{:.0f}"):
        """Draw a single continuous slider."""
        ax.text(-0.5, y_center, label, va="center", ha="left", **label_font)

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

        if vmax > vmin:
            norm_value = (value - vmin) / (vmax - vmin)
        else:
            norm_value = 0.0
        norm_value = max(0.0, min(1.0, norm_value))

        handle_x = slider_left + norm_value * track_width
        handle_y = y_center

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

        handle = patches.Circle(
            (handle_x, handle_y),
            radius=handle_radius,
            facecolor="tab:red",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(handle)

        ax.text(
            1.3,
            y_center,
            fmt.format(value),
            va="center",
            ha="center",
            **value_font,
        )

    def draw_discrete_slider(y_center, label, value, options):
        """Draw a discrete slider with several fixed positions."""
        ax.text(-0.5, y_center, label, va="center", ha="left", **label_font)

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

        for option in options:
            index = options.index(option)
            if num_options == 1:
                norm_pos = 0.5
            else:
                norm_pos = index / (num_options - 1)
            x_pos = slider_left + norm_pos * track_width

            is_selected = abs(option - value) < 1e-6

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

            ax.text(
                x_pos,
                y_center - 0.07,
                f"{option:.2f}",
                va="top",
                ha="center",
                fontsize=10,
                fontname=FONTNAME,
                color="black" if is_selected else "0.4",
            )

        ax.text(
            1.3,
            y_center,
            f"{value:.2f}",
            va="center",
            ha="center",
            **value_font,
        )

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

    residual_variance = np.sum(residuals ** 2) / degrees_of_freedom
    residual_std = np.sqrt(residual_variance)

    mean_x_train = np.mean(x_train)
    sum_squares_x = np.sum((x_train - mean_x_train) ** 2)

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

    ax.grid(color='grey', alpha=0.5, zorder=1)
    ax.set_ylim(-75, 75)
    ax.set_xlim(-5, 105)
    ax.set_xlabel(x_label, fontdict=FONTDICT)
    ax.set_ylabel(y_label, fontdict=FONTDICT)

    return residuals


def annotations_by_language(mode: str):
    if mode == "eng":
        x_label_res = "Predictions"
        y_label_res = "Residuals (actual - predicted)"
        title = "Data, model and prediction intervals"
        interval_label = "Prediction interval"
        model_label = "Model"
        n_component = "n (sample size)"
        noise_component = "noise"
        conf_component = "confidence level"
        calculations_title = "Prediction interval calculation"
        model_title = "1. Model:"
        residuals_title = "2. Residuals:"
        noise_title = "3. Noise estimate (residual standard deviation):"
        se_title_template = "4. Prediction standard error at point $x_0$ ($x_0 = {x0:.0f}$):"
        bounds_title_template = "5. Prediction interval bounds ({confidence:.0f}%):"
        lower_label = "Lower"
        upper_label = "Upper"
        where_label = "Where:"
        x_desc = "$x$ is the feature values;"
        y_desc = "$y$ is the actual target values;"
        yhat_desc = "$\\hat y$ (predicted) is the model prediction;"
        coef_desc = "$b_0$ (intercept) and $b_1$ (slope) are the model coefficients;"
        n_desc = "$n$ is the sample size."
    elif mode == "rus":
        x_label_res = "Предсказания"
        y_label_res = "Остатки (реальные - предсказанные)"
        title = "Данные, модель и предсказательные интервалы"
        interval_label = "Предсказательный интервал"
        model_label = "Модель"
        n_component = "n (размер выборки)"
        noise_component = "шум"
        conf_component = "уровень доверия"
        calculations_title = "Расчёт предсказательного интервала"
        model_title = "1. Модель:"
        residuals_title = "2. Остатки:"
        noise_title = "3. Оценка шума (остаточное стандартное отклонение):"
        se_title_template = "4. Стандартная ошибка предсказания в точке $x_0$ ($x_0 = {x0:.0f}$):"
        bounds_title_template = "5. Границы предсказательного интервала ({confidence:.0f}%):"
        lower_label = "Нижняя"
        upper_label = "Верхняя"
        where_label = "Где:"
        x_desc = "$x$ — значения признака;"
        y_desc = "$y$ — реальные значения отклика;"
        yhat_desc = "$\\hat y$ (predicted) — прогноз модели;"
        coef_desc = "$b_0$ (intercept) и $b_1$ (slope) — коэффициенты модели;"
        n_desc = "$n$ — размер выборки."
    else:
        raise NotImplementedError(f"Language {mode} is not supported")

    return {
        "x_label_res": x_label_res,
        "y_label_res": y_label_res,
        "title": title,
        "interval_label": interval_label,
        "model_label": model_label,
        "n_component": n_component,
        "noise_component": noise_component,
        "conf_component": conf_component,
        "calculations_title": calculations_title,
        "model_title": model_title,
        "residuals_title": residuals_title,
        "noise_title": noise_title,
        "se_title_template": se_title_template,
        "bounds_title_template": bounds_title_template,
        "lower_label": lower_label,
        "upper_label": upper_label,
        "where_label": where_label,
        "x_desc": x_desc,
        "y_desc": y_desc,
        "yhat_desc": yhat_desc,
        "coef_desc": coef_desc,
        "n_desc": n_desc,
    }


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


def generate_dataset(
    n: int,
    noise_component: float = None,
    intercept: float = FIXED_INTERCEPT,
    slope: float = FIXED_SLOPE,
):
    x = np.linspace(0, 100, n)
    y_true = intercept + slope * x

    if noise_component is None or noise_component == 0:
        y = np.copy(y_true)
    else:
        rng = np.random.default_rng(seed=RNG_SEED)
        y = y_true + 50 * rng.normal(0, noise_component, n)

    return x, y


def generate_cases():
    ns = [30, 30, 30,
          50, 70, 90,
          90, 90, 90,
          90, 90, 90]
    noises = [0.0, 0.1, 0.2,
              0.2, 0.2, 0.2,
              0.2, 0.2, 0.2,
              0.2, 0.2, 0.2]
    confidence_levels = [0.90, 0.90, 0.90,
                         0.90, 0.90, 0.90,
                         0.95, 0.95, 0.95,
                         0.99, 0.99, 0.99]
    xs = [70, 70, 70,
          70, 70, 70,
          70, 71, 72,
          71, 70, 69]
    for case in range(len(ns)):
        yield ns[case], noises[case], confidence_levels[case], xs[case]


def plot_prediction_intervals(mode: str = "eng"):
    labels = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists() and len(list(tmp_dir.iterdir())) > 0:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    frames = []
    for n_value, noise_component, confidence_level, x_point_to_show in generate_cases():
        x, y = generate_dataset(
            n=n_value,
            noise_component=noise_component,
            intercept=FIXED_INTERCEPT,
            slope=FIXED_SLOPE,
        )

        intercept = FIXED_INTERCEPT
        slope = FIXED_SLOPE
        predicted = intercept + slope * x

        predicted_lower, predicted_upper = _compute_prediction_interval(
            x_train=x,
            y_train=y,
            intercept=intercept,
            slope=slope,
            confidence_level=confidence_level,
        )
        prediction_point_to_show = intercept + slope * x_point_to_show

        fig_size = (16, 12)
        fig = plt.figure(figsize=fig_size)
        gs = GridSpec(4, 3, figure=fig)
        gs.update(hspace=0.7, wspace=0.7)

        ax_intervals = fig.add_subplot(gs[0:2, 0:2])
        ax_residuals = fig.add_subplot(gs[2:4, 0:2])
        ax_parameters = fig.add_subplot(gs[0, 2])
        ax_calculations = fig.add_subplot(gs[1:4, 2])

        fig.suptitle(
            labels["title"],
            fontsize=18,
            fontdict={'fontname': FONTNAME},
            y=0.97,
            x=0.55,
        )

        ax_intervals.scatter(x, y, s=30, c="grey", alpha=0.5, edgecolor="black", zorder=3)
        ax_intervals.grid(color='grey', alpha=0.5, zorder=2)
        ax_intervals.plot(x, predicted, '--', c="black", zorder=3, label=labels["model_label"])
        ax_intervals.fill_between(
            x,
            predicted_lower,
            predicted_upper,
            color="red",
            alpha=0.1,
            zorder=1,
            label=labels["interval_label"],
        )
        ax_intervals.plot(x, predicted_lower, color="red", linewidth=1, alpha=0.5, zorder=1)
        ax_intervals.plot(x, predicted_upper, color="red", linewidth=1, alpha=0.5, zorder=1)
        ax_intervals.set_ylim(-75, 175)
        ax_intervals.set_xlim(-5, 105)
        ax_intervals.legend(loc='upper left', prop={'family': FONTNAME, 'size': 14})
        ax_intervals.set_ylabel("y", fontsize=14, fontdict=FONTDICT)
        ax_intervals.set_xlabel("x", fontsize=14, fontdict=FONTDICT)
        ax_intervals.plot(
            [-5, x_point_to_show],
            [prediction_point_to_show, prediction_point_to_show],
            c='grey',
            alpha=0.8,
            zorder=2,
        )
        ax_intervals.plot(
            [x_point_to_show, x_point_to_show],
            [-75, prediction_point_to_show],
            c='grey',
            alpha=0.8,
            zorder=2,
        )

        ax_intervals.scatter(x_point_to_show, -50, s=500, c="white", alpha=1.0, zorder=3)
        ax_intervals.text(
            x_point_to_show,
            -50,
            r"$x_0$",
            va="center",
            ha="center",
            fontsize=14,
            fontname=FONTNAME,
            zorder=4,
        )

        ax_intervals.scatter(10, prediction_point_to_show, s=1500, c="white", alpha=1.0, zorder=3)
        ax_intervals.text(
            10,
            prediction_point_to_show,
            r"$\hat y(x_0)$",
            va="center",
            ha="center",
            fontsize=14,
            fontname=FONTNAME,
            zorder=4,
        )

        draw_parameter_sliders(
            ax_parameters,
            n_value=n_value,
            n_min=10,
            n_max=100,
            n_label=labels["n_component"],
            noise_value=noise_component,
            noise_min=0.0,
            noise_max=0.5,
            noise_label=labels["noise_component"],
            confidence_value=confidence_level,
            confidence_options=(0.90, 0.95, 0.99),
            confidence_label=labels["conf_component"],
        )

        residuals = _plot_residuals(
            ax_residuals,
            y,
            predicted,
            labels["x_label_res"],
            labels["y_label_res"],
        )
        ax_residuals.plot([prediction_point_to_show, prediction_point_to_show], [-75, 0], c='grey', alpha=0.8)
        ax_residuals.scatter(prediction_point_to_show, -60, s=500, c="white", alpha=1.0, zorder=3)
        ax_residuals.text(
            prediction_point_to_show,
            -60,
            r"$\hat y(x_0)$",
            va="center",
            ha="center",
            fontsize=14,
            fontname=FONTNAME,
            zorder=4,
        )

        ax_calculations.axis("off")
        lines = []

        lines.append(labels["calculations_title"])
        lines.append("")

        lines.append(labels["model_title"])
        lines.append(r"   $\hat y = b_0 + b_1 x$")
        sign = "+" if slope >= 0 else "-"
        lines.append(
            rf"   $\hat y = {intercept:.1f} {sign} {abs(slope):.1f} \cdot x$"
        )
        lines.append("")

        lines.append(labels["residuals_title"])
        lines.append(r"   $e = y - \hat y$")
        lines.append("")

        lines.append(labels["noise_title"])
        s = np.sqrt(np.sum(residuals ** 2) / (n_value - 2))
        lines.append(
            rf"   $s = \sqrt{{\dfrac{{\sum e^2}}{{n - 2}}}} = {s:.1f}$"
        )
        lines.append("")

        mean_x = np.mean(x)
        sum_squares_x = np.sum((x - mean_x) ** 2)
        se_pred_x0 = s * np.sqrt(
            1.0
            + 1.0 / n_value
            + (x_point_to_show - mean_x) ** 2 / sum_squares_x
        )

        lines.append(
            labels["se_title_template"].format(x0=x_point_to_show)
        )
        lines.append(
            r"   $se_{\mathrm{predicted}}(x_0) = "
            r"s \sqrt{1 + \dfrac{1}{n} + \dfrac{(x_0 - \bar x)^2}{\sum (x - \bar x)^2}}$"
        )
        lines.append(
            rf"   $se_{{\mathrm{{predicted}}}}(x_0) = "
            rf"{s:.1f} \cdot \sqrt{{1 + \dfrac{{1}}{{{n_value}}} + "
            rf"\dfrac{{({x_point_to_show:.1f} - {mean_x:.1f})^2}}{{{sum_squares_x:.1f}}}}}"
            rf" = {se_pred_x0:.1f}$"
        )
        lines.append("")

        dof = n_value - 2
        alpha_level = 1.0 - confidence_level
        t_crit = stats.t.ppf(1.0 - alpha_level / 2.0, df=dof)
        y_hat_0 = intercept + slope * x_point_to_show
        margin_x0 = t_crit * se_pred_x0

        lower_x0 = y_hat_0 - margin_x0
        upper_x0 = y_hat_0 + margin_x0
        ax_intervals.scatter(
            [x_point_to_show, x_point_to_show],
            [lower_x0, upper_x0],
            marker="x",
            s=50,
            c="black",
            alpha=1.0,
            zorder=3,
        )

        lines.append(
            labels["bounds_title_template"].format(confidence=confidence_level * 100)
        )
        lines.append(
            rf"   {labels['lower_label']}: "
            rf"$\hat y_0 - t_{{\alpha/2,\,{dof}}} \cdot se_{{\mathrm{{pred}}}}(x_0)"
            rf" = {y_hat_0:.0f} - {t_crit:.1f} \cdot {se_pred_x0:.1f}"
            rf" = {lower_x0:.1f}$"
        )
        lines.append(
            rf"   {labels['upper_label']}: "
            rf"$\hat y_0 + t_{{\alpha/2,\,{dof}}} \cdot se_{{\mathrm{{pred}}}}(x_0)"
            rf" = {y_hat_0:.0f} + {t_crit:.1f} \cdot {se_pred_x0:.1f}"
            rf" = {upper_x0:.1f}$"
        )
        lines.append("")

        lines.append(labels["where_label"])
        lines.append(f"   {labels['x_desc']}")
        lines.append(f"   {labels['y_desc']}")
        lines.append(f"   {labels['yhat_desc']}")
        lines.append(f"   {labels['coef_desc']}")
        lines.append(f"   {labels['n_desc']}")

        text = "\n".join(lines)

        ax_calculations.text(
            -0.5,
            1.0,
            text,
            transform=ax_calculations.transAxes,
            va="top",
            fontsize=12,
            fontname=FONTNAME,
        )

        raw_svg_file = Path(tmp_dir, f"animation_4_prediction_intervals_{mode}.svg")
        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close()

        final_plot = Path(tmp_dir, f"animation_4_intervals_{mode}_{i}.png")
        save_plot_according_to_template(raw_svg_file, final_plot, dpi=DPI)
        frames.append(final_plot)
        i += 1

    gif_path = Path(get_plots_path(), f"animation_4_prediction_intervals_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for img in frames:
            writer.append_data(imageio.imread(img))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    plot_prediction_intervals("rus")
    plot_prediction_intervals("eng")
