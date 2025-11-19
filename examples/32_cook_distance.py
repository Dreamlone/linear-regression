import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, get_datasets, split_train_test_manual

import warnings
warnings.filterwarnings('ignore')

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 12, 'fontname': FONTNAME}
VMAX = 0.5

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


def annotations_by_language(mode: str):
    if mode == "eng":
        leverage_label = ""
        model_label = ""
        best_model_label = ""
        current_model_label = ""
        new_model_label = ""
        build_model = ""
        influence_label = ""
        cook_distance_label = ""
        excluding_title = ""
    elif mode == "rus":
        leverage_label = "Рычаг"
        model_label = "Модель"
        best_model_label = "Эталонная модель"
        current_model_label = "Текущая модель по данным"
        new_model_label = "Если исключим точку из обучения"
        build_model = "Строим модель"
        influence_label = "Оценка влияния точек на модель"
        cook_distance_label = "Расстояние Кука (D)"
        excluding_title = ("Исключаем объект с большим значением расстояния Кука"
                           "\nи проверяем, как изменится модель")
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (leverage_label, model_label, best_model_label, current_model_label, new_model_label,
            build_model, influence_label, cook_distance_label, excluding_title)


def calculate_leverage(x: np.array):
    x_ = np.column_stack([np.ones_like(x), x])
    inv = np.linalg.pinv(x_.T @ x_)
    h = np.einsum('ij,jk,ik->i', x_, inv, x_)
    return h


def annotate_leverage(ax, x, x_i, loc=(0.02, 0.98)):
    x = np.asarray(x, dtype=float)
    n = x.size
    xbar = np.nanmean(x)
    sxx = np.nansum((x - xbar)**2)
    h_i = (1.0 / n) + ((x_i - xbar)**2 / sxx) if sxx > 0 else (1.0 / n)

    eq_general = r"$h_i=\frac{1}{n}+\frac{(x_i-\bar{x})^2}{\sum_{j=1}^{n}(x_j-\bar{x})^2}$"
    eq_values  = rf"$h_i=\frac{{1}}{{{n}}}+\frac{{({x_i:.3g}-{xbar:.3g})^2}}{{{sxx:.3g}}}={h_i:.2f}$"

    ax.text(loc[0], loc[1], eq_general + "\n" + eq_values,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontname=FONTNAME)
    return h_i


def calculate_cooks_distance(x: np.array, y:np.array):
    """ Cook's distance (OLS y ~ 1 + x) """
    x_ = np.column_stack([np.ones_like(x), x])
    inv = np.linalg.pinv(x_.T @ x_)
    beta = inv @ x_.T @ y
    yhat = x_ @ beta
    e = y - yhat
    n, p = x_.shape
    mse = (e @ e) / max(n - p, 1)
    h = np.einsum('ij,jk,ik->i', x_, inv, x_)
    den = np.clip(1.0 - h, 1e-12, None)
    d = (e ** 2 / (p * mse)) * (h / den ** 2)

    return d


def annotate_cook(ax,
                  h_i: float,
                  p: int = 2,
                  e_i: float | None = None,
                  y_i: float | None = None,
                  pred_i: float | None = None,
                  mse: float | None = None,
                  y: np.ndarray | None = None,
                  y_pred: np.ndarray | None = None,
                  loc=(0.02, 0.98)):
    def fmt(v, precision=2):
        return np.format_float_positional(v, precision=precision, trim='-')

    if e_i is None:
        e_i = float(y_i) - float(pred_i)
    if mse is None:
        y = np.asarray(y, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = y.size
        e = y - y_pred
        mse = (np.nansum(e**2)) / max(n - p, 1)
    denom = max(1e-12, 1.0 - float(h_i))
    d = (e_i**2 / (p * mse)) * (h_i / (denom**2))

    eq_general = r"$D_i=\frac{e_i^2}{p\,\mathrm{MSE}}\cdot\frac{h_i}{(1-h_i)^2}$"
    eq_values = rf"$D=\frac{{({fmt(e_i)})^2}}{{{p}\cdot{fmt(mse)}}}\cdot\frac{{{fmt(h_i)}}}{{(1-{fmt(h_i)})^2}}={fmt(d)}$"

    ax.text(loc[0], loc[1], eq_general + "\n" + eq_values,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontname=FONTNAME)
    return d


def generate_base_frame(x: np.array, y: np.array, h: np.array, x_for_fit: np.array, dropped_points: list,
                        predicted: np.array, leverage_label, model_label, best_model_label, current_model_label,
                        cook_distance_label, influence_label):
    # Create figure and subplots
    fig_size = (14, 4)
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 3, figure=fig)
    gs.update(wspace=0.4)

    ax_leverage = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_cook = fig.add_subplot(gs[0, 2])
    for ax in [ax_leverage, ax_model, ax_cook]:
        ax.set_xlabel("x", fontdict={'fontsize': 12, 'fontname': FONTNAME})
        ax.xaxis.set_ticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.2)

        ax.set_ylim(0, 90000)
        ax.set_xlim(0, 6)

    sc = ax_leverage.scatter(x, y, c=h, cmap='coolwarm', s=50,
                             edgecolor="black", linewidths=0.5, alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax_leverage, shrink=0.9, pad=0.02)
    ax_leverage.set_title(leverage_label, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_leverage.set_ylabel("y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_leverage.spines[['right', 'top']].set_visible(False)

    ax_model.plot([1, 5], [11133.333333333334, 46600], '--', c='black', alpha=0.5, zorder=2,
                  label=best_model_label)
    ax_model.plot(x_for_fit, predicted, c='red', alpha=1.0, zorder=1, label=current_model_label)
    ax_model.scatter(x, y, facecolors='none', edgecolor="black", s=50, alpha=0.8, linewidths=0.5)
    ax_model.yaxis.set_ticklabels([])
    ax_model.set_title(model_label, fontsize=12, fontdict={'fontname': FONTNAME})

    for point_id in dropped_points:
        ax_leverage.scatter([x[point_id]], [y[point_id]], marker='x', s=50, c='black', zorder=4)
        ax_model.scatter([x[point_id]], [y[point_id]], marker='x', s=50, c='black', zorder=4)
        ax_cook.scatter([x[point_id]], [y[point_id]], marker='x', s=50, c='black', zorder=4)

    ax_cook.spines[['right', 'top']].set_visible(False)
    ax_cook.yaxis.set_ticklabels([])
    ax_cook.set_title(influence_label, fontsize=12, fontdict={'fontname': FONTNAME})

    sc_cook = ax_cook.scatter(x, y, c=np.zeros(len(x)), cmap='Reds', s=1,
                              edgecolor="black", linewidths=0.5, alpha=0.9, zorder=1, vmin=0, vmax=VMAX)
    cbar_cook = fig.colorbar(sc_cook, ax=ax_cook, shrink=0.9, pad=0.02)
    cbar_cook.set_label(cook_distance_label, fontsize=9, fontname=FONTNAME)
    # Put blank ones on top of them
    ax_cook.scatter(x, y, c='white', edgecolor="black", linewidths=0.5, s=50, zorder=2)

    return fig, ax_leverage, ax_model, ax_cook


def plot_animation_cooks_distance(mode: str = "eng", animation_duration: float = 3500):
    (leverage_label, model_label, best_model_label, current_model_label, new_model_label,
     build_model, influence_label, cook_distance_label, excluding_title) = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if len(list(tmp_dir.iterdir())) > 1:
        # Clean the directory
        shutil.rmtree(tmp_dir)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concat([rooms, rooms, rooms])
    common_target = np.concat([good_prices, bad_prices_first, bad_prices_second])
    x, y, distorted_x, distorted_y = split_train_test_manual(common_features, common_target, apply_distortion=True)

    h = calculate_leverage(x)

    image_files = []
    image_index = 0
    x_for_fit = np.copy(x)
    y_for_fit = np.copy(y)
    dropped_points = []
    for point_to_check, conclusion in zip([11, 19, 14, 2], ["drop", "keep", "drop", "keep"]):
        x_i = x[point_to_check]
        y_i = y[point_to_check]
        fit_array_index = np.argwhere(y_for_fit == y_i)
        if len(fit_array_index) > 1:
            # FIXME: Not robust. Taking the latest id
            fit_array_index = int(np.max(fit_array_index))
        else:
            fit_array_index = int(fit_array_index)
        print(f"Generating the plots for case number {image_index}")

        predicted, intercept, slope = _build_model(x_for_fit, y_for_fit)

        #############################
        # Base frame with new model #
        #############################
        fig_base, ax_leverage, ax_model, ax_cook = generate_base_frame(x, y, h, x_for_fit, dropped_points,
                                                                       predicted, leverage_label, model_label,
                                                                       best_model_label, current_model_label,
                                                                       cook_distance_label, influence_label)
        ax_cook.plot(x_for_fit, predicted, c='red', alpha=1.0, zorder=5)
        ax_model.legend(loc='upper left', prop={'family': FONTNAME, 'size': 8})

        fig_base.suptitle(f"1 - {build_model}", fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=1.2)
        raw_svg_file = Path(tmp_dir, f"32_cooks_distance_base_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"32_cooks_distance_base_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg")
        image_files.append(path_to_final_path)

        ######################
        # Calculate leverage #
        ######################
        fig_point, ax_leverage, ax_model, ax_cook = generate_base_frame(x, y, h, x_for_fit, dropped_points,
                                                                        predicted, leverage_label, model_label,
                                                                        best_model_label, current_model_label,
                                                                        cook_distance_label, influence_label)
        # Patching the subplots
        ax_leverage.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        h_i = annotate_leverage(ax_leverage, x, x_i=x_i)

        ax_model.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        ax_cook.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        ax_model.legend(loc='upper left', prop={'family': FONTNAME, 'size': 8})
        ax_cook.plot(x_for_fit, predicted, c='red', alpha=1.0, zorder=5)

        fig_point.suptitle(f"2 - {excluding_title}", fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=1.2)
        raw_svg_file = Path(tmp_dir, f"32_cooks_distance_leverage_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"32_cooks_distance_leverage_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg")
        image_files.append(path_to_final_path)

        #################################
        # Checking the individual point #
        #################################
        fig_point, ax_leverage, ax_model, ax_cook = generate_base_frame(x, y, h, x_for_fit, dropped_points,
                                                                        predicted, leverage_label, model_label,
                                                                        best_model_label, current_model_label,
                                                                        cook_distance_label, influence_label)
        # Patching the subplots
        ax_leverage.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        annotate_leverage(ax_leverage, x, x_i=x_i)

        ax_model.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        residual = y_i - predicted[fit_array_index]
        ax_model.text(x_i + 0.6, (y_i + predicted[fit_array_index]) / 2, f"e =\n{residual:.0f}",
                      ha="center", va="center", fontsize=7, color='red', fontname=FONTNAME)

        ax_model.plot([x_i + 0.2, x_i + 0.2], [predicted[fit_array_index], y_i], '--', c='red', linewidth=1)
        ax_model.plot([x_i + 0.1, x_i + 0.2], [y_i, y_i], '--', c='red', linewidth=1)
        ax_model.plot([x_i + 0.1, x_i + 0.2], [predicted[fit_array_index], predicted[fit_array_index]],
                      '--', c='red', linewidth=1)
        ax_model.legend(loc='upper left', prop={'family': FONTNAME, 'size': 8})
        ax_cook.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        d = calculate_cooks_distance(x_for_fit, y_for_fit)
        ax_cook.scatter(x_for_fit, y_for_fit, c=d, cmap='Reds', s=50, zorder=5,
                        edgecolor="black", linewidths=0.5, alpha=0.9, vmin=0, vmax=VMAX)
        ax_cook.plot(x_for_fit, predicted, c='red', alpha=1.0, zorder=5)
        annotate_cook(ax_cook, h_i=h_i, p=2, y_i=y_i, pred_i=predicted[fit_array_index], y=y_for_fit, y_pred=predicted)

        fig_point.suptitle(f"3 - {excluding_title}", fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=1.2)
        raw_svg_file = Path(tmp_dir, f"32_cooks_distance_check_point_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"32_cooks_distance_check_point_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg")
        image_files.append(path_to_final_path)

        ##############
        # Next model #
        ##############
        fig_point, ax_leverage, ax_model, ax_cook = generate_base_frame(x, y, h, x_for_fit, dropped_points,
                                                                        predicted, leverage_label, model_label,
                                                                        best_model_label, current_model_label,
                                                                        cook_distance_label, influence_label)
        # Patching the subplots
        ax_leverage.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        annotate_leverage(ax_leverage, x, x_i=x_i)

        ax_model.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        residual = y_i - predicted[fit_array_index]
        ax_model.text(x_i + 0.6, (y_i + predicted[fit_array_index]) / 2, f"e =\n{residual:.0f}",
                      ha="center", va="center", fontsize=7, color='red', fontname=FONTNAME)

        ax_model.plot([x_i + 0.2, x_i + 0.2], [predicted[fit_array_index], y_i], '--', c='red', linewidth=1)
        ax_model.plot([x_i + 0.1, x_i + 0.2], [y_i, y_i], '--', c='red', linewidth=1)
        ax_model.plot([x_i + 0.1, x_i + 0.2], [predicted[fit_array_index], predicted[fit_array_index]],
                      '--', c='red', linewidth=1)

        # Estimating new model without a point
        x_for_fit_next = np.copy(x_for_fit)
        y_for_fit_next = np.copy(y_for_fit)
        x_for_fit_next = np.delete(x_for_fit_next, fit_array_index)
        y_for_fit_next = np.delete(y_for_fit_next, fit_array_index)
        predicted_next, intercept_next, slope_next = _build_model(x_for_fit_next, y_for_fit_next)
        ax_model.plot(x_for_fit_next, predicted_next, c='red', alpha=0.4, zorder=1, label=new_model_label)
        ax_model.legend(loc='upper left', prop={'family': FONTNAME, 'size': 8})

        ax_cook.scatter([x_i], [y_i], facecolors='none', edgecolor="red", s=200)
        d = calculate_cooks_distance(x_for_fit, y_for_fit)
        ax_cook.scatter(x_for_fit, y_for_fit, c=d, cmap='Reds', s=50, zorder=5,
                        edgecolor="black", linewidths=0.5, alpha=0.9, vmin=0, vmax=VMAX)
        ax_cook.plot(x_for_fit, predicted, c='red', alpha=1.0, zorder=5)
        annotate_cook(ax_cook, h_i=h_i, p=2, y_i=y_i, pred_i=predicted[fit_array_index], y=y_for_fit, y_pred=predicted)

        fig_point.suptitle(f"4 - {excluding_title}", fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=1.2)
        raw_svg_file = Path(tmp_dir, f"32_cooks_distance_next_model_{mode}_{image_index}.svg")
        plt.savefig(raw_svg_file, bbox_inches='tight')
        plt.close()
        path_to_final_path = Path(tmp_dir, f"32_cooks_distance_next_model_{mode}_{image_index}.png")
        save_plot_according_to_template(raw_svg_file, path_to_final_path, template_name="template_small.svg")

        image_files.append(path_to_final_path)
        image_index += 1

        if conclusion == "drop":
            # Drop the point from fitting
            x_for_fit = x_for_fit_next
            y_for_fit = y_for_fit_next
            dropped_points.append(point_to_check)

    # Generate animation from the files
    gif_path = Path(get_plots_path(), f"32_cooks_distance_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=animation_duration, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    plot_animation_cooks_distance("rus")
