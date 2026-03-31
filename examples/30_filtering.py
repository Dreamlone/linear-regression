from pathlib import Path

from matplotlib.gridspec import GridSpec
from scipy import stats

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 62
N_BINS = 12


def filter_iqr(ax, x, y, whisker=1.5):
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

    ax.scatter(x_in, y_in, s=20, c="#5b94e5", alpha=0.8)
    ax.scatter(x_out, y_out, s=20, c="red", alpha=0.8)
    ax.plot([min(x), max(x)], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax_cleanup(ax)

    return q1, q3, lower_bound, upper_bound


def explain_iqr(ax_iqr, y, q1, q3, lower_bound, upper_bound):
    iqr = q3 - q1
    ax_iqr_boxplot = ax_iqr.twinx()
    ax_source_data = ax_iqr_boxplot.twinx()

    kde = stats.gaussian_kde(y)
    ax_iqr.hist(y, density=True, range=(MIN_Y, MAX_Y),
                alpha=0.8, rwidth=0.9, bins=N_BINS,
                color="#5b94e5", orientation='horizontal')
    xx = np.linspace(MIN_Y, MAX_Y, 1000)
    ax_iqr.plot([0, 0.23], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax_iqr.plot([0, 0.23], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax_iqr.plot([0, 0.23], [q1, q1], '--', c='grey', alpha=0.6, linewidth=1)
    ax_iqr.plot([0, 0.23], [q3, q3], '--', c='grey', alpha=0.6, linewidth=1)

    ax_iqr.plot(kde(xx), xx, color="#5b94e5", linewidth=1)
    ax_iqr.grid(color='grey', alpha=0.1)
    ax_iqr.set_xlim(0, 0.25)
    ax_iqr.set_ylim(MIN_Y, MAX_Y)

    ax_iqr_boxplot.set_xlim(0, 0.25)

    # If we need to show the custom boxplot
    # box_data = [{
    #     'med': np.median(y),
    #     'q1': q1,
    #     'q3': q3,
    #     'whislo': lower_bound,
    #     'whishi': upper_bound,
    #     'fliers': []
    # }]
    #
    # ax_iqr_boxplot.bxp(
    #     box_data,
    #     positions=[0.2],
    #     widths=0.015,
    #     patch_artist=True,
    #     boxprops=dict(facecolor='grey', color='grey'),
    #     medianprops=dict(color='grey'),
    #     whiskerprops=dict(color='grey'),
    #     capprops=dict(color='grey'),
    #     zorder=1,
    # )
    ax_iqr_boxplot.boxplot(
        y,
        positions=[0.2],
        widths=0.015,
        patch_artist=True,
        boxprops=dict(facecolor='#5b94e5', color='grey'),
        medianprops=dict(color='white'),
        whiskerprops=dict(color='#5b94e5'),
        capprops=dict(color='#5b94e5'),
        flierprops=dict(markerfacecolor='#5b94e5', marker='o', markersize=3, linestyle='none', alpha=0.3),
        zorder=2
    )

    # Add text labels
    arrow_x = 0.215
    arrow = FancyArrowPatch(
        (arrow_x, q1), (arrow_x, q3),
        arrowstyle='<->',
        mutation_scale=5,
        color='red',
        linewidth=1
    )
    ax_iqr.add_patch(arrow)
    ax_iqr.text(arrow_x + 0.005, (q1 + q3) / 2, "IQR", va='center', ha='left',
                fontsize=8, color='red', fontname=FONTNAME)

    ax_iqr_boxplot.set_ylim(MIN_Y, MAX_Y)
    ax_iqr_boxplot.text(0.24, q1, "Q1", fontname=FONTNAME, fontsize=8, va='center', ha='center', c='grey')
    ax_iqr_boxplot.text(0.24, q3, "Q3", fontname=FONTNAME, fontsize=8, va='center', ha='center', c='grey')

    ax_iqr_boxplot.text(0.22, q1 - 1.1 * iqr, r"Q1$- 1.5\cdot$IQR",
                        fontname=FONTNAME, fontsize=6, va='center', ha='center', c='red')
    ax_iqr_boxplot.text(0.22, q3 + 1.1 * iqr, r"Q3$+ 1.5\cdot$IQR",
                        fontname=FONTNAME, fontsize=6, va='center', ha='center', c='red')

    ax_iqr.plot([0.085, 0.085], [MIN_Y, MAX_Y], c='black')
    ax_iqr.plot([0.17, 0.17], [MIN_Y, MAX_Y], c='black')

    ax_source_data.scatter(np.random.uniform(0.11, 0.14, len(y)), y, s=3, c="#5b94e5")

    for ax in [ax_iqr, ax_iqr_boxplot, ax_source_data]:
        ax_cleanup(ax)
        ax.spines[['right', 'top']].set_visible(False)


def filter_zscore(ax, x, y, threshold):
    y = np.asarray(y)
    mean_y = np.mean(y)
    std_y = np.std(y)

    z_scores = (y - mean_y) / std_y
    mask_inliers = np.abs(z_scores) <= threshold

    x_in = x[mask_inliers]
    y_in = y[mask_inliers]
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]

    ax.scatter(x_in, y_in, s=20, c="#5b94e5", alpha=0.8)
    ax.scatter(x_out, y_out, s=20, c="red", alpha=0.8)

    y_lower = mean_y - threshold * std_y
    y_upper = mean_y + threshold * std_y
    ax.plot([min(x), max(x)], [y_lower, y_lower], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [y_upper, y_upper], '--', c='red', alpha=0.9, linewidth=1)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax_cleanup(ax)

    return mean_y, std_y, y_lower, y_upper, z_scores


def explain_z_score(ax, x, y, z, mean_y, std_y, y_lower, y_upper, mean_label: str, z_scores_label: str):
    palette = 'rainbow'
    sc = ax.scatter(x, y, s=20, c=z, cmap=palette, alpha=0.8, vmin=-3.65, vmax=3.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(z_scores_label, fontsize=8, fontname=FONTNAME)
    cbar.ax.tick_params(labelsize=5)

    ax.plot([min(x) + 3.8, max(x)], [mean_y, mean_y], '--', c='grey', linewidth=1)
    ax.text(min(x) + 1, mean_y, mean_label, va='center', ha='left', fontsize=8, color='grey', fontname=FONTNAME)
    ax.text(min(x), max(y), fr"z-score = ($y_i$ - {mean_label}) / $σ$",
            va='center', ha='left', fontsize=6, color='black', fontname=FONTNAME)
    ax.set_ylim(MIN_Y, MAX_Y)

    ax.plot([min(x), max(x)], [y_lower, y_lower], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [y_upper, y_upper], '--', c='red', alpha=0.9, linewidth=1)

    ax.plot([min(x) + 0.5, min(x) + 0.75], [mean_y + 0.5 * std_y, mean_y + 0.5 * std_y], c="grey", linewidth=1)
    ax.plot([min(x) + 0.5, min(x) + 0.75], [mean_y - 0.5 * std_y, mean_y - 0.5 * std_y], c="grey", linewidth=1)
    ax.plot([min(x) + 0.5, min(x) + 0.5], [mean_y - 0.5 * std_y, mean_y + 0.5 * std_y], c="grey", linewidth=1)
    ax.plot([min(x) + 0.75, max(x)], [mean_y + 0.5 * std_y, mean_y + 0.5 * std_y], '--',
            alpha=0.2, c="grey", linewidth=1)
    ax.plot([min(x) + 0.75, max(x)], [mean_y - 0.5 * std_y, mean_y - 0.5 * std_y], '--',
            alpha=0.2, c="grey", linewidth=1)
    ax.text(min(x), mean_y, "σ", fontname=FONTNAME, fontsize=8, va='center', ha='left', c='grey')

    mask_inliers = (y >= y_lower) & (y <= y_upper)
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]
    z_out = z[~mask_inliers]
    norm = matplotlib.colors.Normalize(vmin=-3.65, vmax=3.5)
    cmap = plt.get_cmap(palette)

    for x_i, y_i, z_i in zip(x_out, y_out, z_out):
        color = cmap(norm(z_i))
        ax.text(x_i + 0.7, y_i, f"{z_i:.1f}", va='center', ha='center', fontsize=5, color=color, fontname=FONTNAME)

    ax_cleanup(ax)
    ax.spines[['right', 'top']].set_visible(False)


def filter_modified_zscore(ax, x, y, threshold):
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

    ax.scatter(x_in, y_in, s=20, c="#5b94e5", alpha=0.8)
    ax.scatter(x_out, y_out, s=20, c="red", alpha=0.8)

    lower_bound = median_y - (threshold * mad_y) / 0.6745
    upper_bound = median_y + (threshold * mad_y) / 0.6745
    ax.plot([min(x), max(x)], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax_cleanup(ax)

    return median_y, mad_y, lower_bound, upper_bound, modified_z_scores


def explain_modified_z_score(ax, x, y, modified_z_scores, median_y, mad_y, y_lower, y_upper,
                             median_label: str, z_scores_label: str):
    palette = "coolwarm"
    sc = ax.scatter(x, y, s=20, c=modified_z_scores, cmap=palette, alpha=0.8, vmin=-4.3, vmax=4.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(z_scores_label, fontsize=8, fontname=FONTNAME)
    cbar.ax.tick_params(labelsize=5)

    ax.plot([min(x) + 3.8, max(x)], [median_y, median_y], '--', c='grey', linewidth=1)
    ax.text(min(x) + 1, median_y, median_label, va='center', ha='left', fontsize=8, color='grey', fontname=FONTNAME)

    eq = r"z-score$_{modified}$ = $\frac{0.6745 \cdot (y_i - \mathrm{median_label})}{\mathrm{median}(|y_i - \mathrm{median_label}|)}$"
    eq = eq.replace("median_label", median_label)
    ax.text(min(x), max(y), eq, va='center', ha='left', fontsize=6, color='black', fontname=FONTNAME)
    ax.set_ylim(MIN_Y, MAX_Y)

    ax.plot([min(x), max(x)], [y_lower, y_lower], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [y_upper, y_upper], '--', c='red', alpha=0.9, linewidth=1)

    mask_inliers = (y >= y_lower) & (y <= y_upper)
    x_out = x[~mask_inliers]
    y_out = y[~mask_inliers]
    z_out = modified_z_scores[~mask_inliers]
    norm = matplotlib.colors.Normalize(vmin=-4.3, vmax=4.0)
    cmap = plt.get_cmap(palette)

    for x_i, y_i, z_i in zip(x_out, y_out, z_out):
        color = cmap(norm(z_i))
        ax.text(x_i + 0.7, y_i, f"{z_i:.1f}", va='center', ha='center', fontsize=5, color=color, fontname=FONTNAME)

    ax_cleanup(ax)
    ax.spines[['right', 'top']].set_visible(False)


def filter_3sigma(ax, x, y):
    threshold = 3.0
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

    ax.scatter(x_in, y_in, s=20, c="#5b94e5", alpha=0.8)
    ax.scatter(x_out, y_out, s=20, c="red", alpha=0.8)
    ax.plot([min(x), max(x)], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.plot([min(x), max(x)], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax_cleanup(ax)

    return mean_y, std_y, lower_bound, upper_bound


def explain_3_sigma(ax_3_sigma, y, mean_y, std_y, lower_bound, upper_bound, mean_label: str, std_label: str):
    kde = stats.gaussian_kde(y)
    ax_3_sigma.hist(y, density=True, range=(MIN_Y, MAX_Y),
                alpha=0.8, rwidth=0.9, bins=N_BINS,
                color="#5b94e5", orientation='horizontal')
    xx = np.linspace(MIN_Y, MAX_Y, 1000)
    ax_3_sigma.plot([0, 0.11], [mean_y, mean_y], '--', c='grey', linewidth=1)
    ax_3_sigma.text(0.115, mean_y, mean_label, fontname=FONTNAME, fontsize=8, va='center', ha='left', c='grey')

    ax_3_sigma.plot([0, 0.12], [mean_y - std_y, mean_y - std_y], '--', c='red', alpha=0.1, linewidth=1)
    ax_3_sigma.plot([0, 0.12], [mean_y + std_y, mean_y + std_y], '--', c='red', alpha=0.1, linewidth=1)
    ax_3_sigma.text(0.125, mean_y - std_y, "-σ", fontname=FONTNAME, fontsize=8, va='center',
                    ha='left', c='red', alpha=0.4)
    ax_3_sigma.text(0.125, mean_y + std_y, "+σ", fontname=FONTNAME, fontsize=8, va='center',
                    ha='left', c='red', alpha=0.4)

    ax_3_sigma.plot([0, 0.13], [mean_y - 2 * std_y, mean_y - 2 * std_y], '--', c='red', alpha=0.2, linewidth=1)
    ax_3_sigma.plot([0, 0.13], [mean_y + 2 * std_y, mean_y + 2 * std_y], '--', c='red', alpha=0.2, linewidth=1)
    ax_3_sigma.text(0.135, mean_y - 2 * std_y, "-2σ", fontname=FONTNAME, fontsize=8, va='center', ha='left',
                    c='red', alpha=0.6)
    ax_3_sigma.text(0.135, mean_y + 2 * std_y, "+2σ", fontname=FONTNAME, fontsize=8, va='center', ha='left',
                    c='red', alpha=0.6)

    ax_3_sigma.plot([0, 0.14], [lower_bound, lower_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax_3_sigma.plot([0, 0.14], [upper_bound, upper_bound], '--', c='red', alpha=0.9, linewidth=1)
    ax_3_sigma.text(0.145, lower_bound, "-3σ", fontname=FONTNAME, fontsize=8, va='center', ha='left',
                    c='red', alpha=0.6)
    ax_3_sigma.text(0.145, upper_bound, "+3σ", fontname=FONTNAME, fontsize=8, va='center', ha='left',
                    c='red', alpha=0.6)

    ax_3_sigma.plot([0.0, 0.168], [mean_y - 0.5 * std_y, mean_y - 0.5 * std_y],
                    '--', alpha=0.2, c="grey", linewidth=1)
    ax_3_sigma.plot([0.0, 0.168], [mean_y + 0.5 * std_y, mean_y + 0.5 * std_y],
                    '--', alpha=0.2, c="grey", linewidth=1)

    ax_3_sigma.plot([0.17, 0.175], [mean_y + 0.5 * std_y, mean_y + 0.5 * std_y], c="grey", linewidth=1)
    ax_3_sigma.plot([0.17, 0.175], [mean_y - 0.5 * std_y, mean_y - 0.5 * std_y], c="grey", linewidth=1)
    ax_3_sigma.plot([0.175, 0.175], [mean_y - 0.5 * std_y, mean_y + 0.5 * std_y], c="grey", linewidth=1)
    ax_3_sigma.text(0.18, mean_y, std_label, fontname=FONTNAME, fontsize=8, va='center', ha='left', c='grey')

    ax_3_sigma.plot(kde(xx), xx, color="#5b94e5", linewidth=1)
    ax_3_sigma.grid(color='grey', alpha=0.1)
    ax_3_sigma.set_xlim(0, 0.25)
    ax_3_sigma.set_ylim(MIN_Y, MAX_Y)

    ax_3_sigma.set_xlim(0, 0.25)
    ax_3_sigma.spines[['right', 'top']].set_visible(False)


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Data filtering methods (one-dimensional)"
        iqr_plot = "Interquartile range"
        iqr_plot_x_label = "Distribution                   Data                    Calculation"
        sigma_plot = "Three-sigma rule (3σ)"
        z_score_plot = "Z-score"
        z_score_threshold = "Z-score threshold"
        z_score_modified_plot = "Modified Z-score"
        main_plot = "Data with outliers"
        y_label = "Detected outliers"
        mean_label = "mean"
        median_label = "median"
        std_label = "standard\ndeviation\n      σ"
        z_scores_label = "Z-scores"
        modified_z_scores_label = "Modified Z-scores"
    elif mode == "rus":
        title = "Способы фильтрации данных (одномерные методы)"
        iqr_plot = "Межквартильный размах"
        iqr_plot_x_label = "Распределение                 Данные                 Метод расчета"
        sigma_plot = "Метод 3х сигм (3σ)"
        z_score_plot = "Z-оценка"
        z_score_threshold = "Порог Z-оценки "
        z_score_modified_plot = "Модифицированная Z-оценка"
        main_plot = "Данные с выбросами"
        y_label = "Обнаруженные выбросы"
        mean_label = "среднее"
        median_label = "медиана"
        std_label = "стандартное\nотклонение\n         σ"
        z_scores_label = "Значения z"
        modified_z_scores_label = "Модифицированные z значения"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return (title, iqr_plot, iqr_plot_x_label, sigma_plot, mean_label, median_label, std_label, z_score_plot,
            z_score_threshold, z_score_modified_plot, z_scores_label, modified_z_scores_label, main_plot, y_label)


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


def ax_cleanup(ax):
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])


def plot_filtering_method_concepts(mode: str = "eng"):
    (title, iqr_plot, iqr_plot_x_label, sigma_plot, mean_label, median_label, std_label, z_score_plot, z_score_threshold,
     z_score_modified_plot, z_scores_label, modified_z_scores_label, main_plot, y_label) = annotations_by_language(mode)
    x, y = generate_synthetic_data()

    fig_size = (15, 11)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[0:2, 0:4])
    ax_main.set_title(main_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_main.scatter(x, y, s=60, c="#5b94e5", alpha=0.8)
    ax_main.set_ylim(MIN_Y, MAX_Y)
    ax_main.set_ylabel("y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_cleanup(ax_main)

    #########
    # 1 IQR #
    #########
    ax_iqr = fig.add_subplot(gs[2, 0])
    ax_iqr.set_title(iqr_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_iqr_data = fig.add_subplot(gs[3, 0])
    q1, q3, lower_bound, upper_bound = filter_iqr(ax_iqr_data, x, y)

    explain_iqr(ax_iqr, y, q1, q3, lower_bound, upper_bound)
    ax_iqr.set_xlabel(iqr_plot_x_label, fontdict={'fontsize': 6, 'fontname': FONTNAME})
    ax_iqr_data.set_ylabel(y_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})

    ###########
    # 2 Sigma #
    ###########
    ax_3_sigma = fig.add_subplot(gs[2, 1])
    ax_3_sigma.set_title(sigma_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_cleanup(ax_3_sigma)

    ax_3_sigma_data = fig.add_subplot(gs[3, 1])
    mean_y, std_y, lower_bound, upper_bound = filter_3sigma(ax_3_sigma_data, x, y)
    explain_3_sigma(ax_3_sigma, y, mean_y, std_y, lower_bound, upper_bound, mean_label, std_label)
    ax_3_sigma_data.plot([min(x), max(x)], [mean_y - std_y, mean_y - std_y],
                         '--', c='red', alpha=0.1, linewidth=1)
    ax_3_sigma_data.plot([min(x), max(x)], [mean_y + std_y, mean_y + std_y],
                         '--', c='red', alpha=0.1, linewidth=1)
    ax_3_sigma_data.plot([min(x), max(x)], [mean_y - 2 * std_y, mean_y - 2 * std_y],
                         '--', c='red', alpha=0.2, linewidth=1)
    ax_3_sigma_data.plot([min(x), max(x)], [mean_y + 2 * std_y, mean_y + 2 * std_y],
                         '--', c='red', alpha=0.2, linewidth=1)

    #############
    # 3 Z score #
    #############
    ax_z_score = fig.add_subplot(gs[2, 2])
    ax_z_score.set_title(z_score_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_cleanup(ax_z_score)

    ax_z_score_data = fig.add_subplot(gs[3, 2])
    threshold = 2.0
    ax_z_score_data.set_title(f"{z_score_threshold}: {threshold:.1f}",
                              fontsize=10, fontdict={'fontname': FONTNAME})
    mean_y, std_y, y_lower, y_upper, z_scores = filter_zscore(ax_z_score_data, x, y, threshold)
    explain_z_score(ax_z_score, x, y, z_scores, mean_y, std_y, y_lower, y_upper, mean_label, z_scores_label)

    ######################
    # 4 Z score modified #
    ######################
    threshold = 2.0
    ax_z_score_modified = fig.add_subplot(gs[2, 3])
    ax_z_score_modified.set_title(z_score_modified_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_cleanup(ax_z_score_modified)

    ax_z_score_modified_data = fig.add_subplot(gs[3, 3])
    ax_z_score_modified_data.set_title(f"{z_score_threshold}: {threshold:.1f}",
                                        fontsize=10, fontdict={'fontname': FONTNAME})
    median_y, mad_y, lower_bound, upper_bound, modified_z_scores = filter_modified_zscore(ax_z_score_modified_data,
                                                                                          x, y, threshold)
    explain_modified_z_score(ax_z_score_modified, x, y, modified_z_scores, median_y, mad_y,
                             lower_bound, upper_bound, median_label, modified_z_scores_label)

    raw_svg_file = Path(get_plots_path(), f"30_explain_filtering_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()
    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"30_explain_filtering_{mode}.png"))


if __name__ == '__main__':
    plot_filtering_method_concepts("rus")
    plot_filtering_method_concepts("eng")
