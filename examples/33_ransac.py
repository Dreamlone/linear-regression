import shutil
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # proxy for legend
from matplotlib.patches import Rectangle
from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template, get_datasets, split_train_test_manual

import warnings
warnings.filterwarnings('ignore')

FONTNAME = "Comic Sans MS"
POINT_SIZE = 30
N_TRIALS = 2
MIN_SAMPLES = 8
RANDOM_STATE = 34
BAND_FACTOR = 2.0

ANIM_DURATION = 380
MAIN_FRAMES_DURATION_MULTIPLIER = 5
DPI = 100


def annotations_by_language(mode: str):
    if mode == "rus":
        best_model_label = "Эталонная модель"
        sample_fit_label = "Модель построенная по всей выборке"
        model = "Модель"
        refit = "Обучение на не-выбросах"
        t1 = "1. Исходные данные"
        t2 = "2. Случайная выборка для инициализации"
        t3 = "3. Обучение модели по выборке"
        t4 = "4. Отсечение выбросов"
        t5 = "5. Обучение на не-выбросах"
        t6 = "6. Повторное отсечение и подсчет не-выбросов"
        columns = ["Модель", "Количество\nне-выбросов"]
        outliers_label = "выбросы"
        final_frame_title = "Выбор финальной модели"
        final_line_label = "Финальная модель"
        table_title_final = "Модель с наибольшим\nколичеством не-выбросов"
    else:
        best_model_label = "Best model"
        sample_fit_label = "Model fit on all data"
        model = "Model"
        refit = "Train on inliers"
        t1 = "1. Raw data"
        t2 = "2. Random subset for init"
        t3 = "3. Fit model on subset"
        t4 = "4. Outlier rejection"
        t5 = "5. Train on inliers"
        t6 = "6. Recheck & final count"
        columns = ["Model", "Inliers count"]
        outliers_label = "outliers"
        final_frame_title = "Choose the final model"
        final_line_label = "Final model"
        table_title_final = "Model with the\nlargest inliers count"
    return (best_model_label, sample_fit_label, model, refit, t1, t2, t3, t4, t5, t6,
            columns, outliers_label, final_frame_title, final_line_label, table_title_final)


def _fit_ols(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm)**2)
    b1 = 0.0 if sxx == 0 else np.sum((x - xm)*(y - ym)) / sxx
    b0 = ym - b1 * xm
    yhat = b0 + b1 * x
    return b0, b1, yhat


def _auto_threshold(y: np.ndarray, yhat: np.ndarray, factor: float = 2.0):
    e = y - yhat
    mad = np.median(np.abs(e - np.median(e)))
    sigma = 1.4826 * mad if mad > 0 else np.std(e)
    return factor * sigma


def _ransac_one_trial(x: np.ndarray, y: np.ndarray, min_samples: int, residual_threshold: float, rng: np.random.Generator):
    n = len(x)
    for _ in range(100):
        idx = rng.choice(n, size=min_samples, replace=False)
        xs, ys = x[idx], y[idx]
        if np.ptp(xs) > 0:
            break
    b0, b1, _ = _fit_ols(xs, ys)
    yhat_all = b0 + b1 * x
    inliers = np.abs(y - yhat_all) <= residual_threshold
    return b0, b1, yhat_all, inliers, idx


def _make_figure():
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_tbl  = fig.add_subplot(gs[0, 1])
    ax_plot.set_xlabel("x", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_plot.set_ylabel("y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_plot.grid(alpha=0.2)
    ax_plot.set_xlim(0, 6)
    ax_plot.set_ylim(0, 90000)
    ax_plot.set_xticks([1,2,3,4,5])
    ax_plot.spines[['right','top']].set_visible(False)
    return fig, ax_plot, ax_tbl


def _draw_table(ax_tbl, columns: list[str], data_rows: list[list[str]]):
    ax_tbl.clear()
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=data_rows,
                       colLabels=columns,
                       loc='center',
                       cellLoc='center',
                       colLoc='center')
    tbl.scale(1, 3)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for _, cell in tbl.get_celld().items():
        cell.get_text().set_fontname(FONTNAME)
    return tbl


def _ensure_legend_order(ax, baseline_label: str, sample_fit_label: str, outliers_label: str):
    """
    Force legend order: [baseline, sample-fit, outliers, ...others...].
    If some labels are missing, create proxy handles so the order stays consistent.
    """
    handles, labels = ax.get_legend_handles_labels()

    # Build unique label->handle (keep last occurrence)
    label2handle = {}
    for h, l in zip(handles, labels):
        label2handle[l] = h

    # Proxies for required entries (if not present)
    if baseline_label not in label2handle:
        label2handle[baseline_label] = Line2D([0], [0], linestyle='--', color='black')

    if sample_fit_label not in label2handle:
        label2handle[sample_fit_label] = Line2D([0], [0], linestyle='-', color='orange', linewidth=2)

    if outliers_label not in label2handle:
        label2handle[outliers_label] = Line2D([0], [0], marker='o', linestyle='None',
                                              markerfacecolor='grey', markeredgecolor='black', markersize=6)

    # Desired order head
    head = [baseline_label, sample_fit_label, outliers_label]
    tail = [l for l in label2handle.keys() if l not in head]

    ordered_labels = head + tail
    ordered_handles = [label2handle[l] for l in ordered_labels]

    ax.legend(ordered_handles, ordered_labels, loc='upper left',
              prop={'family': FONTNAME, 'size': 9})


def plot_animation_ransac(mode: str):
    """Visualization of the RANSAC algorithm"""
    (best_model_label, sample_fit_label, model_label, refit_label,
     title1, title2, title3, title4, title5, title6,
     columns, outliers_label, final_frame_title, final_line_label, table_title_final) = annotations_by_language(mode)

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists() and len(list(tmp_dir.iterdir())) > 0:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concatenate([rooms, rooms, rooms])
    common_target  = np.concatenate([good_prices, bad_prices_first, bad_prices_second])
    x, y, _, _ = split_train_test_manual(common_features, common_target, apply_distortion=True)

    rng = np.random.default_rng(RANDOM_STATE)
    # OLS on all data (orange line)
    b0_base, b1_base, yhat_base = _fit_ols(x, y)
    tau = _auto_threshold(y, yhat_base, factor=BAND_FACTOR)
    xx = np.array([x.min(), x.max()])

    table_rows = [["", ""] for _ in range(N_TRIALS)]
    frames = []
    frame_id = 0

    final_models = [None] * N_TRIALS
    final_counts = [0]    * N_TRIALS

    def _save(fig, name):
        svg_path = Path(tmp_dir, f"{name}.svg")
        fig.savefig(svg_path, bbox_inches='tight')
        plt.close(fig)
        out_png = Path(tmp_dir, f"{name}.png")
        save_plot_according_to_template(svg_path, out_png, template_name="template_small.svg", dpi=DPI)
        frames.append(out_png)

    for trial in range(N_TRIALS):
        # 1) Raw data + baseline + sample-fit (orange)
        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=2)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        # sample-fit line (orange)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title1, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_base_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_base_{mode}_{frame_id}.png"))
        frame_id += 1

        # 2) Random subset
        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=2)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        b0_s, b1_s, _, _, picked_idx = _ransac_one_trial(x, y, MIN_SAMPLES, tau, rng)
        ax.scatter(x[picked_idx], y[picked_idx], s=POINT_SIZE, color='red',
                   edgecolors='black', linewidths=0.6, zorder=3)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title2, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_subset_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_subset_{mode}_{frame_id}.png"))
        frame_id += 1

        # 3) Model from subset
        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=2)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        ax.scatter(x[picked_idx], y[picked_idx], s=POINT_SIZE, color='red',
                   edgecolors='black', linewidths=0.6, zorder=3)
        ax.plot(xx, b0_s + b1_s*xx, c='red', alpha=0.9, linewidth=2.0,
                label=model_label, zorder=4)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        eq_s = rf"${b0_s:.0f} + {b1_s:.0f}\cdot x$"
        table_rows[trial] = [eq_s, ""]
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title3, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_model_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_model_{mode}_{frame_id}.png"))
        frame_id += 1

        # 4) Outlier rejection
        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=1)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        y_line_s = b0_s + b1_s * xx
        ax.plot(xx, y_line_s, c='red', alpha=0.9, linewidth=2.0,
                label=model_label, zorder=3)
        ax.plot(xx, y_line_s + tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        ax.plot(xx, y_line_s - tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        yhat_all_s = b0_s + b1_s * x
        out_s = np.abs(y - yhat_all_s) > tau
        if out_s.any():
            ax.scatter(x[out_s], y[out_s], s=POINT_SIZE, color='grey',
                       edgecolors='black', linewidths=0.8, zorder=5, label=None)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title4, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_reject1_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_reject1_{mode}_{frame_id}.png"))
        frame_id += 1

        # 5) Train on non-outliers
        inliers_mask = ~out_s
        if inliers_mask.sum() >= MIN_SAMPLES:
            b0_r, b1_r, _ = _fit_ols(x[inliers_mask], y[inliers_mask])
        else:
            b0_r, b1_r = b0_s, b1_s
        final_models[trial] = (b0_r, b1_r)

        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=1)
        if inliers_mask.any():
            ax.scatter(x[inliers_mask], y[inliers_mask], s=POINT_SIZE, color='red',
                       edgecolors='black', linewidths=0.6, zorder=2)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        ax.plot(xx, b0_s + b1_s*xx, c='red', alpha=0.3, linewidth=2.0,
                label=model_label, zorder=3)
        ax.plot(xx, b0_r + b1_r*xx, c='red', alpha=0.9, linewidth=2.0,
                label=refit_label, zorder=4)
        ax.plot(xx, y_line_s + tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        ax.plot(xx, y_line_s - tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        eq_r = rf"${b0_r:.0f} + {b1_r:.0f}\cdot x$"
        table_rows[trial] = [eq_r, table_rows[trial][1]]
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title5, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_refit_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_refit_{mode}_{frame_id}.png"))
        frame_id += 1

        # 6) Recheck with refit model
        fig, ax, ax_tbl = _make_figure()
        ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                   s=POINT_SIZE, zorder=1)
        ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                zorder=1, label=best_model_label)
        ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                zorder=1, label=sample_fit_label)
        y_line_r = b0_r + b1_r * xx
        ax.plot(xx, y_line_r, c='red', alpha=0.9, linewidth=2.0,
                label=refit_label, zorder=3)
        ax.plot(xx, y_line_r + tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        ax.plot(xx, y_line_r - tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
        yhat_all_r = b0_r + b1_r * x
        out_r = np.abs(y - yhat_all_r) > tau
        if out_r.any():
            ax.scatter(x[out_r], y[out_r], s=POINT_SIZE, color='grey',
                       edgecolors='black', linewidths=0.8, zorder=5, label=None)
        _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
        _draw_table(ax_tbl, columns, table_rows)
        fig.suptitle(title6, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
        _save(fig, f"33_ransac_reject2_{mode}_{frame_id}")
        for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
            frames.append(Path(tmp_dir, f"33_ransac_reject2_{mode}_{frame_id}.png"))
        frame_id += 1

        # Fast counting (lexsort by x, then y)
        inlier_idx = np.where(~out_r)[0]
        if inlier_idx.size > 0:
            order = np.lexsort((y[inlier_idx], x[inlier_idx]))
            inlier_ordered = inlier_idx[order]
            dy, dx = 100, 0.1

            for k in range(1, inlier_ordered.size + 1):
                fig, ax, ax_tbl = _make_figure()
                ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                           s=POINT_SIZE, zorder=1)
                ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                        zorder=1, label=best_model_label)
                ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                        zorder=1, label=sample_fit_label)
                ax.plot(xx, y_line_r, c='red', alpha=0.9, linewidth=2.0,
                        label=refit_label, zorder=3)
                ax.plot(xx, y_line_r + tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
                ax.plot(xx, y_line_r - tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
                if out_r.any():
                    ax.scatter(x[out_r], y[out_r], s=POINT_SIZE, color='grey',
                               edgecolors='black', linewidths=0.8, zorder=5, label=None)

                sel = inlier_ordered[:k]
                for j, idx_pt in enumerate(sel, start=1):
                    off_x = dx if (j % 2 == 1) else -dx
                    ax.text(x[idx_pt] + off_x, y[idx_pt] + dy, f"{j}",
                            ha="center", va="bottom", fontsize=9, fontname=FONTNAME, color='red')

                _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
                _draw_table(ax_tbl, columns, table_rows)
                fig.suptitle(title6, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
                _save(fig, f"33_ransac_count_{mode}_{frame_id}_{k}")

            n_inliers_r = int(inlier_ordered.size)
            final_counts[trial] = n_inliers_r
            table_rows[trial] = [table_rows[trial][0], f"{n_inliers_r}"]

            fig, ax, ax_tbl = _make_figure()
            ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                       s=POINT_SIZE, zorder=1)
            ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                    zorder=1, label=best_model_label)
            ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                    zorder=1, label=sample_fit_label)
            ax.plot(xx, y_line_r, c='red', alpha=0.9, linewidth=2.0,
                    label=refit_label, zorder=3)
            ax.plot(xx, y_line_r + tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
            ax.plot(xx, y_line_r - tau, '--', c='red', alpha=0.6, linewidth=1.0, zorder=2)
            if out_r.any():
                ax.scatter(x[out_r], y[out_r], s=POINT_SIZE, color='grey',
                           edgecolors='black', linewidths=0.8, zorder=5, label=None)
            _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
            _draw_table(ax_tbl, columns, table_rows)
            fig.suptitle(title6, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
            _save(fig, f"33_ransac_table_update_{mode}_{frame_id}")
            for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
                frames.append(Path(tmp_dir, f"33_ransac_table_update_{mode}_{frame_id}.png"))
            frame_id += 1
        else:
            final_counts[trial] = 0
            fig, ax, ax_tbl = _make_figure()
            ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
                       s=POINT_SIZE, zorder=1)
            ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
                    zorder=1, label=best_model_label)
            ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
                    zorder=1, label=sample_fit_label)
            _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)
            _draw_table(ax_tbl, columns, table_rows)
            fig.suptitle(title6, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
            _save(fig, f"33_ransac_table_update_{mode}_{frame_id}")
            for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER):
                frames.append(Path(tmp_dir, f"33_ransac_table_update_{mode}_{frame_id}.png"))
            frame_id += 1

    # --- FINAL SELECTION FRAME ---
    counts_int = [int(c) if str(c).strip().isdigit() else 0 for (_, c) in table_rows]
    best_trial = int(np.argmax(counts_int)) if len(counts_int) > 0 else 0
    best_b0, best_b1 = final_models[best_trial]

    fig, ax, ax_tbl = _make_figure()
    ax.scatter(x, y, facecolors='white', edgecolors='black', linewidths=0.6,
               s=POINT_SIZE, zorder=1)
    ax.plot([1, 5], [11133.333333333334, 46600], '--', c='black',
            zorder=1, label=best_model_label)
    ax.plot(xx, b0_base + b1_base*xx, c='orange', linewidth=1.0, alpha=0.9,
            zorder=1, label=sample_fit_label)
    ax.plot(xx, best_b0 + best_b1*xx, c='red', alpha=0.95, linewidth=2.2,
            label=final_line_label, zorder=3)
    _ensure_legend_order(ax, best_model_label, sample_fit_label, outliers_label)

    tbl = _draw_table(ax_tbl, columns, table_rows)
    ax_tbl.set_title(table_title_final, fontname=FONTNAME, color='red', y=0.85)

    fig.canvas.draw()
    row_to_highlight = best_trial + 1
    cols = range(len(columns))
    row_cells = [tbl[(row_to_highlight, c)] for c in cols]
    for c in row_cells:
        c.set_zorder(10)
    x0 = min(c.get_x() for c in row_cells)
    y0 = row_cells[0].get_y()
    w = sum(c.get_width() for c in row_cells)
    h = row_cells[0].get_height()
    outline = Rectangle((x0, y0), w, h,
                        fill=False, edgecolor='red', linewidth=1.5,
                        transform=ax_tbl.transAxes, zorder=11)
    ax_tbl.add_patch(outline)

    fig.suptitle(final_frame_title, fontsize=14, fontdict={'fontname': FONTNAME}, va="top", y=0.98)
    _save(fig, f"33_ransac_final_choice_{mode}_{frame_id}")
    for _ in range(1, MAIN_FRAMES_DURATION_MULTIPLIER * 2):
        frames.append(Path(tmp_dir, f"33_ransac_final_choice_{mode}_{frame_id}.png"))

    gif_path = Path(get_plots_path(), f"33_ransac_{mode}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=ANIM_DURATION, loop=0) as writer:
        for img in frames:
            writer.append_data(imageio.imread(img))
    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    plot_animation_ransac("rus")
