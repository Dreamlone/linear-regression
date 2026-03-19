from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, get_datasets, take_sample_manual

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 70000


def construct_title(model, poly, x_symbol="x", precision=1, latex=False):
    """
    Build a simple equation string like:
      y = b0 + b1 * x + b2 * x^2 + ...
    Works for 1D PolynomialFeatures.
    """
    f = lambda v: np.format_float_positional(float(v), precision=precision, trim='-')

    degree = getattr(poly, "degree", len(model.coef_))
    include_bias = getattr(poly, "include_bias", True)

    b0 = float(model.intercept_)
    coefs = np.ravel(model.coef_)

    # start with intercept
    if latex:
        eq = rf"$\hat{{y}} = {f(b0)}"
    else:
        eq = f"y = {f(b0)}"

    # add terms b_p * x^p
    for p in range(1, degree + 1):
        idx = p if include_bias else (p - 1)
        if idx >= len(coefs):
            break
        c = coefs[idx]
        if abs(c) < 1e-14:
            continue

        sign = " + " if c >= 0 else " - "
        mag = f(abs(c))

        if p == 1:
            term = x_symbol if not latex else rf"{x_symbol}"
        else:
            term = f"{x_symbol}^{p}" if not latex else rf"{x_symbol}^{{{p}}}"

        eq += sign + mag + (" * " if not latex else r"\,") + term

    if latex:
        eq += "$"
    return eq


def fmt(v):
    if not np.isfinite(v):
        return ""
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
        return str(int(round(v)))
    return f"{v:.2f}"


def add_table_ellipsis(ax, table, pad_axes=0.02):
    """
    Draw a centered ellipsis just below the rendered table.
    pad_axes is an offset in Axes coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure the table has a layout/extent

    renderer = fig.canvas.get_renderer()
    bbox_px = table.get_window_extent(renderer=renderer)
    bbox_ax = bbox_px.transformed(ax.transAxes.inverted())

    x_center = 0.5 * (bbox_ax.x0 + bbox_ax.x1)
    y_below  = max(0.0, bbox_ax.y0) - pad_axes  # stay inside axes a bit

    ax.text(x_center, y_below, "…",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, fontname=FONTNAME, clip_on=False)


def draw_poly_feature_table(ax,
                            new_features: np.ndarray,
                            y: np.ndarray,
                            max_rows: int = 10):
    ax.clear()
    ax.axis("off")
    x_symbol = "x"

    n_rows, n_cols = new_features.shape
    col_labels = []

    start = 0

    degree = n_cols - start
    for p in range(1, degree + 1):
        if p == 1:
            col_labels.append(x_symbol)
        else:
            col_labels.append(f"{x_symbol}^{p}")

    col_labels = col_labels + ["y"]

    if n_rows <= max_rows:
        idx = np.arange(n_rows)
    else:
        idx = np.linspace(0, n_rows - 1, max_rows, dtype=int)

    cell_text = []
    for i in idx:
        row_vals = [fmt(v) for v in new_features[i, :]] + [fmt(y[i])]
        cell_text.append(row_vals)

    tbl = ax.table(cellText=cell_text,
                   colLabels=col_labels,
                   loc='center',
                   cellLoc='center',
                   colLoc='center')

    tbl.scale(1, 2.0)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for _, cell in tbl.get_celld().items():
        cell.get_text().set_fontname(FONTNAME)

    return tbl


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Polynomial feature transformation"
        degree_prefix = "Polynomial degree"
        table_note = "first 5 rows of the dataset are shown"
    elif mode == "rus":
        title = "Полиномиальное преобразование"
        degree_prefix = "Степень полинома"
        table_note = "показаны первые 5 строк датасета"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, degree_prefix, table_note


def plot_poly_features_models(mode: str = "eng"):
    title, degree_prefix, table_note = annotations_by_language(mode)

    rooms, good_prices, bad_prices_first, bad_prices_second = get_datasets()
    common_features = np.concatenate([rooms, rooms, rooms])
    common_target = np.concatenate([good_prices, bad_prices_first, bad_prices_second])
    x, y, _, _ = take_sample_manual(common_features, common_target, apply_distortion=True)

    fig_size = (14, 8)
    fig, axs = plt.subplots(2, 3, figsize=fig_size)
    fig.subplots_adjust(left=0.05, right=0.97, hspace=0.25)

    for column_id, degree in zip([0, 1, 2], [2, 3, 6]):
        poly = PolynomialFeatures(degree)
        new_features = poly.fit_transform(x.reshape(-1, 1))

        # Pass only left columns because the first column in new_features is always constant
        table = draw_poly_feature_table(axs[0, column_id], new_features[:, 1:], y, max_rows=5)
        axs[0, column_id].set_title(f"{degree_prefix}: {degree}\n{table_note}",
                                    fontdict={'fontsize': 14, 'fontname': FONTNAME})
        add_table_ellipsis(axs[0, column_id], table, pad_axes=0.02)

        model = LinearRegression()
        model.fit(new_features, y)
        x_all = np.linspace(np.min(x), np.max(x), 100)
        predicted = model.predict(poly.transform(x_all.reshape(-1, 1)))

        axs[1, column_id].scatter(x, y, s=40, facecolors='white', edgecolors='black', linewidths=0.6, zorder=2)
        axs[1, column_id].set_xlabel("X", fontdict={'fontsize': 12, 'fontname': FONTNAME})
        axs[1, column_id].set_ylim(MIN_Y, MAX_Y)
        axs[1, column_id].set_xlim(0, 6)
        axs[1, column_id].set_xticks([1, 2, 3, 4, 5])
        axs[1, column_id].grid(alpha=0.3, color='grey')
        axs[1, column_id].plot(x_all, predicted, color='red')
        title_tex = construct_title(model, poly, x_symbol="x", precision=0, latex=True)
        axs[1, column_id].set_title(title_tex, fontdict={'fontsize': 10, 'fontname': FONTNAME}, color="red")

        if column_id == 0:
            axs[1, column_id].set_ylabel("Y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
        else:
            axs[1, column_id].yaxis.set_ticklabels([])

    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME}, va="top", y=1.04)

    raw_svg_file = Path(get_plots_path(), f"37_poly_features_{mode}.svg")
    final_plot = Path(get_plots_path(), f"37_poly_features_{mode}.png")
    plt.savefig(raw_svg_file, bbox_inches='tight')
    plt.close()

    save_plot_according_to_template(raw_svg_file, final_plot, template_name="template_small.svg")


if __name__ == '__main__':
    plot_poly_features_models("rus")
    plot_poly_features_models("eng")
