from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, take_sample_manual, get_extended_dataset


FONTNAME = "Comic Sans MS"
DPI = 200

FEATURE_COLUMNS = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
NUMERIC_COLUMNS = ["rooms", "area", "metro_distance"]
CITY_COLUMN = "city"
AC_COLUMN = "ac_in_apartment"


@dataclass
class RusAnnotations:
    suptitle: str = "История градиентного спуска для многомерной линейной регрессии"
    top_title: str = "Важность признака для модели зависит от размера коэффициента перед ним. Значения по итерациям"
    top_y: str = "Доля коэффициента перед признаком"
    bottom_title: str = "Ошибка на обучающей выборке по итерациям"
    bottom_x: str = "Итерация градиентного спуска"
    bottom_y: str = "Значение квадратической ошибки"
    legend_title: str = "Признак"


@dataclass
class EngAnnotations:
    suptitle: str = "Gradient descent history for multivariate linear regression"
    top_title: str = "Feature importance depends on coefficient magnitude. Values across iterations"
    top_y: str = "Share of feature coefficient"
    bottom_title: str = "Training error vs iteration"
    bottom_x: str = "Gradient descent iteration"
    bottom_y: str = "Squared error value"
    legend_title: str = "Feature"


def annotations_by_language(mode: str) -> Union[RusAnnotations, EngAnnotations]:
    if mode == "rus":
        return RusAnnotations()
    if mode == "eng":
        return EngAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def _encode_ac_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()

    positive = {"1", "true", "yes", "y", "да", "есть"}
    negative = {"0", "false", "no", "n", "нет", "none", "nan"}

    def to_bin(val: str) -> int:
        if val in positive:
            return 1
        if val in negative:
            return 0
        return 0

    return s.map(to_bin).astype(float)


def _build_readable_feature_names(city_categories_sorted: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {
        "rooms_scaled": "количество комнат",
        "area_scaled": "площадь квартиры",
        "metro_distance_scaled": "расстояние до метро",
        "ac_yes": "есть ли кондиционер",
    }

    letters = ["A", "B", "C", "D", "E"]
    for idx, cat in enumerate(city_categories_sorted):
        letter = letters[idx] if idx < len(letters) else str(idx + 1)
        mapping[f"city__{cat}"] = f"город {letter}"

    return mapping


def load_scaled_data_multifeature() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset = get_extended_dataset()

    features = dataset[FEATURE_COLUMNS]
    target = dataset["price"].to_numpy()

    x_train_raw, y_train_raw, _, _ = take_sample_manual(
        np.array(features), np.array(target), apply_distortion=True
    )

    x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    y_train = np.ravel(np.array(y_train_raw, dtype=float))

    # 1) Scale numeric columns
    numeric_scaler = StandardScaler()
    x_num = x_train_df[NUMERIC_COLUMNS].astype(float)
    x_num_scaled = numeric_scaler.fit_transform(x_num)

    # 2) City one-hot in stable order
    city_series = x_train_df[CITY_COLUMN].astype(str)
    city_categories_sorted = sorted(city_series.unique().tolist())
    city_name_map = _build_readable_feature_names(city_categories_sorted)

    city_dummies = pd.get_dummies(city_series, dtype=float)
    city_dummies = city_dummies.reindex(columns=city_categories_sorted, fill_value=0.0)
    city_dummies.columns = [f"city__{c}" for c in city_dummies.columns]

    # 3) AC -> single column (yes/no), for clarity and no redundancy
    ac_yes = _encode_ac_binary(x_train_df[AC_COLUMN]).astype(float).rename("ac_yes")

    x_processed = np.hstack([
        x_num_scaled,
        city_dummies.to_numpy(dtype=float),
        ac_yes.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    feature_names = (
        [f"{c}_scaled" for c in NUMERIC_COLUMNS]
        + list(city_dummies.columns)
        + ["ac_yes"]
    )

    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(float)

    feature_names_human = [city_name_map.get(name, name) for name in feature_names]
    return x_processed, y_scaled, feature_names_human


def mse_and_gradients_multi(
    intercept_value: float,
    weights: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    MSE(b0, w) = (1/n) sum (y - (b0 + Xw))^2
    residual = y_hat - y

    dMSE/db0 = (2/n) * sum(residual)
    dMSE/dw  = (2/n) * X^T residual
    """
    x_values = np.asarray(features, dtype=float)
    y_values = np.asarray(target, dtype=float).ravel()
    w_values = np.asarray(weights, dtype=float).ravel()

    predicted = float(intercept_value) + x_values @ w_values
    residual = predicted - y_values

    n = float(len(y_values))
    mse_value = float(np.mean(residual ** 2))

    grad_b0 = float((2.0 / n) * np.sum(residual))
    grad_w = (2.0 / n) * (x_values.T @ residual)

    return mse_value, grad_b0, grad_w


def gradient_descent_mse_and_coeff_history(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    grad_tol: float,
) -> Tuple[List[float], List[float], List[np.ndarray]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    n_features = int(x_train.shape[1])
    current_b0 = 0.0
    current_w = np.zeros(n_features, dtype=float)

    mse_history: List[float] = []
    b0_history: List[float] = []
    w_history: List[np.ndarray] = []

    for _ in range(int(max_iterations) + 1):
        mse_value, grad_b0, grad_w = mse_and_gradients_multi(current_b0, current_w, x_train, y_train)

        mse_history.append(float(mse_value))
        b0_history.append(float(current_b0))
        w_history.append(current_w.copy())

        grad_norm = float(np.hypot(grad_b0, np.linalg.norm(grad_w)))
        if grad_norm < float(grad_tol):
            break

        current_b0 = float(current_b0) - float(learning_rate) * float(grad_b0)
        current_w = current_w - float(learning_rate) * grad_w

    return mse_history, b0_history, w_history


def _build_dense_fill_shares(
    b0_history: List[float],
    w_history: List[np.ndarray],
    points_per_step: int = 40,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    iteration_values = np.arange(len(b0_history), dtype=float)

    n_iters = int(len(iteration_values))
    n_features = int(len(w_history[0])) if len(w_history) > 0 else 0
    n_coeffs = 1 + n_features  # b0 + all weights

    coef_matrix = np.zeros((n_coeffs, n_iters), dtype=float)
    coef_matrix[0, :] = np.array(b0_history, dtype=float)
    for idx, w_vec in enumerate(w_history):
        coef_matrix[1:, idx] = np.ravel(np.array(w_vec, dtype=float))

    # Make it look continuous: interpolate in time between iterations
    if n_iters <= 1:
        x_dense = iteration_values.copy()
        abs_dense = np.abs(coef_matrix)
    else:
        n_dense = int((n_iters - 1) * points_per_step + 1)
        x_dense = np.linspace(0.0, float(n_iters - 1), n_dense, dtype=float)

        abs_dense = np.zeros((n_coeffs, n_dense), dtype=float)
        for row_idx in range(n_coeffs):
            abs_series = np.abs(coef_matrix[row_idx, :]).astype(float)
            abs_dense[row_idx, :] = np.interp(x_dense, iteration_values, abs_series)

    denom = np.sum(abs_dense, axis=0, keepdims=True)
    denom = np.maximum(denom, float(eps))
    shares = abs_dense / denom

    return x_dense, shares


def _compute_shares_at_iteration(
    b0_history: List[float],
    w_history: List[np.ndarray],
    iteration_index: int,
    eps: float = 1e-12,
) -> np.ndarray:
    last_abs = np.abs(
        np.concatenate((
            [float(b0_history[iteration_index])],
            np.ravel(np.array(w_history[iteration_index], dtype=float)),
        ))
    )
    denom = float(max(np.sum(last_abs), eps))
    return (last_abs / denom).astype(float)


def _annotate_iteration_shares(
    ax,
    iter_x: int,
    shares_at_iter: np.ndarray,
    x_offset: float = 0.22,
    fontsize: int = 7,  # smaller as requested earlier
):
    cum = np.cumsum(shares_at_iter)
    y_bottoms = np.concatenate(([0.0], cum[:-1]))
    y_tops = cum

    x_text = float(iter_x) + float(x_offset)

    for share, y0, y1 in zip(shares_at_iter, y_bottoms, y_tops):
        pct = int(np.rint(float(share) * 100.0))
        if pct <= 0:
            continue
        y_mid = 0.5 * (float(y0) + float(y1))
        ax.text(
            x_text,
            y_mid,
            f"{pct}%",
            color="black",
            fontsize=int(fontsize),
            fontname=FONTNAME,
            va="center",
            ha="left",
            zorder=7,
        )


def build_grouped_palette(coeff_names: List[str]) -> List[tuple]:
    """
    Группируем цвета по смысловым блокам:
    - города -> Blues
    - кондиционер -> Reds
    - численные -> Greens
    - коэффициент сдвига -> grey
    """
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")
    greens = plt.get_cmap("Greens")
    purples = plt.get_cmap("Purples")

    city_idx = [i for i, name in enumerate(coeff_names) if name.startswith("город ")]
    ac_idx = [i for i, name in enumerate(coeff_names) if "кондиционер" in name]
    numeric_names = {"количество комнат", "площадь квартиры", "расстояние до метро"}
    numeric_idx = [i for i, name in enumerate(coeff_names) if name in numeric_names]
    intercept_idx = [i for i, name in enumerate(coeff_names) if name == "коэффициент сдвига"]

    colors = [None] * len(coeff_names)

    def assign(idx_list: List[int], cmap, lo: float = 0.45, hi: float = 0.85):
        if len(idx_list) == 0:
            return
        vals = np.linspace(lo, hi, len(idx_list))
        for j, idx in enumerate(idx_list):
            colors[idx] = cmap(float(vals[j]))

    # Intercept as grey
    for idx in intercept_idx:
        colors[idx] = (0.35, 0.35, 0.35, 1.0)

    assign(numeric_idx, greens, lo=0.45, hi=0.85)
    assign(city_idx, blues, lo=0.45, hi=0.85)
    assign(ac_idx, reds, lo=0.55, hi=0.75)

    # Fallback for anything else
    fallback_idx = [i for i, c in enumerate(colors) if c is None]
    assign(fallback_idx, purples, lo=0.45, hi=0.85)

    return colors  # order matches layers; legend reversal won't affect these


def show_mse_vs_iteration_static(
    mode: str = "rus",
    learning_rate: float = 0.25,
    max_iterations: int = 25,
    grad_tol: float = 1e-8,
    annotate_iteration_5: bool = True,
):
    sns.set_theme(style="whitegrid")

    annotations = annotations_by_language(mode)

    x_train, y_train, feature_names_human = load_scaled_data_multifeature()

    mse_history, b0_history, w_history = gradient_descent_mse_and_coeff_history(
        x_train=x_train,
        y_train=y_train,
        learning_rate=float(learning_rate),
        max_iterations=int(max_iterations),
        grad_tol=float(grad_tol),
    )

    iterations = np.arange(len(mse_history), dtype=int)

    coeff_names = ["коэффициент сдвига"] + list(feature_names_human)

    x_dense, shares_dense = _build_dense_fill_shares(
        b0_history=b0_history,
        w_history=w_history,
        points_per_step=40,
    )

    fig = plt.figure(figsize=(14, 7))
    gridspec = fig.add_gridspec(
        2, 2,
        height_ratios=[1.05, 1.05],
        width_ratios=[22.0, 4.0],
        hspace=0.35,
        wspace=0.05,
    )

    ax_top = fig.add_subplot(gridspec[0, 0])
    ax_bottom = fig.add_subplot(gridspec[1, 0], sharex=ax_top)
    ax_legend = fig.add_subplot(gridspec[:, 1])
    ax_legend.set_axis_off()

    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Top: stackplot with grouped palette ---
    grouped_colors = build_grouped_palette(coeff_names)

    stack_handles = ax_top.stackplot(
        x_dense,
        shares_dense,
        labels=coeff_names,
        colors=grouped_colors,
        alpha=0.8,
        zorder=2,
    )

    ax_top.set_title(
        annotations.top_title,
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        y=1.1,
    )
    ax_top.set_ylabel(annotations.top_y, fontdict={"fontsize": 11, "fontname": FONTNAME})
    ax_top.set_ylim(0.0, 1.0)

    if len(iterations) > 0:
        last_iter = int(iterations[-1])

        # Add space on right for % labels
        ax_top.set_xlim(-0.5, float(last_iter) + 1.0)

        # Manual vertical iteration markers BEHIND the fill (thin black lines)
        for x in iterations:
            ax_top.axvline(
                x=float(x),
                color="black",
                linewidth=0.6,
                alpha=0.22,
                zorder=1,
            )

        # Emphasize iteration 5
        if annotate_iteration_5 and last_iter >= 5:
            ax_top.axvline(
                x=5.0,
                color="black",
                linewidth=1.4,
                alpha=0.75,
                zorder=6,
            )

        # Annotate shares at last iteration
        shares_last = _compute_shares_at_iteration(b0_history, w_history, iteration_index=last_iter)
        _annotate_iteration_shares(
            ax=ax_top,
            iter_x=last_iter,
            shares_at_iter=shares_last,
            x_offset=0.22,
            fontsize=7,
        )
        ax_top.axvline(
            x=last_iter,
            color="black",
            linewidth=1.4,
            alpha=0.75,
            zorder=6,
        )

        # Annotate shares at iteration 5
        if annotate_iteration_5 and last_iter >= 5:
            shares_5 = _compute_shares_at_iteration(b0_history, w_history, iteration_index=5)
            _annotate_iteration_shares(
                ax=ax_top,
                iter_x=5,
                shares_at_iter=shares_5,
                x_offset=0.22,
                fontsize=7,
            )

    ax_top.tick_params(axis="x", labelbottom=False)

    # --- Legend order: bottom-to-top (b0 at bottom), colors unchanged on plot ---
    ax_legend.legend(
        stack_handles[::-1],
        coeff_names[::-1],
        loc="center",
        frameon=True,
        title=annotations.legend_title,
        fontsize=9,
        title_fontsize=10,
    )

    # --- Bottom: scatter MSE vs iteration ---
    ax_bottom.grid(True, axis="both", which="major", color="grey", alpha=0.3, zorder=1)
    ax_bottom.scatter(
        iterations,
        mse_history,
        s=70,
        c="red",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax_bottom.plot(iterations, mse_history, "-", color="black", alpha=0.35, zorder=2)

    ax_bottom.set_title(
        annotations.bottom_title,
        fontsize=14,
        fontdict={"fontname": FONTNAME},
        y=1.02,
    )
    ax_bottom.set_xticks([0, 5, 10, 15, 20, 25])
    ax_bottom.set_xlabel(annotations.bottom_x, fontdict={"fontsize": 11, "fontname": FONTNAME})
    ax_bottom.set_ylabel(annotations.bottom_y, fontdict={"fontsize": 11, "fontname": FONTNAME})

    fig.suptitle(
        annotations.suptitle,
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        va="top",
        x=0.5,
        y=1.04,
    )

    raw_svg = Path(get_plots_path(), f"62_mse_vs_iteration_static_{mode}.svg")
    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    out_png = Path(get_plots_path(), f"62_mse_vs_iteration_static_{mode}.png")
    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name="template.svg",
        dpi=DPI,
    )

    print(f"Saved: {out_png}")
    # ===== Print final equation (order matches legend) =====
    b0_last = float(b0_history[-1])
    w_last = np.ravel(np.array(w_history[-1], dtype=float))

    # names and coefficients in the same order as legend:
    # legend uses coeff_names[::-1], so we do the same
    coeff_names = ["коэффициент сдвига"] + list(feature_names_human)
    coef_values = np.concatenate(([b0_last], w_last), axis=0)

    legend_names = coeff_names[::-1]
    legend_coefs = coef_values[::-1]

    terms = []
    for name, coef in zip(legend_names, legend_coefs):
        c = float(coef)
        # print all terms (or skip near-zero if you want)
        # if abs(c) < 1e-8:
        #     continue
        sign = "+" if c >= 0 else "-"
        terms.append(f" {sign} {abs(c):.2f}·({name})")

    equation = "y_scaled =" + "".join(terms)

    print("\n" + "-" * 80)
    print("Model:")
    print(equation)
    print("-" * 80 + "\n")


if __name__ == "__main__":
    show_mse_vs_iteration_static(
        mode="rus",
        learning_rate=0.25,
        max_iterations=25,
        grad_tol=1e-8,
        annotate_iteration_5=True,
    )
