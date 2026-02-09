from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template, split_train_test_manual, get_extended_dataset, \
    take_sample_manual

FONTNAME = "Comic Sans MS"
DPI = 200

FEATURE_COLUMNS = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
NUMERIC_COLUMNS = ["rooms", "area", "metro_distance"]
CITY_COLUMN = "city"
AC_COLUMN = "ac_in_apartment"


@dataclass
class RusAnnotations:
    suptitle: str = "Сравнение моделей и метрик ошибки на обучении и тесте при разных типах регуляризации"
    top_title: str = "Важность признаков в модели"
    top_y: str = "Доля коэффициента перед признаком"
    bottom_title: str = "Ошибка на обучающей и тестовой выборках по итерациям"
    bottom_x: str = "Итерация градиентного спуска"
    bottom_y: str = "Значение квадратической ошибки"
    legend_title: str = "Признак"
    legend_train: str = "Обучающая выборка"
    legend_test: str = "Тестовая выборка"


@dataclass
class EngAnnotations:
    suptitle: str = "Gradient descent history for multivariate linear regression"
    top_title: str = "Feature importance depends on coefficient magnitude. Values across iterations"
    top_y: str = "Share of feature coefficient"
    bottom_title: str = "Train and test error vs iteration"
    bottom_x: str = "Gradient descent iteration"
    bottom_y: str = "Squared error value"
    legend_title: str = "Feature"
    legend_train: str = "train"
    legend_test: str = "test"


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
    for idx, category in enumerate(city_categories_sorted):
        letter = letters[idx] if idx < len(letters) else str(idx + 1)
        mapping[f"city__{category}"] = f"город {letter}"

    return mapping


def load_scaled_train_test_multifeature() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset = get_extended_dataset()

    features = dataset[FEATURE_COLUMNS]
    target = dataset["price"].to_numpy()

    x_train_raw, y_train_raw, _, _ = take_sample_manual(
        np.array(features), np.array(target), apply_distortion=True
    )
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = split_train_test_manual(
        x_train_raw, y_train_raw, random_state=52,
    )

    x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    x_test_df = pd.DataFrame(x_test_raw, columns=FEATURE_COLUMNS)

    y_train = np.ravel(np.array(y_train_raw, dtype=float))
    y_test = np.ravel(np.array(y_test_raw, dtype=float))

    # --- Numeric scaling (fit on train) ---
    numeric_scaler = StandardScaler()
    x_train_num = x_train_df[NUMERIC_COLUMNS].astype(float)
    x_test_num = x_test_df[NUMERIC_COLUMNS].astype(float)

    x_train_num_scaled = numeric_scaler.fit_transform(x_train_num)
    x_test_num_scaled = numeric_scaler.transform(x_test_num)

    # --- City one-hot (stable order from train, align test) ---
    city_train = x_train_df[CITY_COLUMN].astype(str)
    city_test = x_test_df[CITY_COLUMN].astype(str)

    city_categories_sorted = sorted(city_train.unique().tolist())
    readable_map = _build_readable_feature_names(city_categories_sorted)

    city_train_dummies = pd.get_dummies(city_train, dtype=float).reindex(columns=city_categories_sorted, fill_value=0.0)
    city_test_dummies = pd.get_dummies(city_test, dtype=float).reindex(columns=city_categories_sorted, fill_value=0.0)

    city_train_dummies.columns = [f"city__{c}" for c in city_train_dummies.columns]
    city_test_dummies.columns = [f"city__{c}" for c in city_test_dummies.columns]

    # --- AC as a single binary column ---
    ac_train_yes = _encode_ac_binary(x_train_df[AC_COLUMN]).astype(float).rename("ac_yes")
    ac_test_yes = _encode_ac_binary(x_test_df[AC_COLUMN]).astype(float).rename("ac_yes")

    x_train_processed = np.hstack([
        x_train_num_scaled,
        city_train_dummies.to_numpy(dtype=float),
        ac_train_yes.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    x_test_processed = np.hstack([
        x_test_num_scaled,
        city_test_dummies.to_numpy(dtype=float),
        ac_test_yes.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    feature_names = (
        [f"{c}_scaled" for c in NUMERIC_COLUMNS]
        + list(city_train_dummies.columns)
        + ["ac_yes"]
    )
    feature_names_human = [readable_map.get(name, name) for name in feature_names]

    # --- Target scaling (fit on train, apply to test) ---
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(float)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(float)

    return x_train_processed, y_train_scaled, x_test_processed, y_test_scaled, feature_names_human


def mse_and_gradients_multi(
    intercept_value: float,
    weights: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    y_hat = b0 + X w
    residual = y_hat - y

    MSE = mean(residual^2)

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


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    thr = float(threshold)
    return np.sign(values) * np.maximum(np.abs(values) - thr, 0.0)


def history_lasso_prox_gd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    lasso_lambda: float,
) -> Tuple[List[float], List[float], List[float], List[np.ndarray]]:
    """
    Proximal gradient descent for Lasso:
      w_temp = w - lr * grad_mse_w
      w = S(w_temp, lr * lambda)
    b0 is NOT regularized.
    Metrics stored: pure MSE on train/test (without penalty).
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if lasso_lambda < 0:
        raise ValueError("lasso_lambda must be non-negative.")

    n_features = int(x_train.shape[1])
    current_b0 = 0.0
    current_w = np.zeros(n_features, dtype=float)

    train_hist: List[float] = []
    test_hist: List[float] = []
    b0_hist: List[float] = []
    w_hist: List[np.ndarray] = []

    for _ in range(int(max_iterations) + 1):
        mse_train, grad_b0, grad_w_mse = mse_and_gradients_multi(current_b0, current_w, x_train, y_train)
        mse_test, _, _ = mse_and_gradients_multi(current_b0, current_w, x_test, y_test)

        train_hist.append(float(mse_train))
        test_hist.append(float(mse_test))
        b0_hist.append(float(current_b0))
        w_hist.append(current_w.copy())

        # b0 update (no regularization)
        current_b0 = float(current_b0) - float(learning_rate) * float(grad_b0)

        # proximal step for w
        w_temp = current_w - float(learning_rate) * grad_w_mse
        current_w = soft_threshold(w_temp, float(learning_rate) * float(lasso_lambda))

    return train_hist, test_hist, b0_hist, w_hist


def history_ridge_gd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    ridge_lambda: float,
) -> Tuple[List[float], List[float], List[float], List[np.ndarray]]:
    """
    Gradient descent for Ridge:
      grad_w_total = grad_mse_w + 2*lambda*w
    b0 is NOT regularized.
    Metrics stored: pure MSE on train/test (without penalty).
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative.")

    n_features = int(x_train.shape[1])
    current_b0 = 0.0
    current_w = np.zeros(n_features, dtype=float)

    train_hist: List[float] = []
    test_hist: List[float] = []
    b0_hist: List[float] = []
    w_hist: List[np.ndarray] = []

    for _ in range(int(max_iterations) + 1):
        mse_train, grad_b0, grad_w_mse = mse_and_gradients_multi(current_b0, current_w, x_train, y_train)
        mse_test, _, _ = mse_and_gradients_multi(current_b0, current_w, x_test, y_test)

        train_hist.append(float(mse_train))
        test_hist.append(float(mse_test))
        b0_hist.append(float(current_b0))
        w_hist.append(current_w.copy())

        # b0 update (no regularization)
        current_b0 = float(current_b0) - float(learning_rate) * float(grad_b0)

        # ridge update for w
        grad_w_total = grad_w_mse + 2.0 * float(ridge_lambda) * current_w
        current_w = current_w - float(learning_rate) * grad_w_total

    return train_hist, test_hist, b0_hist, w_hist


def _build_dense_fill_shares(
    b0_history: List[float],
    w_history: List[np.ndarray],
    points_per_step: int = 40,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    iteration_values = np.arange(len(b0_history), dtype=float)
    n_iters = int(len(iteration_values))

    n_features = int(len(w_history[0])) if len(w_history) > 0 else 0
    n_coeffs = 1 + n_features  # b0 + w

    coef_matrix = np.zeros((n_coeffs, n_iters), dtype=float)
    coef_matrix[0, :] = np.array(b0_history, dtype=float)
    for idx, w_vec in enumerate(w_history):
        coef_matrix[1:, idx] = np.ravel(np.array(w_vec, dtype=float))

    # interpolate abs(coefs) to look continuous
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

    denom_raw = np.sum(abs_dense, axis=0, keepdims=True)
    denom = np.maximum(denom_raw, float(eps))
    shares = abs_dense / denom

    # if all coefs ~ 0 -> show 100% on intercept
    near_zero_mask = (denom_raw.ravel() < float(eps))
    if np.any(near_zero_mask):
        shares[:, near_zero_mask] = 0.0
        shares[0, near_zero_mask] = 1.0

    return x_dense, shares


def _compute_shares_at_iteration(
    b0_history: List[float],
    w_history: List[np.ndarray],
    iteration_index: int,
    eps: float = 1e-12,
) -> np.ndarray:
    abs_values = np.abs(
        np.concatenate((
            [float(b0_history[iteration_index])],
            np.ravel(np.array(w_history[iteration_index], dtype=float)),
        ))
    )
    denom = float(np.sum(abs_values))
    if denom < float(eps):
        out = np.zeros_like(abs_values, dtype=float)
        out[0] = 1.0
        return out
    return (abs_values / denom).astype(float)


def _annotate_iteration_shares(
    ax,
    iter_x: int,
    shares_at_iter: np.ndarray,
    x_offset: float = 0.22,
    fontsize: int = 7,
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

    for idx in intercept_idx:
        colors[idx] = (0.35, 0.35, 0.35, 1.0)

    assign(numeric_idx, greens, lo=0.45, hi=0.85)
    assign(city_idx, blues, lo=0.45, hi=0.85)
    assign(ac_idx, reds, lo=0.55, hi=0.75)

    fallback_idx = [i for i, c in enumerate(colors) if c is None]
    assign(fallback_idx, purples, lo=0.45, hi=0.85)

    return colors


def _plot_top_panel(
    ax_top,
    x_dense: np.ndarray,
    shares_dense: np.ndarray,
    coeff_names: List[str],
    colors: List[tuple],
    iterations: np.ndarray,
    b0_hist: List[float],
    w_hist: List[np.ndarray],
    annotations,
    subtitle: str,
    annotate_iteration_5: bool,
    max_iter_global: int,
):
    stack_handles = ax_top.stackplot(
        x_dense,
        shares_dense,
        labels=coeff_names,
        colors=colors,
        alpha=0.8,
        zorder=2,
    )

    ax_top.set_title(
        f"{annotations.top_title}\n({subtitle})",
        fontsize=13,
        fontdict={"fontname": FONTNAME},
        y=1.06,
    )
    ax_top.set_ylabel(annotations.top_y, fontdict={"fontsize": 11, "fontname": FONTNAME})
    ax_top.set_ylim(0.0, 1.0)
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_top.set_xlim(-0.5, float(max_iter_global) + 1.0)

    # Vertical iteration markers behind fill
    for x_val in range(0, int(max_iter_global) + 1):
        ax_top.axvline(x=float(x_val), color="black", linewidth=0.6, alpha=0.22, zorder=1)

    if annotate_iteration_5 and max_iter_global >= 5:
        ax_top.axvline(x=5.0, color="black", linewidth=1.4, alpha=0.75, zorder=6)

    if len(iterations) > 0:
        last_iter = int(iterations[-1])

        ax_top.axvline(x=float(last_iter), color="black", linewidth=1.4, alpha=0.75, zorder=6)

        shares_last = _compute_shares_at_iteration(b0_hist, w_hist, iteration_index=last_iter)
        _annotate_iteration_shares(ax_top, iter_x=last_iter, shares_at_iter=shares_last, fontsize=7)

        if annotate_iteration_5 and last_iter >= 5:
            shares_5 = _compute_shares_at_iteration(b0_hist, w_hist, iteration_index=5)
            _annotate_iteration_shares(ax_top, iter_x=5, shares_at_iter=shares_5, fontsize=7)

    return stack_handles


def _plot_bottom_panel(
    ax_bottom,
    iterations: np.ndarray,
    mse_train: List[float],
    mse_test: List[float],
    annotations,
    subtitle: str,
    xticks: List[int],
):
    ax_bottom.grid(True, axis="both", which="major", color="grey", alpha=0.3, zorder=1)
    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_bottom.set_xticks(xticks)

    ax_bottom.scatter(
        iterations, mse_train,
        s=35, c="red", edgecolor="black", linewidth=0.6, zorder=3,
        label=annotations.legend_train,
    )
    ax_bottom.plot(iterations, mse_train, "-", color="red", alpha=0.45, zorder=2)

    ax_bottom.scatter(
        iterations, mse_test,
        s=35, c="black", edgecolor="black", linewidth=0.6, zorder=3,
        label=annotations.legend_test,
    )
    ax_bottom.plot(iterations, mse_test, "-", color="black", alpha=0.35, zorder=2)

    ax_bottom.set_title(
        f"{annotations.bottom_title}\n({subtitle})",
        fontsize=13,
        fontdict={"fontname": FONTNAME},
        y=1.02,
    )
    ax_bottom.set_xlabel(annotations.bottom_x, fontdict={"fontsize": 11, "fontname": FONTNAME})
    ax_bottom.set_ylabel(annotations.bottom_y, fontdict={"fontsize": 11, "fontname": FONTNAME})
    ax_bottom.legend(loc="upper right")


def _print_equation(model_name: str, b0_hist: List[float], w_hist: List[np.ndarray], feature_names: List[str]) -> None:
    b0_last = float(b0_hist[-1])
    w_last = np.ravel(np.array(w_hist[-1], dtype=float))

    feature_names_ordered = list(feature_names)[::-1]
    w_ordered = w_last[::-1]

    lines: List[str] = []
    lines.append(f"{model_name}:")
    lines.append(f"y_scaled = {b0_last:+.2f}")  # intercept as constant

    for name, coef in zip(feature_names_ordered, w_ordered):
        c = float(coef)
        lines.append(f"          {c:+.2f} · ({name})")

    print("\n" + "-" * 90)
    print("\n".join(lines))
    print("-" * 90 + "\n")


def show_lasso_vs_ridge_history_static(
    mode: str = "rus",
    learning_rate: float = 0.05,
    max_iterations: int = 25,
    lasso_lambda: float = 0.2,
    ridge_lambda: float = 0.9,
    annotate_iteration_5: bool = True,
):
    sns.set_theme(style="whitegrid")
    annotations = annotations_by_language(mode)

    x_train, y_train, x_test, y_test, feature_names_human = load_scaled_train_test_multifeature()
    coeff_names = ["коэффициент сдвига"] + list(feature_names_human)

    # --- Histories ---
    lasso_train_mse, lasso_test_mse, lasso_b0_hist, lasso_w_hist = history_lasso_prox_gd(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_rate=float(learning_rate),
        max_iterations=int(max_iterations),
        lasso_lambda=float(lasso_lambda),
    )

    ridge_train_mse, ridge_test_mse, ridge_b0_hist, ridge_w_hist = history_ridge_gd(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_rate=float(learning_rate),
        max_iterations=int(max_iterations),
        ridge_lambda=float(ridge_lambda),
    )

    lasso_iters = np.arange(len(lasso_train_mse), dtype=int)
    ridge_iters = np.arange(len(ridge_train_mse), dtype=int)
    max_iter_global = int(
        max(lasso_iters[-1] if len(lasso_iters) else 0, ridge_iters[-1] if len(ridge_iters) else 0)
    )
    xticks = list(range(0, max_iter_global + 1, 5)) if max_iter_global >= 5 else list(range(0, max_iter_global + 1))

    # Dense fill for top plots
    lasso_x_dense, lasso_shares_dense = _build_dense_fill_shares(lasso_b0_hist, lasso_w_hist, points_per_step=40)
    ridge_x_dense, ridge_shares_dense = _build_dense_fill_shares(ridge_b0_hist, ridge_w_hist, points_per_step=40)

    # Same palette in both columns
    grouped_colors = build_grouped_palette(coeff_names)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

    ax_top_lasso = fig.add_subplot(gs[0, 0])
    ax_bottom_lasso = fig.add_subplot(gs[1, 0], sharex=ax_top_lasso)
    ax_bottom_lasso.set_ylim(0, 2)

    ax_top_ridge = fig.add_subplot(gs[0, 1])
    ax_bottom_ridge = fig.add_subplot(gs[1, 1], sharex=ax_top_ridge)
    ax_bottom_ridge.set_ylim(0, 2)

    # Make room for legend below bottom row
    fig.subplots_adjust(bottom=0.18)

    # --- Lasso column ---
    handles_lasso = _plot_top_panel(
        ax_top=ax_top_lasso,
        x_dense=lasso_x_dense,
        shares_dense=lasso_shares_dense,
        coeff_names=coeff_names,
        colors=grouped_colors,
        iterations=lasso_iters,
        b0_hist=lasso_b0_hist,
        w_hist=lasso_w_hist,
        annotations=annotations,
        subtitle=f"Lasso, λ={lasso_lambda:.2f}",
        annotate_iteration_5=bool(annotate_iteration_5),
        max_iter_global=max_iter_global,
    )
    _plot_bottom_panel(
        ax_bottom=ax_bottom_lasso,
        iterations=lasso_iters,
        mse_train=lasso_train_mse,
        mse_test=lasso_test_mse,
        annotations=annotations,
        subtitle=f"Lasso, λ={lasso_lambda:.2f}",
        xticks=xticks,
    )

    # --- Ridge column ---
    _plot_top_panel(
        ax_top=ax_top_ridge,
        x_dense=ridge_x_dense,
        shares_dense=ridge_shares_dense,
        coeff_names=coeff_names,
        colors=grouped_colors,
        iterations=ridge_iters,
        b0_hist=ridge_b0_hist,
        w_hist=ridge_w_hist,
        annotations=annotations,
        subtitle=f"Ridge, λ={ridge_lambda:.2f}",
        annotate_iteration_5=bool(annotate_iteration_5),
        max_iter_global=max_iter_global,
    )
    _plot_bottom_panel(
        ax_bottom=ax_bottom_ridge,
        iterations=ridge_iters,
        mse_train=ridge_train_mse,
        mse_test=ridge_test_mse,
        annotations=annotations,
        subtitle=f"Ridge, λ={ridge_lambda:.2f}",
        xticks=xticks,
    )

    # --- Feature legend: move to bottom, centered, below ax_bottom ---
    fig.legend(
        handles_lasso[::-1],
        coeff_names[::-1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        title=annotations.legend_title,
        fontsize=9,
        title_fontsize=10,
    )

    fig.suptitle(
        annotations.suptitle,
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        va="top",
        x=0.5,
        y=1.05,
    )

    raw_svg = Path(get_plots_path(), f"63_lasso_vs_ridge_history_static_{mode}.svg")
    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    out_png = Path(get_plots_path(), f"63_lasso_vs_ridge_history_static_{mode}.png")
    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name="template.svg",
        dpi=DPI,
    )

    print(f"Saved: {out_png}")

    _print_equation(
        model_name=f"Lasso (λ={float(lasso_lambda):.3f})",
        b0_hist=lasso_b0_hist,
        w_hist=lasso_w_hist,
        feature_names=feature_names_human,
    )

    _print_equation(
        model_name=f"Ridge (λ={float(ridge_lambda):.3f})",
        b0_hist=ridge_b0_hist,
        w_hist=ridge_w_hist,
        feature_names=feature_names_human,
    )


if __name__ == "__main__":
    show_lasso_vs_ridge_history_static(
        mode="rus",
        learning_rate=0.05,
        max_iterations=25,
        lasso_lambda=0.2,
        ridge_lambda=0.2,
        annotate_iteration_5=True,
    )
