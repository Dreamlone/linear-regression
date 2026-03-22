import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import (
    save_plot_according_to_template,
    split_train_test_manual,
    get_extended_dataset,
    take_sample_manual,
)

FONTNAME = "Comic Sans MS"
DPI = 200
ANIMATION_DURATION: int = 180

REGULARIZATION_GRID_SIZE: int = 35
MIN_REGULARIZATION: float = 0.05
MAX_REGULARIZATION: float = 2.0

RMSE_MIN: float = 0.0
RMSE_MAX: float = 2.0

COEFF_MIN: float = -2.5
COEFF_MAX: float = 2.5

FEATURE_COLUMNS = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
NUMERIC_COLUMNS = ["rooms", "area", "metro_distance"]
CITY_COLUMN = "city"
AC_COLUMN = "ac_in_apartment"


@dataclass
class RusAnnotations:
    suptitle: str = "Влияние параметра регуляризации λ на модель"

    # Biplot labels
    biplot_x: str = "Отклик (y)"
    biplot_y: str = "Предсказания (ŷ)"

    # Row titles
    lr_title: str = "Модель без регуляризации"
    lr_coef_title: str = "Коэффициенты модели без регуляризации"

    lasso_rmse_title: str = "Lasso: зависимость RMSE от λ"
    lasso_coef_title: str = "Lasso: коэффициенты признаков vs λ"
    lasso_biplot_title: str = "L1 регуляризация"

    ridge_rmse_title: str = "Ridge: зависимость RMSE от λ"
    ridge_coef_title: str = "Ridge: коэффициенты признаков vs λ"
    ridge_biplot_title: str = "L2 регуляризация"

    rmse_y: str = "RMSE"
    gamma_x: str = "Степень регуляризации λ"
    coef_y: str = "Коэффициент"

    legend_train: str = "обучение"
    legend_test: str = "тест"


@dataclass
class EngAnnotations:
    suptitle: str = "Effect of regularization strength λ on the model"

    # Biplot labels
    biplot_x: str = "Target (y)"
    biplot_y: str = "Predictions (ŷ)"

    # Row titles
    lr_title: str = "Model without regularization"
    lr_coef_title: str = "Feature coefficients without regularization"

    lasso_rmse_title: str = "Lasso: RMSE vs λ"
    lasso_coef_title: str = "Lasso: feature coefficients vs λ"
    lasso_biplot_title: str = "L1 regularization"

    ridge_rmse_title: str = "Ridge: RMSE vs λ"
    ridge_coef_title: str = "Ridge: feature coefficients vs λ"
    ridge_biplot_title: str = "L2 regularization"

    rmse_y: str = "RMSE"
    gamma_x: str = "Regularization strength λ"
    coef_y: str = "Coefficient"

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


def _build_readable_feature_names(
    city_categories_sorted: List[str],
    mode: str,
) -> Dict[str, str]:
    if mode == "rus":
        mapping: Dict[str, str] = {
            "rooms_scaled": "количество комнат",
            "area_scaled": "площадь квартиры",
            "metro_distance_scaled": "расстояние до метро",
            "ac_yes": "есть ли кондиционер",
        }
        city_prefix = "город "
    elif mode == "eng":
        mapping = {
            "rooms_scaled": "number of rooms",
            "area_scaled": "apartment area",
            "metro_distance_scaled": "distance to metro",
            "ac_yes": "air conditioning",
        }
        city_prefix = "city "
    else:
        raise NotImplementedError(f"Language {mode} is not supported")

    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for idx, cat in enumerate(city_categories_sorted):
        letter = letters[idx] if idx < len(letters) else str(idx + 1)
        mapping[f"city__{cat}"] = f"{city_prefix}{letter}"

    return mapping


def load_scaled_train_test_multifeature(
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    dataset = get_extended_dataset()

    features = dataset[FEATURE_COLUMNS]
    target = dataset["price"].to_numpy()

    x_train_raw, y_train_raw, _, _ = take_sample_manual(
        np.array(features),
        np.array(target),
        apply_distortion=True,
    )
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = split_train_test_manual(
        x_train_raw,
        y_train_raw,
        random_state=52,
    )

    x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    x_test_df = pd.DataFrame(x_test_raw, columns=FEATURE_COLUMNS)

    y_train = np.ravel(np.array(y_train_raw, dtype=float))
    y_test = np.ravel(np.array(y_test_raw, dtype=float))

    # 1) Scale numeric columns, fit on train and apply to both
    numeric_scaler = StandardScaler()
    x_train_num = x_train_df[NUMERIC_COLUMNS].astype(float)
    x_test_num = x_test_df[NUMERIC_COLUMNS].astype(float)
    x_train_num_scaled = numeric_scaler.fit_transform(x_train_num)
    x_test_num_scaled = numeric_scaler.transform(x_test_num)

    # 2) One hot encode city using a stable full order
    full_city_series = dataset[CITY_COLUMN].astype(str)
    city_categories_sorted = sorted(full_city_series.unique().tolist())
    city_name_map = _build_readable_feature_names(city_categories_sorted, mode=mode)

    train_city = x_train_df[CITY_COLUMN].astype(str)
    test_city = x_test_df[CITY_COLUMN].astype(str)

    train_dummies = pd.get_dummies(train_city, dtype=float)
    test_dummies = pd.get_dummies(test_city, dtype=float)

    train_dummies = train_dummies.reindex(columns=city_categories_sorted, fill_value=0.0)
    test_dummies = test_dummies.reindex(columns=city_categories_sorted, fill_value=0.0)

    train_dummies.columns = [f"city__{c}" for c in train_dummies.columns]
    test_dummies.columns = [f"city__{c}" for c in test_dummies.columns]

    # 3) Convert AC to a binary column
    ac_train = _encode_ac_binary(x_train_df[AC_COLUMN]).astype(float).rename("ac_yes")
    ac_test = _encode_ac_binary(x_test_df[AC_COLUMN]).astype(float).rename("ac_yes")

    x_train_processed = np.hstack([
        x_train_num_scaled,
        train_dummies.to_numpy(dtype=float),
        ac_train.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    x_test_processed = np.hstack([
        x_test_num_scaled,
        test_dummies.to_numpy(dtype=float),
        ac_test.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    feature_keys = (
        [f"{c}_scaled" for c in NUMERIC_COLUMNS]
        + list(train_dummies.columns)
        + ["ac_yes"]
    )
    feature_names_human = [city_name_map.get(name, name) for name in feature_keys]

    # Scale target, fit on train and apply to test
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(float)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(float)

    return x_train_processed, x_test_processed, y_train_scaled, y_test_scaled, feature_keys, feature_names_human


def build_grouped_palette(feature_keys: List[str]) -> List[tuple]:
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")
    greens = plt.get_cmap("Greens")
    purples = plt.get_cmap("Purples")

    city_idx = [i for i, name in enumerate(feature_keys) if str(name).startswith("city__")]
    ac_idx = [i for i, name in enumerate(feature_keys) if str(name) == "ac_yes"]
    numeric_names = {"rooms_scaled", "area_scaled", "metro_distance_scaled"}
    numeric_idx = [i for i, name in enumerate(feature_keys) if str(name) in numeric_names]

    colors: List[Optional[tuple]] = [None] * len(feature_keys)

    def assign(idx_list: List[int], cmap, lo: float = 0.45, hi: float = 0.85):
        if len(idx_list) == 0:
            return
        vals = np.linspace(lo, hi, len(idx_list))
        for j, idx in enumerate(idx_list):
            colors[idx] = cmap(float(vals[j]))

    assign(numeric_idx, greens, lo=0.45, hi=0.85)
    assign(city_idx, blues, lo=0.45, hi=0.85)
    assign(ac_idx, reds, lo=0.55, hi=0.75)

    fallback_idx = [i for i, c in enumerate(colors) if c is None]
    assign(fallback_idx, purples, lo=0.45, hi=0.85)

    return [c for c in colors if c is not None]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


def collect_regularization_path(
    model_kind: str,
    regularization_values: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lasso_max_iter: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    n_steps = int(len(regularization_values))
    n_features = int(x_train.shape[1])

    rmse_train = np.zeros(n_steps, dtype=float)
    rmse_test = np.zeros(n_steps, dtype=float)
    coef_matrix = np.zeros((n_steps, n_features), dtype=float)
    pred_train_list: List[np.ndarray] = []
    pred_test_list: List[np.ndarray] = []

    for idx, alpha_value in enumerate(regularization_values):
        alpha_value_f = float(alpha_value)

        if model_kind == "lasso":
            model = Lasso(
                alpha=alpha_value_f,
                fit_intercept=True,
                max_iter=int(lasso_max_iter),
                tol=1e-4,
                selection="cyclic",
            )
        elif model_kind == "ridge":
            model = Ridge(
                alpha=alpha_value_f,
                fit_intercept=True,
            )
        else:
            raise ValueError("model_kind must be 'lasso' or 'ridge'.")

        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        rmse_train[idx] = rmse(y_train, y_pred_train)
        rmse_test[idx] = rmse(y_test, y_pred_test)

        coef_matrix[idx, :] = np.ravel(np.array(model.coef_, dtype=float))

        pred_train_list.append(np.ravel(y_pred_train).astype(float))
        pred_test_list.append(np.ravel(y_pred_test).astype(float))

    return rmse_train, rmse_test, coef_matrix, pred_train_list, pred_test_list


def compute_global_biplot_limits(
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_groups: List[List[np.ndarray]],
    pad_ratio: float = 0.08,
) -> Tuple[float, float]:
    values: List[np.ndarray] = [np.ravel(y_train).astype(float), np.ravel(y_test).astype(float)]
    for group in pred_groups:
        for arr in group:
            values.append(np.ravel(arr).astype(float))

    all_vals = np.concatenate(values, axis=0)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    span = float(max(vmax - vmin, 1e-12))
    pad = float(pad_ratio) * span
    return vmin - pad, vmax + pad


def plot_rmse_text_cell(
    ax,
    annotations,
    rmse_train_value: float,
    rmse_test_value: float,
):
    ax.set_axis_off()

    ax.text(
        0.5,
        0.58,
        f"RMSE {annotations.legend_train} = {rmse_train_value:.2f}",
        ha="center",
        va="center",
        fontsize=20,
        fontname=FONTNAME,
        color="red",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        f"RMSE {annotations.legend_test} = {rmse_test_value:.2f}",
        ha="center",
        va="center",
        fontsize=20,
        fontname=FONTNAME,
        color="black",
        transform=ax.transAxes,
    )


def plot_lr_coef_barplot(
    ax,
    annotations,
    lr_coefs: np.ndarray,
    feature_names_human: List[str],
    feature_colors: List[tuple],
):
    coefs = np.ravel(np.asarray(lr_coefs, dtype=float))
    n_features = int(len(coefs))
    x = np.arange(n_features, dtype=int)

    ax.grid(color="grey", alpha=0.25, zorder=1)
    ax.bar(x, coefs, color=feature_colors, edgecolor="black", linewidth=0.35, zorder=3)
    ax.axhline(y=0.0, color="black", linewidth=1.0, alpha=0.45, zorder=2)

    ax.set_ylim(float(COEFF_MIN), float(COEFF_MAX))
    ax.set_xlim(-0.6, float(n_features) - 0.4)

    ax.set_ylabel(annotations.coef_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names_human, rotation=50, fontsize=5, fontname=FONTNAME)

    ax.set_title(
        annotations.lr_coef_title,
        fontsize=12,
        fontdict={"fontname": FONTNAME},
        y=1.05,
    )


def plot_biplot(
    ax,
    annotations,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    title: str,
    lims: Tuple[float, float],
):
    ax.grid(color="grey", alpha=0.3, zorder=1)

    ax.scatter(
        y_train,
        y_pred_train,
        s=45,
        c="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_train,
    )
    ax.scatter(
        y_test,
        y_pred_test,
        s=45,
        c="black",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_test,
    )

    x0, x1 = float(lims[0]), float(lims[1])
    ax.plot([x0, x1], [x0, x1], color="black", linewidth=1.6, zorder=2)

    ax.set_xlim(x0, x1)
    ax.set_ylim(x0, x1)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(annotations.biplot_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.biplot_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)

    ax.legend(loc="upper left", frameon=True, fontsize=9)


def plot_rmse_curves(
    ax,
    annotations,
    regularization_values: np.ndarray,
    rmse_train_values: np.ndarray,
    rmse_test_values: np.ndarray,
    current_alpha: float,
    title: str,
):
    ax.grid(color="grey", alpha=0.3, zorder=1)

    ax.plot(
        regularization_values,
        rmse_train_values,
        "-",
        color="red",
        linewidth=2.0,
        zorder=3,
        label=annotations.legend_train,
    )
    ax.plot(
        regularization_values,
        rmse_test_values,
        "-",
        color="black",
        linewidth=2.0,
        alpha=0.9,
        zorder=3,
        label=annotations.legend_test,
    )

    ax.axvline(x=float(current_alpha), color="black", linewidth=1.6, alpha=0.9, zorder=4)
    ax.set_ylim(float(RMSE_MIN), float(RMSE_MAX))

    idx = int(np.argmin(np.abs(np.asarray(regularization_values, dtype=float) - float(current_alpha))))

    x_val = float(np.asarray(regularization_values, dtype=float)[idx])
    y_train_cur = float(np.asarray(rmse_train_values, dtype=float)[idx])
    y_test_cur = float(np.asarray(rmse_test_values, dtype=float)[idx])

    ax.scatter([x_val], [y_train_cur], s=80, c="red", edgecolor="black", linewidth=0.8, zorder=6)
    ax.scatter([x_val], [y_test_cur], s=80, c="black", edgecolor="black", linewidth=0.8, zorder=6)

    x_vals = np.asarray(regularization_values, dtype=float)
    x_span = float(np.max(x_vals) - np.min(x_vals)) if len(x_vals) > 1 else 1.0
    x_offset = 0.02 * x_span

    x_text = x_val + x_offset
    ha = "left"
    if x_val > float(np.max(x_vals)) - 0.08 * x_span:
        x_text = x_val - x_offset
        ha = "right"

    ax.text(
        x_text,
        y_train_cur - 0.2,
        f"{y_train_cur:.2f}",
        color="red",
        fontsize=11,
        fontname=FONTNAME,
        ha=ha,
        va="bottom",
        zorder=7,
    )
    ax.text(
        x_text,
        y_test_cur + 0.15,
        f"{y_test_cur:.2f}",
        color="black",
        fontsize=11,
        fontname=FONTNAME,
        ha=ha,
        va="top",
        zorder=7,
    )

    ax.set_xlabel(annotations.gamma_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.rmse_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)
    ax.legend(loc="upper left", frameon=True, fontsize=9)


def plot_coef_paths(
    ax,
    annotations,
    regularization_values: np.ndarray,
    coef_matrix: np.ndarray,
    feature_names_human: List[str],
    colors: List[tuple],
    current_alpha: float,
    title: str,
    show_legend: bool = True,
):
    ax.grid(color="grey", alpha=0.3, zorder=1)

    n_features = int(coef_matrix.shape[1])
    for j in range(n_features):
        ax.plot(
            regularization_values,
            coef_matrix[:, j],
            linewidth=2.0,
            color=colors[j],
            alpha=0.95,
            zorder=3,
            label=str(feature_names_human[j]),
        )

    ax.axhline(y=0.0, color="black", linewidth=1.0, alpha=0.35, zorder=2)
    ax.axvline(x=float(current_alpha), color="black", linewidth=1.6, alpha=0.9, zorder=4)

    ax.set_ylim(float(COEFF_MIN), float(COEFF_MAX))

    ax.set_xlabel(annotations.gamma_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.coef_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)

    if show_legend:
        ax.legend(loc="lower right", frameon=True, fontsize=7, ncol=1)


def create_axes(fig):
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.32, hspace=0.42)

    ax_lr_rmse_text = fig.add_subplot(gs[0, 0])
    ax_lr_coef_bar = fig.add_subplot(gs[0, 1])
    ax_lr_biplot = fig.add_subplot(gs[0, 2])

    ax_lasso_rmse = fig.add_subplot(gs[1, 0])
    ax_lasso_coef = fig.add_subplot(gs[1, 1])
    ax_lasso_biplot = fig.add_subplot(gs[1, 2])

    ax_ridge_rmse = fig.add_subplot(gs[2, 0])
    ax_ridge_coef = fig.add_subplot(gs[2, 1])
    ax_ridge_biplot = fig.add_subplot(gs[2, 2])

    return (
        ax_lr_rmse_text,
        ax_lr_coef_bar,
        ax_lr_biplot,
        ax_lasso_rmse,
        ax_lasso_coef,
        ax_lasso_biplot,
        ax_ridge_rmse,
        ax_ridge_coef,
        ax_ridge_biplot,
    )


def generate_frame(
    frame_index: int,
    mode: str,
    regularization_values: np.ndarray,
    biplot_lims: Tuple[float, float],
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_lr_train: np.ndarray,
    pred_lr_test: np.ndarray,
    rmse_lr_train: float,
    rmse_lr_test: float,
    lr_coefs: np.ndarray,
    lasso_rmse_train: np.ndarray,
    lasso_rmse_test: np.ndarray,
    lasso_coef_matrix: np.ndarray,
    lasso_pred_train_list: List[np.ndarray],
    lasso_pred_test_list: List[np.ndarray],
    ridge_rmse_train: np.ndarray,
    ridge_rmse_test: np.ndarray,
    ridge_coef_matrix: np.ndarray,
    ridge_pred_train_list: List[np.ndarray],
    ridge_pred_test_list: List[np.ndarray],
    feature_names_human: List[str],
    feature_colors: List[tuple],
) -> plt.Figure:
    annotations = annotations_by_language(mode)
    current_alpha = float(regularization_values[frame_index])

    fig = plt.figure(figsize=(16, 12))
    (
        ax_lr_rmse_text,
        ax_lr_coef_bar,
        ax_lr_biplot,
        ax_lasso_rmse,
        ax_lasso_coef,
        ax_lasso_biplot,
        ax_ridge_rmse,
        ax_ridge_coef,
        ax_ridge_biplot,
    ) = create_axes(fig)

    plot_rmse_text_cell(
        ax=ax_lr_rmse_text,
        annotations=annotations,
        rmse_train_value=float(rmse_lr_train),
        rmse_test_value=float(rmse_lr_test),
    )

    plot_lr_coef_barplot(
        ax=ax_lr_coef_bar,
        annotations=annotations,
        lr_coefs=lr_coefs,
        feature_names_human=feature_names_human,
        feature_colors=feature_colors,
    )

    plot_biplot(
        ax=ax_lr_biplot,
        annotations=annotations,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=pred_lr_train,
        y_pred_test=pred_lr_test,
        title=annotations.lr_title,
        lims=biplot_lims,
    )

    plot_rmse_curves(
        ax=ax_lasso_rmse,
        annotations=annotations,
        regularization_values=regularization_values,
        rmse_train_values=lasso_rmse_train,
        rmse_test_values=lasso_rmse_test,
        current_alpha=current_alpha,
        title=annotations.lasso_rmse_title,
    )

    plot_coef_paths(
        ax=ax_lasso_coef,
        annotations=annotations,
        regularization_values=regularization_values,
        coef_matrix=lasso_coef_matrix,
        feature_names_human=feature_names_human,
        colors=feature_colors,
        current_alpha=current_alpha,
        title=annotations.lasso_coef_title,
        show_legend=True,
    )

    plot_biplot(
        ax=ax_lasso_biplot,
        annotations=annotations,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=lasso_pred_train_list[frame_index],
        y_pred_test=lasso_pred_test_list[frame_index],
        title=annotations.lasso_biplot_title,
        lims=biplot_lims,
    )

    plot_rmse_curves(
        ax=ax_ridge_rmse,
        annotations=annotations,
        regularization_values=regularization_values,
        rmse_train_values=ridge_rmse_train,
        rmse_test_values=ridge_rmse_test,
        current_alpha=current_alpha,
        title=annotations.ridge_rmse_title,
    )

    plot_coef_paths(
        ax=ax_ridge_coef,
        annotations=annotations,
        regularization_values=regularization_values,
        coef_matrix=ridge_coef_matrix,
        feature_names_human=feature_names_human,
        colors=feature_colors,
        current_alpha=current_alpha,
        title=annotations.ridge_coef_title,
        show_legend=False,
    )

    plot_biplot(
        ax=ax_ridge_biplot,
        annotations=annotations,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=ridge_pred_train_list[frame_index],
        y_pred_test=ridge_pred_test_list[frame_index],
        title=annotations.ridge_biplot_title,
        lims=biplot_lims,
    )

    fig.suptitle(
        f"{annotations.suptitle}. λ = {current_alpha:.2f}",
        fontsize=18,
        fontdict={"fontname": FONTNAME},
        x=0.5,
        y=0.99,
    )

    return fig


def show_regularization_gamma_effect(
    mode: str = "rus",
    regularization_grid_size: int = REGULARIZATION_GRID_SIZE,
    min_regularization: float = MIN_REGULARIZATION,
    max_regularization: float = MAX_REGULARIZATION,
    noise: Optional[float] = None,
    pause_frames: int = 3,
    template_name: str = "template.svg",
):
    if noise is not None and float(noise) < 0.0:
        raise ValueError("noise must be non-negative or None.")

    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir = get_tmp_animation_directory()

    x_train, x_test, y_train, y_test, feature_keys, feature_names_human = load_scaled_train_test_multifeature(mode=mode)
    feature_colors = build_grouped_palette(feature_keys)

    # Optional noise on the training target, in scaled units
    if noise is not None and float(noise) > 0.0:
        rng = np.random.default_rng(42)
        y_train = (
            np.ravel(y_train).astype(float)
            + rng.normal(0.0, float(noise), size=len(y_train)).astype(float)
        )

    regularization_values = np.linspace(
        float(min_regularization),
        float(max_regularization),
        int(regularization_grid_size),
        dtype=float,
    )

    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(x_train, y_train)
    pred_lr_train = np.ravel(lr_model.predict(x_train)).astype(float)
    pred_lr_test = np.ravel(lr_model.predict(x_test)).astype(float)

    rmse_lr_train = rmse(y_train, pred_lr_train)
    rmse_lr_test = rmse(y_test, pred_lr_test)
    lr_coefs = np.ravel(np.array(lr_model.coef_, dtype=float))

    lasso_rmse_train, lasso_rmse_test, lasso_coef_matrix, lasso_pred_train_list, lasso_pred_test_list = (
        collect_regularization_path(
            model_kind="lasso",
            regularization_values=regularization_values,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            lasso_max_iter=9000,
        )
    )

    ridge_rmse_train, ridge_rmse_test, ridge_coef_matrix, ridge_pred_train_list, ridge_pred_test_list = (
        collect_regularization_path(
            model_kind="ridge",
            regularization_values=regularization_values,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
    )

    biplot_lims = compute_global_biplot_limits(
        y_train=y_train,
        y_test=y_test,
        pred_groups=[
            [pred_lr_train, pred_lr_test],
            lasso_pred_train_list,
            lasso_pred_test_list,
            ridge_pred_train_list,
            ridge_pred_test_list,
        ],
        pad_ratio=0.08,
    )

    image_files: List[Path] = []
    raw_svg_file = Path(tmp_dir, f"animation_26_regularization_gamma_{mode}.svg")

    for frame_index in range(len(regularization_values)):
        fig = generate_frame(
            frame_index=int(frame_index),
            mode=mode,
            regularization_values=regularization_values,
            biplot_lims=biplot_lims,
            y_train=y_train,
            y_test=y_test,
            pred_lr_train=pred_lr_train,
            pred_lr_test=pred_lr_test,
            rmse_lr_train=rmse_lr_train,
            rmse_lr_test=rmse_lr_test,
            lr_coefs=lr_coefs,
            lasso_rmse_train=lasso_rmse_train,
            lasso_rmse_test=lasso_rmse_test,
            lasso_coef_matrix=lasso_coef_matrix,
            lasso_pred_train_list=lasso_pred_train_list,
            lasso_pred_test_list=lasso_pred_test_list,
            ridge_rmse_train=ridge_rmse_train,
            ridge_rmse_test=ridge_rmse_test,
            ridge_coef_matrix=ridge_coef_matrix,
            ridge_pred_train_list=ridge_pred_train_list,
            ridge_pred_test_list=ridge_pred_test_list,
            feature_names_human=feature_names_human,
            feature_colors=feature_colors,
        )

        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close(fig)

        frame_png = Path(tmp_dir, f"animation_26_regularization_gamma_{mode}_{frame_index}.png")
        save_plot_according_to_template(
            raw_svg_file,
            frame_png,
            template_name=str(template_name),
            dpi=int(DPI),
        )
        image_files.append(frame_png)

    if len(image_files) > 0:
        for _ in range(int(pause_frames)):
            image_files.append(image_files[-1])

    grid_str = f"{int(regularization_grid_size)}"
    if noise is None:
        noise_str = "none"
    else:
        noise_str = f"{float(noise):.2f}".replace(".", "_")

    gif_path = Path(
        get_plots_path(),
        f"animation_26_regularization_gamma_effect_grid_{grid_str}_noise_{noise_str}_{mode}.gif",
    )

    with imageio.get_writer(gif_path, mode="I", duration=ANIMATION_DURATION, loop=0) as writer:
        for image_file in image_files:
            writer.append_data(imageio.imread(image_file))

    print(f"GIF saved at {gif_path}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    for mode in ["rus", "eng"]:
        show_regularization_gamma_effect(
            mode=mode,
            regularization_grid_size=REGULARIZATION_GRID_SIZE,
            min_regularization=MIN_REGULARIZATION,
            max_regularization=MAX_REGULARIZATION,
            noise=None,
            pause_frames=8,
            template_name="template.svg",
        )