from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from examples.paths import get_plots_path
from examples.utils import (
    save_plot_according_to_template,
    split_train_test_manual,
    get_extended_dataset,
    take_sample_manual,
)


FONTNAME = "Comic Sans MS"
DPI = 200

FEATURE_COLUMNS = ["rooms", "area", "metro_distance", "city", "ac_in_apartment"]
NUMERIC_COLUMNS = ["rooms", "area", "metro_distance"]
CITY_COLUMN = "city"
AC_COLUMN = "ac_in_apartment"

RANDOM_STATE = 52

ALPHA_MIN = 0.0
ALPHA_MAX = 2.0
N_COMBINATIONS = 4  # fixed


@dataclass
class RusAnnotations:
    top_x: str = "Количество комнат"
    top_y: str = "Цена"
    rmse_y: str = "RMSE"
    legend_train: str = "обучение"
    legend_valid: str = "валидация"


@dataclass
class EngAnnotations:
    top_x: str = "Rooms"
    top_y: str = "Price"
    rmse_y: str = "RMSE"
    legend_train: str = "train"
    legend_valid: str = "validation"


def annotations_by_language(mode: str):
    if mode == "rus":
        return RusAnnotations()
    if mode == "eng":
        return EngAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


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


def preprocess_train_test_for_ridge(
    dataset_full: pd.DataFrame,
    x_train_raw: np.ndarray,
    x_valid_raw: np.ndarray,
    y_train_raw: np.ndarray,
    y_valid_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Builds processed feature matrices (scaled numeric + one-hot city + binary AC),
    and scales target (fit on train).
    Returns:
      x_train_processed, x_valid_processed, y_train_scaled, y_valid_scaled, target_scaler
    """
    x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    x_valid_df = pd.DataFrame(x_valid_raw, columns=FEATURE_COLUMNS)

    y_train = np.ravel(np.asarray(y_train_raw, dtype=float))
    y_valid = np.ravel(np.asarray(y_valid_raw, dtype=float))

    # 1) Scale numeric (fit on train, apply to valid)
    numeric_scaler = StandardScaler()
    x_train_num = x_train_df[NUMERIC_COLUMNS].astype(float)
    x_valid_num = x_valid_df[NUMERIC_COLUMNS].astype(float)
    x_train_num_scaled = numeric_scaler.fit_transform(x_train_num)
    x_valid_num_scaled = numeric_scaler.transform(x_valid_num)

    # 2) City one-hot with stable full order (from full dataset)
    full_city_series = dataset_full[CITY_COLUMN].astype(str)
    city_categories_sorted = sorted(full_city_series.unique().tolist())

    train_city = x_train_df[CITY_COLUMN].astype(str)
    valid_city = x_valid_df[CITY_COLUMN].astype(str)

    train_dummies = pd.get_dummies(train_city, dtype=float)
    valid_dummies = pd.get_dummies(valid_city, dtype=float)

    train_dummies = train_dummies.reindex(columns=city_categories_sorted, fill_value=0.0)
    valid_dummies = valid_dummies.reindex(columns=city_categories_sorted, fill_value=0.0)

    train_dummies.columns = [f"city__{c}" for c in train_dummies.columns]
    valid_dummies.columns = [f"city__{c}" for c in valid_dummies.columns]

    # 3) AC -> binary
    ac_train = _encode_ac_binary(x_train_df[AC_COLUMN]).astype(float).rename("ac_yes")
    ac_valid = _encode_ac_binary(x_valid_df[AC_COLUMN]).astype(float).rename("ac_yes")

    x_train_processed = np.hstack([
        x_train_num_scaled,
        train_dummies.to_numpy(dtype=float),
        ac_train.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    x_valid_processed = np.hstack([
        x_valid_num_scaled,
        valid_dummies.to_numpy(dtype=float),
        ac_valid.to_numpy(dtype=float).reshape(-1, 1),
    ]).astype(float)

    # Scale target (fit on train, apply to valid)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(float)
    y_valid_scaled = target_scaler.transform(y_valid.reshape(-1, 1)).ravel().astype(float)

    return x_train_processed, x_valid_processed, y_train_scaled, y_valid_scaled, target_scaler


def plot_rooms_vs_price_split(
    ax,
    annotations,
    rooms_train: np.ndarray,
    price_train: np.ndarray,
    rooms_valid: np.ndarray,
    price_valid: np.ndarray,
    title: str,
):
    ax.grid(color="grey", alpha=0.3, zorder=1)

    ax.scatter(
        rooms_train,
        price_train,
        s=60,
        c="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_train,
    )
    ax.scatter(
        rooms_valid,
        price_valid,
        s=60,
        c="black",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_valid,
    )

    ax.set_xlabel(annotations.top_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.top_y, fontdict={"fontsize": 10, "fontname": FONTNAME})

    ax.set_ylim(0.0, 75000.0)
    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.03)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)


def _short_xticks_for_group(mode: str):
    if mode == "rus":
        train_ticks = ["обуч 1", "обуч 2", "обуч 3", "обуч ср"]
        valid_ticks = ["вал 1", "вал 2", "вал 3", "вал ср"]
    else:
        train_ticks = ["tr 1", "tr 2", "tr 3", "tr mean"]
        valid_ticks = ["val 1", "val 2", "val 3", "val mean"]
    return train_ticks + valid_ticks


def plot_rmse_bars_for_lambda_simple(
    ax,
    annotations,
    lambda_value: float,
    rmse_train_value: float,
    rmse_valid_value: float,
    y_max: float,
):
    x = np.array([0, 1], dtype=int)
    heights = np.array([rmse_train_value, rmse_valid_value], dtype=float)

    bars = ax.bar(
        x,
        heights,
        color=["red", "black"],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        width=0.55,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([annotations.legend_train, annotations.legend_valid], fontname=FONTNAME, fontsize=9)
    ax.set_ylabel(annotations.rmse_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylim(0.0, float(y_max))
    ax.set_title(f"λ = {lambda_value:.2f}", fontsize=12, fontdict={"fontname": FONTNAME}, y=1.03)

    for b_idx, bar in enumerate(bars):
        val = float(heights[b_idx])
        color = "red" if b_idx == 0 else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontname=FONTNAME,
            color=color,
            zorder=5,
        )


def plot_rmse_bars_for_lambda_kfold(
    ax,
    annotations,
    mode: str,
    lambda_value: float,
    rmse_train_folds: List[float],
    rmse_valid_folds: List[float],
    y_max: float,
):
    rmse_train_arr = np.asarray(rmse_train_folds, dtype=float)
    rmse_valid_arr = np.asarray(rmse_valid_folds, dtype=float)

    rmse_train_mean = float(np.mean(rmse_train_arr))
    rmse_valid_mean = float(np.mean(rmse_valid_arr))

    heights = np.array(
        [
            float(rmse_train_arr[0]),
            float(rmse_train_arr[1]),
            float(rmse_train_arr[2]),
            rmse_train_mean,
            float(rmse_valid_arr[0]),
            float(rmse_valid_arr[1]),
            float(rmse_valid_arr[2]),
            rmse_valid_mean,
        ],
        dtype=float,
    )
    x = np.arange(len(heights), dtype=int)

    colors = ["red", "red", "red", "red", "black", "black", "black", "black"]

    ax.grid(color="grey", alpha=0.25, zorder=1)
    bars = ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        width=0.6,
    )

    ax.set_ylabel(annotations.rmse_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylim(0.0, float(y_max))
    ax.set_title(f"λ = {lambda_value:.2f}", fontsize=12, fontdict={"fontname": FONTNAME}, y=1.03)

    ax.set_xticks(x)
    ax.set_xticklabels(_short_xticks_for_group(mode), fontname=FONTNAME, fontsize=8, rotation=0)

    for idx, bar in enumerate(bars):
        val = float(heights[idx])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontname=FONTNAME,
            color=colors[idx],
            zorder=5,
        )


def make_ridge_alpha_gridsearch_figure(mode: str = "rus", k_folds: Optional[int] = None):
    if k_folds not in (None, 3):
        raise ValueError("k_folds must be either None or 3.")

    annotations = annotations_by_language(mode)

    dataset = get_extended_dataset()
    features_full = dataset[FEATURE_COLUMNS]
    target_full = dataset["price"].to_numpy()

    # Sample first (exactly your pipeline)
    x_sample_raw, y_sample_raw, _, _ = take_sample_manual(
        np.array(features_full),
        np.array(target_full),
        apply_distortion=True,
    )

    lambda_values = np.linspace(float(ALPHA_MIN), float(ALPHA_MAX), int(N_COMBINATIONS), dtype=float)

    # ---------
    # Case A: no CV (single split)
    # ---------
    if k_folds is None:
        x_train_raw, y_train_raw, x_valid_raw, y_valid_raw = split_train_test_manual(
            x_sample_raw,
            y_sample_raw,
            random_state=int(RANDOM_STATE),
        )

        # Top plot data (raw)
        x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
        x_valid_df = pd.DataFrame(x_valid_raw, columns=FEATURE_COLUMNS)

        rooms_train = np.ravel(np.asarray(x_train_df["rooms"], dtype=float))
        rooms_valid = np.ravel(np.asarray(x_valid_df["rooms"], dtype=float))
        price_train = np.ravel(np.asarray(y_train_raw, dtype=float))
        price_valid = np.ravel(np.asarray(y_valid_raw, dtype=float))

        # Preprocess (train -> valid)
        (
            x_train_processed,
            x_valid_processed,
            y_train_scaled,
            y_valid_scaled,
            target_scaler,
        ) = preprocess_train_test_for_ridge(
            dataset_full=dataset,
            x_train_raw=x_train_raw,
            x_valid_raw=x_valid_raw,
            y_train_raw=y_train_raw,
            y_valid_raw=y_valid_raw,
        )

        rmse_train_list: List[float] = []
        rmse_valid_list: List[float] = []

        y_train_price = np.ravel(np.asarray(y_train_raw, dtype=float))
        y_valid_price = np.ravel(np.asarray(y_valid_raw, dtype=float))

        for lambda_value in lambda_values:
            model = Ridge(alpha=float(lambda_value), fit_intercept=True)
            model.fit(x_train_processed, y_train_scaled)

            pred_train_scaled = np.ravel(model.predict(x_train_processed)).astype(float)
            pred_valid_scaled = np.ravel(model.predict(x_valid_processed)).astype(float)

            pred_train_price = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).ravel().astype(float)
            pred_valid_price = target_scaler.inverse_transform(pred_valid_scaled.reshape(-1, 1)).ravel().astype(float)

            rmse_train_list.append(rmse(y_train_price, pred_train_price))
            rmse_valid_list.append(rmse(y_valid_price, pred_valid_price))

        rmse_ylim = 1.15 * float(max(rmse_train_list + rmse_valid_list + [1e-9]))

        fig = plt.figure(figsize=(4.3 * N_COMBINATIONS, 8.0))
        outer = fig.add_gridspec(2, 1, hspace=0.45)

        top_gs = outer[0].subgridspec(1, 1)
        bottom_gs = outer[1].subgridspec(1, N_COMBINATIONS, wspace=0.35)

        ax_top = fig.add_subplot(top_gs[0, 0])
        title = (
            f"разбиение обучение/валидация (random_state={RANDOM_STATE})"
            if mode == "rus"
            else f"train/validation split (random_state={RANDOM_STATE})"
        )
        plot_rooms_vs_price_split(
            ax=ax_top,
            annotations=annotations,
            rooms_train=rooms_train,
            price_train=price_train,
            rooms_valid=rooms_valid,
            price_valid=price_valid,
            title=title,
        )

        for col_idx, lambda_value in enumerate(lambda_values):
            ax_bottom = fig.add_subplot(bottom_gs[0, col_idx])
            plot_rmse_bars_for_lambda_simple(
                ax=ax_bottom,
                annotations=annotations,
                lambda_value=float(lambda_value),
                rmse_train_value=float(rmse_train_list[col_idx]),
                rmse_valid_value=float(rmse_valid_list[col_idx]),
                y_max=float(rmse_ylim),
            )

        plots_path = Path(get_plots_path())
        raw_svg = plots_path / f"67_hyperparameters_optimization_{mode}.svg"
        out_png = plots_path / f"67_hyperparameters_optimization_{mode}.png"

        plt.savefig(raw_svg, bbox_inches="tight")
        plt.close(fig)

        save_plot_according_to_template(
            raw_svg,
            out_png,
            template_name="template.svg",
            dpi=int(DPI),
        )
        print(f"Saved: {out_png}")
        return

    # ---------
    # Case B: 3-fold CV
    # ---------
    kfold = KFold(n_splits=3, shuffle=True, random_state=int(RANDOM_STATE))
    fold_splits = list(kfold.split(x_sample_raw))
    if len(fold_splits) != 3:
        raise RuntimeError("Expected exactly 3 folds.")

    # Prepare top-row (3 axes): rooms vs price for each fold split (raw)
    top_fold_data = []
    for fold_index, (train_idx, valid_idx) in enumerate(fold_splits, start=1):
        x_train_raw = np.asarray(x_sample_raw)[train_idx]
        y_train_raw = np.asarray(y_sample_raw)[train_idx]
        x_valid_raw = np.asarray(x_sample_raw)[valid_idx]
        y_valid_raw = np.asarray(y_sample_raw)[valid_idx]

        x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
        x_valid_df = pd.DataFrame(x_valid_raw, columns=FEATURE_COLUMNS)

        rooms_train = np.ravel(np.asarray(x_train_df["rooms"], dtype=float))
        rooms_valid = np.ravel(np.asarray(x_valid_df["rooms"], dtype=float))
        price_train = np.ravel(np.asarray(y_train_raw, dtype=float))
        price_valid = np.ravel(np.asarray(y_valid_raw, dtype=float))

        top_fold_data.append((rooms_train, price_train, rooms_valid, price_valid, fold_index))

    # Metrics per lambda: store 3-fold arrays (in PRICE units)
    rmse_train_per_lambda: List[List[float]] = [[] for _ in range(len(lambda_values))]
    rmse_valid_per_lambda: List[List[float]] = [[] for _ in range(len(lambda_values))]

    for fold_index, (train_idx, valid_idx) in enumerate(fold_splits, start=1):
        x_train_raw = np.asarray(x_sample_raw)[train_idx]
        y_train_raw = np.asarray(y_sample_raw)[train_idx]
        x_valid_raw = np.asarray(x_sample_raw)[valid_idx]
        y_valid_raw = np.asarray(y_sample_raw)[valid_idx]

        (
            x_train_processed,
            x_valid_processed,
            y_train_scaled,
            y_valid_scaled,
            target_scaler,
        ) = preprocess_train_test_for_ridge(
            dataset_full=dataset,
            x_train_raw=x_train_raw,
            x_valid_raw=x_valid_raw,
            y_train_raw=y_train_raw,
            y_valid_raw=y_valid_raw,
        )

        y_train_price = np.ravel(np.asarray(y_train_raw, dtype=float))
        y_valid_price = np.ravel(np.asarray(y_valid_raw, dtype=float))

        for lambda_idx, lambda_value in enumerate(lambda_values):
            model = Ridge(alpha=float(lambda_value), fit_intercept=True)
            model.fit(x_train_processed, y_train_scaled)

            pred_train_scaled = np.ravel(model.predict(x_train_processed)).astype(float)
            pred_valid_scaled = np.ravel(model.predict(x_valid_processed)).astype(float)

            pred_train_price = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).ravel().astype(float)
            pred_valid_price = target_scaler.inverse_transform(pred_valid_scaled.reshape(-1, 1)).ravel().astype(float)

            rmse_train_per_lambda[lambda_idx].append(rmse(y_train_price, pred_train_price))
            rmse_valid_per_lambda[lambda_idx].append(rmse(y_valid_price, pred_valid_price))

    # Common y-limit for all barplots
    all_vals: List[float] = []
    for lambda_idx in range(len(lambda_values)):
        all_vals.extend(rmse_train_per_lambda[lambda_idx])
        all_vals.append(float(np.mean(rmse_train_per_lambda[lambda_idx])))
        all_vals.extend(rmse_valid_per_lambda[lambda_idx])
        all_vals.append(float(np.mean(rmse_valid_per_lambda[lambda_idx])))

    rmse_ylim = 1.15 * float(max(all_vals + [1e-9]))

    # Plot
    fig = plt.figure(figsize=(4.3 * N_COMBINATIONS, 8.0))
    outer = fig.add_gridspec(2, 1, hspace=0.45)

    top_gs = outer[0].subgridspec(1, 3, wspace=0.35)
    bottom_gs = outer[1].subgridspec(1, N_COMBINATIONS, wspace=0.35)

    for ax_index in range(3):
        rooms_train, price_train, rooms_valid, price_valid, fold_index = top_fold_data[ax_index]
        ax_top = fig.add_subplot(top_gs[0, ax_index])

        title = (
            f"fold {fold_index}: train/test"
            if mode == "rus"
            else f"fold {fold_index}: train/validation"
        )
        plot_rooms_vs_price_split(
            ax=ax_top,
            annotations=annotations,
            rooms_train=rooms_train,
            price_train=price_train,
            rooms_valid=rooms_valid,
            price_valid=price_valid,
            title=title,
        )

    for col_idx, lambda_value in enumerate(lambda_values):
        ax_bottom = fig.add_subplot(bottom_gs[0, col_idx])
        plot_rmse_bars_for_lambda_kfold(
            ax=ax_bottom,
            annotations=annotations,
            mode=mode,
            lambda_value=float(lambda_value),
            rmse_train_folds=rmse_train_per_lambda[col_idx],
            rmse_valid_folds=rmse_valid_per_lambda[col_idx],
            y_max=float(rmse_ylim),
        )

    plots_path = Path(get_plots_path())
    if k_folds is None:
        prefix = "67"
    else:
        prefix = "68"

    raw_svg = plots_path / f"{prefix}_hyperparameters_optimization_kfold3_{mode}.svg"
    out_png = plots_path / f"{prefix}_hyperparameters_optimization_kfold3_{mode}.png"

    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name="template.svg",
        dpi=int(DPI),
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    make_ridge_alpha_gridsearch_figure(mode="rus", k_folds=None)
    make_ridge_alpha_gridsearch_figure(mode="rus", k_folds=3)
