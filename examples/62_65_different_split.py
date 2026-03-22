from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

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

COEFF_MIN: float = -2.5
COEFF_MAX: float = 2.5

RANDOM_STATES = [20, 40, 60]


@dataclass
class RusAnnotations:
    top_x: str = "Количество комнат"
    top_y: str = "Отклик (y)"
    bottom_y: str = "Коэффициент"
    legend_train: str = "обучение"
    legend_test: str = "тест"
    regression_coef_title: str = "Коэффициенты регрессии"
    ridge_coef_title_template: str = "Коэффициенты Ridge (alpha={:.1f})"
    split_template: str = "Разбиение {}"
    linear_regression_label: str = "Линейная регрессия"
    ridge_label_template: str = "Ridge (alpha={:.2f})"

    def get_suptitle(self, alpha):
        if alpha is None:
            return "Разные разбиения train/test приводят к очень разным коэффициентам модели"
        return "Регуляризация делает модель более стабильной"


@dataclass
class EngAnnotations:
    top_x: str = "Number of rooms"
    top_y: str = "Target (y)"
    bottom_y: str = "Coefficient"
    legend_train: str = "train"
    legend_test: str = "test"
    regression_coef_title: str = "Regression coefficients"
    ridge_coef_title_template: str = "Ridge coefficients (alpha={:.1f})"
    split_template: str = "Split {}"
    linear_regression_label: str = "Linear regression"
    ridge_label_template: str = "Ridge (alpha={:.2f})"

    def get_suptitle(self, alpha):
        if alpha is None:
            return "Different train/test splits lead to different regression coefficients"
        return "Regularization makes the model more robust"


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


def build_grouped_palette(feature_keys: List[str]) -> List[tuple]:
    """Group colors by feature blocks. Colors stay consistent across plots."""
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


def preprocess_train_test_for_lr(
    dataset_full: pd.DataFrame,
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build processed feature matrices (scaled numeric features, one-hot city, binary AC)
    and scale the target using the training split only.

    Returns:
      x_train_processed, x_test_processed, y_train_scaled, y_test_scaled,
      feature_keys, feature_names_human
    """
    x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    x_test_df = pd.DataFrame(x_test_raw, columns=FEATURE_COLUMNS)

    y_train = np.ravel(np.asarray(y_train_raw, dtype=float))
    y_test = np.ravel(np.asarray(y_test_raw, dtype=float))

    # 1) Scale numeric columns, fit on train and apply to both splits
    numeric_scaler = StandardScaler()
    x_train_num = x_train_df[NUMERIC_COLUMNS].astype(float)
    x_test_num = x_test_df[NUMERIC_COLUMNS].astype(float)
    x_train_num_scaled = numeric_scaler.fit_transform(x_train_num)
    x_test_num_scaled = numeric_scaler.transform(x_test_num)

    # 2) One hot encode city using a stable order from the full dataset
    full_city_series = dataset_full[CITY_COLUMN].astype(str)
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

    # 3) AC as a single binary column
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

    return (
        x_train_processed,
        x_test_processed,
        y_train_scaled,
        y_test_scaled,
        feature_keys,
        feature_names_human,
    )


def _get_bottom_title(annotations: Union[RusAnnotations, EngAnnotations], alpha: Optional[float]) -> str:
    if alpha is None:
        return annotations.regression_coef_title
    return annotations.ridge_coef_title_template.format(float(alpha))


def plot_feature_vs_target(
    ax,
    annotations,
    rooms_train: np.ndarray,
    y_train: np.ndarray,
    rooms_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
):
    ax.grid(color="grey", alpha=0.3, zorder=1)

    ax.scatter(
        rooms_train,
        y_train,
        s=55,
        c="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_train,
    )
    ax.scatter(
        rooms_test,
        y_test,
        s=55,
        c="black",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=annotations.legend_test,
    )

    ax.set_xlabel(annotations.top_x, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_ylabel(annotations.top_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)
    ax.legend(loc="upper left", frameon=True, fontsize=9)


def plot_lr_coef_barplot(
    ax,
    annotations,
    main_coefs: np.ndarray,
    feature_names_human: List[str],
    feature_colors: List[tuple],
    title: str,
    overlay_lr_coefs: Optional[np.ndarray] = None,
    overlay_label: Optional[str] = None,
    main_label: Optional[str] = None,
    show_legend: bool = True,
):
    """
    Bar plot of coefficients.

    If overlay_lr_coefs is provided, the overlay and main bars are drawn with a
    small horizontal offset so they do not visually merge into one stacked shape.
    """
    main_vals = np.ravel(np.asarray(main_coefs, dtype=float))
    n_features = int(len(main_vals))
    x = np.arange(n_features, dtype=float)

    ax.grid(color="grey", alpha=0.25, zorder=1)

    if overlay_lr_coefs is None:
        bar_width = 0.64
        ax.bar(
            x,
            main_vals,
            width=bar_width,
            color=feature_colors,
            edgecolor="black",
            linewidth=0.35,
            alpha=1.0,
            zorder=3,
        )
        show_legend = False
    else:
        overlay_vals = np.ravel(np.asarray(overlay_lr_coefs, dtype=float))

        # Small horizontal offset so bars are visually separated
        bar_width = 0.34
        offset = 0.18

        ax.bar(
            x - offset,
            overlay_vals,
            width=bar_width,
            color=feature_colors,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.3,
            zorder=2,
            label=overlay_label if overlay_label is not None else "baseline",
        )

        ax.bar(
            x + offset,
            main_vals,
            width=bar_width,
            color=feature_colors,
            edgecolor="black",
            linewidth=0.35,
            alpha=1.0,
            zorder=3,
            label=main_label if main_label is not None else "model",
        )

    ax.axhline(y=0.0, color="black", linewidth=1.0, alpha=0.45, zorder=4)

    ax.set_ylim(float(COEFF_MIN), float(COEFF_MAX))
    ax.set_xlim(-0.6, float(n_features) - 0.4)

    ax.set_ylabel(annotations.bottom_y, fontdict={"fontsize": 10, "fontname": FONTNAME})
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names_human, rotation=50, fontsize=7, fontname=FONTNAME)

    ax.set_title(title, fontsize=12, fontdict={"fontname": FONTNAME}, y=1.05)

    if show_legend and (overlay_lr_coefs is not None or main_label is not None):
        ax.legend(loc="upper right", frameon=True, fontsize=8)


def show_static_splits_2x3(
    mode: str = "rus",
    random_states: Optional[List[int]] = None,
    alpha: Optional[float] = None,
    template_name: str = "template.svg",
    prefix: str = "62",
):
    if random_states is None:
        random_states = list(RANDOM_STATES)

    if alpha is not None and float(alpha) < 0.0:
        raise ValueError("alpha must be non-negative or None.")

    annotations = annotations_by_language(mode)

    dataset = get_extended_dataset()
    features_full = dataset[FEATURE_COLUMNS]
    target_full = dataset["price"].to_numpy()

    # 1) Manual sampling with distortion, only a subset is used below
    x_sample_raw, y_sample_raw, _, _ = take_sample_manual(
        np.array(features_full),
        np.array(target_full),
        apply_distortion=True,
    )

    fig = plt.figure(figsize=(15, 8))
    gridspec = fig.add_gridspec(2, 3)
    gridspec.update(wspace=0.28, hspace=0.35)

    feature_colors: List[tuple] = []
    feature_names_human_cached: List[str] = []

    if alpha is None:
        model_label = "lr"
    else:
        model_label = f"ridge_alpha_{float(alpha):.3f}".replace(".", "_")

    bottom_title = _get_bottom_title(annotations, alpha=alpha)

    for col_index, random_state in enumerate(random_states):
        split_index = col_index + 1

        # 2) Split sampled data into train/test using a given random state
        x_train_raw, y_train_raw, x_test_raw, y_test_raw = split_train_test_manual(
            x_sample_raw,
            y_sample_raw,
            random_state=int(random_state),
        )

        # Top row uses raw rooms vs raw target
        x_train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
        x_test_df = pd.DataFrame(x_test_raw, columns=FEATURE_COLUMNS)
        rooms_train = np.ravel(np.asarray(x_train_df["rooms"], dtype=float))
        rooms_test = np.ravel(np.asarray(x_test_df["rooms"], dtype=float))
        y_train_plot = np.ravel(np.asarray(y_train_raw, dtype=float))
        y_test_plot = np.ravel(np.asarray(y_test_raw, dtype=float))

        # Bottom row uses processed features and scaled target
        (
            x_train_processed,
            x_test_processed,
            y_train_scaled,
            y_test_scaled,
            feature_keys,
            feature_names_human,
        ) = preprocess_train_test_for_lr(
            dataset_full=dataset,
            x_train_raw=x_train_raw,
            x_test_raw=x_test_raw,
            y_train_raw=y_train_raw,
            y_test_raw=y_test_raw,
            mode=mode,
        )

        if len(feature_names_human_cached) == 0:
            feature_names_human_cached = list(feature_names_human)
            feature_colors = build_grouped_palette(feature_keys)

        # Main model
        if alpha is None:
            main_model = LinearRegression(fit_intercept=True)
            main_label = annotations.linear_regression_label
        else:
            main_model = Ridge(alpha=float(alpha), fit_intercept=True)
            main_label = annotations.ridge_label_template.format(float(alpha))

        main_model.fit(x_train_processed, y_train_scaled)
        main_coefs = np.ravel(np.asarray(main_model.coef_, dtype=float))

        # Overlay baseline if Ridge is selected
        overlay_coefs: Optional[np.ndarray] = None
        overlay_label: Optional[str] = None
        if alpha is not None:
            base_lr = LinearRegression(fit_intercept=True)
            base_lr.fit(x_train_processed, y_train_scaled)
            overlay_coefs = np.ravel(np.asarray(base_lr.coef_, dtype=float))
            overlay_label = annotations.linear_regression_label

        # Top axis
        ax_top = fig.add_subplot(gridspec[0, col_index])
        plot_feature_vs_target(
            ax=ax_top,
            annotations=annotations,
            rooms_train=rooms_train,
            y_train=y_train_plot,
            rooms_test=rooms_test,
            y_test=y_test_plot,
            title=annotations.split_template.format(split_index),
        )

        # Bottom axis
        ax_bottom = fig.add_subplot(gridspec[1, col_index])

        show_legend = (col_index == 0)

        plot_lr_coef_barplot(
            ax=ax_bottom,
            annotations=annotations,
            main_coefs=main_coefs,
            feature_names_human=feature_names_human_cached,
            feature_colors=feature_colors,
            title=bottom_title,
            overlay_lr_coefs=overlay_coefs,
            overlay_label=overlay_label,
            main_label=main_label,
            show_legend=show_legend,
        )

    fig.suptitle(
        annotations.get_suptitle(alpha=alpha),
        fontsize=16,
        fontdict={"fontname": FONTNAME},
        x=0.5,
        y=0.98,
    )

    raw_svg = Path(get_plots_path(), f"{prefix}_static_splits_2x3_{model_label}_{mode}.svg")
    plt.savefig(raw_svg, bbox_inches="tight")
    plt.close(fig)

    out_png = Path(get_plots_path(), f"{prefix}_static_splits_2x3_{model_label}_{mode}.png")
    save_plot_according_to_template(
        raw_svg,
        out_png,
        template_name=str(template_name),
        dpi=int(DPI),
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    prefix = "62"
    for mode in ["rus", "eng"]:
        show_static_splits_2x3(
            mode=mode,
            random_states=[10, 52, 90],
            alpha=None,
            template_name="template.svg",
            prefix=prefix,
        )

    prefix = "65"
    for mode in ["rus", "eng"]:
        show_static_splits_2x3(
            mode=mode,
            random_states=[10, 52, 90],
            alpha=0.2,
            template_name="template.svg",
            prefix=prefix,
        )