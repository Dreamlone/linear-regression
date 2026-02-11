import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from examples.paths import get_plots_path, get_tmp_animation_directory
from examples.utils import save_plot_according_to_template


FONTNAME = "Comic Sans MS"
EQUATION_FONTNAME = "DejaVu Sans Mono"
DPI = 200

POLY_DEGREE: int = 10

ALPHA_MIN: float = 0.02
ALPHA_MAX: float = 1.12
FRAMES: int = 35

ANIM_DURATION = 500
PAUSE_FRAMES: int = 20

# Synthetic data controls
NOISE_STD: float = 0.15
N_SAMPLES: int = 30
RANDOM_SEED: int = 1961

# Axes limits for synthetic data in [-1, 1]
MIN_Y = -1.1
MAX_Y = 1.1


@dataclass
class RusAnnotations:
    suptitle: str = "L2 регуляризация модели с полиномиальными признаками (степень полинома 10)"
    legend_train: str = "обучающая выборка"
    legend_test: str = "тестовая выборка"
    legend_model: str = "модель"


@dataclass
class EngAnnotations:
    suptitle: str = "L2 regularization with polynomial features (degree 10)"
    legend_train: str = "train set"
    legend_test: str = "test set"
    legend_model: str = "model"


def annotations_by_language(mode: str):
    if mode == "rus":
        return RusAnnotations()
    if mode == "eng":
        return EngAnnotations()
    raise NotImplementedError(f"Language {mode} is not supported")


def make_synthetic_linear_data(
    n_samples: int,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a clean (noise-free) synthetic dataset:
      x in [-1, 1]
      y = 0.75 * x
    Noise will be injected ONLY into the training targets after train/test split.
    """
    rng = np.random.default_rng(int(random_seed))

    x = rng.uniform(-1.0, 1.0, size=int(n_samples)).astype(float)
    y_clean = 0.75 * x

    y_clean = np.clip(y_clean, -1.0, 1.0)

    # Sort by x for nicer visualization
    order = np.argsort(x)
    x = x[order]
    y_clean = y_clean[order]

    return x, y_clean


def _build_alpha_values(alpha_min: float, alpha_max: float, frames: int) -> np.ndarray:
    return np.linspace(float(alpha_min), float(alpha_max), int(frames), dtype=float)


def _fmt1(v: float, width: int = 2, signed: bool = False) -> str:
    vv = float(np.round(float(v), 1))
    if abs(vv) < 1e-12:
        vv = 0.0
    if signed:
        return f"{vv:+{width}.1f}"
    return f"{vv:{width}.1f}"


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def _to_sup(n: int) -> str:
    return str(n).translate(_SUP)


def build_equation_text(intercept: float, poly_coefs: np.ndarray) -> str:
    """
    Single-line equation with fixed-length terms, rounded to 1 decimal.
    Keeps ALL terms (including 0.0) so string length stays stable.
    """
    b0 = float(intercept)
    coefs = np.ravel(np.asarray(poly_coefs, dtype=float))

    parts: List[str] = [f"ŷ = {_fmt1(b0, width=5, signed=False)}"]

    for k in range(1, len(coefs) + 1):
        c = float(coefs[k - 1])

        x_term = "x" if k == 1 else f"x{_to_sup(k)}"  # x, x², x³, ...
        parts.append(f"{_fmt1(c, width=+1, signed=True)}·{x_term}")

    return "  ".join(parts)


def fit_predict_curve_and_equation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    alpha_value: float,
    poly: PolynomialFeatures,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, str]:
    """
    Fit model for a given lambda and return:
      - predictions on x_grid
      - equation text in terms of x, x^2, ..., in the ORIGINAL (unscaled) polynomial space

    Model is trained on standardized polynomial features. The equation is unscaled back to
    the polynomial feature space:
      s_j = (z_j - mu_j) / sigma_j
      y = b0 + sum w_j * s_j
        = (b0 - sum w_j*mu_j/sigma_j) + sum (w_j/sigma_j) * z_j
    """
    x_train_2d = np.asarray(x_train, dtype=float).reshape(-1, 1)
    x_grid_2d = np.asarray(x_grid, dtype=float).reshape(-1, 1)

    # Polynomial features: [x, x^2, ..., x^d] because include_bias=False
    z_train = poly.fit_transform(x_train_2d)
    z_grid = poly.transform(x_grid_2d)

    # Standardize polynomial features (fit on train only)
    s_train = scaler.fit_transform(z_train)
    s_grid = scaler.transform(z_grid)

    lam = float(alpha_value)
    if abs(lam) < 1e-12:
        model = LinearRegression(fit_intercept=True)
    else:
        model = Ridge(alpha=lam, fit_intercept=True)

    model.fit(s_train, np.asarray(y_train, dtype=float).ravel())

    # Predictions for the curve
    y_grid_pred = np.ravel(model.predict(s_grid)).astype(float)

    # Convert coefficients back to original polynomial feature space (z)
    w = np.ravel(np.asarray(model.coef_, dtype=float))
    b0 = float(model.intercept_)

    mu = np.ravel(np.asarray(scaler.mean_, dtype=float))
    sigma = np.ravel(np.asarray(scaler.scale_, dtype=float))
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # safety

    coefs_unscaled = w / sigma
    intercept_unscaled = b0 - float(np.sum(w * mu / sigma))

    eq_text = build_equation_text(intercept_unscaled, coefs_unscaled)
    return y_grid_pred, eq_text


def _generate_frame(
    ax: plt.Axes,
    annotations,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_grid: np.ndarray,
    y_grid_pred: np.ndarray,
    lambda_value: float,
    xlim: Tuple[float, float],
):
    ax.clear()
    ax.grid(alpha=0.3, color="grey")

    ax.scatter(
        x_train,
        y_train,
        s=40,
        c="red",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label=annotations.legend_train,
    )

    ax.scatter(
        x_test,
        y_test,
        s=40,
        c="black",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
        zorder=3,
        label=annotations.legend_test,
    )

    ax.plot(
        x_grid,
        y_grid_pred,
        color="red",
        linewidth=2.2,
        zorder=4,
        label=annotations.legend_model,
    )

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(MIN_Y), float(MAX_Y) + 0.3)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax.set_xlabel("X", fontdict={"fontsize": 12, "fontname": FONTNAME})
    ax.set_ylabel("Y", fontdict={"fontsize": 12, "fontname": FONTNAME})

    ax.set_title(
        f"λ = {float(lambda_value):.2f}",
        fontdict={"fontsize": 14, "fontname": FONTNAME},
        color="red",
        pad=10,
    )

    ax.legend(loc="upper left", frameon=True, fontsize=10)


def animate_poly_regularization(
    mode: str = "rus",
    template_name: str = "template.svg",
):
    annotations = annotations_by_language(mode)

    # Synthetic data (clean)
    x, y_clean = make_synthetic_linear_data(
        n_samples=int(N_SAMPLES),
        random_seed=int(RANDOM_SEED),
    )
    x = np.ravel(np.asarray(x, dtype=float))
    y_clean = np.ravel(np.asarray(y_clean, dtype=float))

    x_train, x_test, y_train_clean, y_test = train_test_split(
        x,
        y_clean,
        test_size=0.30,
        random_state=int(RANDOM_SEED),
        shuffle=True,
    )

    # Inject noise ONLY into train targets
    rng = np.random.default_rng(int(RANDOM_SEED))
    y_train = (
        np.ravel(np.asarray(y_train_clean, dtype=float))
        + rng.normal(0.0, float(NOISE_STD), size=len(y_train_clean)).astype(float)
    )
    y_train = np.clip(y_train, -1.0, 1.0)
    y_test = np.ravel(np.asarray(y_test, dtype=float))  # test remains clean

    lambda_values = _build_alpha_values(ALPHA_MIN, ALPHA_MAX, FRAMES)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 400, dtype=float)

    poly = PolynomialFeatures(degree=int(POLY_DEGREE), include_bias=False)
    scaler = StandardScaler()

    # Precompute curves + equations (one per frame)
    curves: List[np.ndarray] = []
    equations: List[str] = []
    for lam in lambda_values:
        y_grid_pred, eq_text = fit_predict_curve_and_equation(
            x_train=x_train,
            y_train=y_train,  # noisy train only
            x_grid=x_grid,
            alpha_value=float(lam),
            poly=poly,
            scaler=scaler,
        )
        curves.append(y_grid_pred)
        equations.append(eq_text)

    xlim = (-1.1, 1.1)

    # Temp directory
    tmp_dir = get_tmp_animation_directory()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_svg_file = Path(tmp_dir, f"66_poly_regularization_{mode}.svg")
    frame_files: List[Path] = []

    # Render frames
    for frame_index, lam in enumerate(lambda_values):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Keep text within the figure bounds to avoid clipping with bbox_inches="tight"
        fig.suptitle(
            annotations.suptitle,
            fontsize=16,
            fontdict={"fontname": FONTNAME},
            y=1.05,
        )

        # Equation below suptitle (black)
        fig.text(
            0.5,
            0.975,
            equations[frame_index],
            ha="center",
            va="top",
            fontsize=10,
            fontname=EQUATION_FONTNAME,
            color="black",
        )

        _generate_frame(
            ax=ax,
            annotations=annotations,
            x_train=np.ravel(x_train).astype(float),
            y_train=np.ravel(y_train).astype(float),
            x_test=np.ravel(x_test).astype(float),
            y_test=np.ravel(y_test).astype(float),
            x_grid=x_grid,
            y_grid_pred=curves[frame_index],
            lambda_value=float(lam),
            xlim=xlim,
        )

        plt.savefig(raw_svg_file, bbox_inches="tight")
        plt.close(fig)

        frame_png = Path(tmp_dir, f"66_poly_regularization_{mode}_{frame_index:03d}.png")
        save_plot_according_to_template(
            raw_svg_file,
            frame_png,
            template_name=str(template_name),
            dpi=int(DPI),
        )
        frame_files.append(frame_png)

    # Pause on last frame
    if len(frame_files) > 0:
        for _ in range(int(PAUSE_FRAMES)):
            frame_files.append(frame_files[-1])

    # Save GIF
    gif_path = Path(get_plots_path(), f"66_poly_regularization_{mode}.gif")

    duration_sec = float(ANIM_DURATION) / 1000.0

    with imageio.get_writer(gif_path, mode="I", duration=duration_sec, loop=0) as writer:
        for frame_png in frame_files:
            writer.append_data(imageio.imread(frame_png))

    print(f"GIF saved at {gif_path}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    animate_poly_regularization(
        mode="rus",
        template_name="template.svg",
    )
