from examples.paths import get_plots_path

import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(7)

n = 10
x = np.arange(1, n + 1, dtype=float)
x_centered = x - x.mean()

b0 = 105.0
b0_shift = 2.0
y_min, y_max = 100.0, 110.0

target_stds = [0.6, 1.4, 2.8]
margin = 0.98

MODEL_LINE_WIDTH = 1.4
BASELINE_LINE_WIDTH = 1.2


def make_base_residual(x_centered: np.ndarray, rng) -> np.ndarray:
    r = rng.normal(size=x_centered.size)
    r = r - r.mean()
    r = r - ((r @ x_centered) / (x_centered @ x_centered)) * x_centered
    s = r.std(ddof=0)
    if s > 0:
        r = r / s
    return r


def max_scale_for_bounds(b0: float, r_unit: np.ndarray, y_min: float, y_max: float) -> float:
    s_max = np.inf
    r_pos = r_unit[r_unit > 0]
    r_neg = r_unit[r_unit < 0]
    if r_pos.size:
        s_max = min(s_max, np.min((y_max - b0) / r_pos))
    if r_neg.size:
        s_max = min(s_max, np.min((b0 - y_min) / (-r_neg)))
    if not np.isfinite(s_max):
        return 0.0
    return max(0.0, s_max)


def draw_vertical_gaussians(ax, x_vals, mu, sigma, y_low, y_high, colors, width=0.38, alpha=0.95):
    yy = np.linspace(y_low, y_high, 250)
    pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((yy - mu) / sigma) ** 2)
    pdf = pdf / pdf.max()
    for xi, ci in zip(x_vals, colors):
        ax.plot(xi + width * pdf, yy, color=ci, alpha=alpha, lw=1.4)


def sigma_mle(y: np.ndarray, mu: float) -> float:
    r = y - mu
    return float(np.sqrt(np.mean(r * r)))


def normal_pdf(y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (y - mu) / sigma
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z * z)


def annotate_p_and_ll(ax, x_vals, y_vals, mu: float, sigma: float, colors, y_offset: float):
    p_vals = normal_pdf(y_vals, mu=mu, sigma=sigma)
    for xi, yi, pi, ci in zip(x_vals, y_vals, p_vals, colors):
        ax.text(
            xi,
            yi + y_offset,
            f"{pi:.1f}",
            color=ci,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    lnL = float(np.sum(np.log(np.clip(p_vals, 1e-300, None))))
    ax.text(
        1.02,
        0.5,
        f"ln L = {lnL:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        color="#666666",
        clip_on=False,
    )


if __name__ == "__main__":
    r_base_unit = make_base_residual(x_centered, rng)
    s_max = max_scale_for_bounds(b0, r_base_unit, y_min=y_min, y_max=y_max)

    cmap = plt.cm.coolwarm
    colors_in_column = cmap(np.linspace(0.0, 1.0, n))
    tick_labels = [rf"$x_{{{i}}}$" for i in range(1, n + 1)]

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.20, hspace=0.16)

    y_low, y_high = 99.0, 111.0
    p_label_offset = 0.35

    for col, target_std in enumerate(target_stds):
        s_eff = min(target_std, s_max * margin)

        r = s_eff * r_base_unit
        y = b0 + r

        b0_up = b0 + b0_shift
        b0_down = b0 - b0_shift

        sigma_b0 = sigma_mle(y, b0)
        sigma_up = sigma_mle(y, b0_up)
        sigma_down = sigma_mle(y, b0_down)

        print(
            f"col={col+1} "
            f"sigma(b0|y)={sigma_b0:.4f} "
            f"sigma(b0+shift|y)={sigma_up:.4f} "
            f"sigma(b0-shift|y)={sigma_down:.4f}"
        )

        sigma_vis = max(1e-6, sigma_b0)

        # Row 1
        ax = axes[0, col]
        ax.scatter(x, y, c="black", s=55, edgecolors="none")
        ax.plot([x.min(), x.max()], [b0, b0], color="black", lw=MODEL_LINE_WIDTH)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(y_low, y_high)
        ax.set_title(rf"$\hat{{y}} = {b0:.1f} + 0.0\,x$")
        ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("y")

        # Row 2
        ax = axes[1, col]
        ax.scatter(x, y, c=colors_in_column, s=55, edgecolors="none")
        ax.plot([x.min(), x.max()], [b0, b0], color="black", lw=MODEL_LINE_WIDTH)
        draw_vertical_gaussians(
            ax,
            x_vals=x,
            mu=b0,
            sigma=sigma_vis,
            y_low=y_low,
            y_high=y_high,
            colors=colors_in_column,
        )
        annotate_p_and_ll(ax, x, y, mu=b0, sigma=sigma_vis, colors=colors_in_column, y_offset=p_label_offset)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        if col == 0:
            ax.set_ylabel("y")
            ax.text(0.02, 0.95, "p(y|x)", transform=ax.transAxes, ha="left", va="top", fontsize=10)

        # Row 3
        ax = axes[2, col]
        ax.scatter(x, y, c=colors_in_column, s=55, edgecolors="none")
        ax.plot([x.min(), x.max()], [b0, b0], color="gray", alpha=0.4, lw=BASELINE_LINE_WIDTH, ls="--")
        ax.plot([x.min(), x.max()], [b0_up, b0_up], color="black", lw=MODEL_LINE_WIDTH)
        draw_vertical_gaussians(
            ax,
            x_vals=x,
            mu=b0_up,
            sigma=sigma_vis,
            y_low=y_low,
            y_high=y_high,
            colors=colors_in_column,
        )
        annotate_p_and_ll(ax, x, y, mu=b0_up, sigma=sigma_vis, colors=colors_in_column, y_offset=p_label_offset)
        ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("y")

        # Row 4
        ax = axes[3, col]
        ax.scatter(x, y, c=colors_in_column, s=55, edgecolors="none")
        ax.plot([x.min(), x.max()], [b0, b0], color="gray", alpha=0.2, lw=BASELINE_LINE_WIDTH, ls="--")
        ax.plot([x.min(), x.max()], [b0_down, b0_down], color="black", lw=MODEL_LINE_WIDTH)
        draw_vertical_gaussians(
            ax,
            x_vals=x,
            mu=b0_down,
            sigma=sigma_vis,
            y_low=y_low,
            y_high=y_high,
            colors=colors_in_column,
        )
        annotate_p_and_ll(ax, x, y, mu=b0_down, sigma=sigma_vis, colors=colors_in_column, y_offset=p_label_offset)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(get_plots_path() / "69_probability.svg", dpi=200, bbox_inches="tight")
    plt.close()
