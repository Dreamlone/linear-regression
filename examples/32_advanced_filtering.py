from pathlib import Path

from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from matplotlib.cm import ScalarMappable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from examples.paths import get_plots_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
MIN_Y = 0
MAX_Y = 62
MIN_X = 14
MAX_X = 30
N_BINS = 12


def filter_mahalanobis(ax,
                       x: np.array,
                       y: np.array,
                       alpha: float = 0.1,
                       border_points: int = 360):
    """
    Compute Mahalanobis-based outlier mask for 2D points (x, y) and
    return ellipse border for the chosen chi-square threshold
    """
    # ---- Basic checks ----
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    # Build data and keep finite rows for fitting
    data_matrix = np.column_stack((x, y))
    finite_mask = np.isfinite(data_matrix).all(axis=1)
    if not finite_mask.any():
        raise ValueError("No finite points to compute covariance/mean.")
    valid = data_matrix[finite_mask]

    # Mean and covariance (empirical)
    mean_vec = valid.mean(axis=0)
    centered_valid = valid - mean_vec
    covariance = np.cov(centered_valid, rowvar=False)

    # Whitening transform Σ^{-1/2} via eigen-decomposition
    # covariance = Q * diag(eigvals) * Q^T
    eps = 1e-12
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)  # guard tiny negatives
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    sigma_inv_sqrt = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T  # Σ^{-1/2}

    # Transform all points to whitened coordinates: z = Σ^{-1/2}(x - μ)
    centered_all = data_matrix - mean_vec
    z_all = centered_all @ sigma_inv_sqrt.T  # (n,2)
    # Non-finite original rows -> keep as NaN in z for plotting
    z_all[~finite_mask] = np.nan

    z_valid = z_all[finite_mask]
    d2_valid = np.einsum('ij,ij->i', z_valid, z_valid)

    # Chi-square threshold
    threshold_sq = chi2.ppf(1 - alpha, df=2)
    keep_valid = d2_valid <= threshold_sq

    keep_full = np.zeros(x.shape[0], dtype=bool)
    keep_full[finite_mask] = keep_valid

    x_in, y_in = x[keep_full], y[keep_full]
    x_out, y_out = x[~keep_full], y[~keep_full]

    angles = np.linspace(0, 2 * np.pi, border_points, endpoint=True)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = (eigvecs @ (np.sqrt(eigvals)[:, None] * unit_circle))
    ellipse = mean_vec[:, None] + np.sqrt(threshold_sq) * ellipse
    x_borders = ellipse[0, :]
    y_borders = ellipse[1, :]

    z_in = z_all[keep_full]
    z_out = z_all[~keep_full]

    ax.scatter(z_in[:, 0], z_in[:, 1], c='#5b94e5', s=25, alpha=0.85)
    if z_out.size > 0:
        ax.scatter(z_out[:, 0], z_out[:, 1], s=25, facecolors='none',
                   edgecolors='r', linewidths=1.5, label="outliers")

    radius = float(np.sqrt(threshold_sq))
    circle_x = radius * np.cos(angles)
    circle_y = radius * np.sin(angles)
    ax.plot(circle_x, circle_y, linestyle='--', c='red', linewidth=2, label=f"threshold d={radius:.2f}")
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.4)
    ax.set_xlabel("z1", fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_ylabel("z2", fontdict={'fontsize': 8, 'fontname': FONTNAME})
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-5, 5)
    ax.legend(loc='lower left', prop={'family': FONTNAME, 'size': 8})

    return x_borders, y_borders, x_in, y_in, x_out, y_out


def plot_lof_boundary(ax,
                      x: np.ndarray,
                      y: np.ndarray,
                      lof_label: str,
                      n_neighbors: int = 20,
                      contamination: float = 0.07,
                      grid_size: int = 300):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    X = np.column_stack((x, y))
    finite_mask = np.isfinite(X).all(axis=1)
    Xf = X[finite_mask]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xf)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True
    )
    lof.fit(Xs)

    Xs_full = np.empty_like(X)
    Xs_full[:] = np.nan
    Xs_full[finite_mask] = scaler.transform(Xf)
    decision_function = np.full(X.shape[0], np.nan)
    decision_function[finite_mask] = lof.decision_function(Xs_full[finite_mask])

    keep_full = decision_function >= 0
    keep_full[~finite_mask] = False

    x_in,  y_in  = x[keep_full],  y[keep_full]
    x_out, y_out = x[~keep_full], y[~keep_full]

    pad_x = 0.05 * (MAX_X - MIN_X + 1e-12)
    pad_y = 0.05 * (MAX_Y - MIN_Y + 1e-12)
    gx = np.linspace(MIN_X - pad_x, MAX_X + pad_x, grid_size)
    gy = np.linspace(MIN_Y - pad_y, MAX_Y + pad_y, grid_size)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.column_stack((XX.ravel(), YY.ravel()))
    grid_s = scaler.transform(grid)

    Z = lof.decision_function(grid_s).reshape(XX.shape)

    plt.cla()
    vmin = np.nanmin([np.nanmin(Z), np.nanmin(decision_function[finite_mask])])
    vmax = np.nanmax([np.nanmax(Z), np.nanmax(decision_function[finite_mask])])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cs = ax.contour(XX, YY, Z, linewidths=1, cmap='rainbow', norm=norm, alpha=0.5)

    x_borders, y_borders = np.array([]), np.array([])
    levels = np.asarray(cs.levels)
    idx = int(np.nanargmin(np.abs(levels - 0.0)))
    segs = cs.allsegs[idx]
    if segs:
        verts = max(segs, key=lambda a: a.shape[0])
        x_borders, y_borders = verts[:, 0], verts[:, 1]

    ax.scatter(x[finite_mask], y[finite_mask], s=15, c=decision_function[finite_mask], cmap='rainbow',
               norm=norm, alpha=1.0)

    mappable = ScalarMappable(norm=norm, cmap='rainbow')
    mappable.set_array([])
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(lof_label, fontsize=8, fontname=FONTNAME)

    ticks = np.linspace(norm.vmin, norm.vmax, 7)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=8)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontname(FONTNAME)

    if x_borders.size > 0:
        ax.plot(x_borders, y_borders, c='red', linewidth=1)
    return x_borders, y_borders, x_in, y_in, x_out, y_out


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Data filtering methods (multivariate)"
        mahalanobis_plot = "Mahalanobis distance"
        lof_plot = "Local outlier factor"
        elliptic_envelope_plot = "Elliptic Envelope"
        main_plot = "Data with outliers"
        lof_label = "Local outlier factor (LOF)"
        y_label = "Detected outliers"
    elif mode == "rus":
        title = "Способы фильтрации данных (многомерные методы)"
        mahalanobis_plot = "Махаланобисово расстояние"
        lof_plot = "Локальный фактор выброса"
        elliptic_envelope_plot = "Elliptic Envelope"
        main_plot = "Данные с выбросами"
        lof_label = "Оценка локальной плотности (LOF)"
        y_label = "Обнаруженные выбросы"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title, mahalanobis_plot, lof_plot, elliptic_envelope_plot, main_plot, lof_label, y_label


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
    (title, mahalanobis_plot, lof_plot, elliptic_envelope_plot,
     main_plot, lof_label, y_label) = annotations_by_language(mode)
    x, y = generate_synthetic_data()

    fig_size = (10, 9)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(3, 2, figure=fig)
    gs.update(hspace=0.3)
    ax_main = fig.add_subplot(gs[0, 0:2])
    ax_main.set_title(main_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_main.scatter(x, y, s=60, c="#5b94e5", alpha=0.8)
    ax_main.set_ylim(MIN_Y, MAX_Y)
    ax_main.set_ylabel("y", fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_cleanup(ax_main)

    ##########################
    # 1 Mahalanobis distance #
    ##########################
    ax_mahalanobis = fig.add_subplot(gs[1, 0])
    ax_mahalanobis.set_title(mahalanobis_plot, fontsize=12, fontdict={'fontname': FONTNAME})

    ax_mahalanobis_data = fig.add_subplot(gs[2, 0])
    x_borders, y_borders, x_in, y_in, x_out, y_out = filter_mahalanobis(ax_mahalanobis, x, y, alpha=0.1)
    ax_mahalanobis_data.plot(x_borders, y_borders, '--', c='red')
    ax_mahalanobis_data.scatter(x_in, y_in, s=30, c="#5b94e5", alpha=0.8)
    ax_mahalanobis_data.scatter(x_out, y_out, s=40, c="red", alpha=0.8)
    ax_mahalanobis_data.set_ylim(MIN_Y, MAX_Y)
    ax_mahalanobis_data.set_xlim(MIN_X, MAX_X)
    ax_mahalanobis_data.set_ylabel(y_label, fontdict={'fontsize': 12, 'fontname': FONTNAME})
    ax_cleanup(ax_mahalanobis_data)

    ################################
    # 2 LOF — Local Outlier Factor #
    ################################
    ax_lof = fig.add_subplot(gs[1, 1])

    x_borders, y_borders, x_in, y_in, x_out, y_out = plot_lof_boundary(ax_lof, x, y, lof_label)
    ax_lof.set_title(lof_plot, fontsize=12, fontdict={'fontname': FONTNAME})
    ax_lof.set_ylim(MIN_Y, MAX_Y)
    ax_lof.set_xlim(MIN_X, MAX_X)
    ax_cleanup(ax_lof)
    ax_lof.spines[['right', 'top']].set_visible(False)

    ax_lof_data = fig.add_subplot(gs[2, 1])
    ax_lof_data.plot(x_borders, y_borders, '--', c='red')
    ax_lof_data.scatter(x_in, y_in, s=30, c="#5b94e5", alpha=0.8)
    ax_lof_data.scatter(x_out, y_out, s=40, c="red", alpha=0.8)
    ax_lof_data.set_ylim(MIN_Y, MAX_Y)
    ax_lof_data.set_xlim(MIN_X, MAX_X)
    ax_cleanup(ax_lof_data)

    raw_svg_file = Path(get_plots_path(), f"32_explain_advanced_filtering_{mode}.svg")
    fig.suptitle(title, fontsize=20, fontdict={'fontname': FONTNAME})
    plt.savefig(raw_svg_file)
    plt.close()
    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"32_explain_advanced_filtering_{mode}.png"))


if __name__ == '__main__':
    plot_filtering_method_concepts("rus")
    plot_filtering_method_concepts("eng")
