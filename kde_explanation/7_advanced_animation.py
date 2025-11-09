from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

from kde_explanation.kde_utils import get_kde_simulation_path, get_kde_plots_path

DPI                 = 120
INTERVAL_MS         = 30
FRAME_SKIP          = 100
LINE_INTERP_STEPS   = 1

PARTICLES_RADIUS    = 0.02
GROUND_Y            = -5.0
CIRCLE_RADIUS       = 0.6

ENVELOPE_BINS           = 300
ENVELOPE_DELAY_FRAMES   = 1
ENVELOPE_LINEWIDTH      = 1.0
ENVELOPE_COLOR          = 'red'
ENVELOPE_ALPHA          = 0.7

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}


@dataclass
class S3Bucket:
    bucket: str
    prefix: str

    @staticmethod
    def get_client():
        """
        Creates a reusable boto3 S3 client
        """
        import boto3

        s3 = boto3.client("s3")
        return s3


def smooth_envelope(x, y,
                    method: str = "savgol",
                    savgol_window: int = 21,   # нечётное
                    savgol_poly: int = 3,
                    gaussian_sigma: float = 2.0,
                    spline_s: float | None = None):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = ~np.isnan(y)
    if not np.any(valid):
        return y.copy()
    y_filled = y.copy()
    if np.any(~valid):
        y_filled[~valid] = np.interp(x[~valid], x[valid], y[valid])

    if method == "savgol":
        n = len(y_filled)
        w = min(savgol_window, n if n % 2 == 1 else n - 1)
        w_min = savgol_poly + 2 if (savgol_poly + 2) % 2 == 1 else savgol_poly + 3
        w = max(w, w_min)
        w = max(5, w)
        if w >= 3 and w <= n and w % 2 == 1:
            y_sm = savgol_filter(y_filled, window_length=w, polyorder=savgol_poly, mode="interp")
        else:
            y_sm = y_filled
    elif method == "gaussian":
        y_sm = gaussian_filter1d(y_filled, sigma=gaussian_sigma, mode="nearest")
    elif method == "spline":
        try:
            sp = UnivariateSpline(x, y_filled, s=spline_s)
            y_sm = sp(x)
        except Exception:
            y_sm = y_filled
    else:
        y_sm = y_filled

    y_sm = np.maximum(y_sm, y)
    return y_sm


def list_frames_sorted(parquet_dir):
    files = list(parquet_dir.glob("*.parquet"))
    def key_func(p: Path):
        try:
            return 0, int(p.stem)
        except ValueError:
            return 1, p.name
    return sorted(files, key=key_func)


def frame_reader_parquet(parquet_dir: Path, frame_skip: int = 1):
    """
    Yield (frame_idx, sand_xy (N x 2, float32), line_x (float), labels (N, int32))
    from per-frame Parquet files written by save_frame_parquet().
    """
    files = list_frames_sorted(parquet_dir)
    if frame_skip > 1:
        files = files[::frame_skip]

    # Only load the columns we actually need
    cols = ["time_index", "coordinate_x", "coordinate_y", "type", "initial_circle"]

    for fpath in files:
        df = pd.read_parquet(fpath, columns=cols, engine="pyarrow")
        if df.empty:
            continue

        # Frame index is constant per file
        frame_idx = int(df["time_index"].iloc[0])

        # Filter sand rows
        sand_df = df[df["type"] == "sand"]
        if not sand_df.empty:
            # Ensure compact dtypes
            sand_xy = sand_df[["coordinate_x", "coordinate_y"]].to_numpy(dtype=np.float32, copy=False)
            # initial_circle is written as int32; keep it, fallback to -1 if anything goes wrong
            try:
                labels = sand_df["initial_circle"].to_numpy(dtype=np.int32, copy=False)
            except Exception:
                labels = np.full(sand_xy.shape[0], -1, dtype=np.int32)
        else:
            # No sand this frame
            sand_xy = np.empty((0, 2), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)

        # Line: two rows with the same x; take the first if present
        line_df = df[df["type"] == "line"]
        line_x = float(line_df["coordinate_x"].iloc[0]) if not line_df.empty else -5.0

        yield frame_idx, sand_xy, line_x, labels


def list_s3_frames_sorted(s3, bucket: str, prefix: str) -> list[str]:
    """
    List S3 object keys under `prefix` that end with .parquet, sorted like list_frames_sorted()
    """
    prefix = prefix.lstrip("/")  # S3 keys are not absolute
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".parquet"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    def key_func(k: str):
        name = k.rsplit("/", 1)[-1]  # basename, e.g. 000123.parquet
        stem = name[:-8] if name.endswith(".parquet") else name
        try:
            return 0, int(stem)
        except ValueError:
            return 1, name
    print(f"Number of simulated frames: {len(keys)}")
    return sorted(keys, key=key_func)


def frame_reader_parquet_s3(
    s3,
    bucket: str,
    prefix: str,
    frame_skip: int = 1):
    """
    Yield (frame_idx, sand_xy (N x 2, float32), line_x (float), labels (N, int32))
    from per-frame Parquet files stored in S3 under (bucket, prefix),
    written by save_frame_parquet().
    """
    from io import BytesIO

    keys = list_s3_frames_sorted(s3, bucket, prefix)
    if frame_skip > 1:
        keys = keys[::frame_skip]

    cols = ["time_index", "coordinate_x", "coordinate_y", "type", "initial_circle"]

    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        df = pd.read_parquet(BytesIO(data), columns=cols, engine="pyarrow")
        if df.empty:
            continue

        frame_idx = int(df["time_index"].iloc[0])

        sand_df = df[df["type"] == "sand"]
        if not sand_df.empty:
            sand_xy = sand_df[["coordinate_x", "coordinate_y"]].to_numpy(dtype=np.float32, copy=False)
            try:
                labels = sand_df["initial_circle"].to_numpy(dtype=np.int32, copy=False)
            except Exception:
                labels = np.full(sand_xy.shape[0], -1, dtype=np.int32)
        else:
            sand_xy = np.empty((0, 2), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)

        line_df = df[df["type"] == "line"]
        line_x = float(line_df["coordinate_x"].iloc[0]) if not line_df.empty else -5.0

        yield frame_idx, sand_xy, line_x, labels


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
    min_x, max_x = x.min(), x.max()
    x = (x - min_x) / (max_x - min_x) * 62
    return y, x  # sx, sy


def kde_values_on_grid(samples, grid):
    kde = stats.gaussian_kde(np.asarray(samples))
    return kde(grid)


def compute_upper_envelope_connected(
    sand_xy: np.ndarray,
    prev_sand_xy: np.ndarray | None = None,
    *,
    x_min: float = 0.0,
    x_max: float = 62.0,
    n_bins: int = ENVELOPE_BINS,
    r: float = PARTICLES_RADIUS,
    ground_y: float = GROUND_Y,
    match_radius: float = 0.08,
    vel_eps: float = 1e-3,
    contact_tol: float = 1e-3
):
    centers = np.linspace(x_min, x_max, n_bins, endpoint=False) + (x_max - x_min) / (2 * n_bins)

    if sand_xy.size == 0:
        return centers, np.full(n_bins, ground_y, dtype=float)

    xs = sand_xy[:, 0]
    ys = sand_xy[:, 1]

    in_domain = (xs >= x_min) & (xs <= x_max)
    xs = xs[in_domain]; ys = ys[in_domain]
    if xs.size == 0:
        return centers, np.full(n_bins, ground_y, dtype=float)

    if prev_sand_xy is None or prev_sand_xy.size == 0:
        static_mask = np.ones(xs.shape[0], dtype=bool)
    else:
        tree_prev = cKDTree(prev_sand_xy)
        dist, idx = tree_prev.query(np.c_[xs, ys], distance_upper_bound=match_radius)
        has_match = np.isfinite(dist)
        static_mask = np.zeros(xs.shape[0], dtype=bool)
        if np.any(has_match):
            y_prev = prev_sand_xy[idx[has_match], 1]
            dy = ys[has_match] - y_prev
            static_mask[has_match] = np.abs(dy) < vel_eps

    if not np.any(static_mask):
        return centers, np.full(n_bins, ground_y, dtype=float)

    xs_stat = xs[static_mask]
    ys_stat = ys[static_mask]

    eps_ground = 1e-9
    seeds_mask = ys_stat <= (ground_y + r + eps_ground)
    if not np.any(seeds_mask):
        return centers, np.full(n_bins, ground_y, dtype=float)

    link_radius = 2.0 * r + contact_tol
    tree_stat = cKDTree(np.c_[xs_stat, ys_stat])
    seed_inds = np.where(seeds_mask)[0]

    connected = np.zeros(xs_stat.shape[0], dtype=bool)
    stack = list(seed_inds)
    connected[seed_inds] = True
    while stack:
        i = stack.pop()
        nbrs = tree_stat.query_ball_point([xs_stat[i], ys_stat[i]], r=link_radius)
        for j in nbrs:
            if not connected[j]:
                connected[j] = True
                stack.append(j)

    xs_conn = xs_stat[connected]
    ys_conn = ys_stat[connected]
    if xs_conn.size == 0:
        return centers, np.full(n_bins, ground_y, dtype=float)

    edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.digitize(xs_conn, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    y_max = np.full(n_bins, ground_y, dtype=float)
    for b in range(n_bins):
        sel = (bin_idx == b)
        if np.any(sel):
            y_max[b] = np.max(ys_conn[sel])

    y = smooth_envelope(centers, y_max,
                        method="gaussian",
                        savgol_window=21,
                        savgol_poly=3,
                        gaussian_sigma=10.0,
                        spline_s=None)
    return centers, y


def annotations_by_language(mode: str):
    if mode == "eng":
        title = "Kernel density estimation"
    elif mode == "rus":
        title = "Ядерная оценка плотности"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return title


def compose_animation(output_gif: Path,
                      s3_bucket_data: Union[S3Bucket, None] = None,
                      folder_with_simulation: Union[Path, None] = None,
                      mode: str = "eng"):
    """
    Please run this script only after 6_advanced_simulation.py (that script produces simulation files)
    Create animation based on simulation files
    """
    if s3_bucket_data is None:
        print("Animaition will be created from local simulation files")
        if folder_with_simulation is None:
            folder_with_simulation = get_kde_simulation_path()
    else:
        print("Animaition will be created from S3 bucket simulation files")
        s3_client = s3_bucket_data.get_client()

    # --- static data (KDE and circles)
    sx, sy = generate_synthetic_data()
    n_circles = len(sx)
    x_grid = np.linspace(sx.min(), sx.max(), 1000)
    kde_vals = kde_values_on_grid(sx, x_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # Left plot (circles)
    ax1.set(xlim=[-6, 65], ylim=[-6, 65])
    ax1.set_aspect('equal')
    ax1.xaxis.set_ticklabels([]); ax1.xaxis.set_ticks([])
    ax1.plot([0, 62], [GROUND_Y, GROUND_Y], c='black', linewidth=1.2)
    ax1.yaxis.set_ticklabels([])
    ax1.yaxis.set_ticks([])

    # Source circles
    circles = [patches.Circle((sx[i], sy[i]), CIRCLE_RADIUS) for i in range(n_circles)]
    pc = PatchCollection(circles, facecolor='grey', edgecolor='white', linewidths=0.6, alpha=1.0)
    ax1.add_collection(pc)

    grey_rgba = mcolors.to_rgba('grey', alpha=1.0)
    white_edge_rgba = (1.0, 1.0, 1.0, grey_rgba[3])
    base_edge_colors = np.tile(white_edge_rgba, (n_circles, 1))

    base_face_colors = np.tile(grey_rgba, (n_circles, 1))
    white_rgba = (1.0, 1.0, 1.0, base_face_colors[0, 3])

    # ranks of circles by X (from left to right): correspond to the order of intersection by the vertical line
    order = np.argsort(sx)
    circle_rank = np.empty(n_circles, dtype=int)
    circle_rank[order] = np.arange(n_circles)

    cmap_circles = matplotlib.colormaps['coolwarm']
    norm_circles = mcolors.Normalize(vmin=0, vmax=max(1, n_circles - 1))

    # sand grains
    ec = EllipseCollection(
        [], [], [],
        units='xy',
        offsets=np.empty((0, 2)),
        transOffset=ax1.transData,
        facecolors=np.empty((0, 4)),
        edgecolors=np.empty((0, 4)),
        linewidths=0.3
    )
    ax1.add_collection(ec)

    line1 = ax1.axvline(-5.0, color='red', linewidth=0.2, alpha=1)

    # Right plot (KDE)
    ax2.set_xlim([-6, 65]); ax2.set_ylim([0.0, 1.0])
    ax2.xaxis.set_ticklabels([]); ax2.xaxis.set_ticks([])
    ax2.plot(x_grid, kde_vals, color='grey', linewidth=2)
    ax2.fill_between(x_grid, 0, kde_vals, color='grey', alpha=0.15)
    line2 = ax2.axvline(-5.0, color='red', linewidth=0.2, alpha=1)

    ax2_env = ax2.twinx()
    ax2_env.set_ylim([-5, 45])
    ax2_env.set_yticks([])
    ax2_env.set_ylabel("")
    ax2_env.patch.set_alpha(0)
    ax2_env.set_zorder(ax2.get_zorder() + 1)
    title = annotations_by_language(mode)
    ax2.text(
        0.5, 0.98, title,
        transform=ax2.transAxes,
        ha="center", va="top",
        fontdict={'fontname': FONTNAME}
    )
    env_line_left,  = ax1.plot([], [], color=ENVELOPE_COLOR,
                               linewidth=ENVELOPE_LINEWIDTH, alpha=ENVELOPE_ALPHA)
    env_line_right, = ax2_env.plot([], [], color=ENVELOPE_COLOR,
                                   linewidth=ENVELOPE_LINEWIDTH, alpha=ENVELOPE_ALPHA)

    if s3_bucket_data is None:
        frames_iter = frame_reader_parquet(folder_with_simulation, FRAME_SKIP)
    else:
        frames_iter = frame_reader_parquet_s3(s3_client, s3_bucket_data.bucket,
                                              s3_bucket_data.prefix, FRAME_SKIP)

    frames_list = list(frames_iter)
    n_frames = len(frames_list)
    if n_frames == 0:
        raise RuntimeError(f"Frames have not beedn found")

    # color normalization by initial_circle (for sand)
    all_labels = np.concatenate([fr[3] for fr in frames_list if fr[3] is not None and len(fr[3]) > 0])
    valid_labels = all_labels[all_labels >= 0]
    if valid_labels.size > 0:
        vmin = int(valid_labels.min())
        vmax = int(valid_labels.max())
        if vmin == vmax:
            vmax = vmin + 1
    else:
        vmin, vmax = 0, 1

    cmap_sand = matplotlib.colormaps['coolwarm']
    norm_sand = mcolors.Normalize(vmin=vmin, vmax=vmax)
    neutral_rgba = mcolors.to_rgba('#BBBBBB')

    def recolor_circles_by_line(xline):
        """Color the source circles: uncrossed ones — gray, crossed ones — coolwarm according to their rank"""
        edge_cols = base_edge_colors.copy()
        mask_crossed = sx < xline
        if np.any(mask_crossed):
            idx = np.where(mask_crossed)[0]
            ranks = circle_rank[idx]
            edge_cols[idx] = cmap_circles(norm_circles(ranks))

        pc.set_edgecolor([tuple(c) for c in edge_cols])

        face_cols = base_face_colors.copy()
        if np.any(mask_crossed):
            face_cols[mask_crossed] = white_rgba
        pc.set_facecolor([tuple(c) for c in face_cols])

        pc.set_alpha(None)

    def init():
        pc.set_facecolor([tuple(c) for c in base_face_colors])
        pc.set_edgecolor([tuple(c) for c in base_edge_colors])
        pc.set_alpha(None)

        line1.set_xdata([-5.0, -5.0])
        line2.set_xdata([-5.0, -5.0])
        pc.set_edgecolor([tuple(c) for c in base_edge_colors])
        pc.set_alpha(None)

        ec.set_offsets(np.empty((0, 2)))
        ec.set(facecolors=np.empty((0, 4)), edgecolors=np.empty((0, 4)))
        env_line_left.set_data([], [])
        env_line_right.set_data([], [])
        return line1, line2, pc, ec, env_line_left, env_line_right

    def animate(i):
        fi, sand_xy, line_x, labels = frames_list[i]

        # grains of sand + colors (edge = face)
        if sand_xy.shape[0] > 0:
            widths  = np.full(sand_xy.shape[0], 2 * PARTICLES_RADIUS)
            heights = np.full(sand_xy.shape[0], 2 * PARTICLES_RADIUS)
            ec.set(widths=widths, heights=heights, angles=np.zeros(sand_xy.shape[0]))
            ec.set_offsets(sand_xy)

            cols = np.zeros((sand_xy.shape[0], 4), dtype=float)
            if labels is None or len(labels) != sand_xy.shape[0]:
                cols[:] = neutral_rgba
            else:
                labels = labels.astype(int, copy=False)
                pos_mask = labels >= 0
                if np.any(pos_mask):
                    cols[pos_mask] = cmap_sand(norm_sand(labels[pos_mask]))
                if np.any(~pos_mask):
                    cols[~pos_mask] = neutral_rgba

            ec.set(facecolors=cols, edgecolors=cols, linewidths=0.3)
        else:
            ec.set_offsets(np.empty((0, 2)))
            ec.set(facecolors=np.empty((0, 4)), edgecolors=np.empty((0, 4)))

        # verical lines. Both plots; left and right
        line1.set_xdata([line_x, line_x])
        line2.set_xdata([line_x, line_x])

        # repainting coolwarm circles by rank
        recolor_circles_by_line(line_x)

        prev_sand = frames_list[i - 1][1] if i > 0 else None
        env_x, env_y = compute_upper_envelope_connected(
            sand_xy,
            prev_sand_xy=prev_sand,
            x_min=0.0, x_max=62.0,
            n_bins=ENVELOPE_BINS,
            r=PARTICLES_RADIUS,
            ground_y=GROUND_Y
        )

        delayed_line_x = frames_list[i - ENVELOPE_DELAY_FRAMES][2] if i >= ENVELOPE_DELAY_FRAMES else -5.0
        reveal_mask = env_x <= delayed_line_x

        env_line_left.set_data(env_x[reveal_mask], env_y[reveal_mask])
        env_line_right.set_data(env_x[reveal_mask], env_y[reveal_mask])

        return line1, line2, pc, ec, env_line_left, env_line_right

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=n_frames,
        interval=INTERVAL_MS,
        blit=True
    )

    ani.save(output_gif, writer="pillow", dpi=DPI)
    print(f"Saved gif animation in file: {output_gif}")


if __name__ == "__main__":
    s3_bucket = S3Bucket(bucket="linear-regression-kernel-density", prefix="simulation")

    output_gif = Path(get_kde_plots_path(), 'final_simulation.gif')
    compose_animation(output_gif=output_gif, s3_bucket_data=s3_bucket, mode="rus")
