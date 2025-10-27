import time
from dataclasses import dataclass
from pathlib import Path
from math import floor
from typing import Union
from io import BytesIO

import numpy as np
import pandas as pd
from pandas import CategoricalDtype

from kde_explanation.kde_utils import get_kde_simulation_path

np.random.seed(1999)

# Scale and physics
PARTICLES_RADIUS        = 0.02
NUMBER_OF_PARTICLES     = 230       # particles per circle
NUMBER_OF_FRAMES        = 63000     # "raw" simulation steps
DT                      = 0.02
GROUND_Y                = -5.0
G                       = 9.81
CONTACT_TOL             = 1e-6
LINE_SPEED              = 1.0       # 1 — go through the entire range for the simulation; 0.5 — for half, etc.

# Geometry of circles and lines
CIRCLE_RADIUS           = 0.6
SIGMA_X                 = 0.25
LINE_X_START            = -5.0
LINE_X_END              = 64.0

# Frame (used for vertical lines — bottom/top point)
PLOT_Y_MIN              = -6.0
PLOT_Y_MAX              = 65.0

CONTACT_SOFTNESS        = 0.4    # contacts are “more solid” less interpenetration of sand grains
N_ITERS                 = 28     # the best resolution of contacts at every step
DAMPING                 = 0.16   # speed is damped more strongly (both along X and Y)
GROUND_INFLUENCE_HEIGHT = 1.3    # higher “earth influence” zone -> higher stratification
FREEZE_INFLUENCE        = 0.90
TYPE_DTYPE = CategoricalDtype(categories=["sand", "line"])


@dataclass
class SimulationState:
    x: np.array
    y: np.array
    vx: np.array
    vy: np.array
    frozen: np.array
    alive: np.array
    triggered: np.array
    initial_circle_idx: np.array
    circle_order_label: np.array
    trigger_order_counter: int

    def save(self, frame: int, where_to_save: Path):
        np.savez_compressed(
            where_to_save,
            x=self.x, y=self.y, vx=self.vx, vy=self.vy,
            frozen=self.frozen, alive=self.alive,
            triggered=self.triggered,
            initial_circle_idx=self.initial_circle_idx,
            circle_order_label=self.circle_order_label,
            trigger_order_counter=self.trigger_order_counter,
            last_frame=frame
        )

    def save_s3(self, frame: int, s3, bucket: str, key: str):
        """Save .npz directly to S3 using a boto3 S3 client"""
        buf = BytesIO()
        # Write NPZ into memory buffer
        np.savez_compressed(
            buf,
            x=self.x, y=self.y, vx=self.vx, vy=self.vy,
            frozen=self.frozen, alive=self.alive,
            triggered=self.triggered,
            initial_circle_idx=self.initial_circle_idx,
            circle_order_label=self.circle_order_label,
            trigger_order_counter=self.trigger_order_counter,
            last_frame=frame
        )
        # reset buffer cursor before upload
        buf.seek(0)

        s3.upload_fileobj(
            buf, bucket, key,
            ExtraArgs={
                "ContentType": "application/octet-stream",
                "ServerSideEncryption": "AES256",
            }
        )

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

    def object_exists(self, s3, key: str):
        from botocore.exceptions import ClientError

        try:
            resp = s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise


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
    return y, x  # (sx, sy)


def generate_points_in_circle(n_points, center_x, center_y, radius, sigma_x):
    """Generation of points in a circle, but with normal distribution along X"""
    points = []
    while len(points) < n_points:
        x = np.random.normal(loc=center_x, scale=sigma_x)
        if abs(x - center_x) > radius:
            continue
        y_range = np.sqrt(radius**2 - (x - center_x)**2)
        y = np.random.uniform(center_y - y_range, center_y + y_range)
        points.append((x, y))
    xs, ys = zip(*points)
    return np.array(xs), np.array(ys)


def build_grid_subset(x, y, indices, cell_size):
    """Build cells only for active indexes (we keep global indexes)"""
    cells = {}
    for i in indices:
        xi = x[i]; yi = y[i]
        ix = floor(xi / cell_size)
        iy = floor(yi / cell_size)
        cells.setdefault((ix, iy), []).append(i)
    return cells


def neighbor_pairs_subset(x, y, indices, cell_size):
    cells = build_grid_subset(x, y, indices, cell_size)
    checked = set()
    for (ix, iy), idxs in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                js = cells.get((ix+dx, iy+dy), [])
                for i in idxs:
                    for j in js:
                        if j <= i:
                            continue
                        key = (i, j)
                        if key not in checked:
                            checked.add(key)
                            yield i, j


def step_physics(x, y, vx, vy, frozen, alive):
    dt = DT
    g = G
    ground_y = GROUND_Y
    r = PARTICLES_RADIUS

    CONTACT_TOL_LOCAL = 1e-8
    SUBSTEPS          = 3
    sub_dt            = dt / SUBSTEPS

    def ground_influence(y_val):
        gap = np.maximum(0.0, y_val - (ground_y + r))
        t = np.clip(1.0 - gap / GROUND_INFLUENCE_HEIGHT, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    act = np.where(alive)[0]
    if act.size == 0:
        return

    cell_size = 2.1 * (2 * r)

    for _sub in range(SUBSTEPS):
        # integration
        moving = act[~frozen[act]]
        if moving.size > 0:
            vy[moving] += g * sub_dt

        x_pred = x.copy()
        y_pred = y.copy()
        if moving.size > 0:
            x_pred[moving] += vx[moving] * sub_dt
            y_pred[moving] -= vy[moving] * sub_dt

        # projections
        for __ in range(N_ITERS):
            # surface (aka floor)
            below = y_pred < ground_y + r
            if np.any(below & alive):
                idx_below = np.where(below & alive)[0]
                y_pred[idx_below] = ground_y + r

            # contacts
            for i, j in neighbor_pairs_subset(x_pred, y_pred, act, cell_size):
                dx = x_pred[j] - x_pred[i]
                dy = y_pred[j] - y_pred[i]
                min_d = 2 * r
                dist2 = dx*dx + dy*dy
                if dist2 < (min_d - CONTACT_TOL_LOCAL) ** 2:
                    if dist2 < 1e-18:
                        jitter = 1e-4 * r
                        dx = np.random.uniform(-jitter, jitter)
                        dy = np.random.uniform(-jitter, jitter)
                        dist2 = dx*dx + dy*dy
                    dist = np.sqrt(dist2)
                    nx, ny = dx / dist, dy / dist
                    overlap = (min_d - dist)
                    gi = max(ground_influence(y_pred[i]), ground_influence(y_pred[j]))
                    corr_scale = CONTACT_SOFTNESS + (1.0 - CONTACT_SOFTNESS) * gi
                    corr = corr_scale * overlap

                    w_i = 0.0 if frozen[i] else 1.0
                    w_j = 0.0 if frozen[j] else 1.0
                    w_sum = w_i + w_j
                    if w_sum == 0.0:
                        continue
                    ci = (w_i / w_sum) * corr
                    cj = (w_j / w_sum) * corr
                    x_pred[i] -= nx * ci; y_pred[i] -= ny * ci
                    x_pred[j] += nx * cj; y_pred[j] += ny * cj

            # floor one more time
            below = y_pred < ground_y + r
            if np.any(below & alive):
                idx_below = np.where(below & alive)[0]
                y_pred[idx_below] = ground_y + r

        # updating speeds
        if moving.size > 0:
            dvx = (x_pred[moving] - x[moving]) / sub_dt
            dvy = (y[moving] - y_pred[moving]) / sub_dt
            vx[moving] = dvx * (1.0 - DAMPING)
            vy[moving] = dvy * (1.0 - DAMPING)

        # ground frost
        on_ground = (y_pred <= ground_y + r + 1e-9)
        frozen |= (on_ground & alive)

        # “Layered” freezing only for active ones
        if np.any(alive & ~frozen):
            supports_left  = [[] for _ in range(len(x))]
            supports_right = [[] for _ in range(len(x))]

            for i, j in neighbor_pairs_subset(x_pred, y_pred, act, cell_size):
                for top, bot in ((i, j), (j, i)):
                    if frozen[bot] and alive[top] and not frozen[top]:
                        dx = x_pred[top] - x_pred[bot]
                        dy = y_pred[top] - y_pred[bot]
                        if dy > 0.0 and dx*dx + dy*dy <= (2*r + 1e-8)**2:
                            (supports_left[top] if dx < 0 else supports_right[top]).append(bot)

            newly_stable = np.zeros(len(x), dtype=bool)
            for i in act:
                if frozen[i] or on_ground[i]:
                    continue
                gi = ground_influence(y_pred[i])
                if gi < FREEZE_INFLUENCE:
                    continue
                if supports_left[i] and supports_right[i]:
                    jl = min(supports_left[i],  key=lambda j: abs(x_pred[i]-x_pred[j]))
                    jr = min(supports_right[i], key=lambda j: abs(x_pred[i]-x_pred[j]))
                    if abs(x_pred[jr] - x_pred[jl]) >= 0.6 * r:
                        newly_stable[i] = True

            frozen |= newly_stable

            # single support -> easy rolling
            has_left  = np.array([len(supports_left[i])  > 0 for i in range(len(x))])
            has_right = np.array([len(supports_right[i]) > 0 for i in range(len(x))])
            slide_idx = (alive & ~frozen) & (has_left ^ has_right)
            if np.any(slide_idx):
                gap = np.maximum(0.0, y_pred - (ground_y + r))
                t = np.clip(1.0 - gap / GROUND_INFLUENCE_HEIGHT, 0.0, 1.0)
                gi_vec = t * t * (3.0 - 2.0 * t)
                dir_sign = np.zeros(len(x))
                dir_sign[slide_idx & has_left]  = +1.0
                dir_sign[slide_idx & has_right] = -1.0
                vx[slide_idx] += dir_sign[slide_idx] * (0.25 * gi_vec[slide_idx]) * (dt / SUBSTEPS)

        x[:] = x_pred
        y[:] = y_pred
        vx[frozen] = 0.0
        vy[frozen] = 0.0

def fmt_minutes(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m} мин {s:0.2f} с"


def line_pos_for_frame(frame_idx: int) -> float:
    """The position of the vertical line for the raw frame frame_idx in [0... NUMBER_OF_FRAMES-1]"""
    if NUMBER_OF_FRAMES <= 1:
        return LINE_X_START
    t = (frame_idx * LINE_SPEED) / (NUMBER_OF_FRAMES - 1)
    if t > 1.0:
        t = 1.0
    return (1 - t) * LINE_X_START + t * LINE_X_END


def build_frame_df(frame_idx: int, xline: float,
                   x_all: np.ndarray, y_all: np.ndarray,
                   alive: np.ndarray, initial_circle_idx: np.ndarray) -> pd.DataFrame:
    sand_idx = np.flatnonzero(alive)
    dfs = []

    if sand_idx.size > 0:
        sand_df = pd.DataFrame({
            "time_index": np.full(sand_idx.size, frame_idx, dtype=np.int32),
            "coordinate_x": x_all[sand_idx].astype(np.float32),
            "coordinate_y": y_all[sand_idx].astype(np.float32),
            "type": ["sand"] * sand_idx.size,
            "initial_circle": np.where(initial_circle_idx[sand_idx] >= 0,
                                       initial_circle_idx[sand_idx], -1).astype(np.int32),
        })
        sand_df["type"] = sand_df["type"].astype(TYPE_DTYPE)
        dfs.append(sand_df)

    line_df = pd.DataFrame({
        "time_index": np.array([frame_idx, frame_idx], dtype=np.int32),
        "coordinate_x": np.array([xline, xline], dtype=np.float32),
        "coordinate_y": np.array([PLOT_Y_MIN, PLOT_Y_MAX], dtype=np.float32),
        "type": ["line", "line"],
        "initial_circle": np.array([-1, -1], dtype=np.int32),
    })
    line_df["type"] = line_df["type"].astype(TYPE_DTYPE)
    dfs.append(line_df)

    df = pd.concat(dfs, ignore_index=True)
    return df


def save_frame_parquet(frame_idx: int, xline: float,
                       x_all: np.ndarray, y_all: np.ndarray,
                       alive: np.ndarray, initial_circle_idx: np.ndarray,
                       out_dir: Path) -> None:
    """Write one Parquet per frame with Zstandard compression."""
    if out_dir.exists() is False:
        out_dir.mkdir(parents=True, exist_ok=True)

    df = build_frame_df(frame_idx, xline, x_all, y_all, alive, initial_circle_idx)
    path = out_dir / f"{frame_idx:06d}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def save_frame_parquet_s3(frame_idx: int, xline: float,
                          x_all: np.ndarray, y_all: np.ndarray,
                          alive: np.ndarray, initial_circle_idx: np.ndarray,
                          s3, bucket: str, prefix: str) -> None:
    """Write one Parquet per frame with Zstandard compression to S3."""
    df = build_frame_df(frame_idx, xline, x_all, y_all, alive, initial_circle_idx)
    key = f"{prefix.rstrip('/')}/{frame_idx:06d}.parquet"

    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="zstd")
    buf.seek(0)

    s3.upload_fileobj(
        buf, bucket, key,
        ExtraArgs={"ContentType": "application/x-parquet"}
    )


def spawn_circle_particles(state: SimulationState, source_circle_id: int, circle_slot_start, circle_slot_end, sx, sy):
    """
    Start scattering for the k-th circle: fill its slot with the starting coordinates.
    Fill the initial_circle_idx for the slot with the value circle_order_label[k].
    """
    i0, i1 = circle_slot_start[source_circle_id], circle_slot_end[source_circle_id]
    xs, ys = generate_points_in_circle(
        n_points=NUMBER_OF_PARTICLES,
        center_x=sx[source_circle_id], center_y=sy[source_circle_id],
        radius=CIRCLE_RADIUS, sigma_x=SIGMA_X
    )
    state.x[i0:i1] = xs
    state.y[i0:i1] = ys
    state.vx[i0:i1] = 0.0
    state.vy[i0:i1] = 0.0
    state.frozen[i0:i1] = False
    state.alive[i0:i1] = True
    # enter the “source circle number” for this slot
    state.initial_circle_idx[i0:i1] = state.circle_order_label[source_circle_id]


def _start_simulation_from_scratch(total_slots: int, n_circles: int) -> (SimulationState, int):
    x = np.full(total_slots, np.nan)
    y = np.full(total_slots, np.nan)
    vx = np.zeros(total_slots)
    vy = np.zeros(total_slots)
    frozen = np.zeros(total_slots, dtype=bool)
    alive = np.zeros(total_slots, dtype=bool)
    triggered = np.zeros(n_circles, dtype=bool)

    initial_circle_idx = np.full(total_slots, -1, dtype=int)  # for every grain
    circle_order_label = np.full(n_circles, -1, dtype=int)  # for each circle
    trigger_order_counter = 0

    print("[start] Fresh start from frame 0")
    state = SimulationState(x=x, y=y, vx=vx, vy=vy, frozen=frozen,
                            alive=alive, triggered=triggered, initial_circle_idx=initial_circle_idx,
                            circle_order_label=circle_order_label, trigger_order_counter=trigger_order_counter)
    start_frame = 0
    return state, start_frame


def _continue_simulation_from_state(data, total_slots: int, n_circles: int) -> (SimulationState, int):
    x = data.get("x")
    if x is None:
        x = np.full(total_slots, np.nan)

    y = data.get("y")
    if y is None:
        y = np.full(total_slots, np.nan)

    vx = data.get("vx")
    if vx is None:
        vx = np.zeros(total_slots)

    vy = data.get("vy")
    if vy is None:
        vy = np.zeros(total_slots)

    frozen = data.get("frozen")
    if frozen is None:
        frozen = np.zeros(total_slots, dtype=bool)

    alive = data.get("alive")
    if alive is None:
        alive = np.zeros(total_slots, dtype=bool)

    triggered = data.get("triggered")
    if triggered is None:
        triggered = np.zeros(n_circles, dtype=bool)

    # new fields (can be missed in the old checkpoint)
    initial_circle_idx = data.get("initial_circle_idx")
    if initial_circle_idx is None:
        initial_circle_idx = np.full(total_slots, -1, dtype=int)

    circle_order_label = data.get("circle_order_label")
    if circle_order_label is None:
        circle_order_label = np.full(n_circles, -1, dtype=int)

    trigger_order_counter = int(data["trigger_order_counter"]) if "trigger_order_counter" in data.files else int(
        np.max(circle_order_label) + 1 if np.any(circle_order_label >= 0) else 0)

    if data.get("last_frame") is None:
        print("State file did not reach first frame yet, so starting from scratch")
        start_frame = 0
    else:
        start_frame = int(data["last_frame"]) + 1

    print(f"[resume] Checkpoint loaded: starting from frame {start_frame}")
    state = SimulationState(x=x, y=y, vx=vx, vy=vy, frozen=frozen,
                            alive=alive, triggered=triggered, initial_circle_idx=initial_circle_idx,
                            circle_order_label=circle_order_label, trigger_order_counter=trigger_order_counter)
    return state, start_frame


def run_simulation(s3_bucket_data: Union[S3Bucket, None] = None,
                   folder_where_to_save_results: Union[Path, None] = None):
    """
    This script will run the simulation 'sand grains into KDE'

    This script runs a simulation where, at each step t, a parquet file is generated with information
    about what the system looks like (particles, their coordinates and types).
    There are two options for running this script:
        - local
        - AWS EC2 + S3 bucket

    :param s3_bucket_data: dataclass with credentials and configuration fields for S3 bucket
    :param folder_where_to_save_results: folder where to save results
    """
    if s3_bucket_data is None:
        # Local run
        if folder_where_to_save_results is None:
            folder_where_to_save_results = get_kde_simulation_path()
        folder_where_to_save_results.mkdir(exist_ok=True, parents=True)
        state_path = Path(folder_where_to_save_results, "state.npz")
    else:
        # Every artifact will be uploaded to S3 bucket
        s3_client = s3_bucket_data.get_client()
        # Full path is something like this f"s3://{s3_bucket_data.bucket}/{s3_bucket_data.prefix}/state.npz"
        state_path = f"{s3_bucket_data.prefix}/state.npz"
        print(state_path)

    sx, sy = generate_synthetic_data(n_samples=100, noise_std=1.0,
                                     outlier_fraction=0.1, random_state=2025)
    n_circles = len(sx)

    total_slots = NUMBER_OF_PARTICLES * n_circles

    if s3_bucket_data is None:
        # Local run
        if state_path.exists():
            print("State file exists. Loading")
            data = np.load(state_path, allow_pickle=True)
            state, start_frame = _continue_simulation_from_state(data, total_slots, n_circles)
        else:
            print("No state file. Starting from scratch")
            state, start_frame = _start_simulation_from_scratch(total_slots, n_circles)
    else:
        # Download from S3
        if s3_bucket_data.object_exists(s3_client, state_path):
            print(f"Read state file from S3 bucket")
            obj = s3_client.get_object(Bucket=s3_bucket_data.bucket, Key=state_path)
            data_bytes = obj["Body"].read()
            with BytesIO(data_bytes) as buf:
                data = np.load(buf, allow_pickle=True)
                state, start_frame = _continue_simulation_from_state(data, total_slots, n_circles)
        else:
            print("No state file in S3 bucket. Starting from scratch")
            state, start_frame = _start_simulation_from_scratch(total_slots, n_circles)

    circle_slot_start = NUMBER_OF_PARTICLES * np.arange(n_circles)
    circle_slot_end   = circle_slot_start + NUMBER_OF_PARTICLES

    for frame in range(start_frame, NUMBER_OF_FRAMES):
        xline = line_pos_for_frame(frame)

        # 1) Determine which circles to “activate” in this frame
        mask_crossed = sx < xline
        to_trigger = (~state.triggered) & mask_crossed
        if np.any(to_trigger):
            # sort the new intersecting circles by the X-coordinate of the center (from left to right)
            # so that the serial numbers follow the physical order of intersection
            new_ids = np.where(to_trigger)[0]
            # sx — X center in our notation
            order = np.argsort(sx[new_ids])
            for k in new_ids[order]:
                # assign the circle its intersection sequence number
                state.circle_order_label[k] = state.trigger_order_counter
                state.trigger_order_counter += 1
                # spawn sand and set initial_circle_idx for its slot
                spawn_circle_particles(state, k, circle_slot_start, circle_slot_end, sx, sy)
            state.triggered[to_trigger] = True

        # 2) One physics step (DT) for all active
        step_physics(state.x, state.y, state.vx, state.vy, state.frozen, state.alive)

        # 3) Save frame-by-frame parquet (grains + line)
        if s3_bucket_data is None:
            save_frame_parquet(frame, xline, state.x, state.y,
                               state.alive, state.initial_circle_idx,
                               folder_where_to_save_results)
        else:
            save_frame_parquet_s3(frame, xline, state.x, state.y,
                                  state.alive, state.initial_circle_idx,
                                  s3_client, s3_bucket_data.bucket, s3_bucket_data.prefix)

        if frame % 100 == 0:
            print(f"[{frame}/{NUMBER_OF_FRAMES}] saved")

            # 4) State of the simulation - checkpoint (do it for every 100 frame)
            if s3_bucket_data is None:
                state.save(frame, where_to_save=state_path)
            else:
                state.save_s3(frame=frame,
                              s3=s3_client,
                              bucket=s3_bucket_data.bucket,
                              key=state_path)


if __name__ == "__main__":
    # Configure S3 bucket - set it as none if you want to use local folder
    s3_bucket = S3Bucket(bucket="linear-regression-kernel-density", prefix="simulation")

    t_start = time.perf_counter()
    run_simulation(s3_bucket_data=s3_bucket)
    t_end = time.perf_counter()
    print(f"Finished. Total number of frames: {NUMBER_OF_FRAMES}. Time: {fmt_minutes(t_end - t_start)}")
