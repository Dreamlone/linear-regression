from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection
import matplotlib.patches as patches
from math import floor

from kde_explanation.kde_utils import get_kde_plots_path

############
# settings #
############
SPEEDUP = 2
PARTICLES_RADIUS        = 0.01      # particle radius
CONTACT_SOFTNESS        = 0.02      # 0 — almost no "pushing apart", 1 — rigid
NUMBER_OF_PARTICLES     = 600
NUMBER_OF_FRAMES        = 250
DT                      = 0.02
GROUND_Y                = 0.5
G                       = 9.81
GROUND_INFLUENCE_HEIGHT = 0.6       # ground influence zone (in Y-axis units)
FREEZE_INFLUENCE        = 0.6       # influence threshold for freezing above ground (0..1)

# Sleep / freeze
# speed threshold for "sleep" (used rarely — we have a hard freeze)
SLEEP_VEL_THR         = 0.02
SLEEP_FRAMES          = 5
# HARD FREEZE: a particle on the ground or stably standing becomes frozen=True permanently.

# Solver parameters
N_ITERS               = 10
DAMPING               = 0.08

# Stability / sliding rules
ALIGN_THR             = 0.15      # if |dx| <= ALIGN_THR*r relative to the support below — treat as "directly on top"
SLIDE_SPEED           = 0.25      # sliding speed along the "slope" with a single support (units/s)
CONTACT_TOL           = 1e-6      # distance tolerance in contact


def generate_points_in_circle(n_points=100, center_x=0.0, center_y=3.0, radius=0.5):
    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    x_smalls = center_x + radii * np.cos(angles)
    y_smalls = center_y + radii * np.sin(angles)
    return x_smalls, y_smalls


def build_grid(x, y, cell_size):
    cells = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        ix = floor(xi / cell_size)
        iy = floor(yi / cell_size)
        cells.setdefault((ix, iy), []).append(i)
    return cells


def neighbor_pairs(x, y, cell_size):
    cells = build_grid(x, y, cell_size)
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


def compute_coordinates(number_of_frames: int, x0, y0):
    dt = DT
    g = G
    ground_y = GROUND_Y
    r = PARTICLES_RADIUS

    # geometry/iterations
    CONTACT_TOL   = 1e-8
    MIN_SEP_X     = 0.6 * r
    SLIDE_SPEED   = 0.25
    SUBSTEPS      = 3
    sub_dt        = dt / SUBSTEPS

    def ground_influence(y_val):
        """0 — far from the ground; 1 — right next to the ground"""
        gap = np.maximum(0.0, y_val - (ground_y + r))
        t = np.clip(1.0 - gap / GROUND_INFLUENCE_HEIGHT, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    x = x0.copy()
    y = y0.copy()
    n = len(x)

    vx = np.zeros(n)
    vy = np.zeros(n)
    xs, ys = [], []

    frozen = np.zeros(n, dtype=bool)
    cell_size = 2.1 * (2 * r)

    for _ in range(number_of_frames):
        for _sub in range(SUBSTEPS):
            vy[~frozen] += g * sub_dt
            x_pred = x.copy()
            y_pred = y.copy()
            x_pred[~frozen] += vx[~frozen] * sub_dt
            y_pred[~frozen] -= vy[~frozen] * sub_dt

            for __ in range(N_ITERS):
                # surface
                below = y_pred < ground_y + r
                if np.any(below):
                    y_pred[below] = ground_y + r

                # non-intersection: the force of correction increases near the ground
                for i, j in neighbor_pairs(x_pred, y_pred, cell_size):
                    dx = x_pred[j] - x_pred[i]
                    dy = y_pred[j] - y_pred[i]
                    dist2 = dx*dx + dy*dy
                    min_d = 2 * r
                    if dist2 < (min_d - CONTACT_TOL) ** 2:
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

                below = y_pred < ground_y + r
                if np.any(below):
                    y_pred[below] = ground_y + r

            dvx = (x_pred - x) / sub_dt
            dvy = (y - y_pred) / sub_dt
            vx[~frozen] = dvx[~frozen] * (1.0 - DAMPING)
            vy[~frozen] = dvy[~frozen] * (1.0 - DAMPING)

            on_ground = (y_pred <= ground_y + r + 1e-9)
            frozen = frozen | on_ground

            if not np.all(frozen):
                supports_left  = [[] for _ in range(n)]
                supports_right = [[] for _ in range(n)]

                for i, j in neighbor_pairs(x_pred, y_pred, cell_size):
                    for top, bot in ((i, j), (j, i)):
                        if frozen[bot] and not frozen[top]:
                            dx = x_pred[top] - x_pred[bot]
                            dy = y_pred[top] - y_pred[bot]
                            if dy > 0.0 and dx*dx + dy*dy <= (2*r + CONTACT_TOL)**2:
                                (supports_left[top] if dx < 0 else supports_right[top]).append(bot)

                newly_stable = np.zeros(n, dtype=bool)
                for i in range(n):
                    if frozen[i] or on_ground[i]:
                        continue
                    gi = ground_influence(y_pred[i])
                    if gi < FREEZE_INFLUENCE:
                        continue
                    if supports_left[i] and supports_right[i]:
                        jl = min(supports_left[i],  key=lambda j: abs(x_pred[i]-x_pred[j]))
                        jr = min(supports_right[i], key=lambda j: abs(x_pred[i]-x_pred[j]))
                        if abs(x_pred[jr] - x_pred[jl]) >= MIN_SEP_X:
                            newly_stable[i] = True

                frozen = frozen | newly_stable

                has_left  = np.array([len(supports_left[i])  > 0 for i in range(n)])
                has_right = np.array([len(supports_right[i]) > 0 for i in range(n)])
                slide_idx = (~frozen) & (has_left ^ has_right)
                if np.any(slide_idx):
                    gi_vec = np.clip(ground_influence(y_pred), 0.0, 1.0)
                    dir_sign = np.zeros(n)
                    dir_sign[slide_idx & has_left]  = +1.0
                    dir_sign[slide_idx & has_right] = -1.0
                    vx[slide_idx] += dir_sign[slide_idx] * (SLIDE_SPEED * gi_vec[slide_idx]) * sub_dt

            x, y = x_pred, y_pred
            vx[frozen] = 0.0
            vy[frozen] = 0.0

        xs.append(x.copy())
        ys.append(y.copy())

    return np.asarray(xs), np.asarray(ys)


def init():
    ec.set_offsets(np.c_[XS[0], YS[0]])
    return (ec,)


def animate(frame):
    idx = min(frame * SPEEDUP, len(XS) - 1)
    ec.set_offsets(np.c_[XS[idx], YS[idx]])
    return (ec,)


if __name__ == '__main__':
    """ 
    Fourth attempt to create an animation of a circle “spilling out” like sand onto the ground
    
    Features of implementation:
        - the grains are generated more densely so that the entire volume of the original circle is filled completely
        - "grains of sand" interact with each other when falling
        - the fall stops when "grains of sand" reach the surface
    """

    x0, y0 = generate_points_in_circle(
        n_points=NUMBER_OF_PARTICLES, center_x=0.0, center_y=3.0, radius=0.5
    )
    XS, YS = compute_coordinates(NUMBER_OF_FRAMES, x0, y0)

    fig, ax = plt.subplots()
    ax.set(xlim=[-2.5, 2.5], ylim=[0, 5])
    ax.set_aspect('equal')
    ax.xaxis.set_ticklabels([]); ax.xaxis.set_ticks([])

    ax.plot([-2.25, 2.25], [GROUND_Y, GROUND_Y], c='black', linewidth=4, zorder=1)
    circle = patches.Circle((0, 3), 0.5, edgecolor='r', facecolor='none', linewidth=1)
    ax.add_patch(circle)

    widths  = np.full(NUMBER_OF_PARTICLES, 2 * PARTICLES_RADIUS)
    heights = np.full(NUMBER_OF_PARTICLES, 2 * PARTICLES_RADIUS)
    ec = EllipseCollection(
        widths=widths,
        heights=heights,
        angles=np.zeros(NUMBER_OF_PARTICLES),
        units='xy',
        offsets=np.c_[x0, y0],
        transOffset=ax.transData,
        facecolors='none',
        edgecolors='r',
        linewidths=0.5
    )
    ec.set_animated(True)
    ax.add_collection(ec)

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=(len(XS) + SPEEDUP - 1) // SPEEDUP,  # меньше кадров к показу
        interval=30,   # быстрее обновления (мс)
        blit=True
    )
    name_gif_file = "4_simple_animation.gif"
    save_path = Path(get_kde_plots_path(), name_gif_file)
    ani.save(save_path, writer="pillow", dpi=150)
    print(f"Saved to {name_gif_file}")
