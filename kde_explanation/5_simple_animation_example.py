import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection
import matplotlib.patches as patches
from math import floor

from kde_explanation.kde_utils import get_kde_plots_path

np.random.seed(1999)

# ===================== настройки =====================
PARTICLES_RADIUS        = 0.018     # радиус частицы
CONTACT_SOFTNESS        = 0.02      # 0 — почти нет распирания, 1 — жёстко
NUMBER_OF_PARTICLES     = 200
NUMBER_OF_FRAMES        = 250
DT                      = 0.02
GROUND_Y                = 0.5
G                       = 9.81
GROUND_INFLUENCE_HEIGHT = 0.6       # зона влияния земли (в единицах оси Y)
FREEZE_INFLUENCE        = 0.6       # порог влияния для заморозки над землей (0..1)

# «Сон»/заморозка (для текущей логики важен только "вечный" фриз ниже)
SLEEP_VEL_THR         = 0.02
SLEEP_FRAMES          = 5

# Параметры решателя
N_ITERS               = 10
DAMPING               = 0.08

# Правила устойчивости/скатывания
ALIGN_THR             = 0.15
SLIDE_SPEED           = 0.25
CONTACT_TOL           = 1e-6

# Насколько ускорять показ (брать каждый k-й кадр)
SPEEDUP               = 2

# ===================== генерация начальных точек =====================
def generate_points_in_circle(n_points=100, center_x=0.0, center_y=3.0, radius=PARTICLES_RADIUS, sigma_x=0.2):
    """
    Генерация точек в круге, но с нормальным распределением по X.
    sigma_x — стандартное отклонение нормали (в единицах оси X),
    центр нормали совпадает с center_x.
    """
    points = []
    while len(points) < n_points:
        # Случайный X из нормального распределения
        x = np.random.normal(loc=center_x, scale=sigma_x)
        # Для Y — равномерно в пределах круга
        y_range = np.sqrt(radius**2 - (x - center_x)**2) if abs(x - center_x) <= radius else None
        if y_range is None:
            continue  # X вне круга — пробуем снова
        y = np.random.uniform(center_y - y_range, center_y + y_range)
        points.append((x, y))
    xs, ys = zip(*points)
    return np.array(xs), np.array(ys)

# ===================== простой ячейковый список =====================
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

# ===================== динамика с «заморозкой» при устойчивости =====================
def compute_coordinates(number_of_frames: int, x0, y0):
    dt = DT
    g = G
    ground_y = GROUND_Y
    r = PARTICLES_RADIUS

    # геометрия/итерации
    CONTACT_TOL_LOCAL = 1e-8
    MIN_SEP_X         = 0.6 * r
    SLIDE_SPEED_LOCAL = 0.25
    SUBSTEPS          = 3
    sub_dt            = dt / SUBSTEPS

    def ground_influence(y_val):
        """0 — далеко от земли; 1 — у самой земли. Плавное нарастание (smoothstep)."""
        gap = np.maximum(0.0, y_val - (ground_y + r))
        t = np.clip(1.0 - gap / GROUND_INFLUENCE_HEIGHT, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)  # smoothstep

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
            # интеграция (только не frozen)
            vy[~frozen] += g * sub_dt
            x_pred = x.copy()
            y_pred = y.copy()
            x_pred[~frozen] += vx[~frozen] * sub_dt
            y_pred[~frozen] -= vy[~frozen] * sub_dt

            # PBD-проекции
            for __ in range(N_ITERS):
                # пол
                below = y_pred < ground_y + r
                if np.any(below):
                    y_pred[below] = ground_y + r

                # непересечение: сила коррекции растёт у земли
                for i, j in neighbor_pairs(x_pred, y_pred, cell_size):
                    dx = x_pred[j] - x_pred[i]
                    dy = y_pred[j] - y_pred[i]
                    dist2 = dx*dx + dy*dy
                    min_d = 2 * r
                    if dist2 < (min_d - CONTACT_TOL_LOCAL) ** 2:
                        if dist2 < 1e-18:
                            jitter = 1e-4 * r
                            dx = np.random.uniform(-jitter, jitter)
                            dy = np.random.uniform(-jitter, jitter)
                            dist2 = dx*dx + dy*dy

                        dist = np.sqrt(dist2)
                        nx, ny = dx / dist, dy / dist
                        overlap = (min_d - dist)

                        # влияние земли для пары
                        gi = max(ground_influence(y_pred[i]), ground_influence(y_pred[j]))
                        # мягкость вдали + жёсткость у земли
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

                # ещё раз пол
                below = y_pred < ground_y + r
                if np.any(below):
                    y_pred[below] = ground_y + r

            # скорости (для не frozen)
            dvx = (x_pred - x) / sub_dt
            dvy = (y - y_pred) / sub_dt
            vx[~frozen] = dvx[~frozen] * (1.0 - DAMPING)
            vy[~frozen] = dvy[~frozen] * (1.0 - DAMPING)

            # заморозка: всегда на земле
            on_ground = (y_pred <= ground_y + r + 1e-9)
            frozen = frozen | on_ground

            # слоистая заморозка над землёй — только в зоне влияния
            if not np.all(frozen):
                supports_left  = [[] for _ in range(n)]
                supports_right = [[] for _ in range(n)]

                for i, j in neighbor_pairs(x_pred, y_pred, cell_size):
                    for top, bot in ((i, j), (j, i)):
                        if frozen[bot] and not frozen[top]:
                            dx = x_pred[top] - x_pred[bot]
                            dy = y_pred[top] - y_pred[bot]
                            if dy > 0.0 and dx*dx + dy*dy <= (2*r + CONTACT_TOL_LOCAL)**2:
                                (supports_left[top] if dx < 0 else supports_right[top]).append(bot)

                newly_stable = np.zeros(n, dtype=bool)
                for i in range(n):
                    if frozen[i] or on_ground[i]:
                        continue
                    gi = ground_influence(y_pred[i])
                    if gi < FREEZE_INFLUENCE:
                        continue  # слишком высоко — не замораживаем слой
                    if supports_left[i] and supports_right[i]:
                        jl = min(supports_left[i],  key=lambda j: abs(x_pred[i]-x_pred[j]))
                        jr = min(supports_right[i], key=lambda j: abs(x_pred[i]-x_pred[j]))
                        if abs(x_pred[jr] - x_pred[jl]) >= MIN_SEP_X:
                            newly_stable[i] = True

                frozen = frozen | newly_stable

                # одиночная опора -> скатывание; скорость зависит от близости к земле
                has_left  = np.array([len(supports_left[i])  > 0 for i in range(n)])
                has_right = np.array([len(supports_right[i]) > 0 for i in range(n)])
                slide_idx = (~frozen) & (has_left ^ has_right)
                if np.any(slide_idx):
                    # векторно влияние земли
                    gap = np.maximum(0.0, y_pred - (ground_y + r))
                    t = np.clip(1.0 - gap / GROUND_INFLUENCE_HEIGHT, 0.0, 1.0)
                    gi_vec = t * t * (3.0 - 2.0 * t)
                    dir_sign = np.zeros(n)
                    dir_sign[slide_idx & has_left]  = +1.0  # опора слева -> вправо
                    dir_sign[slide_idx & has_right] = -1.0  # опора справа -> влево
                    vx[slide_idx] += dir_sign[slide_idx] * (SLIDE_SPEED_LOCAL * gi_vec[slide_idx]) * sub_dt

            # применяем субшаг
            x, y = x_pred, y_pred
            vx[frozen] = 0.0
            vy[frozen] = 0.0

        # записываем кадр
        xs.append(x.copy())
        ys.append(y.copy())

    return np.asarray(xs), np.asarray(ys)


def kde1d(x, grid_x=None, bandwidth=None, points=600):
    x = np.asarray(x)
    if grid_x is None:
        pad = 0.2
        xmin, xmax = x.min() - pad, x.max() + pad
        grid_x = np.linspace(xmin, xmax, points)
    if bandwidth is None:
        # Silverman's Rule
        n = len(x)
        std = np.std(x) + 1e-12
        bandwidth = 1.06 * std * n**(-1/5)

    u = (grid_x[None, :] - x[:, None]) / (bandwidth + 1e-12)
    dens = np.exp(-0.5 * u**2).sum(axis=0) / (len(x) * (bandwidth + 1e-12) * np.sqrt(2*np.pi))
    return grid_x, dens


def init():
    ec.set_offsets(np.c_[XS[0], YS[0]])
    return (ec,)


def animate(frame):
    idx = min(frame * SPEEDUP, len(XS) - 1)
    ec.set_offsets(np.c_[XS[idx], YS[idx]])
    return (ec,)


def fmt_minutes(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m} min {s:0.2f} seconds"


if __name__ == '__main__':
    """ 
    Fifth attempt to create an animation of a circle “spilling out” like sand onto the ground

    Features of implementation:
        - the grains are generated more densely so that the entire volume of the original circle is filled completely
        - "grains of sand" interact with each other when falling
        - the fall stops when "grains of sand" reach the surface
        - normal distribution is shown for comparison
    """

    t0 = time.perf_counter()
    x0, y0 = generate_points_in_circle(
        n_points=NUMBER_OF_PARTICLES, center_x=0.0, center_y=3.0, radius=0.5
    )

    # trajectory calculation
    t1 = time.perf_counter()
    XS, YS = compute_coordinates(NUMBER_OF_FRAMES, x0, y0)
    t2 = time.perf_counter()

    fig, ax = plt.subplots()
    ax.set(xlim=[-2.5, 2.5], ylim=[0, 5])
    ax.set_aspect('equal')
    ax.xaxis.set_ticklabels([]); ax.xaxis.set_ticks([])

    # surface
    ax.plot([-2.25, 2.25], [GROUND_Y, GROUND_Y], c='black', linewidth=1, zorder=1)

    x_final = XS[-1]
    gx, gdens = kde1d(x_final, points=600)

    KDE_BAND_HEIGHT = 0.35
    gdens_scaled = gdens / (gdens.max() + 1e-12) * KDE_BAND_HEIGHT
    kde_line, = ax.plot(gx, GROUND_Y + gdens_scaled, linewidth=1.8, color='red', zorder=0, label='KDE(x)')
    ax.fill_between(gx, GROUND_Y, GROUND_Y + gdens_scaled, color='red', alpha=0.15, zorder=0)

    # Normal distribution fitted to the mean/std of the final X
    mu = float(np.mean(x_final))
    sigma = float(np.std(x_final) + 1e-12)
    norm_pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((gx - mu) / sigma) ** 2)
    norm_scaled = norm_pdf / (norm_pdf.max() + 1e-12) * KDE_BAND_HEIGHT
    norm_line, = ax.plot(gx, GROUND_Y + norm_scaled, linewidth=1.8, color='grey', zorder=1, label='Normal fit')

    circle = patches.Circle((0, 3), 0.5, edgecolor='r', facecolor='none', linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)

    widths  = np.full(NUMBER_OF_PARTICLES, 2 * PARTICLES_RADIUS)
    heights = np.full(NUMBER_OF_PARTICLES, 2 * PARTICLES_RADIUS)
    ec = EllipseCollection(
        widths=widths,
        heights=heights,
        angles=np.zeros(NUMBER_OF_PARTICLES),
        units='xy',
        offsets=np.c_[x0, y0],
        transOffset=ax.transData,
        facecolors='#FA7070',
        edgecolors='r',
        linewidths=0.5
    )
    ec.set_animated(True)
    ax.add_collection(ec)

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=(len(XS) + SPEEDUP - 1) // SPEEDUP,
        interval=30,  # мс
        blit=True
    )

    t3 = time.perf_counter()
    name_gif_file = "5_simple_animation.gif"
    save_path = Path(get_kde_plots_path(), name_gif_file)
    ani.save(save_path, writer="pillow", dpi=150)
    t4 = time.perf_counter()

    total = t4 - t0
    compute = t2 - t1
    save = t4 - t3

    print(f"Trajectory calculation time: {fmt_minutes(compute)}")
    print(f"Animation save time: {fmt_minutes(save)}")
    print(f"Total script execution time: {fmt_minutes(total)}")
