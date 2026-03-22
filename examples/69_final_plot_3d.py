import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.paths import get_plots_path


def cube_vertices(center_xyz: np.ndarray, side: float) -> np.ndarray:
    """Return 8 vertices of an axis-aligned cube."""
    half = 0.5 * float(side)
    cx, cy, cz = [float(v) for v in center_xyz]

    offsets = np.array(
        [
            [-half, -half, -half],
            [-half, -half, +half],
            [-half, +half, -half],
            [-half, +half, +half],
            [+half, -half, -half],
            [+half, -half, +half],
            [+half, +half, -half],
            [+half, +half, +half],
        ],
        dtype=float,
    )
    return offsets + np.array([cx, cy, cz], dtype=float)


def cube_faces(vertices: np.ndarray) -> list:
    """Return faces (as lists of 3D points) for a cube from its vertices."""
    return [
        # x = -half
        [vertices[i] for i in [0, 1, 3, 2]],
        # x = +half
        [vertices[i] for i in [4, 5, 7, 6]],
        # y = -half
        [vertices[i] for i in [0, 1, 5, 4]],
        # y = +half
        [vertices[i] for i in [2, 3, 7, 6]],
        # z = -half
        [vertices[i] for i in [0, 2, 6, 4]],
        # z = +half
        [vertices[i] for i in [1, 3, 7, 5]],
    ]


def add_cube(
    ax,
    center_xyz: np.ndarray,
    side: float,
    face_color: str,
    face_alpha: float = 0.35,
    edge_color: str = "black",
    edge_width: float = 0.8,
):
    vertices = cube_vertices(center_xyz=center_xyz, side=side)
    faces = cube_faces(vertices)

    poly = Poly3DCollection(
        faces,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=edge_width,
        alpha=float(face_alpha),
    )
    ax.add_collection3d(poly)

    return vertices


def add_shadow_on_planes(
    ax,
    vertices: np.ndarray,
    shadow_color: str = "black",
    shadow_alpha: float = 0.10,
    plane_x: float = 0.0,
    plane_y: float = 0.0,
    plane_z: float = 0.0,
):
    """
    Add three orthographic 'shadows' of a cube:
      - onto XY plane (z = plane_z)
      - onto XZ plane (y = plane_y)
      - onto YZ plane (x = plane_x)
    """
    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]

    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

    shadow_xy = np.array(
        [
            [x_min, y_min, plane_z],
            [x_max, y_min, plane_z],
            [x_max, y_max, plane_z],
            [x_min, y_max, plane_z],
        ],
        dtype=float,
    )

    shadow_xz = np.array(
        [
            [x_min, plane_y, z_min],
            [x_max, plane_y, z_min],
            [x_max, plane_y, z_max],
            [x_min, plane_y, z_max],
        ],
        dtype=float,
    )

    shadow_yz = np.array(
        [
            [plane_x, y_min, z_min],
            [plane_x, y_max, z_min],
            [plane_x, y_max, z_max],
            [plane_x, y_min, z_max],
        ],
        dtype=float,
    )

    for shadow_quad in [shadow_xy, shadow_xz, shadow_yz]:
        poly = Poly3DCollection(
            [shadow_quad],
            facecolors=shadow_color,
            edgecolors="none",
            alpha=float(shadow_alpha),
        )
        ax.add_collection3d(poly)


def add_arrow(ax, start_xyz: np.ndarray, end_xyz: np.ndarray, color: str = "black"):
    start_xyz = np.asarray(start_xyz, dtype=float)
    end_xyz = np.asarray(end_xyz, dtype=float)
    direction = end_xyz - start_xyz

    ax.quiver(
        start_xyz[0], start_xyz[1], start_xyz[2],
        direction[0], direction[1], direction[2],
        arrow_length_ratio=0.12,
        linewidth=2.0,
        color=color,
        normalize=False,
    )


def add_vectors_in_yz_shadow(
    ax,
    vertices: np.ndarray,
    plane_x: float = 0.0,
    color_a: str = "black",
    color_b: str = "tab:blue",
):
    """Draw two arrows inside the cube's shadow rectangle on the YZ plane (x=plane_x)."""
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

    y_range = max(1e-9, y_max - y_min)
    z_range = max(1e-9, z_max - z_min)

    start_a = np.array([plane_x, y_min + 0.18 * y_range, z_min + 0.25 * z_range], dtype=float)
    dir_a = np.array([0.0, 0.55 * y_range, 0.20 * z_range], dtype=float)

    ax.quiver(
        start_a[0], start_a[1], start_a[2],
        dir_a[0], dir_a[1], dir_a[2],
        arrow_length_ratio=0.20,
        linewidth=2.2,
        color=color_a,
        normalize=False,
    )

    start_b = np.array([plane_x, y_min + 0.35 * y_range, z_min + 0.18 * z_range], dtype=float)
    dir_b = np.array([0.0, 0.25 * y_range, 0.60 * z_range], dtype=float)

    ax.quiver(
        start_b[0], start_b[1], start_b[2],
        dir_b[0], dir_b[1], dir_b[2],
        arrow_length_ratio=0.20,
        linewidth=2.2,
        color=color_b,
        normalize=False,
    )


def generate_linear_cloud_in_xz_shadow(
    vertices: np.ndarray,
    n_points: int = 250,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (x, z) points inside the cube's XZ footprint with a visible linear trend."""
    rng = np.random.default_rng(seed)

    x_vals = vertices[:, 0]
    z_vals = vertices[:, 2]
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

    x_range = max(1e-9, x_max - x_min)
    z_range = max(1e-9, z_max - z_min)

    margin_x = 0.08 * x_range
    margin_z = 0.08 * z_range

    x_data = rng.uniform(x_min + margin_x, x_max - margin_x, size=int(n_points)).astype(float)
    t = (x_data - (x_min + margin_x)) / max(1e-9, (x_max - margin_x) - (x_min + margin_x))

    z_line = (z_min + margin_z) + (0.15 + 0.70 * t) * (z_range - 2.0 * margin_z)
    z_noise = rng.normal(loc=0.0, scale=0.07 * z_range, size=int(n_points))
    z_data = np.clip(z_line + z_noise, z_min + margin_z, z_max - margin_z).astype(float)

    return x_data, z_data


def add_scatter_in_xz_shadow(
    ax,
    x_data: np.ndarray,
    z_data: np.ndarray,
    plane_y: float,
    color: str = "black",
    size: float = 8.0,
):
    """Draw scatter cloud on the XZ shadow plane (y=plane_y)."""
    y_data = np.full_like(x_data, float(plane_y), dtype=float)
    ax.scatter(x_data, y_data, z_data, c=color, s=float(size), depthshade=False)


def kde_2d_gaussian(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth_x: float,
    bandwidth_y: float,
) -> np.ndarray:
    """Simple 2D Gaussian KDE on a grid (no scipy)."""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    grid_xx, grid_yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])  # (m, 2)

    dx = (points[:, 0:1] - x_data[None, :]) / max(1e-12, float(bandwidth_x))
    dy = (points[:, 1:2] - y_data[None, :]) / max(1e-12, float(bandwidth_y))

    kernel = np.exp(-0.5 * (dx * dx + dy * dy))
    density = kernel.mean(axis=1)

    return density.reshape(grid_xx.shape)


def add_kde_on_xy_shadow_from_xz_data(
    ax,
    vertices: np.ndarray,
    x_data: np.ndarray,
    z_data: np.ndarray,
    plane_z: float = 0.0,
):
    """
    Plot KDE on the XY shadow (z=plane_z), using the same (x,z) data,
    mapping z -> y so it fits inside the cube's XY shadow rectangle.
    """
    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]

    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

    y_range = max(1e-9, y_max - y_min)
    z_range = max(1e-9, z_max - z_min)

    y_mapped = y_min + (z_data - z_min) / z_range * y_range

    x_grid = np.linspace(x_min, x_max, 70)
    y_grid = np.linspace(y_min, y_max, 70)

    bandwidth_x = 0.12 * max(1e-9, (x_max - x_min))
    bandwidth_y = 0.12 * max(1e-9, (y_max - y_min))

    density = kde_2d_gaussian(
        x_data=x_data,
        y_data=y_mapped,
        x_grid=x_grid,
        y_grid=y_grid,
        bandwidth_x=bandwidth_x,
        bandwidth_y=bandwidth_y,
    )

    grid_xx, grid_yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    ax.contour(
        grid_xx,
        grid_yy,
        density,
        zdir="z",
        offset=float(plane_z),
        levels=6,
        colors="black",
        linewidths=1.2,
    )


def add_reference_planes(ax, plane_limit: float):
    """
    Draw three faint reference planes:
      - XY at z=0
      - XZ at y=plane_limit (far wall)
      - YZ at x=0
    """
    plane_xy = np.array(
        [[0, 0, 0], [plane_limit, 0, 0], [plane_limit, plane_limit, 0], [0, plane_limit, 0]],
        dtype=float,
    )
    plane_xz = np.array(
        [[0, plane_limit, 0], [plane_limit, plane_limit, 0], [plane_limit, plane_limit, plane_limit], [0, plane_limit, plane_limit]],
        dtype=float,
    )
    plane_yz = np.array(
        [[0, 0, 0], [0, plane_limit, 0], [0, plane_limit, plane_limit], [0, 0, plane_limit]],
        dtype=float,
    )

    ax.add_collection3d(Poly3DCollection([plane_xy], facecolors="grey", edgecolors="none", alpha=0.06))
    ax.add_collection3d(Poly3DCollection([plane_xz], facecolors="grey", edgecolors="none", alpha=0.04))
    ax.add_collection3d(Poly3DCollection([plane_yz], facecolors="grey", edgecolors="none", alpha=0.05))


def main():
    cube_side = 1.0  # 2x vs original

    # +1 to cube coordinates (gentle "floating" above z=0 plane)
    offset = np.array([1.0, 1.0, 1.0], dtype=float)

    base_centers = [
        np.array([1.0, 7.0, 1.0], dtype=float),
        np.array([5.0, 5.0, 3.0], dtype=float),
        np.array([9.0, 3.0, 5.0], dtype=float),
        np.array([13.0, 1.0, 7.0], dtype=float),
    ]
    centers = [c + offset for c in base_centers]
    cube_colors = ["black", "black", "black", "black"]

    max_center_value = float(np.max(np.vstack(centers)))
    plane_limit = max_center_value + 1.2  # margin

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Hide default 3D panes so only our custom planes remain visible.
    try:
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
    except Exception:
        pass

    add_reference_planes(ax, plane_limit=float(plane_limit))

    vertices_first_cube = None

    for idx, (center_xyz, face_color) in enumerate(zip(centers, cube_colors)):
        vertices = add_cube(
            ax=ax,
            center_xyz=center_xyz,
            side=cube_side,
            face_color=face_color,
            face_alpha=0.35,
            edge_color="black",
            edge_width=0.8,
        )

        add_shadow_on_planes(
            ax=ax,
            vertices=vertices,
            shadow_color="black",
            shadow_alpha=0.10,
            plane_x=0.0,
            plane_y=float(plane_limit),  # XZ shadow to far wall (y=max)
            plane_z=0.0,                 # XY shadow to floor (z=0)
        )

        if idx == 0:
            vertices_first_cube = vertices

    # Blue cube extras
    if vertices_first_cube is not None:
        add_vectors_in_yz_shadow(
            ax=ax,
            vertices=vertices_first_cube,
            plane_x=0.0,
            color_a="black",
            color_b="tab:blue",
        )

        x_data, z_data = generate_linear_cloud_in_xz_shadow(
            vertices=vertices_first_cube,
            n_points=260,
            seed=7,
        )

        add_scatter_in_xz_shadow(
            ax=ax,
            x_data=x_data,
            z_data=z_data,
            plane_y=float(plane_limit),
            color="black",
            size=8.0,
        )

        add_kde_on_xy_shadow_from_xz_data(
            ax=ax,
            vertices=vertices_first_cube,
            x_data=x_data,
            z_data=z_data,
            plane_z=0.0,
        )

    # Arrows between cube centers
    add_arrow(ax, centers[0], centers[1], color="black")
    add_arrow(ax, centers[1], centers[2], color="black")
    add_arrow(ax, centers[2], centers[3], color="black")

    # Limits start from 0 (as requested)
    ax.set_xlim(0.0, plane_limit)
    ax.set_ylim(0.0, plane_limit)
    ax.set_zlim(0.0, plane_limit)

    # No ticks and no grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=17, azim=-60)

    output_path = get_plots_path() / "69_final_plot_3d_base.svg"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()