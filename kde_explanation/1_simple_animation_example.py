from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.patches as patches

from kde_explanation.kde_utils import get_kde_plots_path

fig, ax = plt.subplots()


def compute_coordinates(number_of_frames: int, x_smalls, y_smalls):
    offset = np.linspace(0, 2.5, number_of_frames)
    xs = [x_smalls] * len(offset)
    ys = []
    for current_offset in offset:
        ys.append(y_smalls - current_offset)
    return xs, ys


def animate(frame):
    ax.clear()
    ax.set(xlim=[-2.5, 2.5], ylim=[0, 5])
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.plot([-2.25, 2.25], [0.5, 0.5], c='black', linewidth=4)
    ax.set_aspect('equal')

    if frame == 0:
        # First plot with initial circle
        circle = patches.Circle((0, 3), 0.5, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(circle)
    scat = ax.scatter(xs[frame], ys[frame], s=1, alpha=0.5, color='black')
    return scat,


def generate_points_in_circle(n_points=100, center_x=0.0, center_y=3.0, radius=0.5):
    # Random angles from 0 to 2π
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))

    # Conversion to Cartesian coordinates
    x_smalls = center_x + radii * np.cos(angles)
    y_smalls = center_y + radii * np.sin(angles)

    return x_smalls, y_smalls


if __name__ == '__main__':
    """ 
    First attempt to create an animation of a circle “spilling out” like sand onto the ground
    Features of implementation:
        - "grains of sand" do not interact with each other when falling
        - the fall never stops
    """
    number_of_frames_in_animation = 10
    x_dots_in_circle, y_dots_in_circle = generate_points_in_circle()
    xs, ys = compute_coordinates(number_of_frames_in_animation, x_dots_in_circle, y_dots_in_circle)
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=number_of_frames_in_animation,
        interval=400,
        blit=True
    )
    save_path = Path(get_kde_plots_path(), "1_simple_animation.gif")
    ani.save(save_path, writer="pillow")
