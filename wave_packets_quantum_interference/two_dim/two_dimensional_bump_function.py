from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import linalg
from scipy.sparse import diags, eye, kron, csr_matrix
from mayavi import mlab

visualize_with_matplotlib, save_to_file = True, True
fps, num_seconds, time_steps_per_frame = 30, 12, 1
x_c, ko_x, y_c, ko_y, sigma = -3, 4, 0, 0, 1.0
x_min, x_max, num_points_x = -10, 10, 200
y_min, y_max, num_points_y = -10, 10, 200

total_frames = fps * num_seconds
dx, dy = (x_max - x_min) / (num_points_x - 1), (y_max - y_min) / (num_points_y - 1)
dt = 0.125 * dx * dy
x, y = np.linspace(x_min, x_max, num_points_x)[1:-1], np.linspace(y_min, y_max, num_points_y)[1:-1]
x_g, y_g = np.mgrid[x_min + dx:x_max - dx:1j * (num_points_x - 2),
                    y_min + dy:y_max - dy:1j * (num_points_y - 2)]


def calculate_initial_psi(xo, yo, kxo, kyo, s):
    r = np.sqrt(np.square(x_g - xo) + np.square(y_g - yo))
    return (r/s < 1) * np.exp(-1 / (1 - np.square(r / s))) \
           * np.exp(1j * kxo * (x_g - xo) + 1j * kyo * (y_g - yo))


def calculate_evolution_matrices():
    # building hamiltonian
    hx = 1 / np.square(dx)
    hy = 1 / np.square(dy)
    dim = (num_points_x - 2) * (num_points_y - 2)
    offset = [-2, -1, 0, 1, 2]
    diagonals_x = [
        np.concatenate(([0], -np.ones(num_points_x - 6) / 12.0, [0])),
        np.concatenate(([1], 4 * np.ones(num_points_x - 5) / 3.0, [1])),
        np.concatenate(([-2], -2.5 * np.ones(num_points_x - 4), [-2])),
        np.concatenate(([1], 4 * np.ones(num_points_x - 5) / 3.0, [1])),
        np.concatenate(([0], -np.ones(num_points_x - 6) / 12.0, [0]))
    ]
    tx = diags(diagonals_x, offset)
    e2 = eye(num_points_y - 2)
    tx = kron(e2, tx) * hx

    offset = [-2 * (num_points_x - 2), -num_points_x + 2, 0, num_points_x - 2,
              2 * (num_points_x - 2)]
    diagonals_x = [
        np.concatenate(([0], -np.ones(dim - 2 * num_points_x + 2) / 12.0, [0])),
        np.concatenate(([1], 4 * np.ones(dim - num_points_x) / 3.0, [1])),
        np.concatenate(([-2], -2.5 * np.ones(dim), [-2])),
        np.concatenate(([1], 4 * np.ones(dim - num_points_x) / 3.0, [1])),
        np.concatenate(([0], -np.ones(dim - 2 * num_points_x + 2) / 12.0, [0]))
    ]
    ty = diags(diagonals_x, offset) * hy
    t_electron = -csr_matrix(tx + ty)
    left_u = eye(dim) + 0.5j * t_electron * dt
    right_u = eye(dim) - 0.5j * t_electron * dt
    return left_u, right_u


def calculate_next_psi():
    global psi
    psi = psi.flatten()
    psi, _ = linalg.bicgstab(left_h, right_h.dot(psi), x0=psi)
    psi = np.reshape(psi, (num_points_x - 2, num_points_y - 2))


t = 0
frames_count = 0

psi_left = calculate_initial_psi(x_c, y_c, ko_x, ko_y, sigma)
psi_right = calculate_initial_psi(-x_c, y_c, -ko_x, ko_y, sigma)
psi = psi_left + psi_right

left_h, right_h = calculate_evolution_matrices()

if visualize_with_matplotlib:
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    psi_plot = plt.imshow(np.absolute(psi).T, interpolation='spline16',
                          aspect='auto', cmap="hot", vmin=0, vmax=0.5 / np.e,
                          origin='lower', extent=[x_min, x_max, x_min, x_max])


    def update(frame):
        global psi_left, psi_right, t
        if frame % fps == 0:
            print("Starting {} second, t={:.2f}".format(frame // fps, t))
        for _ in range(time_steps_per_frame):
            t += dt
            calculate_next_psi()
        psi_plot.set_array(np.absolute(psi).T)
        return psi_plot,


    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps,
                         blit=False, repeat=True)

    if save_to_file:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'./2dimBumpFunction_{fps}fps_{num_seconds}seconds_{now}.mp4'
        anim.save(file_name, fps=fps, dpi=300, bitrate=-1)
    else:
        plt.show()
else:
    frame_count = 0
    surf = mlab.surf(x_g / x_max, y_g / y_max, np.absolute(psi) * np.e, vmin=0, vmax=0.5,
                     warp_scale="auto")


    @mlab.animate
    def anim():
        global t, psi_left, frame_count
        while frame_count < total_frames:
            frame_count += 1
            if frame_count % fps == 0:
                print("Starting {} second, t={:.2f}".format(frame_count // fps, t))
            for _ in range(time_steps_per_frame):
                t += dt
                calculate_next_psi()

            surf.mlab_source.scalars = np.absolute(psi) * np.e
            yield

    if save_to_file:
        pass
    else:
        anim()
        mlab.show()
