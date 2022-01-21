from datetime import datetime

import numpy as np
from scipy.sparse import linalg
from scipy.sparse import diags, eye, csr_matrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

save_animation, save_as_gif = False, True

x_min, x_max, num_points = -10, 10, 1000
grid = np.linspace(x_min, x_max, num_points)
dx = (x_max - x_min) / (num_points - 1)
dt = dx**2

xo, sigma = 0, 1

fps, dur_of_video, time_steps_per_frame = 10, 5, 15
total_frames = fps * dur_of_video


def calculate_initial_psi(time, x_c, kx, s):
    r = np.absolute(grid - xo)
    den = (1 - np.square(r / s))
    return (r < s) \
           * np.exp(np.divide(-1, den, out=np.zeros_like(den), where=den > 0)) \
           * np.exp(1j * kx * (grid - x_c))


def calculate_evolution_matrices():
    hx = 1 / np.square(dx)
    dim = (num_points - 2)
    offset = [-2, -1, 0, 1, 2]
    diagonals_x = [
        np.concatenate(([0], -np.ones(num_points - 6) / 12.0, [0])),
        np.concatenate(([1], 4 * np.ones(num_points - 5) / 3.0, [1])),
        np.concatenate(([-2], -2.5 * np.ones(num_points - 4), [-2])),
        np.concatenate(([1], 4 * np.ones(num_points - 5) / 3.0, [1])),
        np.concatenate(([0], -np.ones(num_points - 6) / 12.0, [0]))
    ]
    tx = diags(diagonals_x, offset) * hx
    t_electron = -csr_matrix(tx)
    left_u = eye(dim) + 0.5j * t_electron * dt
    right_u = eye(dim) - 0.5j * t_electron * dt
    return left_u, right_u


def calculate_next_psi():
    global not_moving_psi, left_h, right_h
    for _ in range(time_steps_per_frame):
        not_moving_psi, _ = linalg.bicgstab(left_h, right_h.dot(not_moving_psi), x0=not_moving_psi)


t = 0
w_max = 1/np.e
not_moving_psi = calculate_initial_psi(t, 0, 0, sigma)[1:-1]
grid = grid[1:-1]
left_h, right_h = calculate_evolution_matrices()

matplotlib.rc('animation', html='html5')  # TODO: check this
plt.style.use('dark_background')

fig, axs = plt.subplots(1, figsize=(12, 8))
plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1, wspace=0.025, hspace=0)
fig.suptitle("1-dim Bump Wave Packet", fontsize=20, color='w', x=0.55, y=0.99)

axs.set_ylabel(r"|$\psi(x, t={:.2f})$|".format(t), fontdict={"fontsize": 25})
axs.set_title(r"$k_o=0$", fontsize=20, x=0.1, y=0.935)
axs.set_xlim(x_min, x_max)
axs.set_xticks(np.linspace(x_min, x_max, 5))
axs.set_ylim(-1.1*w_max, 1.1*w_max)
axs.set_yticks([])

not_moving_plot, = axs.plot(grid, np.abs(not_moving_psi), '-w', linewidth=3, label=r"|$\psi|$", zorder=3)
not_moving_plot_r, = axs.plot(grid, np.real(not_moving_psi), '-r', linewidth=1, label=r"$\Re(\psi)$", zorder=2)
not_moving_plot_i, = axs.plot(grid, np.imag(not_moving_psi), '-b', linewidth=1, label=r"$\Im(\psi)$", zorder=1)

for side in ["top", "left", "right", "bottom"]:
    axs.spines[side].set_linewidth(1)
axs.set_xlabel(r"$X$", fontdict={"fontsize": 15})
axs.legend(loc=3)
current_frame = 0


def update(frame):
    global t, axs, current_frame

    if current_frame % fps == 0:
        print(f"@ {current_frame//fps} second ...")

    calculate_next_psi()

    not_moving_plot.set_ydata(np.abs(not_moving_psi))
    not_moving_plot_r.set_ydata(np.real(not_moving_psi))
    not_moving_plot_i.set_ydata(np.imag(not_moving_psi))

    axs.set_ylabel(r"|$\psi(x, t={:.2f}\sigma^2)$|".format(t), fontdict={"fontsize": 20})

    t += dt
    current_frame += 1

    return [not_moving_plot, not_moving_plot_r, not_moving_plot_i]


anim = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=True, interval=1000/fps)

if save_animation:
    anim.save("./1dimBump.mp4", writer="ffmpeg", fps=fps, dpi=160, bitrate=-1,
              metadata={
                  "title": "1 dimensional Bump Wave Packets",
                  "artist": "Taher Amlaki",
                  "subject": "Quantum Wave Packets"
              })
elif save_as_gif:
    writer = PillowWriter(fps=fps, metadata={
        "title": "1 dimensional Bump Wave Packets",
        "artist": "Taher Amlaki",
        "subject": "Quantum Mechanics"})
    anim.save('./1dimBump.gif', dpi=80, writer=writer)
else:
    plt.show()
