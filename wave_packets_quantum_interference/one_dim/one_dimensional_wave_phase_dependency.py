from datetime import datetime
import numpy as np
from scipy.sparse import diags, eye, csr_matrix
import scipy.sparse.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fps, num_seconds, dt = 30, 15, 0.005
total_frames = fps * num_seconds

x_min, x_max, num_points, finite_diff_order = -20, 20, 8000, 4
x_c, ko, sigma = 0, 3, 1.0
grid = np.linspace(x_min, x_max, num_points)[1:-1]
dx = (x_max - x_min) / (num_points - 1)
w_max = 1 / np.power(2 * np.pi * sigma ** 2, 0.25)
w_factor = np.power(0.5 * sigma ** 2 / np.pi, 0.25)


class WavePackets:
    def __init__(self):
        self.zero_phase_wave = None
        self.linear_phase_wave = None
        self.quadratic_phase_wave = None
        self.sin_phase_wave = None
        self.left_u = None
        self.right_u = None
        self.w_factor = 1 / np.power(2 * np.pi * sigma ** 2, 0.25)

    def init(self):
        self.calculate_evolution_matrices()
        self.calculate_initial_wave_packets()

    def calculate_next_psi_functions(self):
        self.zero_phase_wave = linalg.spsolve(
            self.left_u, self.right_u.dot(self.zero_phase_wave)
        )
        self.linear_phase_wave = linalg.spsolve(
            self.left_u, self.right_u.dot(self.linear_phase_wave)
        )
        self.quadratic_phase_wave = linalg.spsolve(
            self.left_u, self.right_u.dot(self.quadratic_phase_wave)
        )
        self.sin_phase_wave = linalg.spsolve(
            self.left_u, self.right_u.dot(self.sin_phase_wave)
        )

    def calculate_initial_wave_packets(self):
        psi = self.w_factor * np.exp(-0.25 * np.square(grid - x_c) / sigma ** 2)
        self.zero_phase_wave = psi
        self.linear_phase_wave = np.exp(1j * ko * grid) * psi
        self.quadratic_phase_wave = np.exp(1j * np.square(grid)) * psi
        self.sin_phase_wave = np.exp(1j * np.cos(2*np.pi*grid)) * psi

    def calculate_evolution_matrices(self):
        hx = 1 / np.square(dx)
        if finite_diff_order == 2:
            offset = [-1, 0, 1]
            diagonals_x = [
                np.ones(num_points - 3),
                -2 * np.ones(num_points - 2),
                np.ones(num_points - 3)
            ]
        else:
            offset = [-2, -1, 0, 1, 2]
            diagonals_x = [
                np.concatenate(([0], -np.ones(num_points - 6) / 12.0, [0])),
                np.concatenate(([1], 4 * np.ones(num_points - 5) / 3.0, [1])),
                np.concatenate(([-2], -2.5 * np.ones(num_points - 4), [-2])),
                np.concatenate(([1], 4 * np.ones(num_points - 5) / 3.0, [1])),
                np.concatenate(([0], -np.ones(num_points - 6) / 12.0, [0]))
            ]
        tx = -csr_matrix(diags(diagonals_x, offset)) * hx
        self.left_u = eye(num_points - 2) + 0.5j * tx * dt
        self.right_u = eye(num_points - 2) - 0.5j * tx * dt


def init_plot():
    global fig, axs
    for ax in axs.flatten():
        ax.spines["right"].set_linewidth(3)
        ax.spines["top"].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.set_xlabel("x")

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0, 1.05 * np.square(w_max)])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.legend(loc="upper left")
    axs[0, 0].set_ylabel(r'|$\psi(x, t)$$\|^2$')
    return zero_plot, linear_plot, quadratic_plot, sin_plot


def update(*args):
    global t
    axs[0, 0].set_ylabel(r'|$\psi(x, t={0:.1f})$$\|^2$'.format(t))
    waves.calculate_next_psi_functions()
    zero_plot.set_ydata(np.square(np.absolute(waves.zero_phase_wave)))
    linear_plot.set_ydata(np.square(np.absolute(waves.linear_phase_wave)))
    quadratic_plot.set_ydata(np.square(np.absolute(waves.quadratic_phase_wave)))
    sin_plot.set_ydata(np.square(np.absolute(waves.sin_phase_wave)))
    t += dt
    return zero_plot, linear_plot, quadratic_plot, sin_plot


t = 0
matplotlib.rc('axes', labelsize=20)

waves = WavePackets()
waves.init()

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

zero_plot, = axs[0, 0].plot(grid, np.square(np.absolute(waves.zero_phase_wave)), '-r',
                            linewidth=3, label=r"$\alpha(x)=0$")

linear_plot, = axs[0, 1].plot(grid, np.square(np.absolute(waves.linear_phase_wave)), '-b',
                              linewidth=3, label=r"$\alpha(x)=x$")

quadratic_plot, = axs[1, 0].plot(grid, np.square(np.absolute(waves.quadratic_phase_wave)), '-g',
                                 linewidth=3, label=r"$\alpha(x)=x^2$")

sin_plot, = axs[1, 1].plot(grid, np.square(np.absolute(waves.sin_phase_wave)), '-m',
                           linewidth=3, label=r"$\alpha(x)=cos(2\pi x)$")

anim = FuncAnimation(fig, update, init_func=init_plot, frames=total_frames, interval=1000 / fps,
                     blit=False, repeat=True)

# now = datetime.now().strftime('%Y%m%d_%H%M%S')
# file_name = f'./1dimWavePhaseDependency_{fps}fps_{num_seconds}seconds_{now}.mp4'
# anim.save(file_name, fps=fps, dpi=160, bitrate=-1)
plt.show()
