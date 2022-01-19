from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fps, num_seconds, dt = 30, 15, 0.005
total_frames = fps * num_seconds

x_min, x_max, num_points = -20, 20, 8000
x_c, ko, sigma = 3, 2, 1.0
grid = np.linspace(x_min, x_max, num_points)
dx = (x_max - x_min) / (num_points - 1)
w2_max = 1 / np.power(2 * np.pi * sigma ** 2, 0.5)
w_factor = np.power(0.5 * sigma ** 2 / np.pi, 0.25)


def calculate_psi(xo, k, s):
    s2 = s**2
    psi = np.exp(-0.25 * np.square(grid - xo - 2j * k * s2) / (s2 + 1j * t))
    psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * t)
    psi = np.power(0.5 * s2 / np.pi, 0.25) * psi
    return psi


def calculate_wave_packets():
    not_moving_psi = calculate_psi(0, 0, sigma)
    moving_psi = calculate_psi(0, ko, sigma)
    interference_psi = calculate_psi(-x_c, ko, sigma) + calculate_psi(x_c, -ko, sigma)
    asymmetric_interference_psi = calculate_psi(-x_c, ko, 1.5*sigma) \
                                  + calculate_psi(x_c, -ko, 0.5 * sigma)
    return not_moving_psi, moving_psi, interference_psi, asymmetric_interference_psi


def init_plot():
    global fig, axs
    for ax in axs.flatten():
        ax.spines["right"].set_linewidth(3)
        ax.spines["top"].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.set_xlabel("x")

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0, 1.05 * w2_max])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.legend(loc="upper left")
    axs[1, 0].set_ylim([0, 4 * w2_max])
    axs[1, 1].set_ylim([0, 4 * w2_max])
    axs[0, 0].set_ylabel(r'|$\psi(x, t)$$\|^2$')

    return zero_plot, linear_plot, interference_plot


def update(*args):
    global t
    axs[0, 0].set_ylabel(r'|$\psi(x, t={0:.1f})$$\|^2$'.format(t))
    static_packet_, moving_packet_, interference_, asym_interference_ = calculate_wave_packets()

    zero_plot.set_ydata(np.square(np.absolute(static_packet_)))
    linear_plot.set_ydata(np.square(np.absolute(moving_packet_)))
    interference_plot.set_ydata(np.square(np.absolute(interference_)))
    asym_interference_plot.set_ydata(np.square(np.absolute(asym_interference_)))
    t += dt
    return zero_plot, linear_plot, interference_plot, asym_interference_plot


t = 0
matplotlib.rc('axes', labelsize=20)

fig, axs = plt.subplots(2, 2, figsize=(12, 12), facecolor="k")

static_packet, moving_packet, interference, asym_interference = calculate_wave_packets()
zero_plot, = axs[0, 0].plot(grid, np.square(np.absolute(static_packet)), '-r',
                            linewidth=3, label=r"k=0$")

linear_plot, = axs[0, 1].plot(grid, np.square(np.absolute(moving_packet)), '-b',
                              linewidth=3, label=r"$k={}$".format(ko))

interference_plot, = axs[1, 0].plot(grid, np.square(np.absolute(interference)), '-g',
                                    linewidth=3, label=r"Wave packets interference")

asym_interference_plot, = axs[1, 1].plot(grid, np.square(np.absolute(asym_interference)), '-m',
                                         linewidth=3, label=r"Asymmetric Wave packets interference")

anim = FuncAnimation(fig, update, init_func=init_plot, frames=total_frames, interval=1000 / fps,
                     blit=False, repeat=True)

# now = datetime.now().strftime('%Y%m%d_%H%M%S')
# file_name = f'./1dimWavePhaseDependency_{fps}fps_{num_seconds}seconds_{now}.mp4'
# anim.save(file_name, fps=fps, dpi=160, bitrate=-1)
plt.show()
