from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, PillowWriter

save_animation, save_as_gif = False, False

x_min, x_max, num_points_x = -10, 10, 200
y_min, y_max, num_points_y = -10, 10, 200
x, y = np.linspace(x_min, x_max, num_points_x), np.linspace(y_min, y_max, num_points_y)
xg, yg = np.meshgrid(x, y)
dx, dy = (x_max - x_min) / (num_points_x - 1), (y_max - y_min) / (num_points_y - 1)
dt = dx * dy

xo, kxo, sigma_x = 5, -1, 1
yo, kyo, sigma_y = 0, 0, 1

fps, dur_of_video, time_steps_per_frame = 20, 5, 5
total_frames = fps * dur_of_video


def calculate_wave_packet(time, x_c, kx, sx, y_c, ky, sy):
    psi_ = 0.25 * np.square(xg - x_c - 2j * kx * sx**2) / (sx**2 + 1j * time)
    psi_ = np.exp(1j*kx*xg - kx**2 * sx**2 - psi_) / np.sqrt(sx**2 + 1j * time)
    psi_x = np.power(sx**2/(2*np.pi), 0.25) * psi_
    psi_ = 0.25 * np.square(yg - y_c - 2j * ky * sy ** 2) / (sy ** 2 + 1j * time)
    psi_ = np.exp(1j * ky * yg - ky ** 2 * sy ** 2 - psi_) / np.sqrt(sy ** 2 + 1j * time)
    psi_y = np.power(sy ** 2 / (2 * np.pi), 0.25) * psi_
    return psi_x * psi_y


def calculate_psi(time, x_c, kx, sx, y_c, ky, sy):
    # return calculate_wave_packet(t, xo, kxo, sigma_x, yo, kyo, sigma_y) \
    #       + calculate_wave_packet(t, -xo, -kxo, sigma_x, yo, kyo, sigma_y)

    psi_ = np.zeros((num_points_x, num_points_y), dtype=np.complex128)
    for i in range(6):
        phi = 2 * np.pi * (i - 1) / 6
        xc = x_c * np.cos(phi) - y_c * np.sin(phi)
        yc = -x_c * np.sin(phi) + y_c * np.cos(phi)
        kx_ = kx * np.cos(phi) - ky * np.sin(phi)
        ky_ = -kxo * np.sin(phi) + ky * np.cos(phi)
        psi_ += calculate_wave_packet(time, xc, kx_, sx, yc, ky_, sy)
    return psi_

t = 0
w_max = np.power(2*np.pi*sigma_x**2, -0.25) * np.power(2*np.pi*sigma_y**2, -0.25)
psi = calculate_psi(t, xo, kxo, sigma_x, yo, kyo, sigma_y)

matplotlib.rc('animation', html='html5')  # TODO: check this
plt.style.use('dark_background')

fig = plt.figure(constrained_layout=True, figsize=(12, 8))
sub_figs = fig.subfigures(1, 2, wspace=0.01, hspace=0.01, width_ratios=[1., 2])
axs0 = sub_figs[0].subplots(2, 1)
axs1 = sub_figs[1].subplots(1, 1)
axs0[0].axis('off')
axs0[1].axis('off')
axs1.axis('off')
sub_figs[0].suptitle(r"$\Re(\psi)$ & $\Im(\psi)$", fontsize=20)
sub_figs[1].suptitle(r"|$\psi$| - Hexagonal Initial Formation", fontsize=20)

psi_plot = axs1.imshow(np.abs(psi), interpolation='spline16', aspect='auto',
                              cmap="hot", norm=PowerNorm(vmin=0, vmax=w_max, gamma=0.4),
                              origin='lower', extent=[x_min, x_max, x_min, x_max])
psi_plot_r = axs0[0].imshow(np.real(psi), interpolation='spline16', aspect='auto',
                              cmap="seismic", vmin=-w_max, vmax=w_max,
                              origin='lower', extent=[x_min, x_max, x_min, x_max])
psi_plot_i = axs0[1].imshow(np.imag(psi), interpolation='spline16', aspect='auto',
                              cmap="seismic", vmin=-w_max, vmax=w_max,
                              origin='lower', extent=[x_min, x_max, x_min, x_max])

current_frame = 0


def update(frame):
    global t, current_frame, psi

    if current_frame % fps == 0:
        print(f"@ {current_frame//fps} second ...")

    for _ in range(time_steps_per_frame):
        psi = calculate_psi(t, xo, kxo, sigma_x, yo, kyo, sigma_y)
        t += dt
    psi_plot.set_array(np.abs(psi))
    psi_plot_r.set_array(np.real(psi))
    psi_plot_i.set_array(np.imag(psi))

    current_frame += 1
    return [psi_plot, psi_plot_r, psi_plot_i]


anim = FuncAnimation(fig, update, frames=total_frames, blit=True, repeat=True, interval=1000/fps)

if save_animation:
    anim.save("./2dimGaussian.mp4", writer="ffmpeg", fps=fps, dpi=160, bitrate=-1,
              metadata={
                  "title": "2 dimensional Gaussian Wave Packets",
                  "artist": "Taher Amlaki",
                  "subject": "Gaussian Wave Packets"
              })
elif save_as_gif:
    writer = PillowWriter(fps=fps, metadata={
        "title": "2 dimensional Gaussian Wave Packets",
        "artist": "Taher Amlaki",
        "subject": "Quantum Mechanics"})
    anim.save('./2dimGaussian.gif', dpi=80, writer=writer)
else:
    plt.show()
