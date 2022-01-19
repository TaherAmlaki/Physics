from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation


fps, num_seconds, dt = 30, 15, 0.1
total_frames = fps * num_seconds

x_min, x_max, num_points = -15, 15, 400
x_c, ko_x, sigma_x = 3, 1, 0.5
y_c, ko_y, sigma_y = 0, 0, 1.5

x, y = np.linspace(x_min, x_max, num_points), np.linspace(x_min, x_max, num_points)
x_g, y_g = np.meshgrid(x, y)

w2_max = 1 / np.power(4 * np.pi**2 * sigma_x**2 * sigma_y**2, 0.5)
w_factor = np.power(0.25 * sigma_x ** 2 * sigma_y**2 / np.pi**2, 0.25)


def calculate_psi(xo, yo, kx, ky, sx, sy):
    s2 = sx ** 2
    psi = np.exp(-0.25 * np.square(x_g - xo - 2j * kx * s2) / (s2 + 1j * t))
    psi = psi * np.exp(1j * kx * xo - kx ** 2 * s2) / np.sqrt(s2 + 1j * t)
    psi_x = np.power(0.5 * s2 / np.pi, 0.25) * psi

    s2 = sy ** 2
    psi = np.exp(-0.25 * np.square(y_g - yo - 2j * ky * s2) / (s2 + 1j * t))
    psi = psi * np.exp(1j * ky * yo - ky ** 2 * s2) / np.sqrt(s2 + 1j * t)
    psi_y = np.power(0.5 * s2 / np.pi, 0.25) * psi

    return psi_x * psi_y


def calculate_wave_packets():
    psi = calculate_psi(-x_c, -y_c, ko_x, ko_y, sigma_x, sigma_y) \
          + calculate_psi(x_c, y_c, -ko_x, ko_y, sigma_x, sigma_y)
    return np.square(np.absolute(psi)), np.real(psi), np.imag(psi)


def update(frame):
    global t, frames_count
    w2, w_real, w_imag = calculate_wave_packets()

    if frames_count % fps == 0:
        print("starting {} second, t={:.2f}".format(frames_count//fps, t))
    frames_count += 1

    im_interference.set_array(w2)
    im_real.set_array(w_real)
    im_imag.set_array(w_imag)
    t += dt
    return im_interference, im_real, im_imag


t = 0
frames_count = 0
w2_, w_real, w_imag = calculate_wave_packets()
fig = plt.figure(constrained_layout=True, figsize=(12, 12))
subfigs = fig.subfigures(1, 2, wspace=0.01, hspace=0.01, width_ratios=[1., 1.5])
axs0 = subfigs[0].subplots(2, 1)
subfigs[0].set_facecolor('0.9')
subfigs[1].set_facecolor('0.9')
# subfigs[0].suptitle('Real and Imaginary Parts', fontsize=20)
axs1 = subfigs[1].subplots(1, 1)
# subfigs[1].suptitle('Wave Function Module', fontsize=20)

axs0[0].axis('off')
axs0[1].axis('off')
axs1.axis('off')

im_interference = axs1.imshow(w2_, interpolation='spline16', aspect='auto',
                              cmap="hot", norm=PowerNorm(vmin=0, vmax=0.01, gamma=0.2),
                              origin='lower', extent=[x_min, x_max, x_min, x_max])

im_real = axs0[0].imshow(w_real, interpolation='spline16', aspect='auto',
                         cmap="seismic", vmin=-0.10, vmax=0.10,
                         origin='lower', extent=[x_min, x_max, x_min, x_max])

im_imag = axs0[1].imshow(w_imag, interpolation='spline16', aspect='auto',
                         cmap="seismic", vmin=-0.10, vmax=0.10,
                         origin='lower', extent=[x_min, x_max, x_min, x_max])

anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps,
                     blit=False, repeat=True)

# now = datetime.now().strftime('%Y%m%d_%H%M%S')
# file_name = f'./1dimWavePhaseDependency_{fps}fps_{num_seconds}seconds_{now}.mp4'
# anim.save(file_name, fps=fps, dpi=160, bitrate=-1)
plt.show()
