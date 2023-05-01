import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


plt.style.use([
    'grid',
    'retro'
])


wn     = 2.86986454420447
wn_hat = 8.6095936326134

zeta = np.arange(.1, .5, .1)

s = ct.tf('s')

mags = []
phases = []
omegas = []

fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8), dpi=200)

for z in zeta:
    G1 = wn_hat**2 / (s**2 + wn_hat**2)
    G2 = (2*z*wn*s + wn**2) / (s**2 + 2*z*wn*s + wn**2)

    G = G1*G2
    mag, phase, omega = ct.bode(G, plot=False)
    
    ax[0].loglog(omega, mag, label=f'$\\zeta={z:.1f}$')
    ax[1].semilogx(omega, np.rad2deg(phase), label=f'$\\zeta={z:.1f}$')

    ax[0].set_ylabel('Magnitude (dB)')

    ax[1].set_xlabel('Frequency (rad/s)')
    ax[1].set_ylabel('Phase (deg)')
    ax[1].set_yticks([-180, -90, 0, 90, 180])

ax[0].legend()
plt.show()