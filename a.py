import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use([
    'retro',
    'grid'
])

M_min = 100
M_max = 1000
K_min = 100
K_max = 10000

M = np.linspace(M_min, M_max, 100)
K = np.linspace(K_min, K_max, 100)

Mmesh, Kmesh = np.meshgrid(M, K)

omega = np.sqrt(Kmesh/Mmesh)

plt.figure(figsize=(8, 6))
plt.contourf(Mmesh, Kmesh, omega, cmap='plasma')
plt.colorbar()
plt.xlabel('Massa (m)')
plt.ylabel('Constante de Rigidez (k)')
plt.title('Frequência natural sqrt(k/m)')

plt.savefig('relations.png', dpi=300)
plt.show()