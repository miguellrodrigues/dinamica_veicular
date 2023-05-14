# import numpy as np
# import matplotlib.pyplot as plt
# import scienceplots

# plt.style.use([
#     'retro',
#     'grid'
# ])

# M_min = 750
# M_max = 3000
# K_min = 1000
# K_max = 100000

# M = np.linspace(M_min, M_max, 100)
# K = np.linspace(K_min, K_max, 100)

# Mmesh, Kmesh = np.meshgrid(M, K)

# omega = np.sqrt(Kmesh/Mmesh)

# plt.figure(figsize=(8, 6))
# plt.contourf(Mmesh/1000, Kmesh/1000, omega)
# plt.colorbar()
# plt.xlabel('Massa (t)')
# plt.ylabel('Constante de Rigidez (kN/m)')
# plt.title('Frequência natural (rad/s)')

# plt.savefig('relations.png', dpi=300)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M_min = 750
M_max = 3000
K_min = 1000
K_max = 100000

M = np.linspace(M_min, M_max, 1000)
K = np.linspace(K_min, K_max, 1000)

Mmesh, Kmesh = np.meshgrid(M, K)

omega = np.sqrt(Kmesh/Mmesh)

# # 

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Mmesh/1000., Kmesh/1000., omega, cmap='viridis')
ax.set_xlabel('Massa (t)')
ax.set_ylabel('Rigidez (kN/m)')
ax.set_zlabel('Frequência natural (rad/s)')

# set azimuth angle
ax.view_init(azim=-45)

plt.savefig('relations.png', dpi=300)
plt.show()