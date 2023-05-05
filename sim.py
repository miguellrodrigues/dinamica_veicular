import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import vibration_toolbox as vtb

m_v = 1500
J_v = 2400
m_d = 100
m_t = 100

M = np.diag([m_v, J_v, m_d, m_t])

c_sd = 500
c_st = 500

l_d  = 1.17
l_t  = 1.68

C = array([
    [c_sd+c_st, l_d*c_sd-l_t*c_st, -c_sd, -c_st],
    [l_d*c_sd-l_t*c_st, l_d**2*c_sd+l_t**2*c_st, -l_d*c_sd, l_t*c_st],
    [-c_sd, -l_d*c_sd, c_sd, 0],
    [-c_st, l_t*c_st, 0, c_st]
])

k_sd = 30000
k_st = 30000
kd   = 300000
kt   = 300000

K = array([
    [k_sd+k_st, l_d*k_sd-l_t*k_st, -k_sd, -k_st],
    [l_d*k_sd-l_t*k_st, l_d**2*k_sd+l_t**2*k_st, -l_d*k_sd, l_t*k_st],
    [-k_sd, -l_d*k_sd, k_sd+kd, 0],
    [-k_st, l_t*k_st, 0, k_st+kt]
])


sys = vtb.VibeSystem(M, C, K)

tf = 60
ts = 0.01
t = np.arange(0, tf, ts)

F = np.zeros((len(t), 4))

F[:, 2] = np.sin(2*np.pi*0.5*t)*kd
F[:, 3] = np.sin(2*t)*kt

x0 = np.array([
    1, 0, 0, 0, 0, 0, 0, 0
])

sys.plot_freq_response(0,1)

plt.show()