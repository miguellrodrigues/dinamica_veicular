import cvxpy as cvx
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

m_v = 1500
J_v = 2400
m_d = 100
m_t = 100

M = np.diag([m_v, J_v, m_d, m_t])

l_d  = 1.17
l_t  = 1.68

# # # # # # # # # # # # # # # # # # # # #
wo = np.array([.91, 1.13, 9.15, 9.15])*2*np.pi

w = np.array([1.2, 1.6, 15.2, 15.8])*2*np.pi
W = np.diag(w) ** 2

MI = np.vectorize(lambda x: 1 / x if x != 0 else 0)(M)

kd = cvx.Variable()
kt = kd

k_sd = cvx.Variable()
k_st = k_sd

K = cvx.bmat([
    [k_sd + k_st, l_d * k_sd - l_t * k_st, -k_sd, -k_st],
    [l_d * k_sd - l_t * k_st, l_d ** 2 * k_sd + l_t ** 2 * k_st, -l_d * k_sd, l_t * k_st],
    [-k_sd, -l_d * k_sd, k_sd + kd, 0],
    [-k_st, l_t * k_st, 0, k_st + kt]
])

K_til = MI@K
P = cvx.Variable((4, 4), PSD=True)

constraints = [
    kd >= 0,
    k_sd >= 0,

    kd <= 1e6,
    k_sd <= 1e5,
]

obj = cvx.Minimize(
    (cvx.trace(K_til-W))**2
)

prob = cvx.Problem(obj, constraints)
prob.solve(solver="MOSEK", verbose=False, )

print(' ')
print("status:", prob.status)

K_til = K_til.value
wn, P = np.linalg.eig(K_til)

wn = np.sqrt(wn)

print(' ')
print(wo/2/np.pi)
print(' ')
print(kd.value)
print(kt.value)
print(k_sd.value)
print(k_st.value)
print(' ')
print(wn/2/np.pi)
print(' ')