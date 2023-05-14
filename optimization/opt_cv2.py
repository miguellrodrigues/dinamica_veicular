import cvxpy as cvx
import numpy as np
from numpy import linalg

np.set_printoptions(precision=3, suppress=True)

m_v = 1500
J_v = 2400
m_d = 100
m_t = 100

M = np.diag([m_v, J_v, m_d, m_t])

l_d = 1
l_t = 1.2

# # # # # # # # # # # # # # # # # # # # #
wo = np.array([5.72, 7.153, 57.474, 57.491])

w = np.array([5.72, 7.153, 57.474, 57.491])
W = np.diag(w) ** 2

kd = cvx.Variable()
kt = cvx.Variable()

k_sd = cvx.Variable()
k_st = cvx.Variable()

K = cvx.bmat([
    [k_sd + k_st, l_d * k_sd - l_t * k_st, -k_sd, -k_st],
    [l_d * k_sd - l_t * k_st, l_d ** 2 * k_sd + l_t ** 2 * k_st, -l_d * k_sd, l_t * k_st],
    [-k_sd, -l_d * k_sd, k_sd + kd, 0],
    [-k_st, l_t * k_st, 0, k_st + kt]
])


M_ = np.linalg.inv(M)
K_til = M_@K

constraints = [
    cvx.diag(K_til) == w,

]

obj = cvx.Minimize(0)

prob = cvx.Problem(obj, constraints)
prob.solve(solver="MOSEK", verbose=True)

print(' ')
print("status:", prob.status)

K_til = K_til.value
wn, P = np.linalg.eig(K_til)

wn = np.sqrt(wn)

print(' ')
print(wo / (2 * np.pi))
print(' ')
print(kd.value)
print(kt.value)
print(k_sd.value)
print(k_st.value)
print(' ')
print(wn / (2 * np.pi))
print(' ')
