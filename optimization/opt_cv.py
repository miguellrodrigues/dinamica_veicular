import numpy as np
import cvxpy as cvx
from   numpy import array, linalg

np.set_printoptions(precision=3, suppress=True)

m_v = 1500
J_v = 2400
m_d = 100
m_t = 100

M = np.diag([m_v, J_v, m_d, m_t])

l_d  = 1
l_t  = 1.2

# # # # # # # # # # # # # # # # # # # # #
wo = np.array([5.72, 7.153, 57.474, 57.491])

w = np.array([5.72, 7.153, 57.474, 57.491])*2.5
W = np.diag(w)**2

M2 = np.vectorize(lambda x: 1/np.sqrt(x) if x != 0 else 0)(M)
M_2 = np.vectorize(lambda x: np.sqrt(x) if x != 0 else 0)(M)

kd   = cvx.Variable()
kt   = cvx.Variable()

k_sd = cvx.Variable()
k_st = cvx.Variable()

K = cvx.bmat([
    [k_sd+k_st, l_d*k_sd-l_t*k_st, -k_sd, -k_st],
    [l_d*k_sd-l_t*k_st, l_d**2*k_sd+l_t**2*k_st, -l_d*k_sd, l_t*k_st],
    [-k_sd, -l_d*k_sd, k_sd+kd, 0],
    [-k_st, l_t*k_st, 0, k_st+kt]
])

K_til = M2@K@M2

constraints = [
    kd   >= 1e4,
    kt   >= 1e4,
]

obj = cvx.Minimize(
    cvx.norm(K_til-W)
)

prob = cvx.Problem(obj, constraints)
prob.solve(solver="MOSEK", verbose=False)

print(' ')
print("status:", prob.status)

K_til = K_til.value
wn, P = np.linalg.eig(K_til)

wn = np.sqrt(wn)

print(' ')
print(wo/(2*np.pi))
print(' ')
print(kd.value)
print(kt.value)
print(k_sd.value)
print(k_st.value)
print(' ')
print(wn/(2*np.pi))
print(' ')
