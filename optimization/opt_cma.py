import numpy as np
from scipy.optimize import differential_evolution

np.set_printoptions(precision=3, suppress=True)

# l_d = 1.17
l_t = 1.68

w = np.array([5.72, 7.153, 57.474, 57.491])
W = np.diag(w)**2

mv = 1500
jv = 2400

def cost(x):
    md, mt           = x[0], x[1]
    kd, kt, ksd, kst = x[2], x[3], x[4], x[5]
    l_d              = x[6]

    M = np.diag([mv, jv, md, mt])

    K = np.array([
        [ksd+kst, l_d*ksd-l_t*kst, -ksd, -kst],
        [l_d*ksd-l_t*kst, l_d**2*ksd+l_t**2*kst, -l_d*ksd, l_t*kst],
        [-ksd, -l_d*ksd, ksd+kd, 0],
        [-kst, l_t*kst, 0, kst+kt]
    ])

    M2 = np.vectorize(lambda x: 1/np.sqrt(x) if x != 0 else 0)(M)

    K_til = M2@K@M2

    _, P  = np.linalg.eig(K_til)
    L     = P.T@K_til

    return np.linalg.norm(L-W, 2)


lower_bounds = [50 ,  50, 500, 500, 500, 500, 1]
upper_bounds = [200, 200, 1e6, 1e6, 1e6, 1e6, 2]

res = differential_evolution(
    cost,
    bounds=list(zip(lower_bounds, upper_bounds)),
    maxiter=10000,
    popsize=100,
    tol=1e-7,
    disp=True,
    polish=False,
    mutation=(0.5, 1),
    recombination=0.7,
    strategy='rand1bin',
    updating='deferred',
    init='sobol',
    x0=[100, 100, 300000, 300000, 20000, 20000, 1.5],
    workers=10,
    seed=None,
)

x = res.x

md, mt           = x[0], x[1]
kd, kt, ksd, kst = x[2], x[3], x[4], x[5]
l_d              = x[6]

M = np.diag([mv, jv, md, mt])

K = np.array([
    [ksd+kst, l_d*ksd-l_t*kst, -ksd, -kst],
    [l_d*ksd-l_t*kst, l_d**2*ksd+l_t**2*kst, -l_d*ksd, l_t*kst],
    [-ksd, -l_d*ksd, ksd+kd, 0],
    [-kst, l_t*kst, 0, kst+kt]
])

M2 = np.vectorize(lambda x: 1/np.sqrt(x) if x != 0 else 0)(M)

K_til  = M2@K@M2
wn, P  = np.linalg.eig(K_til)

print(' ')
print(x)
print(' ')
print(wn**.5)
print(' ')