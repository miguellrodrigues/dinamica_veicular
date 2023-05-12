import numpy as np
from scipy.optimize import differential_evolution

np.set_printoptions(precision=3, suppress=True)

l_d = 1.17
l_t = 1.68

w = np.array([1.2, 1.6, 15.2, 15.8])*2*np.pi
W = np.diag(w) ** 2

md = 100
mt = md
mv = 1500
jv = 2400


def cost(x):
    kd, ksd = x[0], x[1]
    kt = kd
    kst = ksd

    M = np.diag([mv, jv, md, mt])

    K = np.array([
        [ksd + kst, l_d * ksd - l_t * kst, -ksd, -kst],
        [l_d * ksd - l_t * kst, l_d ** 2 * ksd + l_t ** 2 * kst, -l_d * ksd, l_t * kst],
        [-ksd, -l_d * ksd, ksd + kd, 0],
        [-kst, l_t * kst, 0, kst + kt]
    ])

    M2 = np.vectorize(lambda x: 1 / np.sqrt(x) if x != 0 else 0)(M)

    K_til = M2 @ K @ M2

    return np.trace((K_til - W)**2)


lower_bounds = [500, 500]
upper_bounds = [1e6, 1e5]

res = differential_evolution(
    cost,
    bounds=list(zip(lower_bounds, upper_bounds)),
    maxiter=10000,
    popsize=100,
    tol=1e-7,
    disp=True,
    polish=False,
    mutation=(0.5, 1),
    recombination=0.4,
    strategy='rand1bin',
    updating='deferred',
    init='sobol',
    workers=10,
    seed=None,
)

x = res.x

kd, ksd = x[0], x[1]
kt = kd
kst = ksd

M = np.diag([mv, jv, md, mt])

K = np.array([
    [ksd + kst, l_d * ksd - l_t * kst, -ksd, -kst],
    [l_d * ksd - l_t * kst, l_d ** 2 * ksd + l_t ** 2 * kst, -l_d * ksd, l_t * kst],
    [-ksd, -l_d * ksd, ksd + kd, 0],
    [-kst, l_t * kst, 0, kst + kt]
])

M2 = np.vectorize(lambda x: 1 / np.sqrt(x) if x != 0 else 0)(M)

K_til = M2 @ K @ M2
wn, P = np.linalg.eig(K_til)

print(' ')
print(x)
print(' ')
print((wn ** .5)/2/np.pi)
print(' ')
