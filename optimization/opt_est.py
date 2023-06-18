import numpy as np
from scipy.optimize import differential_evolution

np.set_printoptions(precision=3, suppress=True)


w = np.array([1.4, 1.5, 16, 16])*2*np.pi
W = np.diag(w) ** 2


def cost(x):
    kd, ksd = x[0], x[1]
    kt = kd
    kst = ksd

    l_d = x[2]
    l_t = x[3]

    md = x[4]
    mt = md

    mv = x[5]
    jv = x[6]

    M = np.diag([mv, jv, md, mt])

    K = np.array([
        [ksd + kst, l_d * ksd - l_t * kst, -ksd, -kst],
        [l_d * ksd - l_t * kst, l_d ** 2 * ksd + l_t ** 2 * kst, -l_d * ksd, l_t * kst],
        [-ksd, -l_d * ksd, ksd + kd, 0],
        [-kst, l_t * kst, 0, kst + kt]
    ])

    M2 = np.vectorize(lambda x: 1 / np.sqrt(x) if x != 0 else 0)(M)

    K_til = M2 @ K @ M2

    return np.linalg.norm((K_til - W),2)


lower_bounds = [500, 500, .5, 1.25, 75, 500, 1500]
upper_bounds = [500*1e3, 70*1e3,  1.5,  2, 150, 1500, 2500]

res = differential_evolution(
    cost,
    bounds=list(zip(lower_bounds, upper_bounds)),
    maxiter=1000,
    popsize=70,
    tol=1e-7,
    disp=True,
    polish=False,
    mutation=(0.5, 1),
    recombination=0.9,
    strategy='best1bin',
    updating='deferred',
    init='sobol',
    workers=10,
    seed=None,
)

x = res.x

kd, ksd = x[0], x[1]
kt = kd
kst = ksd

l_d = x[2]
l_t = x[3]

md = x[4]
mt = md

mv = x[5]
jv = x[6]

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
