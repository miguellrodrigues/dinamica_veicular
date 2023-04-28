import numpy as np
import sympy as sp
from sympy import Matrix, symbols, eye, zeros, solve, Eq, re, Symbol, atan, pi, exp, sin, cos, sqrt
from numpy import array, arange, rad2deg, angle


def modal_analysis(M, K, C, X0, V0, f, B, wf=0.):
    N = M.shape[0]
    t = symbols('t')

    M_minus_half = zeros(N, N)
    M_half = zeros(N, N)

    for i in range(N):
        M_minus_half[i, i] = (1/sqrt(M[i, i])).evalf()
        M_half[i, i] = (sqrt(M[i, i])).evalf()

    # Calculando Ktil
    K_til = M_minus_half @ K @ M_minus_half

    # Calculando Ctil
    C_til = M_minus_half @ C @ M_minus_half

    # Encontrando as frequências naturais
    lmbd = symbols('lambda')

    expr_wn = (K_til - lmbd*eye(N))

    eig_wn = solve(expr_wn.det(), lmbd)
    wn = [re(sqrt(eig_wn[i])) for i in range(N)]

    # Encontrando os auto-vetores normalizados no unitário
    norm_syms = [Symbol(f'norm_{i}') for i in range(N)]

    X = Matrix(norm_syms)

    norm_eq = Eq(
        (X.T@X)[0],
        1
    )

    P = zeros(N, N)
    for i in range(N):
        sys_wn = expr_wn.subs(lmbd, eig_wn[i])
        sys = sys_wn @ X

        sol = solve(
            [sys[j] for j in range(N-1)] + [norm_eq],
            norm_syms,
            dict=True
        )[0]

        v = array([
            sol[norm_syms[j]] for j in range(N)
        ])

        P[:, i] = v

    # Matriz P

    diag_  = P.T @ C_til @ P
    zetas  = [diag_[i, i] / (2*wn[i]) for i in range(N)]

    wds    = [wn[i] * sqrt(1 - zetas[i]**2) for i in range(N)]

    # matrix spectral de P
    S = M_minus_half @ P
    S_inv = S.inv()

    # condições iniciais inversas modais
    r_0 = S_inv @ X0
    r_dot_0 = S_inv @ V0

    r = []
    F = P.T @ M_minus_half @ B @ f

    for i in range(N):
        w = wn[i]
        wd = wds[i]
        zeta = zetas[i]
        
        r_zero = r_0[i]
        rdot_zero = r_dot_0[i]

        Ai = sqrt(
            ((r_zero * wd)**2 + (rdot_zero + zeta*w*r_zero)**2) / wd**2
        )

        phi_i = atan(
            (r_zero*wd) / (rdot_zero + zeta*w*r_zero)
        )

        if rdot_zero == 0:
            phi_i = pi/2
        
        expr_r = Ai*exp(-zeta*w*t)*sin(wd*t + phi_i)

        f0 = F[i]
        A0 = f0 / sqrt( (w**2 - wf**2)**2 + (2*zeta*w*wf)**2 )
        theta = atan( (2*zeta*w*wf) / (w**2 - wf**2) )
        
        if (w**2 - wf**2) == 0:
            theta = pi/2

        expr_r += A0*cos(wf*t - theta)
        
        r.append(
            expr_r
        )

    
    R = Matrix(r)
    # Retornando ao dominio fisico
    x_t = S @ R

    x = [sp.lambdify(t, x_t[i]) for i in range(N)]

    wn    = np.array(wn)
    zetas = np.array(zetas)
    wds   = np.array(wds)

    return x, wn, zetas, wds, P, S, R, t, sp.simplify(x_t)


def simulation(x, tf, ts):
    time = arange(0., tf, ts)

    x_t = [x[i](time) for i in range(len(x))]

    return time, x_t


def fft(signal, sampling_time):
    fft = np.fft.fft(signal)
    N   = signal.shape[0]

    f = np.fft.fftfreq(len(signal), sampling_time)

    K = N // 2

    freqs = f[:K]*2*np.pi
    amplitudes = np.abs(fft)[:K] * (1 / N)

    phase = np.rad2deg(np.angle(fft)[:K])

    return freqs, amplitudes, phase

