import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spectral import chebyshev as cv
from linalg import decomposition as decomp

# TODO: figure out rational chebyshev polynomials

# problem parameters
r = 0.03
sigma = 0.5
K = 1

# solution parameters
dt = 0.01

scale = 5
center = np.log(K)
x_left = -scale + center
x_right = scale + center

N1, N2 = 64, 64   # can differ; more points can go wherever needs it
x_mid = np.log(K)

D1, xi1 = cv.chebyshev_diff_matrix(N1)
half1 = (x_mid - x_left) / 2
D1 = D1 / half1
x1 = xi1 * half1 + (x_mid + x_left) / 2   # domain 1: [x_left, x_mid]
A1 = (sigma ** 2 / 2 * D1 + (r - sigma ** 2 / 2)) @ D1  - r * np.eye(N1 + 1)

D2, xi2 = cv.chebyshev_diff_matrix(N2)
half2 = (x_right - x_mid) / 2
D2 = D2 / half2
x2 = xi2 * half2 + (x_right + x_mid) / 2   # domain 2: [x_mid, x_right]
A2 = (sigma ** 2 / 2 * D2 + (r - sigma ** 2 / 2)) @ D2  - r * np.eye(N2 + 1)

Ntot = N1 + N2 + 1
A_global = np.zeros((Ntot, Ntot))
A_global[0:N1+1, 0:N1+1] += A1
A_global[N1:N1+N2+1, N1:N1+N2+1] += A2

# Crank-Nicolson: (I - dt/2 A) u_{n+1} = (I + dt/2 A) u_n
I_mat = np.eye(Ntot)
M_global = I_mat - dt / 2 * A_global
N_global = I_mat + dt / 2 * A_global

# enforcing smoothness for M_global across domains
M_global[N1, :] = 0.0
M_global[N1, 0:N1+1]        = D1[-1, :]     # domain 1's own last-row derivative stencil
M_global[N1, N1:N1+N2+1]   -= D2[0, :]     # minus domain 2's own first-row derivative stencil

# enforce homogeneous Dirichlet on u_h at both ends: build into the
# LHS operator before factoring (RHS rows get overwritten with 0 anyway)
M_global[0, :] = 0.0
M_global[0, 0] = 1.0
M_global[-1, :] = 0.0
M_global[-1, -1] = 1.0

LU_M, P = decomp.compute_LU_pivot(M_global)

def update_crank_nicolson(u, tau):
    v = N_global @ u
    v[0] = 0.0
    v[-1] = K*np.exp(-r * tau) - np.exp(x1[-1])
    return decomp.solve_LU(LU_M, v, P)

# European put payoff: max(K - S, 0), S = exp(x)
x = np.concatenate((x2[:-1], x1)) # such that x[0] is the greatest value
u0 = np.maximum(0.0, K - np.exp(x))
u = np.copy(u0)

# exact solution setup
def d1(S, t):
    return 1/(sigma * np.sqrt(t)) * (np.log(S/K) + (r + 1/2 * sigma ** 2) * t)

def d2(S, t):
    return d1(S, t) - sigma * np.sqrt(t)

def PBS(S, t):
    if t == 0:
        return np.maximum(K - S, 0.0)
    return K * np.exp(-r * t) * norm.cdf(-d2(S, t)) - S * norm.cdf(-d1(S, t))

T = 1.0
n_steps = int(T / dt)
save_every = 1

history = [u0.copy()]
times = [0.0]
for step in range(1, n_steps + 1):
    u = update_crank_nicolson(u, step * dt)
    if step % save_every == 0:
        history.append(u.copy())
        times.append(step * dt)

plt.plot(x, PBS(np.exp(x), T), label="Analytic")
plt.plot(x, u, label="Computed", linestyle="--")
plt.legend()
plt.show()

# --- animation ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.05, 2.05)
ax.hlines(y=[0],xmin=x[-1],xmax=x[0], linestyles="--", color="black")
ax.set_xlabel("$x$")
ax.set_ylabel("$V(x,\\tau)$")

line, = ax.plot([], [], lw=2)
title = ax.set_title("")


def init():
    line.set_data([], [])
    title.set_text("")
    return line, title


def update(frame):
    line.set_data(x, history[frame])
    title.set_text(f"Black-Scholes, $\\tau$ = {times[frame]:.4f}")
    return line, title


anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=len(history), interval=50, blit=False)
plt.show()