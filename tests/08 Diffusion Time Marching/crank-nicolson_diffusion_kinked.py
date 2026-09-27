import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spectral import chebyshev as cv
from linalg import decomposition as decomp

def apply_bc(u):
    u = u.copy()
    u[0] = 0.0
    u[-1] = 0.0
    return u

N = 64
dt = 0.0001

D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

# adding boundary conditions into the solve!
A = np.eye(N + 1) - dt/2 * D2
A[0, :] = 0
A[0, 0] = 1
A[-1, :] = 0
A[-1, -1] = 1
LU_A, P = decomp.compute_LU_pivot(A)

def update_crank_nicolson(u):
    v = u + dt/2 * D2 @ u
    v[0] = 0.0
    v[-1] = 0.0
    u_p1 = decomp.solve_LU(LU_A, v, P)
    return u_p1

# compute the eigenmodes with the code from 05.
n_modes = 48
mode_indices = np.arange(1, n_modes + 1)
analytic_eigvals = (mode_indices * np.pi / 2) ** 2

# analytic eigenfunctions, evaluated directly at your Chebyshev nodes -- no eigensolve needed
Phi_analytic = np.sin(np.outer(x + 1, mode_indices) * np.pi / 2)  # shape (N+1, n_modes)

u0 = np.sqrt(1 - x ** 2) + (1 + x)/4

# incorporate the particular
u_p = (u0[0] * (1 + x) + u0[-1] * (1 - x))/2
u0_h = u0 - u_p 
u = np.copy(u0_h)

w = cv.clenshaw_curtis_weights(N)
c_eigenbasis = (Phi_analytic.T * w) @ u0_h / ((Phi_analytic**2 * w[:, None]).sum(axis=0))
analytic_eigvals = np.arange(len(c_eigenbasis)) + 1
analytic_eigvals = (analytic_eigvals * np.pi / 2) ** 2

def analytic_diffusion(t):
    return Phi_analytic @ (c_eigenbasis * np.exp(-analytic_eigvals * t)) + u_p

# print initial error
u0_reconstructed = analytic_diffusion(0.0)

T = 1
n_steps = int(T / dt)

# store a subsampled history so the animation has a manageable number of frames
save_every = 50
history = [u0.copy()]
analytic_history = [analytic_diffusion(0.0)]
errors = []
for step in range(n_steps):
    u = update_crank_nicolson(u)
    errors.append(np.max(np.abs(u + u_p - analytic_diffusion(dt * (step + 1)))))
    if step % save_every == 0:
        history.append(u.copy() + u_p)
        analytic_history.append(analytic_diffusion(dt * (step + 1)))

plt.semilogy(errors)
plt.title("|Analytic - Solved|")
plt.xlabel("Step")
plt.ylabel("Error")
plt.show()

# --- animation ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1.05)
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")

line, = ax.plot([], [], lw=2)
aline, = ax.plot([], [], lw=2, linestyle="--")
title = ax.set_title("")

def init():
    line.set_data([], [])
    aline.set_data([], [])
    title.set_text("")
    return line, title

def update(frame):
    t = frame * save_every * dt
    line.set_data(x, history[frame])
    aline.set_data(x, analytic_history[frame])
    title.set_text(f"Diffusion, $t$ = {t:.4f}")
    return line, title

anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=len(history), interval=50, blit=False)
plt.show()