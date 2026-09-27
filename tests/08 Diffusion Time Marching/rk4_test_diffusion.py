import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spectral import chebyshev as cv
from linalg import eigen

def rk4_step(u, dt, f):
    k1 = f(u)
    k2 = f(u + dt/2 * k1)
    k3 = f(u + dt/2 * k2)
    k4 = f(u + dt   * k3)
    u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

N = 16
dt = 0.0009

D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D
# boundary conditions
D2[0,:] = 0
D2[-1,:] = 0

def diffusion_f(u):
    return D2 @ u

# compute the eigenmodes with the code from 05.
n_modes = 16
mode_indices = np.arange(1, n_modes + 1)
analytic_eigvals = (mode_indices * np.pi / 2) ** 2

# analytic eigenfunctions, evaluated directly at your Chebyshev nodes -- no eigensolve needed
Phi_analytic = np.sin(np.outer(x + 1, mode_indices) * np.pi / 2)  # shape (N+1, n_modes)

u0 = (1 - x ** 2) * x ** 2 * 3
u = np.copy(u0)

w = cv.clenshaw_curtis_weights(N)
c_eigenbasis = (Phi_analytic.T * w) @ u0 / ((Phi_analytic**2 * w[:, None]).sum(axis=0))
analytic_eigvals = np.arange(len(c_eigenbasis)) + 1
analytic_eigvals = (analytic_eigvals * np.pi / 2) ** 2

def analytic_diffusion(t):
    return Phi_analytic @ (c_eigenbasis * np.exp(-analytic_eigvals * t))

T = 1
n_steps = int(T / dt)

# store a subsampled history so the animation has a manageable number of frames
save_every = 50
history = [u0.copy()]
analytic_history = [analytic_diffusion(0.0)]
errors = []
for step in range(n_steps):
    u = rk4_step(u, dt, diffusion_f)
    errors.append(np.max(np.abs(u - analytic_diffusion(dt * (step + 1)))))
    if step % save_every == 0:
        history.append(u.copy())
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