import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spectral import chebyshev as cv
from linalg import eigen
from linalg import decomposition as decomp

def rk4_step(u, dt, f):
    k1 = f(u)
    k2 = f(u + dt/2 * k1)
    k3 = f(u + dt/2 * k2)
    k4 = f(u + dt   * k3)
    u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

N = 16

D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D
# boundary conditions
D2[0,:] = 0
D2[-1,:] = 0

# compute the eigenmodes with the code from 05.
n_modes = 16
mode_indices = np.arange(1, n_modes + 1)
analytic_eigvals = (mode_indices * np.pi / 2) ** 2

# analytic eigenfunctions, evaluated directly at your Chebyshev nodes -- no eigensolve needed
Phi_analytic = np.sin(np.outer(x + 1, mode_indices) * np.pi / 2)  # shape (N+1, n_modes)

# define initial conditions
u0 = (1 - x ** 2) * x ** 2 * 3

w = cv.clenshaw_curtis_weights(N)
c_eigenbasis = (Phi_analytic.T * w) @ u0 / ((Phi_analytic**2 * w[:, None]).sum(axis=0))
analytic_eigvals = np.arange(len(c_eigenbasis)) + 1
analytic_eigvals = (analytic_eigvals * np.pi / 2) ** 2

def analytic_diffusion(t):
    return Phi_analytic @ (c_eigenbasis * np.exp(-analytic_eigvals * t))

def diffusion_f(u):
    return D2 @ u

dts = np.logspace(-5,-1,num=128)
error_rk4 = np.zeros_like(dts)
error_ie = np.zeros_like(dts)
error_cn = np.zeros_like(dts)

for idx, dt in enumerate(dts):
    print(f"Starting index {idx}")
    # implicit euler
    A = np.eye(N + 1) - dt * D2
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    LU_A_ie, P_ie = decomp.compute_LU_pivot(A)

    def update_implicit_euler(u):
        rhs = u.copy()
        rhs[0] = 0.0
        rhs[-1] = 0.0
        return decomp.solve_LU(LU_A_ie, rhs, P_ie)

    # crank nicolson
    A = np.eye(N + 1) - dt/2 * D2
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    LU_A_cn, P_cn = decomp.compute_LU_pivot(A)

    def update_crank_nicolson(u):
        v = u + dt/2 * D2 @ u
        v[0] = 0.0
        v[-1] = 0.0
        u_p1 = decomp.solve_LU(LU_A_cn, v, P_cn)
        return u_p1

    # build initial conditions
    u_rk4 = u0.copy()
    u_ie = u0.copy()
    u_cn = u0.copy()

    T = 1.0
    analytic_final = analytic_diffusion(T)
    n_steps = int(T / dt)
    for step in range(n_steps):
        u_rk4 = rk4_step(u_rk4, dt, diffusion_f)
        u_ie = update_implicit_euler(u_ie)
        u_cn = update_crank_nicolson(u_cn)

    error_rk4[idx] = np.max(np.abs(u_rk4 - analytic_final))
    error_ie[idx] = np.max(np.abs(u_ie - analytic_final))
    error_cn[idx] = np.max(np.abs(u_cn - analytic_final))

    print(f"Finished at index {idx}")

import os
outdir = "./tests/08 Diffusion Time Marching/outputs"

results = np.column_stack([dts, error_rk4, error_ie, error_cn])
header = "dt error_rk4 error_ie error_cn"
np.savetxt(
    os.path.join(outdir, "convergence_vs_dt.txt"),
    results,
    header=header,
    comments="",  # avoids the default '#' prefix on the header line
)

# figure out the order for this later!
with open(os.path.join(outdir, "convergence_vs_dt_metadata.txt"), "w") as f:
    f.write(f"N = {N}\n")
    f.write(f"n_modes = {n_modes}\n")
    f.write(f"T = {T}\n")
    f.write(f"u0 = (1 - x**2) * x**2 * 3\n")

plt.loglog(dts, error_rk4, label="rk4")
plt.loglog(dts, error_ie, label="Implicit euler")
plt.loglog(dts, error_cn, label="crank nicolson")
plt.legend()
plt.title("Error vs. dt")
plt.xlabel("dt")
plt.ylabel("Final error")
plt.show()