import numpy as np
import matplotlib.pyplot as plt

from linalg import decomposition as decomp
from linalg import eigen
from spectral import chebyshev as cv

# tunable parameters
N = 128
n_basis = 16
dt = 0.0001

D, x = cv.chebyshev_diff_matrix(N)
Phi = cv.build_dirichlet_basis(x, n_basis)
w = cv.clenshaw_curtis_weights(N)

W = np.diag(w)
D2 = D @ D
M = Phi.T @ W @ Phi
K = Phi.T @ W @ D2 @ Phi

# rk4 update
M_inv_K = np.linalg.solve(M, K)
print(f"Strong condition: {np.linalg.cond(M_inv_K)}")

# compute the max eigenmode of M_inv_K to find the stable dt
evals, evecs = eigen.QR_eigen_givens_algorithm(M_inv_K)
evals_check = np.linalg.eigvals(M_inv_K)
max_eval = evals[np.argmax(np.abs(evals))]
print(f"Max stable dt: {-2.78529356/max_eval}") #computed from the sidequest file

def galerkin_f(c):
    return M_inv_K @ c

def rk4_step(u, dt, f):
    k1 = f(u)
    k2 = f(u + dt/2 * k1)
    k3 = f(u + dt/2 * k2)
    k4 = f(u + dt   * k3)
    u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

# implicit euler update
A_ie = M - dt * K
LU_ie, P_ie = decomp.compute_LU_pivot(A_ie)

def update_ie_galerkin(c):
    rhs = M @ c
    return decomp.solve_LU(LU_ie, rhs, P_ie)

# crank nicolson update
A_cn = M - dt/2 * K
LU_cn, P_cn = decomp.compute_LU_pivot(A_cn)

def update_cn_galerkin(c):
    rhs = (M + dt/2 * K) @ c
    return decomp.solve_LU(LU_cn, rhs, P_cn)

# build the initial condition
u0 = (1 - x ** 2 + np.sin(np.pi * (x + 1)/2)) * x ** 2
c0 = np.linalg.solve(M, Phi.T @ W @ u0)   # Galerkin projection of u0 onto Phi
c_cn = c0.copy()
c_ie = c0.copy()
c_rk4 = c0.copy()

# compute analytic answer
n_modes = 16
mode_indices = np.arange(1, n_modes + 1)
analytic_eigvals = (mode_indices * np.pi / 2) ** 2

# analytic eigenfunctions, evaluated directly at your Chebyshev nodes -- no eigensolve needed
Phi_analytic = np.sin(np.outer(x + 1, mode_indices) * np.pi / 2)  # shape (N+1, n_modess)

w = cv.clenshaw_curtis_weights(N)
c_eigenbasis = (Phi_analytic.T * w) @ u0 / ((Phi_analytic**2 * w[:, None]).sum(axis=0))
analytic_eigvals = np.arange(len(c_eigenbasis)) + 1
analytic_eigvals = (analytic_eigvals * np.pi / 2) ** 2

def analytic_diffusion(t):
    return Phi_analytic @ (c_eigenbasis * np.exp(-analytic_eigvals * t))

# let's do a galerkin crank nicolson for fun
T = 1
n_steps = int(T/dt)
rk4_cn_error = np.zeros((n_steps,))
for step in range(n_steps):
    c_cn = update_cn_galerkin(c_cn)
    c_ie = update_ie_galerkin(c_ie)
    c_rk4 = rk4_step(c_rk4, dt, galerkin_f)
    rk4_cn_error[step] = np.max(np.max(c_cn - c_rk4))

# reconstruction step.
u_reconstructed_cn = Phi @ c_cn
u_reconstructed_ie = Phi @ c_ie
u_reconstructed_rk4 = Phi @ c_rk4

u_final_exact = analytic_diffusion(T)
print(f"rk4 final error: {np.max(np.abs(u_reconstructed_rk4 - u_final_exact))}")
print(f"ie final error: {np.max(np.abs(u_reconstructed_ie - u_final_exact))}")
print(f"cn final error: {np.max(np.abs(u_reconstructed_cn - u_final_exact))}")
plt.plot(x, u0, label="Original")
plt.plot(x, u_reconstructed_ie, label="Final (IE)")
plt.plot(x, u_reconstructed_cn, label="Final (CN)")
plt.plot(x, u_reconstructed_rk4, label="Final (RK4)")
plt.legend()
plt.show()

# QUESTION: why do these agree to machine precision? Does this lend itself to some notion of analytic continuation?
plt.semilogy(rk4_cn_error)
plt.show()