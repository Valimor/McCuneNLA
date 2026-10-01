import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import decomposition as decomp

# --- Problem Parameters ---
r = 0.03
sigma = 0.5
K = 1.0

# --- Domain Setup ---
dt = 0.0001
scale = 15
x_mid = np.log(K)
x_left = x_mid - scale
x_right = x_mid + scale

N1, N2 = 128, 128 # Polynomial degrees in Domain 1 [x_left, x_mid] and Domain 2 [x_mid, x_right]

# Domain 1: [x_left, x_mid]
D1_raw, xi1 = cv.chebyshev_diff_matrix(N1)
# xi1 goes +1 -> -1. Let's map +1 to x_mid, -1 to x_left
half1 = (x_mid - x_left) / 2.0
D1 = D1_raw / half1
x1 = xi1 * half1 + (x_mid + x_left) / 2.0  # x1[0] = x_mid, x1[-1] = x_left

# Domain 2: [x_mid, x_right]
D2_raw, xi2 = cv.chebyshev_diff_matrix(N2)
# xi2 goes +1 -> -1. Let's map +1 to x_right, -1 to x_mid
half2 = (x_right - x_mid) / 2.0
D2 = D2_raw / half2
x2 = xi2 * half2 + (x_right + x_mid) / 2.0  # x2[0] = x_right, x2[-1] = x_mid

# Build smooth global vector x (sorted left-to-right)
# x1[::-1] goes x_left -> x_mid; x2[::-1][1:] goes >x_mid -> x_right
x_dom1 = x1[::-1]          # index 0 is x_left, index N1 is x_mid
x_dom2 = x2[::-1][1:]      # index 0 is x2[-2] (>x_mid), index N2-1 is x_right
x = np.concatenate((x_dom1, x_dom2))

Ntot = len(x)  # N1 + N2 + 1
idx_mid = N1   # Index of x_mid in global array

# Construct individual Black-Scholes PDE operators
A1 = (sigma**2 / 2.0) * (D1 @ D1) + (r - sigma**2 / 2.0) * D1 - r * np.eye(N1 + 1)
A2 = (sigma**2 / 2.0) * (D2 @ D2) + (r - sigma**2 / 2.0) * D2 - r * np.eye(N2 + 1)

# Assemble Global Operators M_global and N_global
# Index mapping in global vector u:
# u[0 : N1+1] correspond to x1[::-1] (Domain 1)
# u[N1 : Ntot] correspond to x2[::-1] (Domain 2)
M_global = np.eye(Ntot)
N_global = np.eye(Ntot)

# Populate interior rows of Domain 1 (indices 1 to N1-1)
for i in range(1, N1):
    # Mapping reversed index back to Chebyshev D1 indexing
    i_raw = N1 - i
    M_global[i, 0:N1+1] = - (dt / 2.0) * A1[i_raw, ::-1]
    M_global[i, i] += 1.0
    N_global[i, 0:N1+1] = (dt / 2.0) * A1[i_raw, ::-1]
    N_global[i, i] += 1.0

# Populate interior rows of Domain 2 (indices N1+1 to Ntot-2)
for i in range(1, N2):
    g_idx = N1 + i
    i_raw = N2 - i
    M_global[g_idx, N1:Ntot] = - (dt / 2.0) * A2[i_raw, ::-1]
    M_global[g_idx, g_idx] += 1.0
    N_global[g_idx, N1:Ntot] = (dt / 2.0) * A2[i_raw, ::-1]
    N_global[g_idx, g_idx] += 1.0

# --- Boundary Conditions ---
# 1. Left boundary x_left (g_idx = 0 -> x1[-1]): u(x_left) = K * exp(-r*tau)
M_global[0, :] = 0.0
M_global[0, 0] = 1.0
N_global[0, :] = 0.0

# 2. Right boundary x_right (g_idx = Ntot-1 -> x2[0]): u(x_right) = 0
M_global[-1, :] = 0.0
M_global[-1, -1] = 1.0
N_global[-1, :] = 0.0

# 3. Interface condition at x_mid (g_idx = N1): Flux matching (d u1 / dx = d u2 / dx)
M_global[idx_mid, :] = 0.0
# D1 derivative at x_mid (which is index 0 in raw D1)
M_global[idx_mid, 0:N1+1] = D1[0, ::-1]
# D2 derivative at x_mid (which is index N2 in raw D2)
M_global[idx_mid, N1:Ntot] -= D2[N2, ::-1]

N_global[idx_mid, :] = 0.0  # Interface derivative mismatch must equal 0 at step n+1

# Factorize LHS
LU_M, P = decomp.compute_LU_pivot(M_global)

def update_crank_nicolson(u_vec, tau):
    v = N_global @ u_vec
    v[0] = K * np.exp(-r * tau)  # Deep ITM Put boundary
    v[-1] = 0.0                  # Deep OTM Put boundary
    v[idx_mid] = 0.0             # Continuous flux matching
    return decomp.solve_LU(LU_M, v, P)

# --- Simulation Run ---
u0 = np.maximum(0.0, K - np.exp(x))
u = np.copy(u0)

T = 1.0
n_steps = int(T / dt)

for step in range(1, n_steps + 1):
    u = update_crank_nicolson(u, step * dt)

# Analytical exact solution for comparison
def PBS(S, t):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

u_exact = PBS(np.exp(x), T)

max_err_idx = np.argmax(np.abs(u - u_exact))
print(f"Max Domain-Decomposition Error at T={T}: {np.max(np.abs(u - u_exact)):.2e} at index {max_err_idx}")
plt.scatter([x[max_err_idx]],[u[max_err_idx]], label="Maximum error")
plt.plot(x, u, label="solved")
plt.plot(x, u_exact, label="exact")
plt.legend()
plt.show()