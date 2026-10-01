import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import decomposition as decomp

# --- Problem Parameters ---
r = 0.03
sigma = 0.5
K = 1.0

def PBS(S, t):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def run_two_domain_solve(N1, N2, dt, scale=15, T=1.0):
    x_mid = np.log(K)
    x_left = x_mid - scale
    x_right = x_mid + scale

    D1_raw, xi1 = cv.chebyshev_diff_matrix(N1)
    half1 = (x_mid - x_left) / 2.0
    D1 = D1_raw / half1
    x1 = xi1 * half1 + (x_mid + x_left) / 2.0  # x1[0]=x_mid, x1[-1]=x_left

    D2_raw, xi2 = cv.chebyshev_diff_matrix(N2)
    half2 = (x_right - x_mid) / 2.0
    D2 = D2_raw / half2
    x2 = xi2 * half2 + (x_right + x_mid) / 2.0  # x2[0]=x_right, x2[-1]=x_mid

    x_dom1 = x1[::-1]
    x_dom2 = x2[::-1][1:]
    x = np.concatenate((x_dom1, x_dom2))

    Ntot = len(x)
    idx_mid = N1

    A1 = (sigma**2 / 2.0) * (D1 @ D1) + (r - sigma**2 / 2.0) * D1 - r * np.eye(N1 + 1)
    A2 = (sigma**2 / 2.0) * (D2 @ D2) + (r - sigma**2 / 2.0) * D2 - r * np.eye(N2 + 1)

    M_global = np.eye(Ntot)
    N_global = np.eye(Ntot)

    for i in range(1, N1):
        i_raw = N1 - i
        M_global[i, 0:N1+1] = -(dt / 2.0) * A1[i_raw, ::-1]
        M_global[i, i] += 1.0
        N_global[i, 0:N1+1] = (dt / 2.0) * A1[i_raw, ::-1]
        N_global[i, i] += 1.0

    for i in range(1, N2):
        g_idx = N1 + i
        i_raw = N2 - i
        M_global[g_idx, N1:Ntot] = -(dt / 2.0) * A2[i_raw, ::-1]
        M_global[g_idx, g_idx] += 1.0
        N_global[g_idx, N1:Ntot] = (dt / 2.0) * A2[i_raw, ::-1]
        N_global[g_idx, g_idx] += 1.0

    # left boundary: u(x_left) = K*exp(-r*tau)
    M_global[0, :] = 0.0
    M_global[0, 0] = 1.0
    N_global[0, :] = 0.0

    # right boundary: u(x_right) = 0
    M_global[-1, :] = 0.0
    M_global[-1, -1] = 1.0
    N_global[-1, :] = 0.0

    # interface: flux matching
    M_global[idx_mid, :] = 0.0
    M_global[idx_mid, 0:N1+1] = D1[0, ::-1]
    M_global[idx_mid, N1:Ntot] -= D2[N2, ::-1]
    N_global[idx_mid, :] = 0.0

    LU_M, P = decomp.compute_LU_pivot(M_global)

    def update_crank_nicolson(u_vec, tau):
        v = N_global @ u_vec
        v[0] = K * np.exp(-r * tau)
        v[-1] = 0.0
        v[idx_mid] = 0.0
        return decomp.solve_LU(LU_M, v, P)

    u = np.maximum(0.0, K - np.exp(x))

    n_steps = round(T / dt)          # avoids the int()-truncation drift from before
    dt_actual = T / n_steps          # lands exactly on T regardless of dt's float rounding

    for step in range(1, n_steps + 1):
        u = update_crank_nicolson(u, step * dt_actual)

    u_exact = PBS(np.exp(x), T)
    return np.max(np.abs(u - u_exact)), x, u, u_exact

Ns = np.logspace(4,7,num=32,base=2,dtype=np.int64)
dt_fixed = 1e-4
errors_N = np.zeros(len(Ns))

for idx, n in enumerate(Ns):
    print(f"Starting N = {n} at idx {idx} out of {len(Ns)}")
    errors_N[idx], *_ = run_two_domain_solve(n, n, dt_fixed)

# find the least squares best fit (convergence order) in log-log space
# first, remove the plateau
fit_mask = errors_N > 2*errors_N.min()
log_Ns = np.log(Ns[fit_mask]).reshape(-1, 1)
log_err = np.log(errors_N[fit_mask]).reshape(-1, 1)

design = np.column_stack([log_Ns, np.ones_like(log_Ns)])
coefs, *_ = np.linalg.lstsq(design, log_err, rcond=None)

slope, intercept = coefs.flatten()
fit_errors = np.exp(intercept) * Ns[fit_mask]**slope
print(f"Estimated order of convergence: {slope:.3f}")

plt.figure()
plt.loglog(Ns, errors_N, marker='o', label="Error")
plt.loglog(Ns[fit_mask], fit_errors, '--', label=f'slope = {slope:.2f}')
plt.legend()
plt.xlabel("N1 = N2")
plt.ylabel("Max error")
plt.title(f"Two-domain CN convergence in N (dt={dt_fixed})")
plt.show()