import numpy as np
import time
import matplotlib.pyplot as plt
import scipy

from linalg import decomposition as decomp
from testing import test_matrices

seed = 42
rng = np.random.default_rng(seed)

"""
=====================================================================================
EXECUTE
=====================================================================================
"""

def standard_growth_error(A, U):
    return np.max(np.abs(U)) / np.max(np.abs(A))

def lu_reconstruction_error(A, L, U, perm):
    M = L @ U
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return np.linalg.norm(A - M[inv_perm])

def lu_backwards_error(A, LU, perm, b_samples):
    errors = []
    for b in b_samples:
        x_hat = decomp.solve_LU(LU, b, perm)
        residual = A @ x_hat - b
        errors.append(np.linalg.norm(residual) / (np.linalg.norm(A) * np.linalg.norm(x_hat) + np.linalg.norm(b)))
    return np.mean(errors)
# --- Uniform wrappers: every method returns (LU_combined, L, U, perm) ---

def run_no_pivot(A):
    n = A.shape[0]
    LU = decomp.compute_LU(A.copy())
    L, U = decomp.make_L_U_from_LU(LU)
    perm = np.arange(n)  # identity permutation
    return LU, L, U, perm

def run_pivot(A):
    LU, perm = decomp.compute_LU_pivot(A.copy())
    L, U = decomp.make_L_U_from_LU(LU)
    return LU, L, U, perm

def run_scipy(A):
    n = A.shape[0]
    P_mat, L, U = scipy.linalg.lu(A)
    # scipy's P_mat satisfies A = P_mat @ L @ U; convert to a perm vector
    # consistent with your own convention (LU = A[perm]):
    perm = np.argmax(P_mat.T, axis=1)  # row i of P_mat.T is a one-hot at perm[i]
    LU = np.tril(L, -1) + U  # repack into combined storage for solve_LU
    return LU, L, U, perm

lu_functions = [run_no_pivot, run_pivot, run_scipy]

fn_names = {
    run_no_pivot: "No pivoting",
    run_pivot: "Partial pivoting",
    run_scipy: "SciPy",
}

growth_error_dict = {}
reconstruction_error_dict = {}
backwards_error_dict = {}
time_dict = {}

N = 128
sizes = np.logspace(3, 10, num=N, base=2, dtype=np.int64)

encountered_error = False
encountered_error_loop = False

for function in lu_functions:
    growth_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    reconstruction_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    backwards_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    time_dict[function] = np.zeros_like(sizes, dtype=np.float64)

n_b_samples = 8
for idx, n in enumerate(sizes):
    A = scipy.linalg.hilbert(n)
    # A = normal_distributed_random(n)
    # A = wilkinson_growth_matrix(n)
    b_samples = [rng.normal(size=n) for _ in range(n_b_samples)]
    for fn_idx, lu_function in enumerate(lu_functions):
        try:
            t0 = time.time()
            LU, L, U, perm = lu_function(A)
            t1 = time.time()
            time_dict[lu_function][idx] = t1 - t0
            growth_error_dict[lu_function][idx] = standard_growth_error(A, U)
            reconstruction_error_dict[lu_function][idx] = lu_reconstruction_error(A, L, U, perm)
            backwards_error_dict[lu_function][idx] = lu_backwards_error(A, LU, perm, b_samples)
        except Exception as e:
            growth_error_dict[lu_function][idx] = np.nan
            reconstruction_error_dict[lu_function][idx] = np.nan
            backwards_error_dict[lu_function][idx] = np.nan
            if not encountered_error_loop:
                print(f"{fn_names[lu_function]} with size {n}x{n} failed: {e}")
                encountered_error = True
        if idx % 5 == 0:
            print(f"{fn_names[lu_function]} with size {n}x{n} computed at index {idx}")
    if encountered_error:
        encountered_error_loop = True

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax_growth, ax_recon), (ax_back, ax_time) = axes

metric_axes = [
    (ax_growth, growth_error_dict, "Growth factor", "Growth Factor vs. Size"),
    (ax_recon, reconstruction_error_dict, "Reconstruction error $\\|A - LU\\|$", "Reconstruction Error vs. Size"),
    (ax_back, backwards_error_dict, "Backward error", "Backward Error vs. Size"),
    (ax_time, time_dict, "Time (s)", "Runtime vs. Size"),
]

for ax, data_dict, ylabel, title in metric_axes:
    for lu_function in lu_functions:
        ax.loglog(sizes, data_dict[lu_function], label=fn_names[lu_function])
    ax.set_xlabel("Matrix dimension $n$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

fig.suptitle("LU Decomposition Methods on Hilbert Matrices")
fig.tight_layout()
plt.show()