import numpy as np
import time
import matplotlib.pyplot as plt
import scipy

from linalg import QR_LU_Decomposition as decomp

seed = 42
rng = np.random.default_rng(seed)

"""
=====================================================================================
EXECUTE
=====================================================================================
"""

def qr_reconstruction_error(A, Q, R):
    return np.linalg.norm(A - Q @ R)

def orthogonality_error(Q):
    return np.linalg.norm(Q @ Q.T - np.eye(*Q.shape))

def qr_backwards_error(A, Q, R, b_samples):
    errors = []
    for b in b_samples:
        y = Q.T @ b
        x_hat = decomp.solve_triangular(R, y, lower=False)
        residual = A @ x_hat - b
        errors.append(np.linalg.norm(residual) / (np.linalg.norm(A) * np.linalg.norm(x_hat) + np.linalg.norm(b)))
    # return np.mean(errors), np.max(errors)
    return np.mean(errors)

def wilkinson_growth_matrix(n):
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = -1
    A[:, -1] = 1
    return A

def normal_distributed_random(n):
    A = rng.normal(size=(n, n))
    return A

# make this dictionary-based
qr_functions = [
    decomp.compute_gram_schmidt_QR, 
    decomp.compute_modified_gram_schmidt_QR,
    decomp.compute_householder_QR,
    np.linalg.qr
    ]

# names
fn_names = {
    decomp.compute_gram_schmidt_QR: "Gram-Schmidt",
    decomp.compute_modified_gram_schmidt_QR: "Modified Gram-Schmidt",
    decomp.compute_householder_QR: "Householder",
    np.linalg.qr: "NumPy",
}

orthogonality_error_dict = {}
reconstruction_error_dict = {}
backwards_error_dict = {}
time_dict = {}

N = 128
sizes = np.logspace(3, 10, num=N, base=2, dtype=np.int64)

encountered_zero_pivot = False
encountered_zero_pivot_loop = False

# initialize the errors
for function in qr_functions:
    orthogonality_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    reconstruction_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    backwards_error_dict[function] = np.zeros_like(sizes, dtype=np.float64)
    time_dict[function] = np.zeros_like(sizes, dtype=np.float64)

n_b_samples = 8
for idx, n in enumerate(sizes):
    # A = scipy.linalg.hilbert(n)
    # A = wilkinson_growth_matrix(n)
    A = normal_distributed_random(n)
    b_samples = [rng.normal(size=n) for _ in range(n_b_samples)]
    for fn_idx, qr_function in enumerate(qr_functions):
        t0 = time.time()
        Q, R = qr_function(A)
        t1 = time.time()
        time_dict[qr_function][idx] = t1 - t0
        orthogonality_error_dict[qr_function][idx] = orthogonality_error(Q)
        reconstruction_error_dict[qr_function][idx] = qr_reconstruction_error(A, Q, R)
        try:
            backwards_error_dict[qr_function][idx] = qr_backwards_error(A, Q, R, b_samples)
        except:
            backwards_error_dict[qr_function][idx] = np.nan
            if not(encountered_zero_pivot_loop):
                print(f"{fn_names[qr_function]} with size {n}x{n} encountered zero pivot")
                encountered_zero_pivot = True
        if idx % 5 == 0:
            print(f"{fn_names[qr_function]} with size {n}x{n} computed at index {idx}")
    if encountered_zero_pivot:
        encountered_zero_pivot_loop = True

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax_orth, ax_recon), (ax_back, ax_time) = axes

metric_axes = [
    (ax_orth, orthogonality_error_dict, "Orthogonality error $\\|Q^TQ - I\\|$", "Orthogonality Error vs. Size"),
    (ax_recon, reconstruction_error_dict, "Reconstruction error $\\|A - QR\\|$", "Reconstruction Error vs. Size"),
    (ax_back, backwards_error_dict, "Backward error", "Backward Error vs. Size"),
    (ax_time, time_dict, "Time (s)", "Runtime vs. Size"),
]

for ax, data_dict, ylabel, title in metric_axes:
    for qr_function in qr_functions:
        ax.loglog(sizes, data_dict[qr_function], label=fn_names[qr_function])
    ax.set_xlabel("Matrix dimension $n$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

fig.suptitle("QR Decomposition Methods on Random Matrices")
fig.tight_layout()
plt.show()