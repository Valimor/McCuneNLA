import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

from linalg import QR_LU_Decomposition as decomp

"""
=====================================================================================
EXECUTE
=====================================================================================
"""

def orthogonality_error(Q):
    return np.linalg.norm(Q @ Q.T - np.eye(*Q.shape))

N = 23
sizes = np.array([round(2**(n/2)) for n in range(2,N)])
error_gs = np.zeros_like(sizes, dtype=np.float64)
ts_gs = np.zeros_like(sizes, dtype=np.float64)
error_gsm = np.zeros_like(sizes, dtype=np.float64)
ts_gsm = np.zeros_like(sizes, dtype=np.float64)
error_hh = np.zeros_like(sizes, dtype=np.float64)
ts_hh = np.zeros_like(sizes, dtype=np.float64)
error_numpy = np.zeros_like(sizes, dtype=np.float64)
ts_numpy = np.zeros_like(sizes, dtype=np.float64)
for idx, n in enumerate(sizes):
    A = hilbert(n)

    t0 = time.time()
    Q_gs, R_gs = decomp.compute_gram_schmidt_QR(A)
    error_gs[idx] = orthogonality_error(Q_gs)
    t1 = time.time()
    ts_gs[idx] = t1 - t0
    print(f"Finished gram-schmidt for {n}")
    Q_gsm, R_gsm = decomp.compute_modified_gram_schmidt_QR(A)
    error_gsm[idx] = orthogonality_error(Q_gsm)
    t2 = time.time()
    ts_gsm[idx] = t2 - t1
    print(f"Finished modified gram-schmidt for {n}")
    Q_hh, R_hh = decomp.compute_householder_QR(A)
    error_hh[idx] = orthogonality_error(Q_hh)
    t3 = time.time()
    ts_hh[idx] = t3 - t2
    print(f"Finished householder for {n}")
    Q_np, R_np = np.linalg.qr(A)
    error_numpy[idx] = orthogonality_error(Q_np)
    t4 = time.time()
    ts_numpy[idx] = t4 - t3
    print(f"Finished numpy for {n}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax_err, ax_time = axes

# --- Left plot: error vs size ---
ax_err.loglog(sizes, error_gs, label="Gram-Schmidt")
ax_err.loglog(sizes, error_gsm, label="Modified Gram-Schmidt")
ax_err.loglog(sizes, error_hh, label="Householder")
ax_err.loglog(sizes, error_numpy, label="Numpy")
ax_err.set_xlabel("Matrix dimension $n$")
ax_err.set_ylabel("Orthogonality error $\\|Q^TQ - I\\|$")
ax_err.set_title("Orthogonality Error vs. Size")
ax_err.legend()

# --- Right plot: time vs size ---
ax_time.loglog(sizes, ts_gs, label="Gram-Schmidt")
ax_time.loglog(sizes, ts_gsm, label="Modified Gram-Schmidt")
ax_time.loglog(sizes, ts_hh, label="Householder")
ax_time.loglog(sizes, ts_numpy, label="Numpy")
ax_time.set_xlabel("Matrix dimension $n$")
ax_time.set_ylabel("Time (s)")
ax_time.set_title("Runtime vs. Size")
ax_time.legend()

fig.suptitle("QR Decomposition Methods on Hilbert Matrices")
fig.tight_layout()
plt.show()