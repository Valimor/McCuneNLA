import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from spectral import chebyshev as cv
from linalg import eigen

N = 256
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

n_basis = 64
low_k = 4

w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

# Locate a point near x = -1, shifted slightly inward (away from the
# boundary) so the derivative estimate is well-behaved for every alpha,
# regardless of whether the Chebyshev grid is stored ascending or
# descending.
_ascending = x[1] > x[0]
_idx_m1 = np.argmin(np.abs(x - (-1.0)))
_offset = 20
_idx_slope_check = _idx_m1 + _offset if _ascending else _idx_m1 - _offset
_idx_slope_check = int(np.clip(_idx_slope_check, 0, len(x) - 1))

def fix_sign_by_slope(mode):
    """Orient the mode so its slope near x = -1 is positive.

    This is a fixed convention evaluated independently for every alpha
    (not relative to the previous frame), so consecutive frames can't
    drift or jump even if eigenvalues swap order or a continuity check
    point happens to sit near a node.
    """
    deriv = D @ mode
    if deriv[_idx_slope_check] < 0:
        mode = -mode
    return mode

def compute_low_modes(alpha):
    Phi = cv.build_robin_basis(x, n_basis, alpha)
    Phi_xx = D2 @ Phi

    K = -1 * Phi.T @ W @ Phi_xx
    M = Phi.T @ W @ Phi

    eigvals, eigvecs = eigen.generalized_eigen(K, M)

    order = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    eigvals_low = eigvals_sorted[:low_k]
    eigvecs_low = eigvecs_sorted[:, :low_k]

    modes = []
    for eigval, eigvec in zip(eigvals_low, eigvecs_low.T):
        reconstructed = Phi @ eigvec
        reconstructed /= np.linalg.norm(reconstructed)
        reconstructed = fix_sign_by_slope(reconstructed.real)
        modes.append((eigval.real, reconstructed))
    return modes

# sweep alpha over a meaningful range: from Neumann-like (alpha -> 0)
# to strongly Robin/Dirichlet-like (large alpha)
alphas = np.concatenate([
    np.linspace(0.0, 2.0, 40),
    np.linspace(2.0, 20.0, 30)
])

history = [compute_low_modes(a) for a in alphas]

# --- animation ---
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 0.85, low_k))
lines = [ax.plot([], [], color=colors[i])[0] for i in range(low_k)]
title = ax.set_title("")

all_values = np.concatenate([mode for frame in history for _, mode in frame])
y_max = np.max(np.abs(all_values)) * 1.15
ax.set_ylim(-y_max, y_max)

ax.set_xlim(x.min(), x.max())
ax.set_xlabel("$x$")
ax.set_ylabel("Eigenmode (normalized)")

def init():
    for line in lines:
        line.set_data([], [])
    title.set_text("")
    return lines + [title]

def update(frame):
    alpha = alphas[frame]
    modes = history[frame]
    for i, (eigval, mode) in enumerate(modes):
        lines[i].set_data(x, mode)
        lines[i].set_label(f"$\\lambda_{i}$ = {eigval:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    title.set_text(f"Robin eigenmodes, $\\alpha$ = {alpha:.2f}")
    return lines + [title]

anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=len(alphas), interval=120, blit=False)

outdir = "figures/05 BVP Eigenmodes"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "05_robin_alpha_variation.gif")
anim.save(outpath, writer="pillow", dpi=100)
plt.close(fig)