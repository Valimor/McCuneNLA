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

def fix_sign_by_continuity(mode, previous_mode):
    if previous_mode is None:
        return mode
    if np.dot(mode, previous_mode) < 0:
        return -mode
    return mode

def compute_low_modes(alpha, previous_modes=None):
    Phi = cv.build_robin_basis(x, alpha, n_basis)
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
    for i, (eigval, eigvec) in enumerate(zip(eigvals_low, eigvecs_low.T)):
        reconstructed = Phi @ eigvec
        reconstructed /= np.linalg.norm(reconstructed)

        prev = previous_modes[i][1] if previous_modes is not None else None
        reconstructed = fix_sign_by_continuity(reconstructed.real, prev)

        modes.append((eigval.real, reconstructed))
    return modes

alphas = np.concatenate([
    np.linspace(0.0, 2.0, 40),
    np.linspace(2.0, 20.0, 30)
])

# build history sequentially, each frame's sign chosen relative to the previous frame
history = []
prev_modes = None
for a in alphas:
    modes = compute_low_modes(a, prev_modes)
    history.append(modes)
    prev_modes = modes

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