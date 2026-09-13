import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from linalg import decomposition, iterative, eigen

def compute_ritz_history(A, max_m=None):
    n = A.shape[0]
    if max_m is None:
        max_m = n

    b = np.ones((n,), dtype=np.float64)
    Q, H = None, None
    ritz_history = []

    for step in range(max_m):
        Q, H = iterative.arnoldi_step(A, b, Q, H)
        m = H.shape[1]  # current Krylov dimension (accounts for early breakdown)

        H_square = H[:m, :m] if H.shape[0] == H.shape[1] else H[:-1, :]
        ritz_values, _ = eigen.QR_eigen_givens_algorithm(H_square)
        ritz_history.append(ritz_values)

        if H.shape[0] == H.shape[1]:
            break  # breakdown occurred, Krylov space exhausted

    return ritz_history

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_arnoldi_gif(A, filepath, max_m=None, interval=150, dpi=100):
    true_eigs = np.linalg.eigvals(A)
    ritz_history = compute_ritz_history(A, max_m)

    fig, ax = plt.subplots(figsize=(5, 5))
    margin = 0.2 * np.max(np.abs(true_eigs))
    ax.set_xlim(true_eigs.real.min() - margin, true_eigs.real.max() + margin)
    ax.set_ylim(true_eigs.imag.min() - margin, true_eigs.imag.max() + margin)
    ax.set_xlabel("Re($\\lambda$)")
    ax.set_ylabel("Im($\\lambda$)")
    ax.set_aspect("equal")

    # true eigenvalues: static, filled gray dots, drawn once
    ax.plot(true_eigs.real, true_eigs.imag, "o", color="lightgray",
            markersize=10, markeredgecolor="gray", zorder=1, label="True eigenvalues")

    # ritz values: filled dots, updated every frame
    ritz_dots, = ax.plot([], [], "o", color="crimson", markersize=7, zorder=2, label="Ritz values")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)

    def init():
        ritz_dots.set_data([], [])
        title.set_text("")
        return ritz_dots, title

    def update(frame):
        rv = ritz_history[frame]
        ritz_dots.set_data(rv.real, rv.imag)
        title.set_text(f"Arnoldi iteration, m = {frame+1}")
        return ritz_dots, title

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=len(ritz_history),
                                    interval=interval, blit=True)
    anim.save(filepath, writer="pillow", dpi=dpi)
    plt.close(fig)

def eigenmode_check(A, evals, evecs):
    for (eval, evec) in zip(evals, evecs.T):
        print(f"Error for eval: {eval}: {np.linalg.norm(A @ evec - eval * evec)}")

def random_spd_with_condition(n, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(0, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

def random_spd_bigger_eigs(n, l, cond_number, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.logspace(l, -np.log10(cond_number), n)
    return Q @ np.diag(eigs) @ Q.T

rng = np.random.default_rng(seed=10)

"""
theta = np.pi/3
A = np.array([
    [np.cos(theta), np.sin(theta), 0, 0],
    [-np.sin(theta), np.cos(theta), 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, -1]
])
"""
n = 50
# A = rng.normal(size=(n,n))
A = rng.random(size=(n,n)) - 0.5
fname = "figures/04 Eigenmodes/04_random_uniform_arnoldi_convergence.gif"
make_arnoldi_gif(A, fname)
print(f"Gif saved to {fname}")