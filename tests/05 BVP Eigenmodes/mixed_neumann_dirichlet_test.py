import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen
from testing import eigenmode_validation

N = 256
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

n_basis = 64
Phi = cv.build_mixed_basis(x, n_basis)   # self-normalized (unit-norm columns), as originally written
Phi_xx = D2 @ Phi   # first derivative only, not second. integration by parts. possible only for neumann

w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

K = -1 * Phi.T @ W @ Phi_xx
M = Phi.T @ W @ Phi

eigvals, eigvecs = eigen.generalized_eigen(K, M)

# keep the same order for both!
order = np.argsort(eigvals.real)
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

low_k = 4
eigvals_low = eigvals_sorted[:low_k]
eigvecs_low = eigvecs_sorted[:, :low_k]

for n, (eigval, eigvec) in enumerate(zip(eigvals_low, eigvecs_low.T)):
    reconstructed_eigvec = Phi @ eigvec
    reconstructed_eigvec /= np.linalg.norm(reconstructed_eigvec)
    if reconstructed_eigvec[np.argmax(np.abs(reconstructed_eigvec))] < 0:
        reconstructed_eigvec = -reconstructed_eigvec
    plt.plot(x, reconstructed_eigvec, label=f"$\\lambda$ = {eigval:.3f}")

    analytic = np.sin((2*n+1) * np.pi/4 * (x+1))
    analytic /= np.linalg.norm(analytic)              # same normalization convention
    if np.sign(analytic[8]) != np.sign(reconstructed_eigvec[8]): # avoiding zero
        analytic *= -1
    plt.plot(x, analytic, "--", label=f"Analytic $\\lambda$ = {(np.pi * (2*n+1)/4) ** 2:.3f}") 
plt.title("Mixed eigenmodes with Clenshaw-Curtis")   
plt.legend()
plt.show()