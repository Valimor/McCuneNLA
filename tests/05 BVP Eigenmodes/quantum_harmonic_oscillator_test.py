import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen
from testing import eigenmode_validation

from scipy.special import eval_hermite

def qho_eigenfunction(n, x):
    Hn = eval_hermite(n, x)
    u = Hn * np.exp(-x**2 / 2)
    return u / np.linalg.norm(u)

N = 256
n_basis = 32
D, x = cv.chebyshev_diff_matrix(N)
Phi = cv.build_dirichlet_basis(x, n_basis)

L = 3 # rescaled space
x = L * x
D = D / L
D2 = D @ D
Phi_x = D @ Phi  
Phi_xx = D2 @ Phi

w_quad = cv.clenshaw_curtis_weights(N)
W_quad = np.diag(w_quad)

p = 1+0*x 
P = np.diag(p)
q = -x ** 2
Q = np.diag(q)
w = 1+0*x # crashes if w is zero. interesting
W = np.diag(w)

# correction for robin
#boundary_term = alpha * p[0] * np.outer(Phi[0, :], Phi[0, :])

K = Phi_x.T @ W_quad @ P @ Phi_x - Phi.T @ W_quad @ Q @ Phi #+ boundary_term
M = Phi.T @ W_quad @ W @ Phi

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
    rayleigh_eigval = eigenmode_validation.generalized_rayleigh_quotient(-D @ P @ D - Q, W, reconstructed_eigvec)
    analytic_eigval = 2*(n+1) - 1

    analytic_eigvec = qho_eigenfunction(n, x)
    analytic_eigvec /= (np.linalg.norm(analytic_eigvec) / np.linalg.norm(reconstructed_eigvec))
    # flip starting from the middle
    i = N // 2
    while np.abs(analytic_eigvec[i]) < 1e-6:
        i += 1
    analytic_eigvec *= np.sign(analytic_eigvec[i] / reconstructed_eigvec[i])
    plt.plot(x, analytic_eigvec, label=f"Analytic $\\lambda$ = {analytic_eigval:.3f}", linestyle="--")
    print(f"Eigenvalue error (numeric vs generalized rayleigh) {np.abs(eigval - analytic_eigval)}")
plt.title("Quantum Harmonic Oscillator Eigenmodes")   
plt.legend()
plt.show()