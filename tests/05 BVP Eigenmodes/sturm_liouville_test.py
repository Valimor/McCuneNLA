import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv
from linalg import eigen
from testing import eigenmode_validation

N = 256
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D

n_basis = 32
#alpha = 0.0
#Phi = cv.build_robin_basis(x, n_basis, alpha=alpha)
Phi = cv.build_dirichlet_basis(x, n_basis)
Phi_x = D @ Phi  
Phi_xx = D2 @ Phi

w_quad = cv.clenshaw_curtis_weights(N)
W_quad = np.diag(w_quad)

# define functions (w = 1, q = 0, so omitting them for the test)
p = (x+2)**2
P = np.diag(p)
q = 0*x
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
    analytic_eigval = ((n+1)*np.pi/np.log(3))**2 + 1/4

    analytic_eigvec = np.pow((x+2), -0.5) * np.sin((n+1)*np.pi*np.log(x+2)/np.log(3))
    analytic_eigvec /= np.linalg.norm(analytic_eigvec) / np.linalg.norm(reconstructed_eigvec)
    plt.plot(x, analytic_eigvec, label=f"Analytic $\\lambda$ = {analytic_eigval:.3f}", linestyle="--")
    print(f"Eigenvalue error (numeric vs generalized rayleigh) {np.abs(eigval - analytic_eigval)}")
plt.title("Dirichlet eigenmodes for Simple Sturm-Liouville")   
plt.legend()
plt.show()

# sympy validation. I should start using this more!
import sympy as sp
x, n = sp.symbols('x n', positive=True)
lam = (n*sp.pi/sp.log(3))**2 + sp.Rational(1,4)
u = (x+2)**sp.Rational(-1,2) * sp.sin(n*sp.pi*sp.log(x+2)/sp.log(3))
lhs = sp.simplify((x+2)**2 * sp.diff(u, x, 2) + 2*(x+2)*sp.diff(u, x) + lam*u)
print(lhs)  # should simplify to 0