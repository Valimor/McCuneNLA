import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

# testing the following:
# u" = 6u^2, u(0) = 1, u(1) = 1/4
#
# to do this I am going to rescale and shift the domain. This should multiply D and Phi by 1/2
# TODO: make a class for chebyshev function generation
# - let it take in endpoints for an interval, then have it return any basis/D I want

# NOTE: whenever rescaling a domain and differentiating, take the derivative of a known function
# on that domain as a check!

N = 256
n_basis = 32

D, x = cv.chebyshev_diff_matrix(N)
D *= 2 #rescaling
D2 = D@D
Phi = cv.build_dirichlet_basis(x, n_basis)
x = 0.5 * x + 0.5 # shifting to [0,1]
Phi_dp = D2 @ Phi
w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

# initial guess. gotta figure out how to incorporate the inhomogenous conditions
# there's no way it's as simple as just keeping it?
# turns out it is!
u = 1 - 3/4 * x #+ x * (1 - x) # interesting, this converges slower! probably bc farther from the correct shape

tol = 1e-8
max_newton_steps = 10

max_r = []

for step in range(max_newton_steps):
    R = D2 @ u - 6 * u ** 2
    max_r.append(np.max(np.abs(R)))
    if np.max(np.abs(R)) < tol:
        print(f"Converged in {step} Newton steps")
        print(np.max(np.abs(R)))
        best_r = R.copy()
        break

    q = -12 * u # from jacobian
    Q = np.diag(q)

    K = Phi.T @ W @ Phi_dp + Phi.T @ W @ Q @ Phi
    rhs = -Phi.T @ W @ R

    delta_coeffs = np.linalg.solve(K, rhs)
    delta = Phi @ delta_coeffs
    u = u + delta

else:
    print("Newton did not converge within max_steps")

plt.plot(x, u, label="final")
plt.plot(x, 1/(x + 1)**2, label="Analytic", linestyle="--")
plt.title(f"Final vs. Analytic. R={np.max(np.abs(R))}")
plt.legend()
plt.show()

plt.plot(range(len(max_r)), np.log10(np.array(max_r)))
plt.title("Errors vs step")
plt.xlabel("Step")
plt.ylabel("log|Error|")
plt.show()