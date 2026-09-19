import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from spectral import chebyshev as cv

# goal: u′′+λe^u=0, u(+-1) = 0
#   - i can solve this ez

# structure: solve u" = f(u, u', x) by linearizing and updating!
# u" + λe^u = R(u). residual, a function of x
# (u + d)" + λe^(u + d) ~ u" + d" + λe^u(1 + d) -> d" + λe^u(d) = -R(u).
#
# let epsilon*delta. ~ u" + epsilon*d" + λe^u(1 + epsilon*d) -> epsilon*d" + epsilon*λe^u(d) = R(u)
#   -hm. feels like the same as just scaling epsilon.
# as a note, e^u is a function of x that doesn't vary with d! therefore, 
#   - d" + q(x)d = R is the eqn we solve at each step, then update with newton-raphson!

# the way to do this: (D2 + Q)d = R. solve for the d at each step and update. do until convergence?
#   - just gotta figure out how to do the NR

# initialize the problem
N = 128
n_basis = 64
D, x = cv.chebyshev_diff_matrix(N)
D2 = D @ D
Phi = cv.build_dirichlet_basis(x, n_basis)
Phi_dp = D2 @ Phi
w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

lam = 0.8

u = np.zeros_like(x)  # u_0 = 0 satisfies Dirichlet BCs trivially -- reasonable starting guess
#u = (1 - x ** 2)/2 # satisfies the BCs

# question: there is a phenomenon where the max value suddenly jumps...
#   -YEP bifurcation. known phenomenon per gemini. check wikipedia

tol = 1e-10
max_newton_steps = 100
max_us = np.zeros((max_newton_steps,))
max_rs = np.zeros((max_newton_steps,))

best_u = u.copy()
best_r = np.zeros_like(x)

# do sympy for newton-raphson
theta_sym = sym.symbols("\\theta")
lam_sym = theta_sym ** 2 / (2 *sym.cosh(theta_sym / 2) ** 2) - lam
lam_sym_dtheta = sym.diff(lam_sym, theta_sym)
nr_step_sym = lam_sym / lam_sym_dtheta
nr_step = sym.lambdify(theta_sym, nr_step_sym)

# start by solving for theta
theta = 1.0
for step in range(max_newton_steps):
    theta = theta - nr_step(theta)
    R = lam - theta ** 2 / (2 * np.cosh(theta/2) ** 2)
    if np.abs(R) < tol:
        break
print(f"Found theta {theta:.3f} with residual {R:.3f} in {step} steps")

max_r = []

for step in range(max_newton_steps):
    # (a) residual, per your question 1
    R = D2 @ u + lam * np.exp(u)
    max_r.append(np.max(np.abs(R)))
    if np.max(np.abs(R)) < tol:
        print(f"Converged in {step} Newton steps")
        print(np.max(np.abs(R)))
        best_r = R.copy()
        break

    max_us[step] = np.max(u)
    max_rs[step] = np.max(np.abs(R))

    if step > 0:
        if max_rs[step] < np.min(max_rs[:step]):
            best_u = u.copy()
            best_r = R.copy()

    q = lam * np.exp(u)  
    Q = np.diag(q)

    K = Phi.T @ W @ Phi_dp + Phi.T @ W @ Q @ Phi
    rhs = -Phi.T @ W @ R # check this logic.

    delta_coeffs = np.linalg.solve(K, rhs)
    delta = Phi @ delta_coeffs
    u = u + delta

else:
    print("Newton did not converge within max_steps")

analytic_u = -2 * np.log(np.cosh(x * theta/2)/np.cosh(theta/2))

print(f"Value of u at x = {x[N//2]} : {u[N // 2]}") # finding the value near 0

plt.plot(x, best_u, label="Best u")
plt.plot(x, analytic_u, label="Analytic u", linestyle="--")
plt.title(f"Largest R: {np.max(np.abs(best_r)):.3f}")
plt.legend()
plt.show()

plt.plot(range(len(max_r)), np.log10(np.array(max_r)))
plt.title("Errors vs step")
plt.xlabel("Step")
plt.ylabel("log|Error|")
plt.show()