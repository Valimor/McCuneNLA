import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from spectral import chebyshev as cv

def validate_frechet_derivative(u_k, delta_test, D, D2, m, eps=1e-6):
    """
    Checks the Fréchet derivative of F(u) = (u^m u')' + 1 at u_k,
    in the direction delta_test, against a direct finite-difference estimate.

    Returns (fd_estimate, linear_estimate, max_abs_diff) so you can inspect
    both the raw arrays and the summary error.
    """
    def F(u):
        return D @ (u**m * (D @ u)) + 1

    # --- finite-difference estimate of the Frechet derivative ---
    F_plus  = F(u_k + eps * delta_test)
    F_minus = F(u_k - eps * delta_test)
    fd_estimate = (F_plus - F_minus) / (2 * eps)   # central difference: O(eps^2) accurate

    # --- your analytic linear operator, applied to delta_test ---
    up = D @ u_k

    p = u_k ** m
    s = 2 * m * u_k**(m - 1) * up
    q = D @ (m * u_k**(m - 1) * up)

    linear_estimate = (
        p * (D2 @ delta_test)
        + s * (D @ delta_test)
        + q * delta_test
    )

    diff = np.max(np.abs(fd_estimate - linear_estimate))
    return fd_estimate, linear_estimate, diff

def validate_term(u_k, delta_test, D, D2, m, eps=1e-6):
    def F(u):
        return D @ (u**m * (D @ u)) + 1

    F_plus  = F(u_k + eps * delta_test)
    F_minus = F(u_k - eps * delta_test)
    fd_estimate = (F_plus - F_minus) / (2 * eps)

    up = D @ u_k
    p = u_k ** m
    s = 2 * m * u_k**(m - 1) * up
    q = D @ (m * u_k**(m - 1) * up)

    term_p = p * (D2 @ delta_test)
    term_s = s * (D @ delta_test)
    term_q = q * delta_test

    print("fd estimate norm:", np.linalg.norm(fd_estimate))
    print("term_p norm:", np.linalg.norm(term_p))
    print("term_s norm:", np.linalg.norm(term_s))
    print("term_q norm:", np.linalg.norm(term_q))
    print("sum norm:", np.linalg.norm(term_p + term_s + term_q))
    print("diff (sum vs fd):", np.max(np.abs(fd_estimate - (term_p+term_s+term_q))))

    return fd_estimate, term_p, term_s, term_q

def newton_step_with_line_search(u, delta, R_func, max_backtrack=20):
    full_R_norm = np.max(np.abs(R_func(u)))
    t = 1.0
    for i in range(max_backtrack):
        u_trial = u + t * delta
        trial_R_norm = np.max(np.abs(R_func(u_trial)))
        if trial_R_norm < full_R_norm:
            return u_trial, t
        t *= 0.5
    return u, 0.0 # no improving step found -- stay put, rather than silently accepting a worse one

# (u^mu′)′+1=0,u(±1)=0
N = 32
n_basis = 30

D, x = cv.chebyshev_diff_matrix(N)
D2 = D@D
Phi = cv.build_dirichlet_basis(x, n_basis)
Phi_p = D @ Phi
Phi_dp = D2 @ Phi
w = cv.clenshaw_curtis_weights(N)
W = np.diag(w)

print("Phi stats")
print(np.max(np.abs(Phi)))
print(np.linalg.cond(Phi.T @ Phi))

print("Phi prime stats")
print(np.max(np.abs(Phi_p)))
print(np.linalg.cond(Phi_p.T @ Phi_p))

print("Phi dprime stats")
print(np.max(np.abs(Phi_dp)))
print(np.linalg.cond(Phi_dp.T @ Phi_dp))

# define the residual.
def R_func(u):
    return D @ (u**m * D @ u) + 1

# initial guess
u = (1 - x ** 2) * (1 + x)
#u = np.sqrt(np.maximum(1 - x**2, 0)) * 1.2
u_0 = u.copy()

# problem parameter
m = 1
tol = 1e-4 # for representability
max_newton_steps = 10
max_r = []
for step in range(max_newton_steps):
    R = D @ (u**m * D @ u) + 1
    max_r.append(np.max(np.abs(R)))
    if np.max(np.abs(R)) < tol:
        print(f"Converged in {step} Newton steps")
        print(np.max(np.abs(R)))
        best_r = R.copy()
        break

    p = u ** m
    P = np.diag(p)

    up = D @ u
    s = 2 * m * u**(m-1) * up
    S = np.diag(s)

    q = D @ (m * u**(m-1) * up)
    Q = np.diag(q)

    K = Phi.T @ W @ P @ Phi_dp + Phi.T @ W @ S @ Phi_p + Phi.T @ W @ Q @ Phi
    # print(f"Condition of K: {np.linalg.cond(K)}")
    rhs = -Phi.T @ W @ R

    delta_coeffs = np.linalg.solve(K, rhs)
    delta = Phi @ delta_coeffs
    term1 = Phi.T @ W @ P @ Phi_dp
    term2 = Phi.T @ W @ S @ Phi_p
    term3 = Phi.T @ W @ Q @ Phi
    u, t_used = newton_step_with_line_search(u, delta, R_func)

else:
    print("Newton did not converge within max_steps")

plt.plot(x, u, label="final")
plt.plot(x, u_0, label="initial")
plt.plot(x, np.sqrt(1 - x ** 2), label="Analytic", linestyle="--")
plt.title(f"Final vs. Analytic. R={np.max(np.abs(R))}")
plt.legend()
plt.show()

plt.plot(range(len(max_r)), np.log10(np.array(max_r)))
plt.title("Errors vs step")
plt.xlabel("Step")
plt.ylabel("log|Error|")
plt.show()