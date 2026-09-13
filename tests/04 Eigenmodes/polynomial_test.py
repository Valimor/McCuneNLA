import numpy as np
import matplotlib.pyplot as plt
import time

from linalg import eigen

# TODO: make this more efficient
# clearly a sparse matrix.
# note: it's also hessenberg!
def polynomial_companion_matrix(p_coefs):
    # recursion
    if p_coefs.shape[0] == 0:
        return np.nan
    if np.abs(p_coefs[0]) < 1e-14:
        return polynomial_companion_matrix(p_coefs[1:])

    # protect
    p_coefs = p_coefs.copy()

    # monic
    p_coefs = p_coefs/p_coefs[0]
    n = p_coefs.shape[0] - 1

    # make the block matrix
    C = np.zeros((n,n), dtype=np.float64)
    C[1:n,0:n-1] = np.eye(n-1)
    C[:,-1] = -p_coefs[1:][::-1]
    return C

def make_callable_poly(p):
    def polynomial(x):
        total = 0
        for coef in p:
            total *= x
            total += coef
        return total
    return polynomial

rng = np.random.default_rng(seed=10)

display_errors = False

# making the for loop to generate all of the data
num = 32 # different resolutions
max_size = 128 # highest dimension

dimensions = np.logspace(2, np.log2(max_size), num=num, endpoint=True, base=2, dtype=np.int64)

# time storage objects
qr_times = np.zeros_like(dimensions, dtype=np.float64)
qr_givens_times = np.zeros_like(dimensions, dtype=np.float64)
np_times = np.zeros_like(dimensions, dtype=np.float64)

# error storage objects
qr_errors = np.zeros_like(dimensions, dtype=np.float64)
qr_givens_errors = np.zeros_like(dimensions, dtype=np.float64)
np_errors = np.zeros_like(dimensions, dtype=np.float64)

# steps storage objects
qr_steps = np.zeros_like(dimensions, dtype=np.float64)
qr_givens_steps = np.zeros_like(dimensions, dtype=np.float64)

# condition storage object
conditions = np.zeros_like(dimensions, dtype=np.float64)

# TODO: store these in a text file so I can change my plotting without having to rerun the code
for idx, dim in enumerate(dimensions):
    # initialize the polynomial
    p_coefs = rng.normal(size=(dim,))
    polynomial = make_callable_poly(p_coefs)

    t0 = time.time()
    C = polynomial_companion_matrix(p_coefs)
    roots, _, steps = eigen.QR_eigen_algorithm(C, convergence_diagnostic=True)
    t1 = time.time()
    qr_steps[idx] = steps
    qr_times[idx] = t1 - t0
    max_error = 0
    for root in roots:
        delta = np.abs(polynomial(root))
        if display_errors:
            print(f"Root {root} has value {delta}")
        if delta > max_error:
            max_error = delta
    qr_errors[idx] = max_error
    conditions[idx] = np.linalg.cond(C)

    t0 = time.time()
    C = polynomial_companion_matrix(p_coefs)
    roots, _, steps = eigen.QR_eigen_givens_algorithm(C, convergence_diagnostic=True)
    t1 = time.time()
    qr_givens_steps[idx] = steps
    qr_givens_times[idx] = t1 - t0
    max_error = 0
    for root in roots:
        delta = np.abs(polynomial(root))
        if display_errors:
            print(f"Root {root} has value {delta}")
        if delta > max_error:
            max_error = delta
    qr_givens_errors[idx] = max_error

    # numpy validation
    t0 = time.time()
    P_np = np.polynomial.Polynomial(p_coefs[::-1])
    P_np_roots = P_np.roots()
    t1 = time.time()
    np_times[idx] = t1 - t0
    max_error = 0
    for root in roots:
        delta = np.abs(P_np(root))
        if display_errors:
            print(f"Root {root} has value {delta}")
        if delta > max_error:
            max_error = delta
    np_errors[idx] = max_error

    print(f"Finished step {idx + 1}/{num} w/ degree {dim}")

# plotting
# do gridspec?
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2)

# time axes
time_ax = fig.add_subplot(gs[0,0])
time_ax.loglog(dimensions, qr_times, label="QR time")
time_ax.loglog(dimensions, qr_givens_times, label="QR Givens time")
time_ax.loglog(dimensions, np_times, label="Numpy times")
time_ax.set_xlabel("Polynomial degree")
time_ax.set_ylabel("Solution time (s)")
time_ax.legend()
time_ax.set_title("Solution time vs. Degree")

# steps axes
steps_ax = fig.add_subplot(gs[0,1])
steps_ax.loglog(dimensions, qr_steps, label="QR Steps")
steps_ax.loglog(dimensions, qr_givens_steps, label="QR Givens Steps")
steps_ax.set_xlabel("Polynomial degree")
steps_ax.set_ylabel("Steps taken to find roots")
steps_ax.set_title("Solution steps vs. Degree")
steps_ax.legend()

# error axes
error_ax = fig.add_subplot(gs[1,0])
error_ax.loglog(dimensions, qr_errors, label="QR Error")
error_ax.loglog(dimensions, qr_givens_errors, label="QR Givens errors")
error_ax.loglog(dimensions, np_errors, label="Numpy errors", linestyle="--")
error_ax.loglog(dimensions, conditions, label="Condition number")
error_ax.set_xlabel("Polynomial degree")
error_ax.set_ylabel("Polynomial error at root")
error_ax.set_title("Polynomial value at root vs. Degree")
error_ax.legend()

fig.suptitle("Evaluating polynomial root finding")
fig.tight_layout()
fig.show()

plt.show()