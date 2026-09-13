# Markdown Note Template

## What was implemented

I implemented a few numerical eigensolvers based on the QR algorithm. The first was QR without any extra steps, then QR with shifting and deflation, then a combination of hessenberg decomposition and givens-rotation based QR, then finally applying that last step to create the arnoldi algorithm.

## Validation

I wrote a function that prints the error of the eigenmode by computing $||Av - \lambda v||$ for all of the eigenmode calculations. For the intermediate steps, I compared the results with `np.linalg.eig` for all of the similar matrices. Lastly, for the hessenberg matrices I wrote a function that checks for the largest entry in the area that should be all zeros. These functions were applied in many different locations to both check for bugs and also validate that the different first steps acquired from a single step of Householder and Givens rotations were still correct.

I also used the written QR algorithms to find the roots of polynomials. By constructing the companion matrix of a polynomial, I could compute the roots of a known polynomial and see how they compared to the analytically known value. I also randomly generated polynomials of a given degree, solved for the roots with numpy, then compared the results to the value I got from my algorithm.

## The interesting numerical behavior

Rate of convergence of the QR algorithm was much slower than the equivalent with all the shifting, deflation, and givens rotations. [todo: make a ]


1. Arnoldi iteration provided a shockingly accurate approximation to the tpomost eigenvalues at low dimension of the krylov subspace.
2. Polynomial root finding was not as accurate as I expected, with error around 1e-4 for the known polynomial $(x+1)^4=0$

## What surprised you or broke

I was surprised at how many iterations the naive QR algorithm (with $A_k = Q_kR_k$ and $A_{k+1} = R_kQ_k$) took to converge. For a $4\times 4$ symmetric positive definite matrix, the values above the diagonal for the computed $R_k$ matrix only got down to values around $1e-12$ (close to machine precision) after nearly 200 iterations. I was surprised at how such a small matrix that was very well-conditioned still took so long to converge.

[TODO: get more information/plots on the matrix convergence]

I was surprised at the behavior for the polynomial root finding. I generated a random polynomial of logarithmically spaced degrees from 2 to 128 and found that the QR algorithm performed almost exactly the same (in terms of time requirement) until it passed the degree 80 mark. I suspect this is because the numpy vectorization was so efficient that it was essentially the same as the single float multiplication from the givens rotation, but by the end the behavior had begun to diverge.

I was also surprised by the magnitude of the errors of the roots of the polynomial, which I defined to be the absolute value of the polynomial evaluated at the root. It generally increased with polynomial degree, but the errors were not consistent between QR and QR givens. However, the errors were exactly identical between the QR givens and the numpy. I looked this up later and found that numpy uses the qr givens algorithm (with many efficiency improvements) to find the roots of a polynomial. It was cool to see that the exact similarity of the behavior serving as a look behind the curtains of numpy.

It was also interesting to plot the magnitude of the errors vs the condition number. The errors vaguely trended with the condition number, with a slight uptick in condition number at higher degrees indicating massive errors. However, the conditions numbers did not vary much with polynomial degree while the errors did. 