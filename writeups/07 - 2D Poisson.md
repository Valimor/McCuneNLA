# 2D Poisson Equation Solver/Eigendecomposer

## What was implemented

I implemented a Galerkin and finite-differences solver for the 2D Poisson equation on a square domain with dirichlet boundary conditions. Then, I used the same matrix setup and my existing eigenmode code to find the eigenmodes for the differential equation with both my Galerkin code and the Finite difference code.

## Validation

I solved against a known solution and compared the results to the analytic solution. Futhermore, when plotting I showed the error specifically.

(need to find the analytic eigenmodes!)

## The interesting numerical behavior

Since the domain was so well-behaved, the solution process was able to be dramatically expedited with the sylvester equation. By implementing a solver for the sylvester equation, the solution speed was able to be significantly faster than for the direct case. It was especially interesting, as Sylvester outperformed the Galerkin methods in essentially every metric. It provided slightly worse error once the machine precision became a limiting factor, but it also had better setup time and better solution time. This was only possible because of the rectangular structure of the domain leading to the symmetries in the sylvester case.

(analytic eigenmodes of this)

## What surprised me or broke

I expected the significance of the number of basis points to be smaller. However, I found that the time required to solve for the eigendecomposition for the equation with $N=40$ was significantly longer than for $N=32$ (ended up being close to 8x as long with my self-constructed eigen library). As a result, I needed to switch to the numpy builtin code to make everything work.

I was also surprised at how the error distributions were different for finite differenes vs. galerkin. Galerkin provided significantly better performance for error with small N than finite differences did, and the error was concentrated differently. Furthermore, the error for Galerkin was not rotationally invariant as the finite differences was. I suspect this is because of the presence of significantly more computations in the setup, so this invariance was broken.

[eigenmodes]