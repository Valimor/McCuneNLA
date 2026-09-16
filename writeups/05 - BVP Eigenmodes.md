# Markdown Note Template

## What was implemented

I implemented a framework that solves BVPs for their eigenmodes. To do this, I had to implement a series of new functionality, including:

- a Cholesky decomposition function
- a generalized eigenvalue solver
- a Chebyshev node generator
- a Chebyshev polynomial generator
- a Chebyshev differentiation matrix generator
- a series of functions that recombine the basis to impose boundary conditions. These include:
  1. Dirichlet
  2. Neumann
  3. Mixed dirichlet/neumann
  4. Robin
- a test suite of functions

## Validation

Validation for the dirichlet and neumann boundary conditions for the system $u'' = -\lambda u$ was done by comparing the analytical solution to the result. This was used as a debugging tool, as there were many times where the analytically derived solutoin failed to agree on both the eigenvalue and the eigenvector.

From there, I worked on solving the aforementioned system with mixed and robin boundary conditions. The former also had a closed form solution, which I validated against. The latter did not have as well-known of an analytical solution, so I validated it with a Rayleigh quotient instead.

After that, I moved onto Sturm-Liouville problems. These were more complex, so I had to validate with a generalized Rayleigh quotient. For some problems (such as the quantum harmonic oscillator) I had an analytically known solution, but for others the generalized Rayleigh quotient was all I had.

## The interesting numerical behavior

First of all, the conditioning of the second-derivative matrix was horrendous. When more and more basis vectors were added, instead of the solution converging to a more accurate answer it instead had many "wobbles" introduced into the eigenmodes. They didn't converge towards the analytical solution, and it seemed like they weren't converging to anything physical at all. To fix this, an integration-by-parts technique was used to replace the $-\Phi^T_{xx}\Phi$ with $\Phi_x^T\Phi_x$, which was tenable because the boundary conditions had $\Phi_x$ at the endpoints evaluate to zero. Once this change was implemented the eigenvectors immediately converged to the analytically expected value. However, the eigenvalue was still incorrect. Only later did I realize that the naive approach for performing the integration did not take into account the variable $dx$ that were introduced by the chebyshev points, and the incorporation of those immediately fixed the eigenvalue. This also fixed some of the bugs in the dirichlet condition case, which I only realized after going back and validating.

## What surprised me or broke

I was surprised at how much the relative complexity of the dirichlet and the neumann boundary conditions differed. The dirichlet boundary conditions were simple - once I had my generalized eigenvalue solver done, I simply plugged in the differentation matrix and voila, out came the expected answer. The neumann case as described above was much harder

I was also surprised at how easy it was to implement the mixed and robin boundary conditions after I implemented the neumann bondary conditions. Furthermore, I was also surprised at how simple the implementation of Sturm-Liouville problems (such as the quantum harmonic oscillator) was on top of the existing architecture.