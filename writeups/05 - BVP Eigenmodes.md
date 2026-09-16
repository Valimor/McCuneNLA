# Markdown Note Template

## What was implemented

I implemented a framework that solves BVPs for their eigenmodes. To do this, I had to implement a series of new functionality, including:

- a Cholesky decomposition function
- a generalized eigenvalue solver
- a Chebyshev node generator
- a Chebyshev polynomial generator
- a Chebyshev differentiation matrix generator
- a function that recombines the basis to impose boundary conditions.
- a test suite of functions

## Validation

Validation for the dirichlet and neumann boundary conditions for the system $u'' = -\lambda u$ was done by comparing the analytical solution to the result. This was used as a debugging tool, as there were many times where the analytically derived solutoin failed to agree on both the eigenvalue and the eigenvector.

## The interesting numerical behavior

[todo]

## What surprised me or broke

I was surprised at how much the relative complexity of the dirichlet and the neumann boundary conditions differed. The dirichlet boundary conditions were simple - once I had my generalized eigenvalue solver done, I simply plugged in the differentation matrix and voila, out came the expected answer. The neumann conditions were quite different.

First of all, the conditioning of the second-derivative matrix was horrendous. When more and more basis vectors were added, instead of the solution converging to a more accurate answer it instead had many "wobbles" introduced into the eigenmodes. They didn't converge towards the analytical solution, and it seemed like they weren't converging to anything physical at all. To fix this, an integration-by-parts technique was used to replace the $-\Phi^T_{xx}\Phi$ with $\Phi_x^T\Phi_x$, which was tenable because the boundary conditions had $\Phi_x$ at the endpoints evaluate to zero. Once this change was implemented the eigenvectors immediately converged to the analytically expected value. However, the eigenvalue was still incorrect. Only later did I realize that the naive approach for performing the integration did not take into account the variable $dx$ that were introduced by the chebyshev points, and the incorporation of those immediately fixed the eigenvalue. This also fixed some of the bugs in the dirichlet condition case, which I only realized after going back and validating.
