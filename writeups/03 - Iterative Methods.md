# Iterative Methods for Solving

Suggested next steps
Implement Jacobi and Gauss-Seidel quickly (reuse your existing test-matrix infrastructure — conditioning sweep generator, Hilbert matrix, maybe a Chebyshev BVP matrix).
Check convergence/non-convergence behavior on a couple of these, including at least one of your actual PDE-derived matrices.
Fold that into the opening of the Krylov writeup as motivation, rather than a separate document.
Move the bulk of your writeup effort into CG and GMRES themselves, which are the methods that'll actually matter going forward.

## What was implemented

- one paragraph, plain description, no code.

## Validation

- how you know it's correct. Comparison against a known analytic solution, against scipy/numpy, or a convergence-order check. This is the most important section for a research audience. it's the difference between "I wrote code" and "I verified my code is right."

## The interesting numerical behavior

- conditioning plots, convergence plots, error vs. N, eigenvalue distributions. This is where the actual research content lives (e.g., "Chebyshev differentiation matrix condition number grows like O(N^4)" is a finding, not a footnote).

## What surprised you or broke

- genuinely valuable for a research portfolio. "I expected X, got Y, here's why" reads as more sophisticated than a clean success story, and it's honest.
Plots, generated from a script you keep alongside the writeup so they're reproducible, not just pasted images
