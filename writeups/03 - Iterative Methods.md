# Iterative Methods for Solving

Check convergence/non-convergence behavior on a couple of these, including at least one of your actual PDE-derived matrices.
Fold that into the opening of the Krylov writeup as motivation, rather than a separate document.
Move the bulk of your writeup effort into CG and GMRES themselves, which are the methods that'll actually matter going forward.

## What was implemented

4 different iterative methods to solve linear systems were programmed:

- Jacobi
- Gauss-Seidel
- Conjugate Gradient
- GMRes

The first two were test programs, while the latter two the strong majority of the time. As a bonus, GMRes also required building tools to solve Arnoldi iterations and required the solution of least-squares problems. Since least-squares was not the focus of this section, the numpy built-in was used.

## Validation

The code behaved as expected for all of the above methods. In other words, Jacobi converged iff the matrix had the desired spectral properties and did not converge otherwise. Furthermore, Gauss-Seidel showed the same behavior. The Krylov subspace methods were more interesting a

## The interesting numerical behavior

It was interesting to me how a solution to the full-rank system was so closely approximated by a non-full rank solution (in the case of the Krylov subspace methods). These methods clearly failed at lower dimensions, so I found it fascinating that the generalization to higher dimensions is easily able to capture most of the behavior.

It was also interesting how the Krylov subspace methods were so much more robust than the naive iterative methods. It showed how the non-krylov subspace methods were simply worse, as they took more computation, had more stringent requirements to be used, and did not proide any other useful computation. In contrast, conjugate gradient and gmres can both be easily used to compute the krylov subspace (with gmres directly using the Arnoldi iteration that is used to approximate the eigenmodes in the next section).

## What surprised you or broke

I was surprised at how poorly behaved a normally-distributed $N\times N$ matrix was for $N$ greater than 10. I knew that the condition number grew very rapidly, but this was a good demonstration of how poorly it really behaved. For the least robust methods, most of the time it outright failed to converge. Even in the most robust (GMRes), it only converged to near machine precision on the very last step.

I was also surprised by how even the Krylov subspace methods were not immune to the effects of conditioning. For $N=10$, the hilbert matrix led to significant errors in GMRes and Conjugate gradient. However, for the high-condition numbered matrices, GMRES performed better (in terms of residual norm error) than Conjugate Gradient.
