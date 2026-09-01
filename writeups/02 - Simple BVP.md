# Simple BVP

## What was implemented

I used my decomposition/solving code to solve a simple BVP $f=u''$. I solved it for different boundary condtiions, then proceeded to extend it to the more general case of $f = au'' + bu' + cu$, where $(a,b,c) \in \mathbb{R}^3$. I also compared my solution methods to `np.linalg.solve` and found that numpy was able to solve much more quickly, so I decided to switch to that for most of the tests.

## Validation

I compared the numerically derived solution against the analytic formulation and found error values that were very small for reasonable resolutions.

## The interesting numerical behavior

The convergence speed was one of the most interested numerical results. For polynomial solutions, the collocation method returned the exact answer to within machine precision. For other smooth functions, the convergence followed a three-phase curve of error vs resolution.

1. Starts low and sharply increases: this is because the low-resolution solutions do not have enough variance in them to produce a horribly inaccurate answer.
2. Reaches maximum error, then exponentially falls off to machine precision: this is because the spectral collocation method has exponential convergence
3. After it reaches machine epsilon it begins to grow again: this is due to increasing roundoff errors compounding and altering the solution.

The convergence for the functions that are not smooth everywhere was polynomial, not exponential.

## What surprised me

I was surprised at how the roundoff error was so prominent. I thought that the error in the solution would hit machine epsilon and stay around there instead of increasing again. I was also surprised at how consistent the growth was after the machine epsilon floor was reached. Most of the growth across the different methods was roughly the same, which indicates some theoretical floor.
