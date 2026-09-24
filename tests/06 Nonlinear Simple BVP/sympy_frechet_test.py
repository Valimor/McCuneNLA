import sympy as sp

x, eps, m = sp.symbols('x epsilon m')
u = sp.Function('u')(x)
delta = sp.Function('delta')(x)

def F(u_expr):
    return sp.diff(u_expr**m * sp.diff(u_expr, x), x) + 1

# perturb u -> u + eps*delta, differentiate w.r.t. eps, evaluate at eps=0
F_perturbed = F(u + eps * delta)
frechet = sp.diff(F_perturbed, eps).subs(eps, 0)
frechet = sp.simplify(frechet)
print(frechet)  