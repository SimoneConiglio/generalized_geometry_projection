import sympy as sp
from sympy.utilities.codegen import codegen

dx, dy, dz = sp.symbols('dx dy dz')
L, T, P = sp.symbols('L T P', real=True)
eps_safe = sp.symbols('eps_safe', real=True, positive=True)

nx = sp.cos(P) * sp.cos(T)
ny = sp.cos(P) * sp.sin(T)
nz = sp.sin(P)

proj = dx*nx + dy*ny + dz*nz

def build_expr(t_val):
    dist2 = (dx - t_val*nx)**2 + (dy - t_val*ny)**2 + (dz - t_val*nz)**2
    upsi = sp.sqrt(dist2 + eps_safe)
    return {
        'upsi': upsi,
        'dX': -sp.diff(upsi, dx),
        'dY': -sp.diff(upsi, dy),
        'dZ': -sp.diff(upsi, dz),
        'dL': sp.diff(upsi, L),
        'dT': sp.diff(upsi, T),
        'dP': sp.diff(upsi, P)
    }

exprs_m = build_expr(-L/2)
exprs_p = build_expr(L/2)
exprs_mid = build_expr(proj)

def cse_and_print(name, exprs_dict):
    print(f"def {name}(dx, dy, dz, L, T, P, eps_safe):")
    # Using CSE to optimize the numpy math
    replacements, reduced = sp.cse([exprs_dict[k] for k in ['upsi', 'dX', 'dY', 'dZ', 'dL', 'dT', 'dP']])
    for sym, expr in replacements:
        print(f"    {sym} = {sp.printing.numpy.NumPyPrinter().doprint(expr)}")
    keys = ['upsi', 'dX', 'dY', 'dZ', 'dL', 'dT', 'dP']
    for i, k in enumerate(keys):
        print(f"    {k} = {sp.printing.numpy.NumPyPrinter().doprint(reduced[i])}")
    print("    return upsi, dX, dY, dZ, dL, dT, dP\n")

print("import numpy as np\n")
cse_and_print("grad_3d_free_neg", exprs_m)
cse_and_print("grad_3d_free_pos", exprs_p)
cse_and_print("grad_3d_free_mid", exprs_mid)
