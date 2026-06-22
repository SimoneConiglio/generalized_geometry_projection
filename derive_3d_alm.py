import sympy as sp

dx, dy, dz = sp.symbols('dx dy dz')
L, W_width, h = sp.symbols('L W_width h', real=True)
T = sp.symbols('T', real=True)
eps_safe = sp.symbols('eps_safe', real=True, positive=True)

cos_t, sin_t = sp.cos(T), sp.sin(T)
x_loc = dx * cos_t + dy * sin_t
y_loc = -dx * sin_t + dy * cos_t
z_loc = dz

# For maximum we can't use sp.Max easily with derivatives without piecewise,
# so we will write out the derivatives manually or use a smooth approximation,
# or we can just derive assuming dx_b > 0 and set them to 0 when dx_b <= 0 in numpy.
#
# Let dx_b = x_loc - L/2
# We know d(dx_b)/d(param) is only non-zero when x_loc > L/2.
# So we can just differentiate x_loc - L/2, y_loc - W/2, z_loc - h/2.
# Then in numpy we just mask out the gradients where dx_b <= 0.
# Same for distance.

def get_alm_gradients():
    # Assume dx_b, dy_b, dz_b > 0 for sympy derivation
    dx_b = x_loc - L/2
    dy_b = y_loc - W_width/2
    dz_b = z_loc - h/2
    
    upsi = sp.sqrt(dx_b**2 + dy_b**2 + dz_b**2 + eps_safe)
    
    return {
        'upsi': upsi,
        'dX': -sp.diff(upsi, dx),
        'dY': -sp.diff(upsi, dy),
        'dZ': -sp.diff(upsi, dz),
        'dL': sp.diff(upsi, L),
        'dW': sp.diff(upsi, W_width),
        'dh': sp.diff(upsi, h),
        'dT': sp.diff(upsi, T)
    }

exprs_alm = get_alm_gradients()

def cse_and_print(name, exprs_dict):
    print(f"def {name}(dx, dy, dz, L, W_width, h, T, eps_safe):")
    replacements, reduced = sp.cse([exprs_dict[k] for k in ['upsi', 'dX', 'dY', 'dZ', 'dL', 'dW', 'dh', 'dT']])
    for sym, expr in replacements:
        print(f"    {sym} = {sp.printing.numpy.NumPyPrinter().doprint(expr)}")
    keys = ['upsi', 'dX', 'dY', 'dZ', 'dL', 'dW', 'dh', 'dT']
    for i, k in enumerate(keys):
        print(f"    {k}_raw = {sp.printing.numpy.NumPyPrinter().doprint(reduced[i])}")
    print("    return upsi_raw, dX_raw, dY_raw, dZ_raw, dL_raw, dW_raw, dh_raw, dT_raw\n")

print("\n")
cse_and_print("grad_3d_alm_raw", exprs_alm)
