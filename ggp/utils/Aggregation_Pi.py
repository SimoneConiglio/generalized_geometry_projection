import numpy as np

def Aggregation_Pi(z, p):
    """
    Function that performs the aggregation of the values in z and also computes
    sensitivities (gradients).
    """
    zm = np.tile(np.max(z, axis=0), (z.shape[0], 1))
    ka = p['ka']
    
    aggregation = p['aggregation']
    if aggregation == 'asymptotic':
        Wa = np.sum(z, axis=0)
        dWa = np.ones_like(z)
    elif aggregation == 'boolean':
        Wa = 1.0 - np.prod(1.0 - z, axis=0)
        mask = (z != 1.0)
        z_mod = np.where(mask, z, 0.0)
        prod_val = np.prod(1.0 - z_mod, axis=0)
        prod_rep = np.tile(prod_val, (z.shape[0], 1))
        dWa = prod_rep / np.where(mask, 1.0 - z, 1.0)
    elif aggregation == 'p-norm':
        zp = p['zp']
        zm = zm + zp
        z = z + zp
        zm_row = zm[0, :]
        z_over_exp_zm = z / np.exp(zm)
        sum_term = np.sum(z_over_exp_zm**ka, axis=0)
        Wa = np.exp(zm_row) * (sum_term**(1.0 / ka)) - zp
        dWa = (z_over_exp_zm**(ka - 1)) * np.tile(sum_term**(1.0 / ka - 1.0), (z.shape[0], 1))
    elif aggregation == 'p-mean':
        zp = p['zp']
        zm = zm + zp
        z = z + zp
        zm_row = zm[0, :]
        z_over_exp_zm = z / np.exp(zm)
        sum_term = np.sum(z_over_exp_zm**ka, axis=0)
        mean_term = sum_term / z.shape[0]
        Wa = np.exp(zm_row) * (mean_term**(1.0 / ka)) - zp
        dWa = (1.0 / (z.shape[0]**(1.0 / ka))) * (z_over_exp_zm**(ka - 1)) * np.tile(sum_term**(1.0 / ka - 1.0), (z.shape[0], 1))
    elif aggregation == 'KS':
        zm_row = zm[0, :]
        exp_term = np.exp(ka * (z - zm))
        sum_exp = np.sum(exp_term, axis=0)
        Wa = zm_row + (1.0 / ka) * np.log(sum_exp)
        dWa = exp_term / np.tile(sum_exp, (z.shape[0], 1))
    elif aggregation == 'KSl':
        zm_row = zm[0, :]
        exp_term = np.exp(ka * (z - zm))
        sum_exp = np.sum(exp_term, axis=0)
        mean_exp = sum_exp / z.shape[0]
        Wa = zm_row + (1.0 / ka) * np.log(mean_exp)
        dWa = exp_term / np.tile(sum_exp, (z.shape[0], 1))
    elif aggregation == 'IE':
        exp_term = np.exp(ka * (z - zm))
        sum_exp = np.sum(exp_term, axis=0)
        z_exp = z * exp_term
        sum_z_exp = np.sum(z_exp, axis=0)
        Wa = sum_z_exp / sum_exp
        
        term1 = (exp_term + ka * z_exp) * np.tile(sum_exp, (z.shape[0], 1))
        term2 = np.tile(sum_z_exp, (z.shape[0], 1)) * ka * exp_term
        dWa = (term1 - term2) / np.tile(sum_exp**2, (z.shape[0], 1))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
        
    return Wa, dWa
