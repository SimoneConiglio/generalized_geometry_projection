"""Outer-loop Le-Norato: inner runs use the RAW (stable) Verbart constraint at a
fixed allowable; between runs the allowable is corrected by the measured true
sigma_max ratio. Fixed point: sigma_max = sigma_lim = 0.75."""
import dataclasses as dc, numpy as np, json
from ggp.problem.loader import load_problem
from ggp.optimization.pipeline import GGPPipeline, _stress_exclusion_mask
from ggp.discretisation.fem import FEMDiscretiser
from ggp.geometry.io.registry import get_reader
from ggp.gemseo_wrappers.physics_discipline import GGPPhysicsDiscipline
from ggp.utils.jax_sensitivity import StressConstraintKernel
from ggp.visualization.plot import save_density_plot_2d

SIGMA_LIM = 0.75
results = []

def build_spec(sig_allow):
    spec = load_problem("ggp/cli/presets/l_shape_stress.yaml")
    cons = []
    for c in spec.constraints:
        if c.name == "stress":
            p = dict(c.params)
            p["sigma_limit"] = sig_allow
            p["adaptive"] = False          # RAW constraint inside: stable for MMA
            c = dc.replace(c, params=p)
        cons.append(c)
    opts = dict(spec.solver.options)
    opts.update(max_optimization_step=0.005, initial_asymptotes_distance=0.005,
                max_asymptote_distance=0.005)
    return dc.replace(spec, constraints=cons,
                      formulation=dc.replace(spec.formulation, fix_mc=True),
                      solver=dc.replace(spec.solver, options=opts, max_iter=300))

def measure(spec, res):
    dom = get_reader(spec.geometries[0].type).read(spec.geometries[0])
    an = FEMDiscretiser().discretise(dom, spec)
    fixed = sorted(set(list(an.point_fixed_dofs) +
        [d for bc in an.bcs_applied for d in bc.get_boundary_values().keys()]))
    phys = GGPPhysicsDiscipline(V_u=an.function_spaces["u"], ke_ref=an.ke_ref,
        fixed_dofs=fixed, f_vec=an.load_vector.copy(), mesh_area=1.0,
        volfrac=spec.volfrac, p_penalty=3.0, Emin=1e-6, E0=1.0, fem_solver="direct",
        empty_elements=an.empty_elements if an.empty_elements else None)
    phys._run({"rho_E": res.density_field, "rho_V": res.density_field})
    cd = np.asarray([np.asarray(phys.cell_dofs[e]) for e in range(phys.num_elements)])
    mask = _stress_exclusion_mask(an, res.eval_coords, 3.0)
    k = StressConstraintKernel(cd, an.se_ref, sigma_lim=1.0, dim=2,
                               kind="verbart", P=8.0, active_mask=mask)
    smax = k.max_true_stress(res.density_field, phys.last_u, rho_solid=0.5)
    return smax, float(np.mean(res.density_field))

x = np.load("scratch_bench/recipe_x1.npy").copy()
x[5::6] = 1.0
sig_allow = SIGMA_LIM
for outer in range(3):
    spec = build_spec(sig_allow)
    res = GGPPipeline(spec, x0=x, overrides={"Emin": 1e-6}).run()
    C = float(np.exp(res.objective_value) - 1.0)
    smax, vol = measure(spec, res)
    print(f"OUTER {outer}: sig_allow={sig_allow:.4f} -> C={C:.2f} vol={vol:.3f} "
          f"true smax={smax:.3f} (target {SIGMA_LIM})", flush=True)
    results.append({"outer": outer, "sig_allow": sig_allow, "C": C, "vol": vol,
                    "sigma_max": smax})
    x = np.asarray(res.design_variables, float).flatten()
    last_res, last_spec = res, spec
    if abs(smax - SIGMA_LIM) / SIGMA_LIM < 0.03:
        print("CONVERGED: true sigma_max within 3% of the limit", flush=True)
        break
    # Le correction, damped: scale the allowable by the measured overshoot ratio
    sig_allow = float(np.clip(sig_allow * (SIGMA_LIM / smax) ** 0.7,
                              SIGMA_LIM / 4, SIGMA_LIM * 4))

smax, vol = measure(last_spec, last_res)
C = float(np.exp(last_res.objective_value) - 1.0)
save_density_plot_2d(last_res.density_field, last_res.eval_coords,
                     "docs/_static/lshape_stress_active.png",
                     f"L-bracket, outer-loop Le allowable, C={C:.1f}, "
                     f"max sigma={smax:.2f}/{SIGMA_LIM}")
json.dump(results, open("scratch_bench/outer_result.json", "w"), indent=2)
print("DONE:", json.dumps(results))
