# Uno solvers on the Short Cantilever 2D

Objective log(C+1), volume <= 40 %, 18 GP bars (108 variables), direct FE solver,
same budget of FE analyses for every run (GEMSEO `max_iter`). C is the final
compliance; *vs MMA* is relative to the preset MMA run. All Uno runs use an
L-BFGS Hessian (`filterslp`: zero Hessian) unless stated. `_cs`: volume
constraint scaled by 0.01 (the response is in percent); `_trR`: initial
trust-region radius R in the normalised design space; `fix`: radius never
enlarged (`TR_increase_factor = 1`), i.e. an MMA-like move limit.

`docs/_static/sc2d_uno_convergence.png` shows C and the volume constraint at
every FE analysis (rejected trial points included). Gradients are analytic
(adjoint): one FE solve per evaluation. The SQP iterates approach the active
volume constraint from slightly outside (violations ~1e-3..1e-2 %-points),
MMA from inside.

Regenerate with `python -m benchmarks.sc2d_uno --report`.

| Config | Evals | Compliance C | vs MMA | Volume con. | Time (s) | Uno options |
|---|---|---|---|---|---|---|
| filterslp_cs | 8 | 42358.85 | +56940.46% | -8.1e+01 | 4 | `{"preset": "filterslp", "constraint_scale": 0.01}` |
| filterslp_cs_tr0.02 | 11 | 1020.97 | +1274.84% | -1.2e+01 | 2 | `{"preset": "filterslp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.02}}` |
| mma | 320 | 74.26 | +0.00% | -3.9e-04 | 163 | `preset MMA` |
| filtersqp_cs_tr0.005fix | 320 | 74.50 | +0.32% | -1.7e-04 | 47 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.005, "TR_increase_factor": 1.0}}` |
| funnelsqp_cs_tr0.005fix | 320 | 74.50 | +0.32% | -1.7e-04 | 48 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.005, "TR_increase_factor": 1.0}}` |
| filtersqp_cs_tr0.01fix | 320 | 74.56 | +0.40% | -5.8e-07 | 44 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01, "TR_increase_factor": 1.0}}` |
| funnelsqp_cs_tr0.01fix | 320 | 74.58 | +0.42% | +4.9e-05 | 45 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01, "TR_increase_factor": 1.0}}` |
| funnelsqp_cs_tr0.02 | 320 | 75.14 | +1.18% | -3.6e-05 | 48 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.02}}` |
| filtersqp_cs_tr0.02 | 320 | 75.23 | +1.31% | -1.4e-03 | 46 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.02}}` |
| filtersqp_cs_tr0.02fix | 320 | 75.61 | +1.82% | +3.3e-05 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.02, "TR_increase_factor": 1.0}}` |
| funnelsqp_cs_tr0.02fix | 320 | 75.63 | +1.84% | +7.6e-05 | 46 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.02, "TR_increase_factor": 1.0}}` |
| filtersqp_cs_tr0.1 | 320 | 76.02 | +2.37% | -3.0e-03 | 48 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.1}}` |
| funnelsqp_cs_tr0.1 | 320 | 76.02 | +2.37% | -3.0e-03 | 45 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.1}}` |
| filterslp_cs_tr0.1 | 320 | 77.98 | +5.01% | -2.1e-03 | 46 | `{"preset": "filterslp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.1}}` |
| filtersqp_cs_m20 | 320 | 81.58 | +9.85% | -7.6e-03 | 48 | `{"preset": "filtersqp", "constraint_scale": 0.01, "quasi_newton_memory_size": 20}` |
| filtersqp_cs_tr0.01 | 320 | 83.54 | +12.50% | -7.0e-05 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01}}` |
| funnelsqp_cs_tr0.01 | 320 | 83.77 | +12.81% | -1.4e-04 | 44 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01}}` |
| filterslp_cs_tr0.01 | 320 | 92.01 | +23.90% | -1.2e-03 | 43 | `{"preset": "filterslp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01}}` |
| filtersqp_cs | 320 | 99.62 | +34.14% | +5.8e-05 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.01}` |
| funnelsqp_cs | 320 | 100.58 | +35.44% | -6.5e-04 | 45 | `{"preset": "funnelsqp", "constraint_scale": 0.01}` |
| ipopt_cs_mu0.01 | 320 | 102.55 | +38.09% | -2.6e-01 | 45 | `{"preset": "ipopt", "constraint_scale": 0.01, "uno_options": {"barrier_initial_parameter": 0.01}}` |
| filtersqp_cs0.001 | 320 | 109.98 | +48.10% | +2.5e-05 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.001}` |
| ipopt_cs_m20 | 320 | 120.94 | +62.85% | -5.4e-01 | 46 | `{"preset": "ipopt", "constraint_scale": 0.01, "quasi_newton_memory_size": 20}` |
| filtersqp_cs0.1 | 320 | 352.42 | +374.57% | +4.9e-05 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.1}` |
| filtersqp | 320 | 359.56 | +384.19% | +3.2e-06 | 45 | `{"preset": "filtersqp"}` |
| filterslp | 320 | 389.10 | +423.96% | -1.0e-05 | 46 | `{"preset": "filterslp"}` |
| filtersqp_cs_id | 320 | 554.23 | +646.33% | +7.5e-06 | 45 | `{"preset": "filtersqp", "constraint_scale": 0.01, "hessian_model": "identity"}` |
| funnelsqp | 320 | 746.49 | +905.22% | -1.9e-02 | 46 | `{"preset": "funnelsqp"}` |
| filterslp_cs_tr0.01fix | 320 | 894.72 | +1104.84% | -8.8e+00 | 44 | `{"preset": "filterslp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01, "TR_increase_factor": 1.0}}` |
| ipopt_cs | 320 | 1113.88 | +1399.95% | -6.3e-03 | 47 | `{"preset": "ipopt", "constraint_scale": 0.01}` |
| ipopt | 320 | 2622.07 | +3430.88% | -8.0e+01 | 47 | `{"preset": "ipopt"}` |
| filtersqp_cs_tr0.01fix | 640 | 74.49 | +0.31% | +1.3e-05 | 93 | `{"preset": "filtersqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01, "TR_increase_factor": 1.0}}` |
| funnelsqp_cs_tr0.01fix | 640 | 74.51 | +0.33% | +3.1e-05 | 91 | `{"preset": "funnelsqp", "constraint_scale": 0.01, "uno_options": {"TR_radius": 0.01, "TR_increase_factor": 1.0}}` |
