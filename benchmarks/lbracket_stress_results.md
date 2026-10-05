# Open-source optimizers on local stress constraints (density L-bracket)

Best V = lowest volume among iterates with max g <= 0.01 (1 % stress overshoot).
Time split: FE = primal solves + adjoint/forward back-substitutions + Jacobian;
opt = the optimizer's own time. Regenerate: `python -m benchmarks.lbracket_stress_report`.

| n = m | Method | Best V (max g <= 1 %) | Final V | Final max g | FE solves | Back-substitutions | Wall (s) | FE (s) | Optimizer (s) | Peak memory (MB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | AL + MMA | 0.3031 | 0.3047 | +1.3e-02 | 1501 | 1.50e+03 | 5 | 4 | 0 | 64 | 150 AL steps |
| 300 | IPOPT | 0.2985 | 0.2985 | +1.0e-08 | 77 | 2.22e+04 | 3 | 1 | 1 | 116 | Algorithm terminated successfully at a l |
| 300 | NLopt MMA | 0.3018 | 0.3018 | +1.0e-06 | 262 | 7.86e+04 | 65 | 5 | 60 | 85 | 3 |
| 300 | ParOpt MMA | 0.3011 | 0.3011 | -6.4e-07 | 301 | 9.03e+04 | 90 | 6 | 85 | 129 | done |
| 300 | ParOpt TR | 0.4766 | 0.4702 | +3.1e-02 | 301 | 9.03e+04 | 87 | 7 | 80 | 128 | done |
| 300 | TAO ALMM | 0.8949 | 0.5107 | +4.2e-02 | 1028 | 1.08e+04 | 6 | 5 | 1 | 100 | -2 |
| 300 | Uno filterSLP | 0.3134 | 0.3136 | +1.1e-07 | 117 | 3.51e+04 | 7 | 2 | 4 | - | SUCCESS |
| 300 | Uno filterSQP | 0.3243 | 0.3243 | +1.1e-13 | 37 | 1.11e+04 | 2 | 1 | 1 | - | SUCCESS |
| 300 | Uno funnelSQP | 0.3149 | 0.3149 | +3.5e-13 | 50 | 1.50e+04 | 3 | 1 | 2 | - | SUCCESS |
| 300 | Uno IPM | 0.3008 | 0.3008 | +9.7e-09 | 153 | 4.56e+04 | 5 | 3 | 3 | - | SUCCESS |
| 1200 | AL + MMA | 0.3856 | 0.4416 | +1.5e-02 | 1501 | 1.50e+03 | 19 | 19 | 0 | 80 | 150 AL steps |
| 1200 | IPOPT | 0.3845 | 0.3845 | +4.3e-08 | 102 | 1.21e+05 | 99 | 49 | 50 | 492 | Algorithm terminated successfully at a l |
| 1200 | NLopt MMA | 0.9920 | 0.8559 | +5.9e+00 | 310 | 3.71e+05 | 204 | 180 | 24 | 327 | 3 |
| 1200 | ParOpt MMA | 0.3787 | 0.3787 | -1.9e-06 | 301 | 3.61e+05 | 4578 | 205 | 4373 | 407 | done |
| 1200 | ParOpt TR | 0.7751 | 0.7751 | +1.1e-03 | 301 | 3.61e+05 | 3745 | 248 | 3497 | 388 | done |
| 1200 | TAO ALMM | nan | 0.5384 | +2.9e-01 | 11010 | 1.30e+04 | 145 | 141 | 4 | 110 | -2 |
| 1200 | Uno filterSLP | 0.3792 | 0.3795 | +1.4e-07 | 122 | 1.46e+05 | 430 | 60 | 370 | - | SUCCESS |
| 1200 | Uno filterSQP | 0.3775 | 0.3775 | +1.1e-12 | 91 | 1.09e+05 | 105 | 43 | 61 | - | SUCCESS |
| 1200 | Uno funnelSQP | 0.3962 | 0.3962 | +1.2e-12 | 227 | 2.72e+05 | 259 | 110 | 148 | - | SUCCESS |
| 1200 | Uno IPM | 0.3821 | 0.3821 | +3.0e-08 | 249 | 2.98e+05 | 327 | 124 | 202 | - | SUCCESS |
| 2700 | AL + MMA | 0.4030 | 0.4159 | +1.9e-02 | 1501 | 1.50e+03 | 55 | 54 | 1 | 100 | 150 AL steps |
| 2700 | IPOPT | 0.4063 | 0.4063 | +6.3e-08 | 227 | 6.08e+05 | 2226 | 812 | 1413 | 1438 | Algorithm terminated successfully at a l |
| 2700 | NLopt MMA | 0.6305 | 0.6305 | -3.5e-03 | 10 | 2.70e+04 | 3600 | 37 | 3563 | 529 | 6 |
| 2700 | TAO ALMM | nan | 0.5518 | +3.6e-01 | 12805 | 1.30e+04 | 677 | 665 | 13 | 205 | -2 |
| 2700 | Uno filterSLP | 0.4127 | 0.4129 | +4.1e-07 | 163 | 4.40e+05 | 2256 | 511 | 1733 | - | SUCCESS |
| 2700 | Uno filterSQP | 0.4035 | 0.4035 | +8.1e-05 | 301 | 8.13e+05 | 2381 | 996 | 1365 | - | ITERATION_LIMIT |
| 2700 | Uno IPM | 0.4057 | 0.4057 | +5.7e-08 | 175 | 4.70e+05 | 3527 | 603 | 2911 | - | SUCCESS |
| 4800 | AL + MMA | 0.4099 | 0.4236 | +1.9e-02 | 1501 | 1.50e+03 | 166 | 164 | 2 | 306 | 150 AL steps |
| 4800 | IPOPT | 0.4127 | 0.4129 | +1.7e-05 | 119 | 5.71e+05 | 3602 | 1880 | 1722 | 3209 | The user call-back function intermediate |
| 4800 | TAO ALMM | nan | 0.5548 | +5.5e-01 | 11589 | 1.33e+04 | 1331 | 1312 | 20 | 196 | -2 |
| 7500 | AL + MMA | 0.4185 | 0.4276 | +1.3e-02 | 1501 | 1.50e+03 | 301 | 298 | 3 | 464 | 150 AL steps |
| 7500 | TAO ALMM | nan | 0.5460 | +8.3e-01 | 12995 | 1.32e+04 | 2764 | 2726 | 38 | 306 | -2 |
| 19200 | AL + MMA | 0.4351 | 0.4395 | +8.2e-03 | 1501 | 1.50e+03 | 1106 | 1091 | 15 | 416 | 150 AL steps |
| 30000 | AL + MMA | 0.4428 | 0.4475 | +2.8e-02 | 1501 | 1.50e+03 | 1927 | 1895 | 32 | 671 | 150 AL steps |
