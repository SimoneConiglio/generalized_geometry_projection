Developer Guide
===============

This guide outlines the mathematical foundation, architectural patterns, and engineering decisions behind the ``ggp`` framework.

Mathematical Foundation
-----------------------

The **Generalized Geometry Projection (GGP)** method parametrizes a design domain using explicit geometric primitives (e.g., rectangles, ellipses) rather than a dense grid of pixel densities. This framework implements the 2D Free formulation based on the original work by Coniglio et al.

### 1. Primitive Mapping

Each component :math:`i` is defined by a set of continuous variables :math:`x_i = [X_c, Y_c, L, H, \theta]`.
The signed distance field :math:`\psi_i(x, y)` of a primitive is smoothed to ensure differentiability. The local density :math:`\rho_i(x, y)` is obtained via a **Regularized Heaviside function**, ensuring that the density smoothly transitions from 1 (inside the component) to 0 (outside) over a narrow band :math:`\epsilon_{mna}`.

### 2. Saturated Kreisselmeier-Steinhauser (KS) Aggregation

To combine multiple overlapping components into a single global density field :math:`\rho(x, y)`, the KS function is used as a smooth approximation of the maximum operator:

.. math::
    KS(\rho) = \frac{1}{\kappa_a} \ln \left( \frac{1}{N} \sum_{i=1}^N \exp(\kappa_a \rho_i) \right)

Because the KS function can exceed 1.0 (causing issues for physics solvers), we apply a **smooth saturation function** to strictly bound the final density :math:`\rho \in [0, 1]`.

Architecture & Object-Oriented Design
-------------------------------------

The ``ggp`` package is structured to decouple the geometry parametrization from the physics solvers, enabling easy extension to 3D or Additive Layer Manufacturing (ALM) constraints without rewriting the optimization loop.

.. mermaid::
    :align: center

    graph TD
        subgraph GEMSEO
            A[MMA Optimizer] -->|Design Variables 'x'| B(GGPMacroDiscipline)
        end
        
        subgraph FEniCS Tape
            B -->|x| C{GeometryFactory: GGP2DMapper}
            C -->|Density Field 'rho'| D{PhysicsFactory: LinearElasticitySolver}
            D -->|Compliance 'J', Volume 'V'| B
        end
        
        subgraph dolfin-adjoint
            B -.->|Request Gradient| E(compute_gradient)
            E -.->|dJ/dx, dV/dx| B
        end
        
        B -->|Objective & Gradients| A

The GEMSEO "Macro-Discipline"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A critical engineering challenge was integrating ``dolfin-adjoint`` (which tracks FEniCS operations) with ``GEMSEO`` (the MDAO orchestration engine).

**Why not a modular MDAChain?**
Initially, we attempted to split the process into two GEMSEO disciplines: a Geometry discipline and a Physics discipline.
However, GEMSEO's chain rule requires explicitly computing the intermediate Jacobian :math:`\frac{\partial \rho}{\partial x}`. For a 50x50 mesh and 50 design variables, this meant computing and passing a dense 2500x50 matrix every iteration. This completely broke the analytical efficiency of the adjoint method.

**The Solution: Monolithic Adjoint Tracking**
We implemented the ``GGPMacroDiscipline``. This single GEMSEO wrapper takes the primitive variables :math:`x`, constructs ``dolfin_adjoint.Constant`` objects, and runs the entire forward pass (Mapping + Physics).
When GEMSEO requests the gradient :math:`\frac{\partial J}{\partial x}`, ``dolfin-adjoint`` solves the adjoint PDE and computes the exact sensitivities analytically in a fraction of a second, entirely bypassing the dense Jacobian.

Tape Management & Safe Re-execution
-----------------------------------

To prevent memory leaks and graph corruption across thousands of optimization iterations, the ``_run`` method executes a strict protocol:

1. ``get_working_tape().clear_tape()`` is called to destroy the previous iteration's computational graph.
2. Fresh ``dolfin_adjoint.Constant`` objects are instantiated.
3. The forward graph is rebuilt from scratch.

Performance Monitoring
----------------------

To track the evolution of code performance and identify bottlenecks, a profiling infrastructure is provided.

### 1. Running the Profiling Suite

To run a fast profile of all standard benchmarks (5 iterations each), execute:

.. code-block:: bash

   PYTHONPATH=$PYTHONPATH:. python profile_suite.py

This script will:
- Run all 4 benchmarks.
- Generate detailed ``.prof`` files in the ``performance_logs/`` directory.
- Log the average time per iteration and the Git commit hash to ``performance_history.json``.

### 2. Analyzing Bottlenecks

Detailed profiling data can be visualized using tools like `snakeviz`:

.. code-block:: bash

   pip install snakeviz
   snakeviz performance_logs/short_cantilever.prof

Citations
---------

- **Original Paper:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). https://doi.org/10.1007/s11831-019-09362-8
- **Original MATLAB Code:** `GGP-Matlab Repository <https://github.com/topggp/GGP-Matlab>`_
y <https://github.com/topggp/GGP-Matlab>`_
