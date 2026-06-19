Developer Guide
===============

This guide outlines the mathematical foundation, architectural patterns, and engineering decisions behind the ``ggp`` framework.

Mathematical Foundation
-----------------------

The **Generalized Geometry Projection (GGP)** method parametrizes a design domain using explicit geometric primitives (e.g., rectangles, ellipses) rather than a dense grid of pixel densities. This framework implements the 2D Free formulation based on the original work by Coniglio et al.

1. Primitive Mapping
^^^^^^^^^^^^^^^^^^^^

Each component :math:`i` is defined by a set of continuous variables :math:`x_i = [X_c, Y_c, L, H, \theta]`, representing its center coordinates, length, thickness, and orientation angle. 

To determine the density contribution of a primitive at any spatial point :math:`(x, y)`, the method computes the projection in four steps:

**Step 1: Local Coordinates**
The coordinates are translated and rotated to the component's local reference frame:

.. math::
    \Delta x = x - X_c, \quad \Delta y = y - Y_c
.. math::
    x_{loc} = \Delta x \cos\theta + \Delta y \sin\theta
.. math::
    y_{loc} = -\Delta x \sin\theta + \Delta y \cos\theta

**Step 2: Skeleton Distance**
The distance :math:`\psi_i(x,y)` from the point to the component's central skeleton (a line segment of length :math:`L`) is calculated:

.. math::
    d_x = \max(0, |x_{loc}| - L/2)
.. math::
    \psi_i(x, y) = \sqrt{d_x^2 + y_{loc}^2}

**Step 3: Signed Distance**
The signed distance variable :math:`\zeta` evaluates whether the point is inside or outside the component's boundary (thickness :math:`H`):

.. math::
    \zeta = \psi_i(x, y) - H/2

(:math:`\zeta < 0` inside, :math:`\zeta > 0` outside).

**Step 4: Regularized Mapping**
To ensure strict differentiability for gradient-based optimization, the local density :math:`\rho_i(x,y)` is mapped from :math:`\zeta` using a smoothed area-fraction function over a narrow transition band :math:`r_{gp}`:

.. math::
    \rho_i(x, y) = 
    \begin{cases} 
    1 & \text{if } \zeta < -r_{gp} \\
    \delta_{min} + (1 - \delta_{min}) \frac{1}{\pi} \left( \arccos(z) - z \sqrt{1 - z^2} \right) & \text{if } -r_{gp} \le \zeta \le r_{gp} \\
    \delta_{min} & \text{if } \zeta > r_{gp} 
    \end{cases}

where :math:`z = \zeta / r_{gp}` is the normalized distance inside the transition band, and :math:`\delta_{min}` is a tiny void density (e.g., :math:`10^{-6}`) to prevent global stiffness matrix singularities.

2. Saturated Kreisselmeier-Steinhauser (KS) Aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To combine multiple overlapping components into a single global density field :math:`\rho(x, y)`, the KS function is used as a smooth approximation of the maximum operator:

.. math::
    KS(\rho) = \frac{1}{\kappa_a} \ln \left( \frac{1}{N} \sum_{i=1}^N \exp(\kappa_a \rho_i) \right)

Because the KS function can exceed 1.0 (causing issues for physics solvers), we apply a **smooth saturation function** to strictly bound the final density :math:`\rho \in [0, 1]`.

3. Continuous Additive Layer Manufacturing (ALM) Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To rigorously enforce Additive Manufacturing overhang constraints natively through the geometry parameterization, the framework employs a continuous layer mapping formulation (matching the original ALM paper).

Instead of deploying floating primitives, the ALM formulation constructs a cohesive part made of :math:`N_y` stacked horizontal layers (or trapezoidal segments).
The design variables represent continuous nodal coordinates exactly at the layer interfaces:

- **Horizontal Coordinates** :math:`X_k \in \mathbb{R}^{N_y \times N_p}`
- **Layer Widths** :math:`L_k \in \mathbb{R}^{N_y \times N_p}`
- **Layer Heights** :math:`h \in \mathbb{R}^{N_y}`
- **Component Densities** :math:`m \in \mathbb{R}^{N_p}`

The total number of optimization parameters is exactly :math:`(2 \times N_y \times N_p) + N_y + N_p`.

The continuous mapping analytically interpolates between the top boundary :math:`(X_{k+1}, L_{k+1}, y_{k+1})` and bottom boundary :math:`(X_k, L_k, y_k)` of each layer :math:`k` to construct continuous trapezoidal blocks. The constraints on these :math:`X_k` coordinates mathematically guarantee maximum overhang angle requirements between consecutive layers without needing heuristic density filters.

Finite Element Solver
---------------------

The physical evaluation is handled by a FEniCS-based Linear Elasticity solver. It takes the projected continuous density field :math:`\rho(x, y)` and evaluates the mechanical compliance (stiffness) of the structure.

**1. Material Penalization (SIMP)**
We use the Solid Isotropic Material with Penalization (SIMP) model to interpolate the Young's modulus :math:`E`:

.. math::
    E(\rho) = E_{min} + \rho^p (E_0 - E_{min})

where :math:`p=3.0` is the penalization power, :math:`E_0 = 1.0` is the solid stiffness, and :math:`E_{min} = 10^{-6}` is a tiny void stiffness to prevent a singular matrix.

**2. Constitutive Relations**
The stress tensor :math:`\sigma(u)` for a displacement field :math:`u` under the assumption of linear isotropic elasticity is:

.. math::
    \sigma(u) = \lambda \text{tr}(\varepsilon(u)) I + 2\mu \varepsilon(u)

where the linear strain is :math:`\varepsilon(u) = \frac{1}{2}(\nabla u + (\nabla u)^T)`. For 2D Plane Stress formulations, the Lamé parameters are defined using Poisson's ratio :math:`\nu = 0.3`:

.. math::
    \lambda = \frac{E \nu}{1 - \nu^2}, \quad \mu = \frac{E}{2(1+\nu)}

**3. Variational Weak Form**
The solver finds the displacement :math:`u \in V` that satisfies the weak form of the equilibrium equation:

.. math::
    \int_\Omega \sigma(u) : \varepsilon(v) d\Omega = \int_{\partial \Omega_N} t \cdot v ds \quad \forall v \in V

where :math:`t` is the traction boundary load applied on :math:`\partial \Omega_N`.

**4. Compliance Objective**
The objective function for stiffness maximization is the compliance :math:`C`, computed as the external work done by the applied loads:

.. math::
    C = \int_{\partial \Omega_N} t \cdot u ds

Architecture & Object-Oriented Design
-------------------------------------

The ``ggp`` package is structured to decouple the geometry parametrization from the physics solvers, enabling easy extension to 3D or Additive Layer Manufacturing (ALM) constraints without rewriting the optimization loop.

.. mermaid::
    :align: center

    graph TD
        subgraph GEMSEO
            A[MMA Optimizer] -->|x_scaled| B(MDAChain)
            B -->|Objective & Gradients| A
        end
        
        subgraph MDAChain
            B -->|x_scaled| C(GGPVectorizedGeometryDiscipline)
            C -->|rho_E, rho_V| D(GGPPhysicsFastDiscipline)
            D -->|Compliance, Volume| B
            
            C -.->|Jacobians: drho/dx| B
            D -.->|Adjoint Gradients: dJ/drho| B
        end

The GEMSEO Modular Fast Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A major contribution of this framework is the implementation of a highly efficient **modular MDAChain** that decouples geometry mapping and physical solving while maintaining high performance.

**1. Decoupled Disciplines**
We implemented two separate GEMSEO disciplines:
- ``GGPVectorizedGeometryDiscipline``: Maps the design parameters :math:`x` to the density fields ``rho_E`` and ``rho_V``. It computes the analytical Jacobian :math:`\frac{\partial \rho}{\partial x}` using fully vectorized NumPy operations.
- ``GGPPhysicsFastDiscipline`` (or ``GGPPhysicsAdjointDiscipline``): Solves the elasticity equations and computes adjoint compliance sensitivities.

**2. High-Performance Sparse Assembly & Linear Solver**
To bypass the overhead of symbolic FEniCS adjoint taping in large-scale optimizations, ``GGPPhysicsFastDiscipline`` uses **SciPy sparse CSR solvers** and vectorized NumPy operations. 

The linear system resolution is highly optimized:
- **Pre-computed Elementary Matrix:** The elementary stiffness matrix is **not** assembled at each iteration. Because the framework uses a structured grid, a single reference unit element stiffness matrix (:math:`K_{ref}` for :math:`E=1.0`) is computed via FEniCS once during initialization.
- **Global Assembly:** At each outer iteration, the global matrix is assembled instantly by computing the penalized Young's modulus :math:`E(\rho_e)` for each element, scaling :math:`K_{ref}`, and injecting the blocks directly into a SciPy Sparse Coordinate (COO) matrix.
- **Direct Resolution:** The COO matrix is converted to a Compressed Sparse Row (CSR) format, Dirichlet boundary conditions are applied directly to the internal data arrays (to bypass slow list-of-lists modifications), and the system :math:`K U = F` is solved using ``scipy.sparse.linalg.spsolve`` (which utilizes the high-performance **SuperLU/UMFPACK** direct solvers). This allows each iteration's state solve and exact analytical gradient calculation to complete in milliseconds.

**3. Trajectory-Level Validation**
Using this fast modular architecture, the script ``Main_ggp.py`` reproduces the exact convergence history and post-processing of the original academic MATLAB code to double-precision accuracy across standard benchmarks (Short Cantilever, MBB, L-Shape).

Tape Management & Safe Re-execution
-----------------------------------

To prevent memory leaks and graph corruption across thousands of optimization iterations in cases using ``GGPPhysicsAdjointDiscipline`` (which tracks FEniCS operations), the ``_run`` method executes a strict protocol:

1. ``get_working_tape().clear_tape()`` is called to destroy the previous iteration's computational graph.
2. Fresh ``dolfin_adjoint.Constant`` objects are instantiated.
3. The forward graph is rebuilt from scratch.

Performance Monitoring
----------------------

To track the evolution of code performance and identify bottlenecks, a profiling infrastructure is provided.

1. Running the Profiling Suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a fast profile of all standard benchmarks (5 iterations each), execute:

.. code-block:: bash

   PYTHONPATH=$PYTHONPATH:. python profile_suite.py

This script will:
- Run all 4 benchmarks.
- Generate detailed ``.prof`` files in the ``performance_logs/`` directory.
- Log the average time per iteration and the Git commit hash to ``performance_history.json``.

2. Analyzing Bottlenecks
^^^^^^^^^^^^^^^^^^^^^^^^

Detailed profiling data can be visualized using tools like `snakeviz`:

.. code-block:: bash

   pip install snakeviz
   snakeviz performance_logs/short_cantilever.prof

Citations
---------

- **Original Paper:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). https://doi.org/10.1007/s11831-019-09362-8
- **Original MATLAB Code:** `GGP-Matlab Repository <https://github.com/topggp/GGP-Matlab>`_
