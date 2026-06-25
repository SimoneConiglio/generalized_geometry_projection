Developer Guide
===============

This guide outlines the mathematical foundation, architectural patterns, and engineering decisions behind the ``ggp`` framework.

Mathematical Foundation
-----------------------

The **Generalized Geometry Projection (GGP)** method parametrizes a design domain using explicit geometric primitives (e.g., rectangles, ellipses) rather than a dense grid of pixel densities. This framework implements the 2D Free formulation based on the original work by Coniglio et al.

1. Primitive Mapping
^^^^^^^^^^^^^^^^^^^^

Each component :math:`i` is defined by a set of continuous variables
:math:`x_i = [X_c, Y_c, L, H, \theta, M_c]`, representing its center
coordinates, length, thickness, orientation angle, and membership density.

To determine the density contribution of a primitive at any spatial point
:math:`(x, y)`, the evaluation proceeds in two steps.

**Step 1: Local Coordinates**
The global coordinates are translated and rotated to the component's local
reference frame:

.. math::
    \Delta x = x - X_c, \quad \Delta y = y - Y_c

.. math::
    x_{loc} = \Delta x \cos\theta + \Delta y \sin\theta, \quad
    y_{loc} = -\Delta x \sin\theta + \Delta y \cos\theta

**Step 2: Characteristic Function (AMNA — default)**

The default method is **AMNA** (Anisotropic Moving Node Approach, called *MNA*
in the original Matlab reference).  It models each component as an axis-aligned
rectangle in local coordinates, using four independent quintic window
evaluations — one per edge:

.. math::
    \zeta_1 = -\tfrac{L}{2} - x_{loc}, \quad
    \zeta_2 = x_{loc} - \tfrac{L}{2}, \quad
    \zeta_3 = y_{loc} - \tfrac{H}{2}, \quad
    \zeta_4 = -\tfrac{H}{2} - y_{loc}

Each :math:`\zeta_k` measures the signed distance from the corresponding
edge (:math:`\zeta_k < 0` means *inside* that half-space).  The quintic window
function with bandwidth :math:`\sigma` (parameter ``r_gp``) is:

.. math::
    W(\zeta) =
    \begin{cases}
        1 & \zeta < -\sigma \\[4pt]
        \tfrac{1}{2} - \tfrac{15}{16\sigma}\zeta
            + \tfrac{5}{8\sigma^3}\zeta^3
            - \tfrac{3}{16\sigma^5}\zeta^5 & |\zeta| \le \sigma \\[4pt]
        0 & \zeta > \sigma
    \end{cases}

The component characteristic value is the product of all four window
evaluations, integrated over a small Gauss-point sampling window of radius
:math:`r_{gp}` centred on each element:

.. math::
    \chi_i(x, y) = W(\zeta_1)\,W(\zeta_2)\,W(\zeta_3)\,W(\zeta_4)

**Default settings (matching the original reference):**
``method='AMNA'``, ``r_gp=1.0`` (one-element transition band), ``minh=3``
(minimum bar thickness of 3 elements).

**Legacy method (GP — circular capsule)**

The original GP formulation uses a single distance-based transition.  It is
still available via ``method='GP'`` in the YAML formulation block, but
produces less crisp bar boundaries and is not recommended for new runs.

2. Saturated Kreisselmeier-Steinhauser (KS) Aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To combine multiple overlapping components into a single global density field :math:`\rho(x, y)`, the KS function is used as a smooth approximation of the maximum operator:

.. math::
    KS(\rho) = \frac{1}{\kappa_a} \ln \left( \frac{1}{N} \sum_{i=1}^N \exp(\kappa_a \rho_i) \right)

Because the KS function can exceed 1.0 (causing issues for physics solvers), we apply a **smooth saturation function** to strictly bound the final density :math:`\rho \in [0, 1]`.

3. Continuous Additive Layer Manufacturing (ALM) Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ALM formulation constructs a cohesive part made of :math:`N_y` stacked
horizontal layers (or trapezoidal segments).  The design variable vector has
the following **interleaved** layout (matching the GGP\_ALM Matlab reference):

.. code-block:: text

    [Xc_0_0, L_0_0,  Xc_0_1, L_0_1,  …,  Xc_{Ny-1}_{Np-1}, L_{Ny-1}_{Np-1},
     h_0, …, h_{Np-1},
     Mc_0, …, Mc_{Np-1},
     y0, theta0]

- :math:`X_{c,k,j}`, :math:`L_{k,j}` — x-centre and width of column :math:`j` in layer :math:`k`
- :math:`h_j \in [0.2,1]` — normalised total print height per column (:math:`N_p` values)
- :math:`M_{c,j}` — membership density per column, shared across all layers
- :math:`y_0`, :math:`\theta_0` — global printing-plane offset and rotation

Total variables: :math:`2 N_y N_p + 2 N_p + 2`.

The characteristic function uses the quintic MNA window and a height cutoff
:math:`\zeta_3 = y_p - L_y h_j` to terminate each column at its total print
height.  Left/right boundary tests are linearly interpolated between adjacent
layer interfaces to produce smooth trapezoidal segments.  Overhang and
bridge-length constraints are linear in the :math:`(X_c, L)` variables; see
:doc:`alm_guide` for full details.

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

Architecture & Design Patterns
------------------------------

The ``ggp`` package is structured around four interlocking design patterns that enable extension without modifying existing code.

Data Flow
^^^^^^^^^

.. mermaid::
    :align: center

    graph TD
        subgraph Config
            A[YAML file / ProblemSpec] -->|load_problem| B(ProblemSpec)
        end

        subgraph GGPPipeline
            B --> C(GeometryReader registry)
            C -->|DomainRepresentation| D(FEMDiscretiser)
            D -->|AnalysisDomain| E(GGPGeometryDiscipline)
            D --> F(GGPPhysicsDiscipline)
            E -->|rho_E, rho_V| F
            E -.->|drho/dx Jacobian| G
            F -->|compliance, volume| G
            F -.->|adjoint gradients| G
        end

        subgraph GEMSEO
            G(MDAChain) -->|objective + gradients| H[MMA / SLP / CONLIN]
            H -->|x_vars| G
        end

1. Declarative Problem Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every optimisation run is fully described by a single :class:`~ggp.problem.spec.ProblemSpec` instance — a frozen dataclass tree that is the single source of truth. It is constructed from a YAML file by :func:`~ggp.problem.loader.load_problem` and can be overridden programmatically using :func:`dataclasses.replace`:

.. code-block:: python

    from dataclasses import replace
    from ggp.problem.loader import load_problem

    spec = load_problem("short_cantilever.yaml")
    spec = replace(spec, volfrac=0.3,
                   solver=replace(spec.solver, algorithm="SLP", max_iter=100))

Because ``ProblemSpec`` is frozen (``@dataclass(frozen=True)``), it is safe to share across threads and to serialize back to YAML without mutation concerns.

2. Self-Registering Geometry Readers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`ggp.geometry.io.registry` module holds a global dictionary mapping geometry type strings to reader classes. Registering a new geometry source requires only a decorator — no changes to any other file:

.. code-block:: python

    from ggp.geometry.io.registry import register_reader
    from ggp.geometry.io.base import GeometryReader, DomainRepresentation

    @register_reader("my_mesh_format")
    class MyMeshReader(GeometryReader):
        def read(self, spec):
            ...
            return DomainRepresentation(...)

The reader can then be referenced in any YAML file as ``type: my_mesh_format``.

3. Self-Registering Projection Mappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`ggp.projection.registry` module follows the same decorator pattern for geometry projection.
Each mapper is a pure-NumPy class (no FEniCS dependency) that implements the :class:`~ggp.projection.base.ProjectionMapper` interface:

- :meth:`~ggp.projection.base.ProjectionMapper.forward` — compute ``(rho_E, rho_V)`` from ``x_vars``
- :meth:`~ggp.projection.base.ProjectionMapper.jacobian` — compute analytic Jacobians ``drho_E/dx``, ``drho_V/dx``
- :meth:`~ggp.projection.base.ProjectionMapper.num_vars_per_component` — design variables per primitive
- :meth:`~ggp.projection.base.ProjectionMapper.default_bounds` — sensible bounds given domain extents

Built-in mappers and their variable counts:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Mode key
     - Class
     - Variables per component
   * - ``Free`` / ``2D_Free``
     - ``Free2DMapper``
     - 6: Xc, Yc, length, thickness, angle, density
   * - ``3D_Free``
     - ``Free3DMapper``
     - 8: Xc, Yc, Zc, length, width, theta, phi, density
   * - ``2D_ALM``
     - ``ALM2DMapper``
     - 3: Xc, width, mass
   * - ``3D_ALM``
     - ``ALM3DMapper``
     - 6: Xc, Yc, width_x, width_y, Zc, mass

Adding a new mapper:

.. code-block:: python

    from ggp.projection.registry import register_mapper
    from ggp.projection.base import ProjectionMapper

    @register_mapper("Lattice_3D")
    class Lattice3DMapper(ProjectionMapper):
        def forward(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
            ...
        def jacobian(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
            ...

The new mode is immediately available in the CLI via ``formulation.mode: Lattice_3D`` in a YAML config.

4. GEMSEO Discipline Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~ggp.gemseo_wrappers.geometry_discipline.GGPGeometryDiscipline` is a thin GEMSEO :class:`~gemseo.core.discipline.discipline.Discipline` that:

1. Fetches the correct mapper from the registry via ``get_mapper(mode, ...)``.
2. Calls ``mapper.forward()`` and ``mapper.jacobian()`` in ``_run()`` and ``_compute_jacobian()``.
3. Applies KS aggregation and smooth saturation to produce final ``rho_E`` / ``rho_V`` fields.

:class:`~ggp.gemseo_wrappers.physics_discipline.GGPPhysicsDiscipline` wraps the FEniCS elasticity solver and computes the compliance (objective) and volume constraint value.

Both disciplines run inside a GEMSEO ``MDAChain`` driven by an ``MDF`` scenario, which handles the gradient chain-rule automatically.

FEM Solver & Pre-computed Reference Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid FEniCS adjoint taping overhead at every iteration, ``GGPPhysicsDiscipline`` uses a pre-computed reference stiffness matrix strategy:

- **Initialisation:** A reference element stiffness matrix :math:`K_{ref}` (for :math:`E = 1`) is computed once via FEniCS during setup.
- **Per-iteration assembly:** At each optimisation step, the global stiffness matrix is assembled by scaling :math:`K_{ref}` with the SIMP-penalized modulus :math:`E(\rho_e)` per element and injecting the result into a SciPy COO sparse matrix (CSR after conversion).
- **Linear solve:** The assembled :math:`K U = F` is dispatched to one of three pluggable backends selected via ``solver.fem_solver`` in the YAML or ``--fem-solver`` on the CLI:

  .. list-table::
     :header-rows: 1
     :widths: 15 30 30 25

     * - Backend
       - Key
       - Implementation
       - Best for
     * - Direct
       - ``direct`` *(default)*
       - ``scipy.sparse.linalg.spsolve`` (SuperLU / UMFPACK)
       - Small–medium 2-D meshes; deterministic and always converges
     * - Iterative
       - ``iterative``
       - PETSc CG with GAMG algebraic multigrid preconditioner
       - Large 3-D meshes; memory-efficient
     * - AMJax
       - ``amjax``
       - PyAMG smoothed-aggregation hierarchy as preconditioner for scipy CG
       - Both 2-D and 3-D; robust, memory-efficient, dependency-free

The assembly code is identical for 2-D and 3-D problems; only the size and sparsity of :math:`K` differ.  All three backends operate on the same CSR matrix and return the same NumPy displacement vector, so switching backends never affects post-processing or sensitivity computation.

**AMJax backend details** (:mod:`ggp.physics.amjax_solver`):

The ``amjax`` backend in :func:`~ggp.physics.amjax_solver.solve_amjax` implements the core approach of the `AMJax library <https://github.com/vboussange/AMJax>`_ — algebraic multigrid (AMG) preconditioning for FEM Krylov solves — using PyAMG and SciPy directly.  It performs the following steps every iteration:

1. Converts :math:`K` to a CSR ``float64`` matrix.
2. Builds a smoothed-aggregation AMG hierarchy via ``pyamg.smoothed_aggregation_solver`` (optimal for SPD FEM matrices).
3. Exposes the hierarchy as a ``scipy.sparse.linalg.LinearOperator`` preconditioner via ``ml.aspreconditioner(cycle=cycle)``.
4. Runs preconditioned CG via ``scipy.sparse.linalg.cg`` with a fallback retry if convergence is not achieved within ``maxiter`` steps.
5. Returns the solution as a NumPy ``float64`` array.

This approach delivers the same convergence benefits as AMJax (AMG-preconditioned CG vs raw iterative solve) while requiring only ``pyamg`` as an additional dependency alongside SciPy.  The ``cycle`` parameter selects the AMG cycle type (``"V"``, ``"W"``, or ``"F"``; default ``"V"``).

Performance Monitoring
----------------------

The ``ggp/utils/profiling.py`` module provides utilities to time individual pipeline stages. Profiling data is written to ``performance_logs/`` and aggregated in ``performance_history.json`` (per Git commit).

To analyze bottlenecks in a saved ``.prof`` file, use snakeviz:

.. code-block:: bash

   pip install snakeviz
   snakeviz performance_logs/<benchmark>.prof

Citations
---------

- **Original Paper:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). https://doi.org/10.1007/s11831-019-09362-8
- **Original MATLAB Code:** `GGP-Matlab Repository <https://github.com/topggp/GGP-Matlab>`_
