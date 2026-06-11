Credits & Dependencies
======================

The ``ggp`` framework relies on a number of high-quality open-source packages. We gratefully acknowledge their developers and communities.

Core Scientific Libraries
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - License
     - Role
   * - `FEniCS <https://fenicsproject.org/>`_ (``dolfin``)
     - LGPL-3.0
     - Finite Element framework providing mesh generation, function spaces, variational forms (UFL), and PDE solvers. Used for all geometry projections and the linear elasticity solver.
   * - `dolfin-adjoint <https://www.dolfin-adjoint.org/>`_
     - LGPL-3.0
     - High-level automated differentiation framework for FEniCS. Originally used for shape derivatives, listed for legacy environment compatibility.
   * - `PETSc <https://petsc.org/>`_ (``petsc4py``)
     - BSD-2-Clause
     - High-performance sparse linear algebra backend. Used for fast global stiffness matrix assembly in CSR format.
   * - `GEMSEO <https://gemseo.readthedocs.io/>`_
     - LGPL-3.0
     - Multidisciplinary Design Analysis and Optimization (MDAO) framework. Provides the discipline abstraction, design space management, and scenario orchestration.
   * - `gemseo-mma <https://pypi.org/project/gemseo-mma/>`_
     - LGPL-3.0
     - GEMSEO plugin providing the Method of Moving Asymptotes (MMA) optimizer used for all topology optimization runs.
   * - `NumPy <https://numpy.org/>`_
     - BSD-3-Clause
     - Fundamental array computing library. Used for vectorized geometry mapping, analytic Jacobians, and all numerical computations.
   * - `SciPy <https://scipy.org/>`_
     - BSD-3-Clause
     - Sparse matrix utilities (CSR format) used for stiffness assembly and linear system solving.

Meshing
-------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - License
     - Role
   * - `mshr <https://bitbucket.org/fenics-project/mshr/>`_
     - LGPL-3.0
     - FEniCS mesh generation component. Used for creating structured and unstructured meshes from CSG geometries.

Development & Documentation
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - License
     - Role
   * - `pytest <https://docs.pytest.org/>`_
     - MIT
     - Testing framework used for the unit and regression test suite.
   * - `pytest-cov <https://github.com/pytest-dev/pytest-cov>`_
     - MIT
     - Code coverage plugin for pytest, used in CI to generate coverage reports.
   * - `Sphinx <https://www.sphinx-doc.org/>`_
     - BSD-2-Clause
     - Documentation generator used to build this documentation site.
   * - `sphinx-book-theme <https://sphinx-book-theme.readthedocs.io/>`_
     - BSD-3-Clause
     - Modern Sphinx theme providing the documentation layout.
   * - `nbsphinx <https://nbsphinx.readthedocs.io/>`_
     - MIT
     - Sphinx extension for rendering Jupyter notebooks as documentation pages.
   * - `JupyterLab <https://jupyter.org/>`_
     - BSD-3-Clause
     - Interactive notebook environment used for example tutorials.
   * - `Matplotlib <https://matplotlib.org/>`_
     - PSF-based
     - 2D plotting library used for density and component visualization in examples and post-processing.
   * - `NLopt <https://nlopt.readthedocs.io/>`_
     - MIT
     - Nonlinear optimization library, available as an alternative optimizer backend.

Academic References
-------------------

This implementation is based on the following publications:

1. **GGP Framework:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). `DOI:10.1007/s11831-019-09362-8 <https://doi.org/10.1007/s11831-019-09362-8>`_

2. **ALM Extension:** Bhat, K.V., Capasso, G., Coniglio, S., Morlier, J., Gogu, C. *On some applications of Generalized Geometric Projection to optimal 3D printing*. Computers & Graphics (2021).

3. **Original MATLAB Code:** `GGP-Matlab <https://github.com/topggp/GGP-Matlab>`_ repository.
