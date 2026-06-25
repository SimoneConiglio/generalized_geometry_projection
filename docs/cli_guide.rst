================================
Command-Line Interface (CLI)
================================

After installing the package (``pip install -e .``), the ``ggp`` command is available in your Conda environment.

.. code-block:: bash

   ggp --help
   ggp --version

Command Overview
================

The CLI provides two top-level commands: ``optimize`` and ``info``.

``info`` Command
----------------

Displays what is available in the current installation.

.. code-block:: bash

   ggp info                  # show everything
   ggp info --presets        # list built-in problem presets
   ggp info --mappers        # list registered projection mappers
   ggp info --backends       # list available linear-algebra backends

``optimize`` Command
--------------------

Runs an end-to-end topology optimisation from a built-in preset or a custom YAML file.

.. code-block:: bash

   ggp optimize --preset <name>   [OPTIONS]
   ggp optimize --config <file>   [OPTIONS]

You must provide exactly one of ``--preset`` or ``--config``.

**Source options:**

* ``--preset <name>`` — use a built-in preset YAML (see ``ggp info --presets``).
  Built-in presets: ``short_cantilever``, ``mbb``, ``l_shape``, ``alm_cantilever``.
* ``--config <path>`` — path to a custom YAML problem definition file.

**Override options** (applied on top of the preset or config file):

* ``--max-iter <int>`` — maximum number of outer optimisation iterations.
* ``--algorithm <str>`` — optimisation algorithm: ``MMA``, ``SLP``, or ``CONLIN``.
* ``--volfrac <float>`` — target volume fraction constraint.
* ``--fem-solver <backend>`` — FEM linear-system backend.  Choices:

  - ``direct`` *(default)* — ``scipy.sparse.linalg.spsolve`` (SuperLU/UMFPACK); best for small-to-medium 2-D meshes.
  - ``iterative`` — PETSc CG + GAMG; recommended for large 3-D meshes.
  - ``amjax`` — AMG-preconditioned CG (PyAMG smoothed-aggregation hierarchy used as a preconditioner for ``scipy.sparse.linalg.cg``); implements the core approach of the `AMJax library <https://github.com/vboussange/AMJax>`_.  Suitable for both 2-D and 3-D problems.  Requires ``pyamg`` (installed via ``environment.yml``).

* ``--iterative`` — shorthand flag equivalent to ``--fem-solver iterative``.  Kept for backward compatibility.
* ``--use-line-search`` — flag: enable monotone backtracking line search.
  Recommended with ``SLP`` or ``CONLIN``.

Examples
========

Run the short cantilever benchmark with default settings:

.. code-block:: bash

   ggp optimize --preset short_cantilever

Run the MBB beam with SLP and line search:

.. code-block:: bash

   ggp optimize --preset mbb --algorithm SLP --use-line-search

Run the ALM cantilever with 30 iterations:

.. code-block:: bash

   ggp optimize --preset alm_cantilever --max-iter 30

Override the volume fraction on any preset:

.. code-block:: bash

   ggp optimize --preset l_shape --volfrac 0.3 --algorithm CONLIN

Run from a custom YAML file with the iterative solver:

.. code-block:: bash

   ggp optimize --config my_3d_problem.yaml --iterative --max-iter 50

Run the short cantilever with the AMJax JAX-accelerated solver:

.. code-block:: bash

   ggp optimize --preset short_cantilever --fem-solver amjax

Run a 3-D problem with AMJax (GPU-compatible, JIT-compiled):

.. code-block:: bash

   ggp optimize --config my_3d_problem.yaml --fem-solver amjax --max-iter 50

YAML Problem Definition Format
===============================

Custom YAML files follow the ``ProblemSpec`` schema. The minimal structure is:

.. code-block:: yaml

   geometries:
     - type: fenics_rectangle
       role: design
       params:
         Lx: 60.0
         Ly: 30.0
         nx: 60
         ny: 30

   boundary_conditions:
     - region: left
       type: fixed

   loads:
     - region: mid_right
       type: point
       value: [0.0, -1.0]

   formulation:
     mode: Free          # Free | 2D_Free | ALM | 2D_ALM | 3D_Free | 3D_ALM
     num_components: 18

   solver:
     algorithm: MMA      # MMA | SLP | CONLIN
     max_iter: 50
     fem_solver: amjax   # direct | iterative | amjax (built-in presets default to amjax)

   volfrac: 0.4

For a 3-D problem, use ``type: fenics_box`` with params ``Lx``, ``Ly``, ``Lz``, ``nx``, ``ny``, ``nz``,
set ``formulation.mode`` to ``3D_Free`` or ``3D_ALM``, and set ``solver.fem_solver`` to ``iterative``
or ``amjax`` in the YAML (or pass the equivalent CLI flag).

The built-in presets in ``ggp/cli/presets/`` serve as ready-to-copy starting points.
