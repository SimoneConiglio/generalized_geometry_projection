================================
Command-Line Interface (CLI)
================================

GGP provides a unified Command-Line Interface to quickly run topology optimizations with varying algorithms, formulations, and test cases.

Running the CLI
===============

From the project root, you can invoke the CLI script ``ggp.py``:

.. code-block:: bash

   python ggp.py optimize [OPTIONS]

Command Overview
================

``optimize`` Command
--------------------

This command runs an end-to-end topology optimization scenario. 

**Options:**

* ``--use-case`` : Choose the structural boundary conditions.
  
  * *Choices:* ``Short_Cantilever``, ``MBB``, ``L-shape``
  * *Default:* ``Short_Cantilever``

* ``--formulation`` : Choose the generalized geometry projection method.
  
  * *Choices:* ``Free``, ``ALM``, ``3D_Free``, ``3D_ALM``
  * *Default:* ``Free``

* ``--max-iter`` : Maximum number of optimization iterations.
  
  * *Default:* ``50``

* ``--algorithm`` : Optimization algorithm.
  
  * *Choices (GEMSEO solvers):* ``MMA``, ``SLP``, ``CONLIN``
  * *Default:* ``MMA``

* ``--use-line-search`` : Enable line search for the optimization algorithm.
  
  * *Usage:* Often recommended when using ``SLP``.

* ``--iterative`` : Toggle the use of high-performance iterative solvers (PETSc GAMG) instead of memory-heavy direct solvers.
  
  * *Usage:* Highly recommended and effectively required for ``3D_Free`` or ``3D_ALM`` formulations due to scale.

* ``--length`` & ``--height`` : Domain length (L) and height (H). For 3D, depth (D) equals height (H).
  
  * *Default:* Depends on the ``--use-case``.

* ``--nelx`` & ``--nely`` : Number of elements in X and Y directions. For 3D, ``nelz`` equals ``nely``.
  
  * *Default:* Depends on the ``--use-case``.

* ``--volfrac`` : Target volume fraction constraint.
  
  * *Default:* ``0.4``

Examples
========

Run a standard Free formulation optimization on an MBB beam with SLP algorithm:

.. code-block:: bash

   python ggp.py optimize --use-case MBB --formulation Free --algorithm SLP --max-iter 100

Run an ALM continuous formulation optimization with a custom mesh and volume fraction:

.. code-block:: bash

   python ggp.py optimize --use-case Short_Cantilever --formulation ALM --max-iter 30 --nelx 120 --nely 60 --volfrac 0.5

Run a 3D topology optimization using scalable PETSc iterative solvers (with automated ParaView XDMF outputs):

.. code-block:: bash

   conda run -n samo_agents python ggp.py optimize --use-case Short_Cantilever --formulation 3D_Free --max-iter 30 --iterative
