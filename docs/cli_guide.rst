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
  
  * *Choices:* ``Free``, ``ALM``, ``2D_Free``, ``3D_Free``
  * *Default:* ``Free``

* ``--max-iter`` : Maximum number of optimization iterations.
  
  * *Default:* ``50``

* ``--algorithm`` : Optimization algorithm.
  
  * *Choices (GEMSEO solvers):* ``MMA``, ``SLP``, ``CONLIN``
  * *Default:* ``MMA``

* ``--length`` & ``--height`` : Domain length (L) and height (H).
  
  * *Default:* Depends on the ``--use-case``.

* ``--nelx`` & ``--nely`` : Number of elements in X and Y directions.
  
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
