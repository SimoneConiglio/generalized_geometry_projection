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
  
  * *Choices:* ``Free``, ``ALM``, ``ALM_Alternating``, ``2D_Free``, ``3D_Free``
  * *Default:* ``Free``

* ``--max-iter`` : Maximum number of optimization iterations.
  
  * *Default:* ``50``

* ``--max-inner`` : Maximum number of inner iterations (for Alternating ALM).
  
  * *Default:* ``10``

Examples
========

Run a standard Free formulation optimization on an MBB beam:

.. code-block:: bash

   python ggp.py optimize --use-case MBB --formulation Free --max-iter 100

Run an ALM continuous formulation optimization using the Alternating Augmented Lagrangian algorithm:

.. code-block:: bash

   python ggp.py optimize --use-case Short_Cantilever --formulation ALM_Alternating --max-iter 30 --max-inner 10
