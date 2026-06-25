Installation Guide
==================

This guide provides a detailed, step-by-step walk-through to reproduce the exact environment required to run the ``ggp`` framework.

System Requirements
-------------------

Topology optimization frameworks rely on robust C++ libraries for finite element analysis. The framework uses **FEniCS 2019.1.0**. 

.. warning::
   FEniCS requires a UNIX-like environment. **Windows users MUST use Windows Subsystem for Linux (WSL2)** or Docker. Native Windows installations of FEniCS are not supported.

Step 1: Install Conda
---------------------

If you do not have Conda installed, we recommend installing Miniconda inside your Linux/WSL terminal:

.. code-block:: bash

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

Restart your terminal after installation.

Step 2: Clone the Repository
----------------------------

Clone the repository to your local machine and navigate into it:

.. code-block:: bash

   git clone https://github.com/SimoneConiglio/generalized_geometry_projection.git
   cd generalized_geometry_projection

Step 3: Create the Conda Environment
------------------------------------

We provide an ``environment.yml`` file that strictly pins the FEniCS version alongside GEMSEO and all required scientific tools (NumPy, SciPy, Jupyter, Sphinx).  It also installs the optional **AMJax** FEM solver backend dependencies (``pyamg``, ``jax``, ``jaxlib``, ``amjax``) via pip.  These are only used when ``solver.fem_solver: amjax`` is selected; the default ``direct`` backend requires only SciPy.

Run the following command from the root of the cloned repository:

.. code-block:: bash

   conda env create -f environment.yml

This process may take a few minutes as it downloads and resolves the FEniCS binaries from ``conda-forge``.

Step 4: Activate the Environment
--------------------------------

Every time you work on this project, you must activate the environment:

.. code-block:: bash

   conda activate ggp

Step 5: Install the Package
---------------------------

Install the ``ggp`` package in editable mode so that the ``ggp`` CLI is available and Python can import the package without manual ``PYTHONPATH`` manipulation:

.. code-block:: bash

   pip install -e .

This registers the ``ggp`` entry-point command and the ``ggp-topo`` package into the active Conda environment.

Step 6: Verify the Installation
-------------------------------

You can verify that all dependencies and internal modules are working by running the provided ``pytest`` suite:

.. code-block:: bash

   pytest tests/

You can also confirm the CLI is working:

.. code-block:: bash

   ggp --version
   ggp info --presets

If the tests pass and the CLI responds, you are ready to run the examples or develop your own topology optimization workflows.
