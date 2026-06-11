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

We provide an ``environment.yml`` file that strictly pins the FEniCS version alongside GEMSEO and all required scientific tools (NumPy, SciPy, Jupyter, Sphinx). This ensures you do not face C++ compilation or MPI errors.

Run the following command from the root of the cloned repository:

.. code-block:: bash

   conda env create -f environment.yml

This process may take a few minutes as it downloads and resolves the FEniCS binaries from ``conda-forge``.

Step 4: Activate the Environment
--------------------------------

Every time you work on this project, you must activate the environment:

.. code-block:: bash

   conda activate ggp

Step 5: Set the Python Path
---------------------------

For Python to recognize the ``ggp`` package without requiring a global pip install, you must append the current directory to your ``PYTHONPATH`` before running scripts.

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:.

*Tip: You can add this line to your `~/.bashrc` or run commands by prefixing them, e.g., `PYTHONPATH=$PYTHONPATH:. python examples/01_short_cantilever.py`*

Step 6: Verify the Installation
-------------------------------

You can verify that all dependencies and internal modules are working by running the provided `pytest` suite:

.. code-block:: bash

   pytest tests/

If the tests pass, you are ready to start running the examples or developing your own topology optimization workflows!
