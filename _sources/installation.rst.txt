Installation Guide
==================

This guide provides instructions on how to install and set up the `samo_ggp` framework.

Prerequisites
-------------

The `samo_ggp` framework is built heavily around **FEniCS** (`dolfin`, `dolfin-adjoint`) and **GEMSEO**. Due to the complex C++ dependencies of FEniCS, it is highly recommended to install the framework within a `conda` environment on a Linux system or Windows Subsystem for Linux (WSL).

Step 1: Clone the Repository
----------------------------

Clone the repository to your local machine:

.. code-block:: bash

   git clone https://github.com/SimoneConiglio/generalized_geometry_projection.git
   cd generalized_geometry_projection

Step 2: Create the Conda Environment
------------------------------------

We provide an `environment.yml` file that correctly resolves the FEniCS version alongside GEMSEO and all required scientific tools (NumPy, SciPy, Jupyter, Sphinx).

.. code-block:: bash

   conda env create -f environment.yml

Once created, activate the environment:

.. code-block:: bash

   conda activate samo_agents

Step 3: Verify the Installation
-------------------------------

You can verify that all dependencies and internal modules are working by running the provided `pytest` suite:

.. code-block:: bash

   PYTHONPATH=$PYTHONPATH:. pytest tests/

If the tests pass, you are ready to start running the examples or developing your own topology optimization workflows!
