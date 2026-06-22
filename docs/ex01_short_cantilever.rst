Short Cantilever
================

A GGP benchmark: a 60 × 60 beam clamped on the left face with a downward
point load at the mid-right edge.  Five vertical bar columns (18 layers each)
optimise their x-position and width per layer using the MNA characteristic
function — the same variable structure as the Matlab reference code.

Running
-------

.. code-block:: bash

   ggp optimize --preset short_cantilever

To save the density image directly into the documentation static folder:

.. code-block:: bash

   ggp optimize --preset short_cantilever --output-dir docs/_static

Result
------

.. figure:: _static/short_cantilever_optimized.png
   :alt: Optimized short cantilever topology
   :width: 80%
   :align: center

   Optimized density field after 150 MMA iterations (40 % volume fraction).

Problem details
---------------

+---------------------+------------------+
| Parameter           | Value            |
+=====================+==================+
| Domain              | 60 × 60          |
+---------------------+------------------+
| Mesh                | 60 × 60 quads    |
+---------------------+------------------+
| Formulation         | ALM (MNA method) |
+---------------------+------------------+
| Layers (nY)         | 18               |
+---------------------+------------------+
| Columns (np)        | 5                |
+---------------------+------------------+
| Layer height        | 3.33             |
+---------------------+------------------+
| ``r_gp`` (R)        | 0.5              |
+---------------------+------------------+
| Ngp                 | 1                |
+---------------------+------------------+
| Volume fraction     | 0.40             |
+---------------------+------------------+
| Algorithm           | MMA              |
+---------------------+------------------+
| Max iterations      | 150              |
+---------------------+------------------+
