MBB Beam
========

The half-symmetry Messerschmidt-Bölkow-Blohm (MBB) beam is a classic
topology optimisation benchmark.  The left edge is a symmetry plane
(horizontal displacement blocked), the bottom-right corner has a vertical
roller, and the top-left corner carries a downward point load.

Running
-------

.. code-block:: bash

   ggp optimize --preset mbb

To save the density image:

.. code-block:: bash

   ggp optimize --preset mbb --output-dir docs/_static

Result
------

.. figure:: _static/mbb_optimized.png
   :alt: Optimized MBB beam topology
   :width: 90%
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
