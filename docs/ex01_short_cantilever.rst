Short Cantilever
================

A standard GGP benchmark: a 60 × 30 beam clamped on the left face with a
downward point load at the mid-right edge.  The Free formulation uses 18
overlapping bar-shaped components whose shapes and positions are optimised
to minimise compliance at a 40 % volume fraction.

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
   :width: 90%
   :align: center

   Optimized density field after 150 MMA iterations (40 % volume fraction).

Problem details
---------------

+---------------------+------------------+
| Parameter           | Value            |
+=====================+==================+
| Domain              | 60 × 30          |
+---------------------+------------------+
| Mesh                | 60 × 30 quads    |
+---------------------+------------------+
| Formulation         | Free 2D          |
+---------------------+------------------+
| Components          | 18               |
+---------------------+------------------+
| Volume fraction     | 0.40             |
+---------------------+------------------+
| Algorithm           | MMA              |
+---------------------+------------------+
| Max iterations      | 150              |
+---------------------+------------------+
