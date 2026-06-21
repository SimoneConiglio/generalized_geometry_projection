ALM Cantilever
==============

A cantilever optimised under Additive Layer Manufacturing (ALM) constraints.
Components are arranged in horizontal layers; linear overhang constraints
enforce a 45° self-support angle between adjacent layers, restricting the
optimiser to shapes that can be printed without support material.

Running
-------

.. code-block:: bash

   ggp optimize --preset alm_cantilever

To save the density image:

.. code-block:: bash

   ggp optimize --preset alm_cantilever --output-dir docs/_static

Result
------

.. figure:: _static/alm_cantilever_optimized.png
   :alt: Optimized ALM cantilever topology
   :width: 90%
   :align: center

   Optimized density field after 150 MMA iterations (40 % volume fraction, 45° overhang limit).

Problem details
---------------

+---------------------+------------------+
| Parameter           | Value            |
+=====================+==================+
| Domain              | 60 × 30          |
+---------------------+------------------+
| Mesh                | 60 × 30 quads    |
+---------------------+------------------+
| Formulation         | ALM 2D           |
+---------------------+------------------+
| Layers              | 10               |
+---------------------+------------------+
| Components / layer  | 5                |
+---------------------+------------------+
| Volume fraction     | 0.40             |
+---------------------+------------------+
| Overhang angle      | 45°              |
+---------------------+------------------+
| Algorithm           | MMA              |
+---------------------+------------------+
| Max iterations      | 150              |
+---------------------+------------------+

The overhang constraint is linear in the design variables and enforces:

.. math::

   |X_{c,k+1} - X_{c,k}| + \tfrac{1}{2}(L_{k+1} - L_k) \leq h \tan(\alpha)

for every pair of vertically adjacent components, where :math:`\alpha = 45°`.
