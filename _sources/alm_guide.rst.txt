Additive Manufacturing Constraints
==================================

The ``ggp`` framework supports **Additive Layer Manufacturing (ALM)** constraints
through the MNA (Moving Node Approach) continuous formulation.  This ensures that
the generated topology is self-supporting, i.e. printable without internal
support structures.

The implementation is based on:

    Bhat, K.V., Capasso, G., Coniglio, S., Morlier, J., Gogu, C.
    *On some applications of Generalized Geometric Projection to optimal 3D printing*.
    Computers & Graphics (2021).

Variable Layout
---------------

Each run of ``ALM2DMapper`` creates a single flat vector of design variables:

.. code-block:: text

    [Xc_0_0, L_0_0,  Xc_0_1, L_0_1,  …,  Xc_{nY-1}_{np-1}, L_{nY-1}_{np-1},
     h_0, h_1, …, h_{np-1},
     Mc_0, Mc_1, …, Mc_{np-1},
     y0, theta0]

where:

- :math:`N_y` = number of layers, :math:`N_p` = components per layer (printed columns)
- :math:`X_{c,k,j}`, :math:`L_{k,j}` — x-centre and width of component :math:`j` in layer :math:`k` (interleaved pairs, F-major ordering so that :math:`j` is the "column" index)
- :math:`h_j \in [0.2,\,1]` — normalised total print height of column :math:`j`; the physical height cutoff is :math:`L_y \cdot h_j`
- :math:`M_{c,j} \in [0,\,1]` — membership density for the whole printed component :math:`j` (shared across all layers)
- :math:`y_0` — y-translation of the printing plane (shifts the build origin)
- :math:`\theta_0` — rotation of the printing plane (tilts the build direction)

Total variables: :math:`2 N_y N_p + 2 N_p + 2`.

Printing Plane Transform
------------------------

When :math:`y_0 \neq 0` or :math:`\theta_0 \neq 0`, the evaluation coordinates
:math:`(x, y)` are mapped to print-space before the characteristic function is
evaluated:

.. math::

    x_p = x \cos\theta_0 - y \sin\theta_0

    y_p = y_0 + x \sin\theta_0 + y \cos\theta_0

This allows non-horizontal build directions, e.g. inclined printing at angle
:math:`\theta_0`.

MNA Characteristic Function
----------------------------

For each evaluation point :math:`(x_p, y_p)` and each printed component
:math:`j`, the framework:

1. Finds the layer index :math:`k_e = \lfloor y_p / \Delta h \rfloor` (clamped to
   :math:`[0, N_y - 2]`).
2. Computes linearly interpolated left/right boundaries between layer :math:`k_e`
   and :math:`k_e + 1`:

   .. math::

       l_y = (X_{k_e} - L_{k_e}/2) + \alpha \bigl[(X_{k_e+1} - L_{k_e+1}/2) - (X_{k_e} - L_{k_e}/2)\bigr]

       u_y = (X_{k_e} + L_{k_e}/2) + \alpha \bigl[(X_{k_e+1} + L_{k_e+1}/2) - (X_{k_e} + L_{k_e}/2)\bigr]

   where :math:`\alpha = (y_p - Y_{k_e}) / (Y_{k_e+1} - Y_{k_e})` is the local
   interpolation factor.

3. Evaluates three signed-distance tests:

   .. math::

       \zeta_1 = -x_p + l_y, \quad \zeta_2 = x_p - u_y, \quad \zeta_3 = y_p - L_y h_j

4. Computes the characteristic function via the quintic MNA window:

   .. math::

       W(\zeta) = \frac{1}{2} - \frac{15}{16\sigma}\zeta + \frac{5}{8\sigma^3}\zeta^3 - \frac{3}{16\sigma^5}\zeta^5
       \quad \text{for } |\zeta| \le \sigma

   .. math::

       W_j = W(\zeta_1) \cdot W(\zeta_2) \cdot W(\zeta_3)

   The height test :math:`\zeta_3` cuts off the component above its total
   print height :math:`L_y h_j`.

5. Returns the density contribution: :math:`\rho_j^{el} = M_{c,j}^p \cdot W_j`.

Overhang Angle Constraints
---------------------------

Linear constraints enforce that each layer is supported by the one below.
For layer interface :math:`k \to k+1` and component :math:`j`:

.. math::

    (X_{c,k+1,j} + L_{k+1,j}/2) - (X_{c,k,j} + L_{k,j}/2) &\le \Delta h \tan\alpha

    (X_{c,k,j} - L_{k,j}/2) - (X_{c,k+1,j} - L_{k+1,j}/2) &\le \Delta h \tan\alpha

where :math:`\alpha` is the maximum allowable overhang angle (default 45°).
When :math:`\theta_0 \neq 0`, :math:`\alpha` is corrected to
:math:`\alpha \pm \theta_0` for the two constraint directions.

These are activated by adding an ``overhang`` constraint entry to the preset YAML:

.. code-block:: yaml

    constraints:
      - name: overhang
        type: ineq
        bound: 0.0
        params:
          alpha_deg: 45.0

Bridge Length Constraints
--------------------------

To limit the lateral drift of a component across layers relative to its base,
bridge-length constraints bound the horizontal extent of each upper layer
relative to the base-layer (layer 0) x-centre:

.. math::

    X_{c,k,j} + L_{k,j}/2 - X_{c,0,j} &\le BL

    X_{c,0,j} - X_{c,k,j} + L_{k,j}/2 &\le BL

Enable by adding ``bridge_length`` to the overhang constraint params:

.. code-block:: yaml

    constraints:
      - name: overhang
        type: ineq
        bound: 0.0
        params:
          alpha_deg: 45.0
          bridge_length: 12.5   # in the same units as Lx

Feature Summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Feature
     - Status
   * - MNA continuous mapping
     - ✅ Implemented
   * - Per-column print height :math:`h_j`
     - ✅ Implemented
   * - Per-column membership :math:`M_{c,j}`
     - ✅ Implemented (shared across layers)
   * - Overhang angle constraints
     - ✅ Implemented
   * - Bridge length constraints (2D)
     - ✅ Implemented
   * - Printing plane y-offset :math:`y_0`
     - ✅ Implemented
   * - Printing plane rotation :math:`\theta_0`
     - ✅ Implemented
   * - 3D bridge length constraints
     - ✅ Implemented (``ALM3DMapper``)
