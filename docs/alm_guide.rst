Additive Manufacturing Constraints
==================================

The ``ggp`` framework extends the standard Generalized Geometry Projection (GGP) method to support **Additive Layer Manufacturing (ALM)** constraints. This ensures that the generated topology is printable without internal support structures.

This implementation follows the AMNA (Adapted Moving Node Approach) formulation from:

    Bhat, K.V., Capasso, G., Coniglio, S., Morlier, J., Gogu, C.
    *On some applications of Generalized Geometric Projection to optimal 3D printing*.
    Computers & Graphics (2021).

Layer-by-Layer Parameterization
-------------------------------

In standard GGP, components can change their orientation (:math:`\theta`) and vertical position (:math:`y_c`) freely. In the ALM formulation (provided by ``GGP2DALMMapper``), components are restricted to horizontal tracks of a fixed height (matching the layer height).

Each component :math:`j` in layer :math:`i` is defined by only three variables:
:math:`x_{i,j} = [X_{c,i,j}, w_{i,j}, M_{i,j}]`

where:

- :math:`X_{c}` is the horizontal center.
- :math:`w` is the track width (length of the segment).
- :math:`M` is the track mass variable.

AMNA Box Formulation (Paper Eq. 40-41)
---------------------------------------

The characteristic function for each layer segment uses a **product of five regularized Heaviside functions**, one for each boundary of the rectangular layer primitive:

.. math::
    \rho^{el}_i = \prod_{k=1}^{5} W(\zeta_k)

where the :math:`\zeta` variables represent signed distances from a Gauss point to each boundary edge:

.. math::
    \zeta_1 = -L/2 - x_{loc}, \quad
    \zeta_2 = x_{loc} - L/2, \quad
    \zeta_3 = y_{loc} - h/2, \quad
    \zeta_4 = -h/2 - y_{loc}, \quad
    \zeta_5 = y_{loc} - h/2

and the smoothed Heaviside function :math:`W(\zeta)` is the 5th-order polynomial:

.. math::
    W(\zeta) = \frac{1}{2} - \frac{15}{16\sigma}\zeta + \frac{5}{8\sigma^3}\zeta^3 - \frac{3}{16\sigma^5}\zeta^5

This produces a **flat box/block** density field for each layer, rather than the rounded bar shape used in the Free GGP formulation (GP method). The transition width :math:`\sigma` controls the sharpness of the boundary.

Overhang Angle Constraints
--------------------------

To be printable without supports, each layer must be sufficiently supported by the layer beneath it. For a layer height :math:`\Delta h` and a maximum allowable overhang angle :math:`\alpha` (typically 45°), the following linear constraints are enforced between layer :math:`i` and layer :math:`i+1` (Paper Eq. 42-45):

.. math::
    (x_{c, i+1} + \frac{w_{i+1}}{2}) - (x_{c, i} + \frac{w_i}{2}) \le \Delta h \cdot \tan(\alpha)

    (x_{c, i} - \frac{w_i}{2}) - (x_{c, i+1} - \frac{w_{i+1}}{2}) \le \Delta h \cdot \tan(\alpha)

Implementation in GEMSEO
------------------------

These constraints are linear with respect to the design variables. In the provided examples, they are wrapped in an ``ALMConstraintsDiscipline`` and passed to the GEMSEO optimizer (MMA), which handles them alongside the volume and compliance objectives.

See the :ref:`alm_example` for a complete implementation.

Current Limitations
-------------------

Compared to the full paper formulation, the following features are **not yet implemented**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Paper Feature
     - Status
   * - AMNA box formulation (Eq. 40-41)
     - ✅ Implemented
   * - Overhang angle constraints (Eq. 42-45)
     - ✅ Implemented
   * - Bridge length constraints (Eq. 46)
     - ✅ Implemented (3D only)
   * - Design space rotation :math:`\theta_0`
     - ❌ Not implemented — the build direction is fixed (bottom-to-top)
   * - Design space translation :math:`y_0`
     - ❌ Not implemented — the build origin is fixed
   * - Per-component print height :math:`h_i`
     - ❌ Not implemented — all layers share a fixed height
   * - KS constraint aggregation (Eq. 47)
     - ✅ Implemented via ``ks_aggregation`` in ``math_utils``
