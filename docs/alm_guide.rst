Additive Manufacturing Constraints
==================================

The `samo_ggp` framework extends the standard Generalized Geometry Projection (GGP) method to support **Additive Layer Manufacturing (ALM)** constraints. This ensures that the generated topology is printable without internal support structures.

Layer-by-Layer Parameterization
-------------------------------

In standard GGP, components can change their orientation (:math:`\theta`) and vertical position (:math:`y_c`) freely. In the ALM formulation (provided by ``GGP2DALMMapper``), components are restricted to horizontal tracks of a fixed height (matching the layer height).

Each component :math:`j` in layer :math:`i` is defined by only three variables:
:math:`x_{i,j} = [X_{c,i,j}, w_{i,j}, M_{i,j}]`

where:
- :math:`X_{c}` is the horizontal center.
- :math:`w` is the track width.
- :math:`M` is the track mass variable.

Overhang Angle Constraints
--------------------------

To be printable without supports, each layer must be sufficiently supported by the layer beneath it. For a layer height :math:`\Delta h` and a maximum allowable overhang angle :math:`\alpha` (typically 45°), the following linear constraints are enforced between layer :math:`i` and layer :math:`i+1`:

.. math::
    (x_{c, i+1} - \frac{w_{i+1}}{2}) \ge (x_{c, i} - \frac{w_i}{2}) - \Delta h \cdot \tan(\alpha)

    (x_{c, i+1} + \frac{w_{i+1}}{2}) \le (x_{c, i} + \frac{w_i}{2}) + \Delta h \cdot \tan(\alpha)

Implementation in GEMSEO
------------------------

These constraints are linear with respect to the design variables. In the provided examples, they are wrapped in an ``ALMConstraintsDiscipline`` and passed to the GEMSEO optimizer (MMA), which handles them alongside the volume and compliance objectives.

See the :ref:`alm_example` for a complete implementation.
