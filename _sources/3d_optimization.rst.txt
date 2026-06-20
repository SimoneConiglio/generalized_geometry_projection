3D Topology Optimization Framework
==================================

The Generalized Geometry Projection (GGP) framework provides comprehensive support for **3D Topology Optimization**. Expanding beyond planar models, the 3D pipeline enables the synthesis of highly complex structural geometries under various loading conditions, fully integrated with high-performance physical solvers and state-of-the-art visualization pipelines.

3D Formulations
---------------

The 3D framework introduces volumetric formulations designed to map continuous geometric entities into a 3D finite element mesh:

- ``3D_Free``: This mode projects independent volumetric components (e.g., 3D round-ended capsules/cylinders) seamlessly into a 3D hexahedral grid. Each component is defined by 8 design variables:
  
  1. ``Xc``: Center X-coordinate
  2. ``Yc``: Center Y-coordinate
  3. ``Zc``: Center Z-coordinate
  4. ``L``: Capsule Length
  5. ``W``: Capsule Width/Thickness
  6. ``Theta``: Rotation Angle 1
  7. ``Phi``: Rotation Angle 2
  8. ``M``: Component Density/Presence
  
These continuous parameters are mapped analytically onto the 3D domain using a regularized Saturated KS function, ensuring fully differentiable constraints and volume evaluations.

High-Performance Solving (FEniCS + PETSc)
-----------------------------------------

Scaling topology optimization from 2D to 3D exponentially increases the degrees of freedom (e.g., a standard 60x30x30 mesh contains 54,000 hexahedral elements and over 150,000 DOFs). To handle this, the 3D physics pipeline is deeply integrated with **FEniCS** and **PETSc4py**.

Instead of utilizing direct LU solvers which suffer from massive memory overhead in 3D, the physics discipline automatically switches to scalable iterative solvers:

- **GAMG Preconditioner**: The solver uses PETSc's Algebraic Multigrid (GAMG) preconditioner to dramatically accelerate convergence.
- **CG Krylov Solver**: The system evaluates linear elasticity deformations via the Conjugate Gradient (CG) method.
- **Multithreading**: The JIT-compiled C++ expressions assemble the massive 3D stiffness kernels rapidly using multi-threading.

XDMF Visualization & ParaView
-----------------------------

In 3D, relying on generic Python plotting libraries (such as ``matplotlib``) for rendering intersecting polygons becomes computationally intractable and visually cluttered. 

To solve this, the framework implements a native **XDMF Exporter**. During the optimization cycles, the GGP iterative loop seamlessly writes the instantaneous volumetric density field (``rho_V``) to standard `.xdmf` and `.h5` files.

**Viewing in ParaView:**
1. Open the generated ``density.xdmf`` file within **ParaView**.
2. ParaView automatically loads the data as a time-series dataset, allowing you to "Play" the optimization animation from iteration 0 to the end.
3. Apply a **Threshold** filter to the ``topology`` scalar field (e.g., extracting regions where density > 0.5) to view the solid material distribution.
4. This method accurately reveals the precise 3D capsules mapped by the Saturated KS function natively within the finite element space.

CLI Usage
---------

You can launch a 3D topology optimization task utilizing the iterative solvers directly from the CLI. 

.. code-block:: bash

   conda run -n samo_agents python ggp.py optimize \
     --use-case Short_Cantilever \
     --formulation 3D_Free \
     --max-iter 30 \
     --iterative

This automatically establishes the 3D bounding box dimensions, launches the GAMG iterative solver, and produces the XDMF sequences for subsequent ParaView rendering.
