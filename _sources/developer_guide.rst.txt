Developer Guide
===============

This guide outlines the mathematical foundation, architectural patterns, and engineering decisions behind the `samo_ggp` framework.

Mathematical Foundation
-----------------------

The **Generalized Geometry Projection (GGP)** method parametrizes a design domain using explicit geometric primitives (e.g., rectangles, ellipses) rather than a dense grid of pixel densities. This framework implements the 2D Free formulation based on the original work.

### 1. Primitive Mapping
Each component $i$ is defined by a set of continuous variables $x_i = [X_c, Y_c, L, H, \theta]$.
The signed distance field $\psi_i(x, y)$ of a primitive is smoothed to ensure differentiability. The local density $\rho_i(x, y)$ is obtained via a **Regularized Heaviside function**, ensuring that the density smoothly transitions from 1 (inside the component) to 0 (outside) over a narrow band $\epsilon_{mna}$.

### 2. Saturated Kreisselmeier-Steinhauser (KS) Aggregation
To combine multiple overlapping components into a single global density field $\rho(x, y)$, the KS function is used as a smooth approximation of the maximum operator:

.. math::
    KS(\rho) = \frac{1}{\kappa_a} \ln \left( \frac{1}{N} \sum_{i=1}^N \exp(\kappa_a \rho_i) \right)

Because the KS function can exceed 1.0 (causing issues for physics solvers), we apply a **smooth saturation function** to strictly bound the final density $\rho \in [0, 1]$.

Architecture & Object-Oriented Design
-------------------------------------

The `samo_ggp` package is structured to decouple the geometry parametrization from the physics solvers, enabling easy extension to 3D or Additive Layer Manufacturing (ALM) constraints without rewriting the optimization loop.

### 1. The Factory Pattern
- **`BaseMapper` & `GeometryFactory`:** Encapsulates the GGP mapping logic. To add 3D ALM track mapping, a new `GGP3DMapper` can be introduced and registered in the factory.
- **`BaseSolver` & `PhysicsFactory`:** Encapsulates the FEniCS PDE solvers. Currently implements `LinearElasticitySolver` (using SIMP penalization).

### 2. The GEMSEO "Macro-Discipline"
A critical engineering challenge was integrating `dolfin-adjoint` (which tracks FEniCS operations) with `GEMSEO` (the MDAO orchestration engine).

**Why not a modular `MDAChain`?**
Initially, we attempted to split the process into two GEMSEO disciplines:
1. `GeometryDiscipline`: $x \rightarrow \rho$
2. `PhysicsDiscipline`: $\rho \rightarrow \text{Compliance}$

However, GEMSEO's chain rule requires explicitly computing the intermediate Jacobian $\frac{\partial \rho}{\partial x}$. For a $50 \times 50$ mesh and 50 design variables, this meant computing and passing a dense $2500 \times 50$ matrix every iteration. This completely broke the analytical efficiency of the adjoint method.

**The Solution: Monolithic Adjoint Tracking**
We implemented the `GGPMacroDiscipline`. This single GEMSEO wrapper takes the primitive variables $x$, constructs `dolfin_adjoint.Constant` objects, and runs the entire forward pass (Mapping + Physics).
When GEMSEO requests the gradient $\frac{\partial J}{\partial x}$, `dolfin-adjoint` solves the adjoint PDE and computes the exact sensitivities analytically in a fraction of a second, entirely bypassing the dense $\frac{\partial \rho}{\partial x}$ Jacobian.

Tape Management & Safe Re-execution
-----------------------------------

To prevent memory leaks and graph corruption across thousands of optimization iterations, the `_run` method executes a strict protocol:
1. `get_working_tape().clear_tape()` is called to destroy the previous iteration's computational graph.
2. Fresh `dolfin_adjoint.Constant` objects are instantiated.
3. The forward graph is rebuilt from scratch.

Citations
---------

This framework is a Python/FEniCS re-implementation and extension of the original MATLAB framework.

1. **Original Paper:**
   Bhat, A., Capasso, M., Coniglio, S., et al. *"On some applications of Generalized Geometric Projection to optimal 3D printing"*.
2. **Original MATLAB Code:**
   The foundational KS aggregation and Heaviside mappings are heavily inspired by the associated GGP-Matlab repository provided by the original authors.
