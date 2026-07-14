---
template: home.html
title: OpenSCvx
hide:
  - navigation
  - toc
description: >-
  OpenSCvx is an open-source Python library for nonlinear trajectory
  optimization built on successive convexification (SCvx), with a composable
  problem API and a JAX backend.
social:
  cards_layout_options:
    title: Fast and Easy Nonconvex Trajectory Optimization
---

# OpenSCvx — Trajectory Optimization via Successive Convexification in Python/JAX

OpenSCvx is an open-source Python library for nonlinear trajectory optimization
built on successive convexification (SCvx). You write dynamics, costs, and
constraints as a single composable problem object; the library lowers that
symbolic definition to JAX, discretizes it on a time grid, and solves the
resulting nonconvex problem through a sequence of convex subproblems. The same
problem definition runs on CPU and GPU, and vectorizes across decision nodes and
scenario batches.

## Why OpenSCvx

**Modeling.** Dynamics, costs, and constraints are defined together in a structured problem object.
The Users Guide is the place to see how the pieces fit from a first model to a solve.

**JAX.** The symbolic layer lowers to JAX. On the compiled hot paths we **`jax.jit`** and **immediately `jax.export`**: export is always part of that pipeline, not an extra step you opt into later. CPU and GPU use the same problem definition; performance still depends on problem size and settings.

**Vectorization.** Two ideas show up in practice: the discretized dynamics and many nonlinear terms
are evaluated **across decision nodes** in parallel inside the solver (see
[Vectorization and vmapping](UnderTheHood/vectorization_and_vmapping.md)). When
your model has a **batch axis** you want to treat uniformly—many obstacles, repeated geometry, and
similar—**`ox.Vmap`** writes that data-parallel piece symbolically and lowers to **`jax.vmap`**. It
helps avoid hand-written Python loops over those batch dimensions; it is not a general substitute
for every outer loop (see [Obstacle avoidance with Vmap](UsersGuide/03_obstacle_avoidance_vmap.md)).

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: __[Run examples](examples.md)__ — locate scripts in the repo, run them locally, and browse problem categories.
- :material-book-open-variant: __[Users guide](UsersGuide/00_introduction.md)__ — modeling, constraints, and visualization.
- :material-code-braces: __[API reference](Reference/index.md)__ — modules and functions generated from source with docstrings.
- :material-view-dashboard: __[Examples](Examples/index.md)__ — runnable scripts organized by topic.

</div>