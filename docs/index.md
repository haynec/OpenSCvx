---
template: home.html
title: OpenSCvx
hide:
  - navigation
  - toc
description: >-
  An Open-Source Modular and Extensible Nonlinear Trajectory Optimization Package
social:
  cards_layout_options:
    title: Fast and Easy Nonconvex Trajectory Optimization
---

## Why OpenSCvx

**Interface.** Dynamics, costs, and constraints live in one structured problem object—enough rigor for
research-grade models, enough flexibility to iterate quickly. Pair with the Users Guide when you
want a guided path from sketch to solve.

**JAX & XLA.** Inner loops are shaped for **`jax.jit`** and **`jax.export`**: successive convexification
dispatches as compiled programs where it matters, with the same code path on CPU or GPU.

**Vectorization.** Batch scenarios, ensembles, and parameter grids with **`vmap`** so one program
describes many solves without a slow Python outer loop.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: __[Getting started](getting-started.md)__ — install the package and run your first workflow.
- :material-book-open-variant: __[Users guide](UsersGuide/00_introduction.md)__ — modeling, constraints, and visualization.
- :material-code-braces: __[API reference](Reference/index.md)__ — modules and functions generated from source with docstrings.
- :material-view-dashboard: __[Examples](Examples/index.md)__ — runnable scripts organized by topic.

</div>

## API documentation

The **Reference** section is generated from the `openscvx` Python package using [mkdocstrings](https://mkdocstrings.github.io/)
with the Python handler. Public APIs should use **Google-style** docstrings so signatures,
parameters, and descriptions render consistently in the docs.
