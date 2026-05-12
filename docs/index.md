---
template: home.html
title: OpenSCvx
description: >-
  Open-source JAX trajectory optimization: composable models, compiled SCP iterations on CPU and GPU,
  and vmap-friendly batches for parameter sweeps and scenarios.
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

## Documentation tooling

The site is built with **MkDocs** and **Material for MkDocs**. The team behind Material is also
developing **[Zensical](https://zensical.org/)**, which can read existing `mkdocs.yml` files and
offers preliminary [mkdocstrings](https://zensical.org/docs/setup/extensions/mkdocstrings/)
integration. A future switch is possible once plugin coverage matches this project’s needs
(for example **mkdocs-gen-files**, **literate-nav**, **section-index**, and **mike** versioning).
