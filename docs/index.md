---
template: home.html
title: OpenSCvx
description: >-
  JAX trajectory optimization: composable API, JAX compiling, and vectorization for batched
  scenarios.
social:
  cards_layout_options:
    title: Fast and Easy Nonconvex Trajectory Optimization
---

## Why OpenSCvx

<div class="osc-home-strip" aria-hidden="true" markdown="0">
<svg class="osc-home-strip__svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 58" fill="none">
  <rect class="osc-home-strip__layer" x="2" y="6" width="436" height="14" rx="4" />
  <rect class="osc-home-strip__layer" x="2" y="24" width="436" height="14" rx="4" />
  <rect class="osc-home-strip__layer osc-home-strip__layer--accent" x="2" y="42" width="436" height="14" rx="4" />
  <text class="osc-home-strip__text" x="220" y="16" text-anchor="middle">Model in Python</text>
  <text class="osc-home-strip__text" x="220" y="34" text-anchor="middle">Compile with JAX / XLA</text>
  <text class="osc-home-strip__text osc-home-strip__text--strong" x="220" y="52" text-anchor="middle">Scale with vmap &amp; batches</text>
</svg>
</div>

**Interface.** Specify dynamics, constraints, and costs in one place—structured enough for serious
problems, flexible enough to iterate. Use the docs here and the Users Guide to go from a sketch to
a running solve; optional **visualization** hooks help you inspect trajectories when you need them.

**JAX compilation.** Core numerical paths are set up for **compiled JAX** (including
**`jax.export`** where that fits your deployment story—not only `jax.jit`): repeated SCP iterations
dispatch as compiled programs where it pays off, with **XLA** laying out fast kernels on CPU or GPU.
Differentiable pieces line up with JAX’s **grad** / **custom_vjp** patterns when you need
sensitivities.

**Vectorization.** Treat parallel instances—parameter sweeps, ensembles, or scenarios—as batch
dimensions using **`vmap`** (and friends) so one program describes many solves without a slow
Python outer loop.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: __[Getting started](getting-started.md)__ — install the package and run your first workflow.
- :material-book-open-variant: __[Users guide](UsersGuide/00_introduction.md)__ — modeling, constraints, and visualization.
- :material-code-braces: __[API reference](Reference/)__ — modules and functions generated from source with docstrings.
- :material-view-dashboard: __[Examples](examples.md)__ — runnable examples and deeper demos.

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
