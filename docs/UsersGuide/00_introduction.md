# Users Guide

Welcome to the OpenSCvx Users Guide. This section aims to provides a progressive and useful introduction to trajectory optimization with OpenSCvx, starting from first principles and building toward complex, representative problems.

## Learning Path

The tutorials are designed to be read in order. Each builds on concepts from the previous, introducing new features in the context of increasingly realistic problems.

| Tutorial | Problem | You Will Learn |
|----------|---------|----------------|
| [01 Hello Brachistochrone](01_hello_world_brachistochrone.md) | Minimum-time descent curve | Core API: states, controls, dynamics, time, CTCS constraints, solving |
| [02 Drone Racing](02_drone_racing_constraints.md) | Racing through gates | Nodal constraints, `.at()`, `.over()`, `.convex()`, keyframe initialization |
| [03 Obstacle Avoidance](03_obstacle_avoidance_vmap.md) | 6-DOF navigation | Quaternion dynamics, spatial utilities, `ox.Parameter`, `ox.Vmap` |
| [04 Viewpoint Constraints](04_viewpoint_constraints.md) | Perception-constrained racing | Custom symbolic functions, Vmap with custom functions, attitude initialization |
| [05 Visualization](05_visualization.md) | — | 2D plots with Plotly, 3D interactive visualization with viser, Plotly-in-viser |
| [06 Dubin's Car](06_logic.md) | Conditional path planning | Conditional statements, signal temporal logic (STL) |
| [07 Multi-Link Arms](07_lie.md) | Articulated robot control | Lie algebra, propagated states |

## Quick Start

If you're new to OpenSCvx, start with [Hello Brachistochrone](01_hello_world_brachistochrone.md). By the end of that tutorial you will have solved your first trajectory optimization problem and understand the core workflow:

1. Define states and controls with `ox.State` and `ox.Control`
2. Specify dynamics as a dictionary of symbolic expressions
3. Add constraints using `ox.ctcs()` for continuous enforcement
4. Create a `Problem`, initialize, solve, and post-process

From there, each subsequent tutorial introduces new capabilities while reinforcing the fundamentals.

## Interactive Notebooks

Some tutorials include Google Colab notebooks for interactive learning:

- [03 6-DOF Obstacle Avoidance ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xLPC_UJWC35oPRIAY3vkxi8WEYnHCysQ?usp=sharing)
- [04 Viewpoint Constraints ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b3NEx288h4r4HuvCOj-fexmt90PPhKUw?usp=sharing)

These notebooks let you run the examples without local setup and experiment with parameters in real-time.

## Beyond the Tutorials

After completing the tutorials, explore:

- [Examples](../Examples/abstract/brachistochrone.md) — Complete problem implementations across domains
- [API Reference](../Reference/problem.md) — Detailed documentation for all classes and functions
