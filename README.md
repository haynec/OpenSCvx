<a id="readme-top"></a>

<img src="figures/openscvx_logo.svg" width="1200"/>
<p align="center">
    <a href="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/lint.yml"><img src="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/lint.yml/badge.svg"/></a>
    <a href="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/tests-unit.yml"><img src="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/tests-unit.yml/badge.svg"/></a>
    <a href="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/tests-integration.yml"><img src="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/tests-integration.yml/badge.svg"/></a>
    <a href="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/nightly.yml"><img src="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/nightly.yml/badge.svg"/></a>
    <a href="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/release.yml"><img src="https://github.com/OpenSCvx/OpenSCvx/actions/workflows/release.yml/badge.svg?event=release"/></a>
</p>
<p align="center">
    <a href="https://arxiv.org/abs/2410.22596"><img src="http://img.shields.io/badge/arXiv-2410.22596-B31B1B.svg"/></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/></a>
</p>

<!-- PROJECT LOGO -->
<br />

OpenSCvx is a general python-based successive convexification implementation which uses a JAX backend.

## Installation

OpenSCvx is available on [PyPI](https://pypi.org/project/openscvx/) and can be trivially installed with pip.

It is recommended to install OpenSCvx inside a virtual environment (venv, conda, uv, *etc.*). If you don't already have one set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Using pip

```bash
pip install openscvx
```

> [!TIP]
> **Optional Dependencies**
>
> For GUI support or CVXPYGen code generation:
> ```bash
> pip install openscvx[gui,cvxpygen]
> ```

> [!TIP]
> **Nightly Builds**
>
> To install the latest development version (nightly), use the `--pre` flag:
> ```bash
> pip install --pre openscvx
> ```

### Using uv

If you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) you can prefix the commands with `uv` for faster installation:

```bash
uv pip install openscvx
```

## Installing From Source

### Using pip

```bash
git clone https://github.com/OpenSCvx/OpenSCvx.git
cd OpenSCvx

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Using uv

```bash
git clone https://github.com/OpenSCvx/OpenSCvx.git
cd OpenSCvx

uv venv
source .venv/bin/activate
uv pip install -e .
```

## Getting Started

Check out the OpenSCvx documentation to help you get started

- [Getting Started Docs](https://openscvx.github.io/OpenSCvx/latest/getting-started/)
- [Users Guide](https://openscvx.github.io/OpenSCvx/latest/UsersGuide/00_introduction/)
- [API Reference](https://openscvx.github.io/OpenSCvx/latest/Reference/problem/)

### Running the Examples

We also have a selection of problems in the `examples/` folder as well as on the [Examples page](https://openscvx.github.io/OpenSCvx/latest/Examples/abstract/brachistochrone/) of the documentation. The example trajectory optimization problems are grouped by application and represent some of the problem types that can be solved by OpenSCvx.

> [!Note]
> To run the examples, you'll need to clone this repository and install OpenSCvx in editable mode (`pip install -e .`). See the [Installing From Source](#installing-from-source) section above for detailed installation instructions.

To run a problem simply run any of the examples directly, for example:

```sh
python3 examples/abstract/brachistochrone.py
```

and adjust the plotting as needed.

Check out the problem definitions inside `examples/` to see how to define your own problems.

## Code Structure

<img src="figures/oscvx_structure_full_dark.svg" width="1200"/>

## What is implemented

This repo has the following features:

1. Free Final Time
2. Fully adaptive time dilation (```s``` is appended to the control vector)
3. Continuous-Time Constraint Satisfaction
4. FOH and ZOH exact discretization (```t``` is a state so you can bring your own scheme)
6. Vectorized and Ahead-of-Time (AOT) Compiled Multishooting Discretization
7. JAX Autodiff for Jacobians

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements

This work was supported by a NASA Space Technology Graduate Research Opportunity and the Office of Naval Research under grant N00014-17-1-2433. The authors would like to acknowledge Natalia Pavlasek, Fabio Spada, Samuel Buckner, Abhi Kamath, Govind Chari, and Purnanand Elango as well as the other Autonomous Controls Laboratory members, for their many helpful discussions and support throughout this work.

## Citation

Please cite the following works if you use the repository,

```tex
@ARTICLE{hayner2025los,
        author={Hayner, Christopher R. and Carson III, John M. and Açıkmeşe, Behçet and Leung, Karen},
        journal={IEEE Robotics and Automation Letters}, 
        title={Continuous-Time Line-of-Sight Constrained Trajectory Planning for 6-Degree of Freedom Systems}, 
        year={2025},
        volume={},
        number={},
        pages={1-8},
        keywords={Robot sensing systems;Vectors;Vehicle dynamics;Line-of-sight propagation;Trajectory planning;Trajectory optimization;Quadrotors;Nonlinear dynamical systems;Heuristic algorithms;Convergence;Constrained Motion Planning;Optimization and Optimal Control;Aerial Systems: Perception and Autonomy},
        doi={10.1109/LRA.2025.3545299}}
```

```tex
@misc{elango2024ctscvx,
      title={Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction}, 
      author={Purnanand Elango and Dayou Luo and Abhinav G. Kamath and Samet Uzun and Taewan Kim and Behçet Açıkmeşe},
      year={2024},
      eprint={2404.16826},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2404.16826}, 
}
```

```tex
@misc{chari2025qoco,
  title = {QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation},
  author = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year = {2025},
  eprint = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass = {math.OC},
}
```
