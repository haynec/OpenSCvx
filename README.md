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

<!-- GETTING STARTED -->
## Getting Started

### Installation

<details>
<summary>Stable</summary>

To grab the latest stable release simply run

```sh
pip install openscvx
```

to install OpenSCVx in your python environment.

Or using uv:

```sh
uv pip install openscvx
```

For optional dependencies:

```sh
pip install openscvx[gui,cvxpygen]
# or with uv
uv pip install openscvx[gui,cvxpygen]
```
</details>

<details>
<summary>Nightly</summary>

To install the latest development version (nightly):

```sh
pip install --pre openscvx
```

With optional dependencies:

```sh
pip install --pre openscvx[gui,cvxpygen]
```

Or using uv:

```sh
uv pip install --pre openscvx
# With optional dependencies
uv pip install --pre openscvx[gui,cvxpygen]
```

**Note:** The `--pre` flag tells pip/uv to install pre-release versions (e.g., `1.2.4.dev3`) from PyPI.

Alternatively, for local development with the latest source:

```sh
# Clone the repo
git clone https://github.com/OpenSCvx/OpenSCvx.git
cd OpenSCvx

# Install in editable/development mode
pip install -e .
# or with uv
uv pip install -e .
```

</details>

### Dependencies

The main packages are:

- `cvxpy` - is used to formulate and solve the convex subproblems
- `jax` - is used for determining the Jacobians using automatic differentiation, vectorization, and ahead-of-time (AOT) compilation of the dynamics and their Jacobians 
- `numpy` - is used for numerical operations
- `diffrax` - is used for the numerical integration of the dynamics
- `termcolor` - is used for pretty command line output
- `plotly` - is used for all visualizations

These will be installed automatically, but can be installed via conda or pip if you are building from source.

#### GUI Dependencies (Optional)

For interactive 3D plotting and real-time visualization, additional packages are required:

- `pyqtgraph` - is used for interactive 3D plotting and real-time visualization
- `PyQt5` - provides the Qt5 GUI framework for pyqtgraph
- `scipy` - is used for spatial transformations in plotting functions
- `PyOpenGL` - provides OpenGL bindings for Python, required for 3D plotting
- `PyOpenGL_accelerate` - (optional) speeds up PyOpenGL


For local development:

```sh
pip install -e ".[gui]"
```

#### CVXPYGen Dependencies (Optional)

For code generation and faster solver performance, CVXPYGen can be installed:

- `cvxpygen` - enables code generation for faster solver performance
- `qocogen` - custom solver backend for CVXPYGen (included with cvxpygen extras)

To install with CVXPYGen support:

```sh
pip install openscvx[cvxpygen]
```

Or for both GUI and CVXPYGen:

```sh
pip install openscvx[gui,cvxpygen]
```

CVXPYGen features include:
- Automatic C++ code generation for optimization problems
- Faster solver performance through compiled code
- Support for custom solver backends like QOCOGen

### Local Development

This git repository can be installed using https

```sh
git clone https://github.com/OpenSCvx/OpenSCvx.git
```

or ssh

```sh
git clone git@github.com:OpenSCvx/OpenSCvx.git
```

Dependencies can then be installed using Conda or Pip

<details>
<summary>Via Conda</summary>

1. Clone the repo using https or ssh
2. Create a conda environment with Python:
   ```sh
   conda create -n openscvx python>=3.9
   ```
3. Activate the environment:
   ```sh
   conda activate openscvx
   ```
4. Install the package with dependencies:
   ```sh
   pip install -e .
   ```

   Or install with optional dependencies:
   ```sh
   pip install -e ".[gui,cvxpygen]"
   ```
</details>

<details>
<summary>Via uv</summary>

1. Prerequisites
   - Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repo using https or ssh
3. Create virtual environment and install the package:
   ```sh
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or install with optional dependencies:
   ```sh
   uv pip install -e ".[gui,cvxpygen]"
   ```
</details>

<details>
<summary>Via pip</summary>

1. Prerequisites
   Python >= 3.9
2. Clone the repo using https or ssh
3. Create virtual environment (called `venv` here) and source it
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the package with dependencies:
   ```sh
   pip install -e .
   ```

   Or install with optional dependencies:
   ```sh
   pip install -e ".[gui,cvxpygen]"
   ```
</details>

### Running Trajectory Optimization

See `examples/` folder for several example trajectory optimization problems grouped by application.
To run a problem simply run any of the examples directly, for example:

```sh
python3 examples/abstract/brachistochrone.py
```

> **Note:** To run the examples, you'll need to clone this repository and install OpenSCvx in editable mode (`pip install -e .`). See the [Local Development](#local-development) section above for detailed installation instructions.

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
