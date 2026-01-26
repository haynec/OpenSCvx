# Getting Started

OpenSCvx is a JAX-based Python library for trajectory optimization using Successive Convexification (SCvx). It provides a simple interface for formulating and solving trajectory optimization problems with continuous-time constraint satisfaction.

!!! danger "Important"
    The library is currently in beta testing. Please report any issues on the GitHub repository.

## Key Features

- **JAX-based**: Automatic differentiation, vectorization, and compilation
- **Continuous-time constraints**: Support for path constraints that must be satisfied at all times
- **Successive Convexification**: Robust optimization algorithm for non-convex problems
- **Multiple constraint types**: Continuous-time, nodal, and boundary constraints
- **Interactive visualization**: 3D plotting and real-time optimization visualization
- **Code generation**: Automatic C++ code generation for optimization problems
- Faster solver performance through compiled code for smaller problems
- Support for customized solver backends like QOCOGen

## Installation

You can install OpenSCvx using pip or uv. For the most common use case, which includes support for interactive plotting and code generation, you can install the library with the `gui` and `cvxpygen` extras:

```sh
pip install openscvx[gui,cvxpygen]
# or with uv
uv pip install openscvx[gui,cvxpygen]
```

If you only need the core library without the optional features, you can run:

```sh
pip install openscvx
# or with uv
uv pip install openscvx
```

### Development Version (Nightly)

To install the latest development version (nightly) from PyPI:

```sh
pip install --pre openscvx[gui,cvxpygen]
# or with uv
uv pip install --pre openscvx[gui,cvxpygen]
```

Or for just the core library:

```sh
pip install --pre openscvx
# or with uv
uv pip install --pre openscvx
```

!!! note "Pre-release Versions"
    The `--pre` flag tells pip/uv to install pre-release versions (e.g., `1.2.4.dev3`). These nightly builds contain the latest features and bug fixes but may be less stable than official releases.

### Local Development

For local development, you can clone the repository and install it in editable mode:

```sh
# Clone the repo
git clone https://github.com/OpenSCvx/OpenSCvx.git
cd OpenSCvx

# Install in editable mode with all optional dependencies
pip install -e ".[gui,cvxpygen]"
# or with uv
uv pip install -e ".[gui,cvxpygen]"
```

### Dependencies

OpenSCvx has a few optional dependency groups:

The core dependencies are installed automatically with `openscvx`:

- `cvxpy` - for convex optimization
- `jax` - for fast linear algebra, automatic differentiation, and vectorization
- `numpy` - for numerical operations
- `diffrax` - for automatic differentiation
- `termcolor` - for colored terminal output
- `plotly` - for basic interactive 3D plotting


- **`gui`**: For interactive 3D plotting and real-time visualization. This includes:
    - `pyqtgraph` - for realtime 3D plotting
    - `PyQt5` - for GUI
    - `scipy` - for spatial operations
    - `PyOpenGL` - for 3D plotting
    - `PyOpenGL_accelerate` (optional, for speed) - for 3D plotting

- **`cvxpygen`**: For C++ code generation, enabling faster solver performance on smaller problems. This includes:
    - `cvxpygen` - for C++ code generation
    - `qocogen` - fast SOCP solver

### Local Development

For setting up a local development environment, we recommend using Conda to manage environments.

<details>
<summary>Via Conda</summary>

1.  Clone the repository:
    ```sh
    git clone https://github.com/OpenSCvx/OpenSCvx.git
    cd OpenSCvx
    ```
2.  Create and activate a conda environment with Python:
    ```sh
    conda create -n openscvx python>=3.9
    conda activate openscvx
    ```
3.  Install the package in editable mode with all optional dependencies:
    ```sh
    pip install -e ".[gui,cvxpygen]"
    ```
</details>

<details>
<summary>Via uv</summary>

1.  Prerequisites: Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2.  Clone the repository:
    ```sh
    git clone https://github.com/OpenSCvx/OpenSCvx.git
    cd OpenSCvx
    ```
3.  Create and activate a virtual environment:
    ```sh
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
4.  Install the package in editable mode with all optional dependencies:
    ```sh
    uv pip install -e ".[gui,cvxpygen]"
    ```
</details>

<details>
<summary>Via pip and venv</summary>

1.  Clone the repository:
    ```sh
    git clone https://github.com/OpenSCvx/OpenSCvx.git
    cd OpenSCvx
    ```
2.  Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the package in editable mode with all optional dependencies:
    ```sh
    pip install -e ".[gui,cvxpygen]"
    ```
</details>

## Quick Example

Here's a simple example to get you started with OpenSCvx. This demonstrates a minimum-time problem where a vehicle moves from the origin to a target position:

```python
import numpy as np
import openscvx as ox
from openscvx import Problem

# Define state variables
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.min = np.array([-10.0, -10.0])
position.max = np.array([10.0, 10.0])
position.initial = np.array([0.0, 0.0])
position.final = np.array([5.0, 5.0])

# Define control variables
velocity = ox.Control("velocity", shape=(2,))  # Velocity [vx, vy]
velocity.min = np.array([-2.0, -2.0])
velocity.max = np.array([2.0, 2.0])

# Set initial guesses
position.guess = np.linspace(position.initial, position.final, 20)
velocity.guess = np.repeat(
    np.expand_dims(np.array([1.0, 1.0]), axis=0), 20, axis=0
)

# Collect states and controls
states = [position]
controls = [velocity]

# Define dynamics using symbolic expressions
dynamics = {
    "position": velocity,  # position derivative is velocity
}

# Define time (minimize final time)
time = ox.Time(
    initial=0.0,
    final=("minimize", 5.0),  # Minimize final time with initial guess of 5.0
    min=0.0,
    max=10.0,
)

# Create the problem
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=[],
    N=20,
)

# Solve the problem
problem.initialize()
result = problem.solve()
result = problem.post_process(result)

# Access results
print(f"Converged: {result.converged}")
print(f"Optimal time: {result.t_final:.3f}")
print(f"Final position: {result.trajectory['position'][-1]}")
print(f"Total cost: {result.cost:.3f}")
```

!!! note "Note"
    This is a basic example. For a more detailed introduction, see the [Users Guide](UsersGuide/00_introduction.md).

## Next Steps

- **[Users Guide](UsersGuide/00_introduction.md)**: Progressive tutorials from basics to advanced topics
    - [01 Hello Brachistochrone](UsersGuide/01_hello_world_brachistochrone.md): Core API fundamentals
    - [02 Drone Racing](UsersGuide/02_drone_racing_constraints.md): Constraint types and 3D dynamics
    - [03 Obstacle Avoidance](UsersGuide/03_obstacle_avoidance_vmap.md): 6-DOF dynamics, Parameters, and Vmap
    - [04 Viewpoint Constraints](UsersGuide/04_viewpoint_constraints.md): Custom functions and perception
    - [05 Visualization](UsersGuide/05_visualization.md): 2D plots and 3D interactive visualization
- **[Examples](Examples/abstract/brachistochrone.md)**: Complete problem implementations across domains
- **[API Reference](Reference/problem.md)**: Detailed documentation for all classes and functions
- **[Citation](citation.md)**: Information for citing OpenSCvx in your research
