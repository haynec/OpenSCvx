---
description: >-
  Runnable OpenSCvx trajectory optimization examples spanning aerospace, robotics, and abstract optimal-control problems.
---

# Examples

OpenSCvx comes with a comprehensive set of examples demonstrating various trajectory optimization problems. These examples are located in the `examples/` folder and cover different applications and complexity levels.

## Running Examples

See `examples/` folder for several example trajectory optimization problems.
To run a problem simply run any of the examples directly, for example:

```sh
python3 examples/brachistochrone.py
```

and adjust the plotting as needed (see [Tutorial 05: Visualization](UsersGuide/05_visualization.md)).

## Creating Your Own Problems

Check out the problem definitions inside `examples/params` to see how to define your own problems. Each example demonstrates:

- State and control variable definition
- Dynamics specification
- Constraint formulation
- Problem instantiation and solving
- Results visualization

## Example Structure

Most examples follow this structure:

1. **Imports**: Import necessary OpenSCvx modules
2. **Problem Setup**: Define parameters, state, and control variables
3. **Dynamics**: Specify the system dynamics
4. **Constraints**: Define path and boundary constraints
5. **Problem Instantiation**: Create and configure the Problem
6. **Solving**: Run the optimization
7. **Visualization**: Plot and analyze results