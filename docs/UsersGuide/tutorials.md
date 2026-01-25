# Tutorials

This section provides step-by-step tutorials for using OpenSCvx to solve trajectory optimization problems. Each tutorial includes a Google Colab notebook for interactive learning.

## Getting Started

If you're new to OpenSCvx, we recommend:

1. Start with the [Basic Problem Setup](basic_problem_setup.md) guide
2. Work through the 6DoF Obstacle Avoidance tutorial
3. Progress to the Line-of-Sight Guidance tutorial for advanced concepts
4. Explore the [Examples](../examples.md) section for additional problems

## Available Tutorials

### [6DoF Obstacle Avoidance](tutorial_6dof_obstacle_avoidance.md)

<a href="https://colab.research.google.com/drive/1xLPC_UJWC35oPRIAY3vkxi8WEYnHCysQ?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

Learn how to solve a minimum-time trajectory optimization problem for a 6-degree-of-freedom drone navigating around obstacles. This tutorial covers:

- State and control variable definition
- 6DoF dynamics with quaternions
- Obstacle avoidance constraints
- Continuous-time constraint satisfaction
- Problem instantiation and solving

### [6DoF Line-of-Sight Guidance](tutorial_6dof_los_guidance.md)

<a href="https://colab.research.google.com/drive/1b3NEx288h4r4HuvCOj-fexmt90PPhKUw?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

Solve a complex trajectory optimization problem involving gate navigation and line-of-sight constraints. This advanced tutorial demonstrates:

- Multi-gate navigation with sequence constraints
- Line-of-sight guidance constraints
- Mixed continuous and discrete constraints
- Advanced initial guess generation
- SCP parameter tuning

## Interactive Learning

Each tutorial includes a Google Colab notebook that you can run interactively without setting up a local environment. The notebooks contain the complete code examples and allow you to experiment with parameters and see results in real-time.