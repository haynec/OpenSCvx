# 05 Visualizing Results

Once you've solved a trajectory optimization problem, you'll want to see the results. OpenSCvx provides two complementary visualization systems:

- **2D Plots (Plotly)**: Time series, projections, and SCP debugging heatmaps
- **3D Interactive (Viser)**: Animated trajectory playback with gates, obstacles, thrust vectors, and more

This tutorial covers both systems, progressing from quick diagnostic plots to fully customized 3D visualizations.

This tutorial covers:

- Time series plots with `plot_states()` and `plot_controls()`
- 2D projections and vector norms
- Interactive 3D visualization with viser
- SCP convergence debugging
- Building custom visualizations from primitives

## 2D Plots with Plotly

The `openscvx.plotting` module provides high-level functions for common visualization tasks. These work with any problem and require no additional setup.

### Time Series: States and Controls

The most common visualization is plotting state and control trajectories over time:

```python
from openscvx.plotting import plot_states, plot_controls

# Plot all states in a subplot grid
plot_states(results).show()

# Plot specific states
plot_states(results, ["position", "velocity"]).show()

# Plot all controls
plot_controls(results).show()
```

Each component gets its own subplot with:

- **Green line**: High-fidelity propagated trajectory (from `post_process()`)
- **Cyan markers**: Discrete optimization nodes
- **Red dashed lines**: Variable bounds (if defined)

For single-component plots:

```python
from openscvx.plotting import plot_state_component, plot_control_component

# Plot just the z-component of position
plot_state_component(results, "position", component=2).show()

# Plot the first torque component
plot_control_component(results, "torque", component=0).show()
```

### 2D Projections

For 3D trajectories, viewing XY, XZ, and YZ plane projections is often more informative than 3D plots:

```python
from openscvx.plotting import plot_projections_2d

# Basic projections
plot_projections_2d(results, var_name="position").show()

# Color by velocity magnitude
plot_projections_2d(
    results,
    var_name="position",
    velocity_var_name="velocity",
    cmap="viridis"
).show()
```

### Vector Norms

For constraints on vector magnitudes (thrust limits, velocity bounds), plotting the norm over time is essential:

```python
from openscvx.plotting import plot_vector_norm

# Plot thrust magnitude with bounds
rho_min, rho_max = 5.0, 40.0
plot_vector_norm(results, "thrust_force", bounds=(rho_min, rho_max)).show()

# Plot velocity magnitude
plot_vector_norm(results, "velocity").show()
```

### SCP Debugging

When convergence is slow or the solution looks wrong, these heatmaps help diagnose issues:

```python
from openscvx.plotting import plot_trust_region_heatmap, plot_virtual_control_heatmap

# Which variables/nodes have large trust region violations?
plot_trust_region_heatmap(results).show()

# Where is virtual control being applied? (indicates constraint/dynamics violations)
plot_virtual_control_heatmap(results).show()
```

Large values in the virtual control heatmap indicate where the linearized dynamics or constraints are being artificially satisfied. Persistent hot spots suggest the problem may be infeasible or needs better initialization.

---

## 3D Interactive Visualization with Viser

For spatial problems (drones, rockets, spacecraft), 2D plots only tell part of the story. The `openscvx.plotting.viser` module provides interactive 3D visualization using [viser](https://github.com/nerfstudio-project/viser).
This excellent library provides an easy, performant API for animating 3D data and meshes in your browser. It also nicely supports link sharing and interactive elements. The team over at [Nerfstudio](https://github.com/nerfstudio-project) deserve a huge shout-out.

### Design Philosophy

We can't anticipate every visualization need, your problem might have unique constraints, custom geometry, or domain-specific elements we've never considered.
Rather than building a monolithic set of plotting functions that try to handle every case (and inevitably fails), we provide **composable building blocks** that you can mix and match.
These are designed to integrate cleanly with the existing viser API, encapsulating common utility functions rather than locking the user into our design choices.
If you want to do it yourself, the existing `result.trajectory["name"]` syntax should allow you to easily access your data.

The `openscvx.plotting.viser` module gives you:

- **Primitives**: Small, focused functions that each do one thing well (`create_server`, `add_gates`, `add_thrust_vector`, `add_animated_trail`, etc.)
- **Direct viser access**: Nothing stops you from using viser's API directly alongside our utilities

This approach means you're never fighting against our assumptions about what a "correct" visualization looks like.

### Animation Mechanism

The animation system uses a simple callback pattern. Understanding this upfront makes everything else click:

1. **Static elements** (gates, obstacles, ghost trajectory) are added once and never change
2. **Animated elements** return a tuple `(handle, update_callback)` where the callback is a function that takes a frame index and updates that element's state
3. **You collect all callbacks** into a list
4. **`add_animation_controls()`** wires the callbacks to GUI controls (play/pause, time slider, speed)

When the user plays the animation or scrubs the timeline, all callbacks are invoked with the current frame index, and each element updates itself.

```python
# Each animated primitive returns (handle, update_callback)
_, update_trail = add_animated_trail(server, positions, colors)
_, update_thrust = add_thrust_vector(server, positions, thrust, attitude=attitude)

# Collect callbacks
update_callbacks = [update_trail, update_thrust]

# Wire to GUI controls
add_animation_controls(server, time_array, update_callbacks)
```

This pattern is what makes the system composable: you can add or remove animated elements by simply including or excluding their callbacks.
The templates used in the examples in [`examples/plotting_viser.py`](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/plotting_viser.py) are just specific compositions of these primitives.

### Plotly Integration

Viser also supports embedding Plotly figures directly in its GUI sidebar.
We extend this with utilities that synchronize 2D plots with the 3D animation timeline, as you scrub through the trajectory, a time marker on the Plotly plot moves in sync.

This is useful for displaying time-series data (thrust magnitude, constraint values, _etc._) alongside the 3D view, giving you a complete picture of the trajectory in one window.

The integration follows the same callback pattern:

```python
from openscvx.plotting import plot_controls
from openscvx.plotting.viser import add_animated_plotly_vline

# Create any Plotly figure
fig = plot_controls(results, ["thrust_force"])

# Embed it with an animated vertical time marker
_, update_plot = add_animated_plotly_vline(server, fig, time_array=traj_time)
update_callbacks.append(update_plot)
```

For common cases like plotting vector norms, there's a convenience wrapper:

```python
from openscvx.plotting.viser import add_animated_vector_norm_plot

# Creates the plot and adds the animated marker in one call
_, update_norm = add_animated_vector_norm_plot(server, results, "thrust_force")
if update_norm:
    update_callbacks.append(update_norm)
```

### Template Reference

To see how these primitives compose into full visualizations, we provide template functions in [`examples/plotting_viser.py`](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/plotting_viser.py).
These are **not part of the openscvx package!** Rather, they are reference implementations meant to serve as an example or to be copied into your project and customized as necessary.

The templates demonstrate common patterns:

| Template | Use Case |
|----------|----------|
| `create_animated_plotting_server()` | General 3D trajectory animation with trail, thrust vectors, attitude frames, viewcones |
| `create_scp_animated_plotting_server()` | 3D SCP iteration convergence visualization |
| `create_pdg_animated_plotting_server()` | Powered descent guidance (rockets) with glideslope cone |

The next section walks through exactly how `create_animated_plotting_server` is built, teaching you the pattern, so you can create your own.

---

## Building Custom Visualizations

The templates in [`examples/plotting_viser.py`](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/plotting_viser.py) are convenient starting points, but they're intentionally just examples.
When your problem has unique requirements—custom constraint geometry, problem-specific annotations, or novel visual elements—you'll compose primitives from `openscvx.plotting.viser` directly.

The best way to learn the pattern is to walk through how `create_animated_plotting_server` is built. Understanding its structure teaches you how to create your own visualizations.

### Extract Data and Compute Colors

First, we can pull the trajectory data from results and compute velocity-based colors for the trail:

```python
from openscvx.plotting.viser import compute_velocity_colors, create_server

# Extract trajectory arrays (handles JAX arrays automatically)
pos = results.trajectory["position"]
vel = results.trajectory["velocity"]
thrust = results.trajectory.get("thrust_force")  # May be None
attitude = results.trajectory.get("attitude")    # May be None for 3-DOF
traj_time = results.trajectory["time"]

# Map velocity magnitude to viridis colormap
colors = compute_velocity_colors(vel)  # Shape: (N, 3), dtype uint8
```

### Create the Server

The `create_server()` helper creates a viser server and auto-sizes the grid based on your trajectory bounds:

```python
server = create_server(pos, show_grid=True)
```

At this point you have a browser-based 3D viewer (typically at `http://localhost:8080`) with an empty scene.

### Add Static Elements

Static elements are added once and don't change during animation. Add whatever is relevant to your problem:

```python
from openscvx.plotting.viser import (
    add_gates,
    add_ellipsoid_obstacles,
    add_ghost_trajectory,
)

# Racing gates (if you have gate vertices)
if "vertices" in results:
    add_gates(server, results["vertices"])

# Ellipsoidal obstacles (if present)
if "obstacles_centers" in results:
    add_ellipsoid_obstacles(
        server,
        centers=results["obstacles_centers"],
        radii=results.get("obstacles_radii"),
        axes=results.get("obstacles_axes"),
    )

# Ghost trajectory: faint full path so you can see the complete route
add_ghost_trajectory(server, pos, colors)
```

Other static primitives include `add_glideslope_cone()` for rocket landing problems.

### Add Animated Elements

As described in [Animation Mechanism](#animation-mechanism), each animated primitive returns `(handle, update_callback)`. Collect the callbacks into a list:

```python
from openscvx.plotting.viser import (
    add_animated_trail,
    add_attitude_frame,
    add_position_marker,
    add_thrust_vector,
    add_viewcone,
    add_target_markers,
)

update_callbacks = []

# Growing trail colored by velocity
_, update_trail = add_animated_trail(server, pos, colors)
update_callbacks.append(update_trail)

# Current position indicator: use attitude frame for 6-DOF, sphere for 3-DOF
if attitude is not None:
    _, update_attitude = add_attitude_frame(
        server, pos, attitude,
        axes_length=2.0
    )
    update_callbacks.append(update_attitude)
else:
    _, update_marker = add_position_marker(server, pos)
    update_callbacks.append(update_marker)

# Thrust vector arrow (body-frame thrust needs attitude)
_, update_thrust = add_thrust_vector(
    server, pos, thrust,
    attitude=attitude,
    scale=0.3
)
if update_thrust is not None:  # None if thrust data wasn't provided
    update_callbacks.append(update_thrust)
```

For viewplanning problems, you can add a camera viewcone and target markers:

```python
# Camera field-of-view cone (requires sensor parameters)
if attitude is not None and "R_sb" in results:
    _, update_viewcone = add_viewcone(
        server, pos, attitude,
        half_angle_x=np.pi / 6,  # 30 degree half-angle
        half_angle_y=np.pi / 6,
        scale=10.0,
        R_sb=results["R_sb"],
        opacity=0.4,
    )
    update_callbacks.append(update_viewcone)

# Target markers for viewplanning
if "init_poses" in results:
    target_results = add_target_markers(server, results["init_poses"], radius=1.0)
    for _, update in target_results:
        if update is not None:
            update_callbacks.append(update)
```

### Embed Plotly Panels

As described in [Plotly Integration](#plotly-integration), you can embed 2D plots in the viser sidebar with synchronized time markers. Here's a practical example with additional options:

```python
from openscvx.plotting.viser import add_animated_vector_norm_plot

# Add thrust magnitude plot with bounds and custom folder
_, update_norm = add_animated_vector_norm_plot(
    server, results, "thrust_force",
    title="Thrust Magnitude",
    bounds=(0.0, max_thrust),  # Show constraint bounds
    folder_name="Control Plots"  # Organize in GUI folder
)
if update_norm is not None:
    update_callbacks.append(update_norm)
```

You can add multiple plots—each appears in the sidebar and stays synchronized with the 3D animation.

### Wire Up Animation Controls

Finally, connect all the update callbacks to the animation system. This adds play/pause, a time slider, and speed controls to the GUI:

```python
from openscvx.plotting.viser import add_animation_controls

add_animation_controls(
    server, traj_time, update_callbacks,
    loop=True  # Loop when animation reaches the end
)
```

### Run Server

Keep the server alive so you can interact with the visualization:

```python
server.sleep_forever()
```

### Extending with Raw Viser

Our primitives don't lock you in. The `server` object is a standard `viser.ViserServer`, so you can freely mix our utilities with viser's native API:

```python
# Add custom scene elements using viser directly
server.scene.add_label("/custom/label", "Mission Start", position=(0, 0, 5))
server.scene.add_box("/custom/landing_pad", dimensions=(2, 2, 0.1), position=(0, 0, 0))

# Add GUI elements
with server.gui.add_folder("Custom Controls"):
    reset_button = server.gui.add_button("Reset View")
```

This interoperability means you're never limited to what we anticipated—if viser can do it, you can add it. See the [viser documentation](https://viser.studio/) for the full API.

### Putting It Together

Here's the complete pattern in one place:

```python
from openscvx.plotting.viser import (
    add_animated_trail,
    add_animated_vector_norm_plot,
    add_animation_controls,
    add_attitude_frame,
    add_gates,
    add_ghost_trajectory,
    add_thrust_vector,
    compute_velocity_colors,
    create_server,
)

# 1. Extract data
pos = results.trajectory["position"]
vel = results.trajectory["velocity"]
thrust = results.trajectory.get("thrust_force")
attitude = results.trajectory.get("attitude")
traj_time = results.trajectory["time"]
colors = compute_velocity_colors(vel)

# 2. Create server
server = create_server(pos)

# 3. Static elements
add_gates(server, gate_vertices)
add_ghost_trajectory(server, pos, colors)

# 4. Animated 3D elements
update_callbacks = []
_, cb = add_animated_trail(server, pos, colors)
update_callbacks.append(cb)
_, cb = add_attitude_frame(server, pos, attitude)
update_callbacks.append(cb)
_, cb = add_thrust_vector(server, pos, thrust, attitude=attitude, scale=0.3)
if cb: update_callbacks.append(cb)

# 5. Embedded Plotly panels (optional)
_, cb = add_animated_vector_norm_plot(server, results, "thrust_force")
if cb: update_callbacks.append(cb)

# 6. Wire up controls
add_animation_controls(server, traj_time, update_callbacks, loop=True)

# 7. Run
server.sleep_forever()
```

Copy this pattern, add or remove primitives as needed, and you have a custom visualization tailored to your problem.

## Further Reading

- [API Reference: Plotting](../Reference/plotting/index.md)
- [Viser Documentation](https://viser.studio/)
- [Complete Drone Racing with Viewplanning Example](../Examples/drone/dr_vp.md)
