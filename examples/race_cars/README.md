This folder contains the code that has been used for the simulations and experiments associated with the 
publication: *NMPC for Racing Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance - Daniel Kloeser, Tobias Schoels, Tommaso Sartor, Andrea Zanelli, Gianluca Frison, Moritz Diehl. Proceedings of the 21th IFAC World Congress, Berlin, Germany - July 2020*. 
A video of the experiments can be found on youtube: https://www.youtube.com/watch?v=1JDBQXVrZbo.

`race_car_hybrid.py` extends the minimum-lap-time problem with a hybrid
power unit patterned on the 2026 Formula 1 regulations — a ~55/45
combustion/electric power split, a battery, and a per-lap recovery cap, all
scaled to the RC car by dimensionless ratios (see the module docstring).
It races a 4x-scaled LMS track (`tracks/LMS_Track_x4.txt`) whose straights
push the car into its power-limited regime, and replaces the box acceleration
bounds with a friction ellipse so deployment and harvesting compete with
cornering for tyre grip. The lap is a flying qualifying lap — the driving
states are periodic across the lap (convex cross-node equalities) while the
state of charge is free at both ends — and the per-lap recovery cap
actively binds at the optimum. The example solves three power-unit variants — hybrid,
MGU-K failure, and an unrestricted full-envelope ICE — as a single batched
solve over the power-unit parameters, then races all three cars on one
Viser track to make the electric system's lap-time worth directly visible.

`race_car_mpc.py` swaps the single global minimum-time solve for a
receding-horizon controller: a short fixed horizon that maximises arc-length
progress, applied one node at a time in closed loop, mirroring the acados
benchmark's NMPC formulation.

`race_car_multi_agent.py` combines the two: a whole grid of the hybrid cars
races `M_LAPS` laps wheel-to-wheel from an F1-style standing start. One symbolic problem
describes a single car's MPC horizon; each race step advances the entire
field with a single `solve_batched` call, batching over per-car boundary
pins, spec parameters (power, mass, battery size), and each car's forecast
of its opponents' trajectories. Cars keep an elliptical clearance in track
coordinates from the plans their opponents published on the previous step —
decentralized MPC with communicated plans — so overtakes and energy strategy
emerge from the optimization. The `AGENTS` roster at the top of the file is
the only thing to edit to grow the field or tweak a car's spec.

`race_car_viser.py` and `race_car_plots.py` are the shared visualisation
toolkit behind all of the above. The former builds the 3D Viser scene — track
mesh, low-poly cars, single-lap, chase-camera, and multi-car comparison
servers. The latter holds the Plotly vocabulary the examples compose their
figures from: the bird's-eye track scaffold, the friction-ellipse trace, and
the live telemetry panels (striplines and g-g with per-car playback markers)
that run inside the Viser sidebar during a replay.
