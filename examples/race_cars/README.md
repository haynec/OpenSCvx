This folder contains the code that has been used for the simulations and experiments associated with the 
publication: *NMPC for Racing Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance - Daniel Kloeser, Tobias Schoels, Tommaso Sartor, Andrea Zanelli, Gianluca Frison, Moritz Diehl. Proceedings of the 21th IFAC World Congress, Berlin, Germany - July 2020*. 
A video of the experiments can be found on youtube: https://www.youtube.com/watch?v=1JDBQXVrZbo.

Every example races the same spatial (path-parametric) bicycle model on the
same LMS kart track, so the files read as one family: start at
`race_car_openscvx.py` and each later file adds exactly one idea — a power
unit, a receding horizon, a field of opponents.

## Single car, one lap

`race_car_openscvx.py` is the baseline everything else is an ablation of: the
acados minimum-lap-time benchmark written as one global free-final-time solve
over the whole lap, with box acceleration limits.

`race_car_ice.py` moves the baseline to the 4x-scaled track
(`tracks/LMS_Track_x4.txt`), swaps the box limits for a friction ellipse, and
makes the lap a flying one — the same car as `race_car_hybrid.py` with the
hybrid power unit removed, so the pair measures what the energy management
costs in solve time and convergence.

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

## Closed loop

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

`race_car_multi_agent_mpcc.py` races the same field against a *reference*
instead of raw progress: phase 1 solves each spec's minimum-time flying lap,
phase 2 races while every car regulates around that shared lap through pchip
reference splines and a per-car pace scalar. Tracking a nominal lap solves far
faster than maximising progress, and the racing shows up as the deviation each
car buys from the reference.

`race_car_multi_agent_mpcc_ice.py` is the pure-combustion twin of the tracking
race: same field, track, and objective with the battery, recovery accounting,
and deploy/harvest controls removed, so the pair measures the per-step cost of
the hybrid system in a closed-loop race.

## Shared modules (not runnable)

`_viser.py` builds the 3D scenes: the LMS track mesh, the low-poly car, and
the overview/chase/comparison playback servers every example above imports.

`_plotting.py` builds the shared 2D Plotly figures: the track outline, the
g-g diagram against the friction ellipse, the acceleration-vs-bounds trace,
and the dark telemetry panels that play back in the Viser sidebar.

`time2spatial.py` and `tracks/readDataFcn.py` are vendored acados helpers
(2-clause BSD) that convert between path and Cartesian coordinates and load
the track tables; their upstream names are kept deliberately.
