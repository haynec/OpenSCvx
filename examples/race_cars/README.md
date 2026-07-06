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
cornering for tyre grip. The lap is a qualifying lap — boundary state of
charge is free at both ends — and the per-lap recovery cap actively binds
at the optimum. The example solves three power-unit variants — hybrid,
MGU-K failure, and an unrestricted full-envelope ICE — as a single batched
solve over the power-unit parameters, then races all three cars on one
Viser track to make the electric system's lap-time worth directly visible.
