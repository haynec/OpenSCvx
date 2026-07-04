This folder contains the code that has been used for the simulations and experiments associated with the 
publication: *NMPC for Racing Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance - Daniel Kloeser, Tobias Schoels, Tommaso Sartor, Andrea Zanelli, Gianluca Frison, Moritz Diehl. Proceedings of the 21th IFAC World Congress, Berlin, Germany - July 2020*. 
A video of the experiments can be found on youtube: https://www.youtube.com/watch?v=1JDBQXVrZbo.

`race_car_f1_energy.py` extends the minimum-lap-time problem with a hybrid
power unit patterned on the 2026 Formula 1 regulations — a ~55/45
combustion/electric power split, a battery, and a per-lap recovery cap, all
scaled to the RC car by dimensionless ratios (see the module docstring). The
lap is charge-sustaining, so every joule deployed must first be harvested
under braking; on this tight kart track the solution is harvest-limited
rather than cap-limited, mirroring the real 2026 concern that recovering the
full per-lap allowance is hard on most circuits.
