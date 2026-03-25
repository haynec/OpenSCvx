"""Numerical integration schemes for trajectory optimization.

This module provides implementations of numerical integrators used for simulating
continuous-time dynamics.

Current Implementations:
    RK45 Integration: Explicit Runge-Kutta-Fehlberg method (4th/5th order)
        with both fixed-step and adaptive implementations via Diffrax.
        Supports a variety of explicit and implicit ODE solvers through the
        Diffrax backend (Dopri5/8, Tsit5, KenCarp3/4/5, etc.).

Planned Architecture (ABC-based):

A base class will be introduced to enable pluggable integrator implementations.
This will enable users to implement custom integrators.
Future integrators will implement the Integrator interface:

```python
# integrators/base.py (planned):
class Integrator(ABC):
    @abstractmethod
    def step(self, f: Callable, x: Array, u: Array, t: float, dt: float) -> Array:
        '''Take one integration step from state x at time t with step dt.'''
        ...

    @abstractmethod
    def integrate(self, f: Callable, x0: Array, u_traj: Array,
                    t_span: tuple[float, float], num_steps: int) -> Array:
        '''Integrate over a time span with given control trajectory.'''
        ...
```
"""

from .diffrax import (
    DEFAULT_DIFFRAX_ATOL,
    DEFAULT_DIFFRAX_RTOL,
    SOLVER_MAP,
    solve_ivp_diffrax,
    solve_ivp_diffrax_prop,
)
from .runge_kutta import (
    rk45_step,
    solve_ivp_rk45,
)

__all__ = [
    "DEFAULT_DIFFRAX_RTOL",
    "DEFAULT_DIFFRAX_ATOL",
    "SOLVER_MAP",
    "rk45_step",
    "solve_ivp_rk45",
    "solve_ivp_diffrax",
    "solve_ivp_diffrax_prop",
]
