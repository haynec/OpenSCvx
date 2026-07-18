---
title: Time Dilation
description: >-
  How OpenSCvx handles free-final-time and minimum-time problems by optimizing a
  time-dilation control on a fixed normalized-time grid.
---

# Time Dilation

*Time dilation* is how OpenSCvx supports **free-final-time** problems — including
minimum-time problems — on a fixed transcription grid. Rather than optimize over
physical time directly, OpenSCvx discretizes the trajectory on a **normalized
time** $\tau \in [0, 1]$ with fixed node spacing, and lets the solver stretch or
compress physical time through an extra control.

## Normalized time and the dilation factor

Let $\tau$ be normalized time and $t$ physical time. Their relationship is
governed by a scalar **time-dilation factor** $s(\tau)$:

$$
\frac{\mathrm{d}t}{\mathrm{d}\tau} = s(\tau).
$$

Because every physical derivative is taken with respect to $t$, expressing the
dynamics on the $\tau$ grid scales them by $s$:

$$
\frac{\mathrm{d}x}{\mathrm{d}\tau}
  = \frac{\mathrm{d}x}{\mathrm{d}t}\,\frac{\mathrm{d}t}{\mathrm{d}\tau}
  = s(\tau)\, f\big(x(\tau), u(\tau)\big).
$$

OpenSCvx treats time as a state (with $\mathrm{d}t/\mathrm{d}\tau = s \cdot 1$)
and $s$ as an ordinary per-node control. Making $s(\tau)$ a decision variable is
what turns a fixed-grid transcription into a free-final-time problem: a larger
$s$ spends more physical time per grid interval, so the solver can slow down near
constraint boundaries and speed up elsewhere. The dilation multiplication is
applied symbolically when the problem is built, so JAX autodiff produces the
correct Jacobians — including $\partial f/\partial s = f$ — automatically.

## Declaring a free final time

Free and minimum-time problems are declared through the `ox.Time` object passed
to `ox.Problem`. The `final` field accepts the same boundary markers as any
state — `"free"`, `"minimize"`, or `"maximize"` — so a minimum-time problem is
simply a free final time that is minimized:

```python
import openscvx as ox

# Minimum-time: final time is free and minimized.
time = ox.Time(initial=0.0, final=("minimize", tf_guess), min=0.0, max=800.0)

problem = ox.Problem(dynamics, ..., time=time)
```

OpenSCvx augments the problem with a dedicated dilation control automatically. By
default its per-node bounds are $0.3\,t_f \le s \le 3\,t_f$; you can override
them, seed a dilation guess, or force a uniform physical time step:

```python
time = ox.Time(
    initial=0.0,
    final=time_final,
    guess=time_guess,
    time_dilation_min=time_dilation_min,
    time_dilation_max=time_dilation_max,
    uniform_time_grid=False,  # allow s to vary node to node
)
```

Setting `uniform_time_grid=True` adds equality constraints forcing $s$ equal
across all nodes, recovering a single free scalar duration with evenly spaced
physical time steps.

## Related reading

- [Exact Discretization](discretization.md) — how the dilated dynamics are
  integrated on the $\tau$ grid.
- [Control Parameterization](control_parameterization.md) — how controls,
  including the dilation control, are interpolated between nodes.
