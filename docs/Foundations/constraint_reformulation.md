---
title: Constraint Reformulation (CTCS)
description: >-
  How OpenSCvx enforces path constraints continuously in time by integrating an
  augmented penalty state, rather than only at the discretization nodes.
---

# Isoperimetric Constraint Reformulation

![CTCS_dark]{ align=right width="60%" }
![CTCS_light]{ align=right width="60%" }

Path constraints are meant to hold for *all* time, but a node-based
transcription only checks them at the discretization nodes — leaving the solution
free to violate them *between* nodes. OpenSCvx closes this gap with
**continuous-time constraint satisfaction (CTCS)**: it reformulates a path
constraint as an integral condition on an augmented state, so satisfying it at
the nodes guarantees satisfaction throughout the interval.

## The augmented penalty state

Write each path constraint in canonical form $g(x, u) \le 0$. For a group of
CTCS constraints, OpenSCvx adds one augmented state $y$ whose derivative
accumulates a non-negative penalty of the constraint violations:

$$
\frac{\mathrm{d}y}{\mathrm{d}\tau}
  = s(\tau)\, \sum_j \mathrm{penalty}\big(g_j(x, u)\big),
$$

where the default penalty is the squared ReLU $\max(0, g_j)^2$ (Huber and smooth
ReLU are also available), and $s$ is the [time-dilation](time_dilation.md)
factor. Because the integrand is non-negative, $y$ is monotonically
non-decreasing and accumulates *any* violation occurring between nodes.

Enforcing $y \equiv 0$ then drives every penalty — and hence every constraint —
to zero continuously. In practice the convex subproblem (i) resets $y$ to $0$ at
the start of each enforcement interval and (ii) bounds the per-segment increment,

$$
y_0 = 0, \qquad |y_k - y_{k-1}| \le y_{\max},
$$

so a small $y_{\max}$ makes the between-node violation provably small. This is
the isoperimetric-style reformulation: a bounded integral of the violation stands
in for an infinite family of pointwise constraints.

## Declaring CTCS constraints

Wrap any path constraint with `ox.ctcs`, or call `.over(...)` on a constraint to
restrict it to a node interval:

```python
import openscvx as ox

constraints = [
    ox.ctcs(state <= state.max),
    ox.ctcs(state.min <= state, penalty="huber"),
    (distance >= obstacle_radius).over((0, N)),
]
```

Constraints sharing an enforcement interval are grouped onto one augmented state;
different intervals get separate states. The bound $y_{\max}$ is the constraint's
`licq_max` (default $10^{-4}$); tighten it for stricter between-node enforcement.
After solving, the achieved continuous-time violation is reported as
`results.ctcs_violation`.

## Related reading

- [Optimal Control Problem](ocp.md) — where the CTCS reset-and-bound constraints
  enter the convex subproblem.
- [Time Dilation](time_dilation.md) — the $s$ factor that also scales the CTCS
  dynamics.

[CTCS_dark]: ../assets/images/ctcs_dark.png#only-dark
[CTCS_light]: ../assets/images/ctcs_light.png#only-light
