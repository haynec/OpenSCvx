---
title: Control Parameterization
description: >-
  Zero-order-hold (ZOH) versus first-order-hold (FOH) control parameterization
  between nodes in OpenSCvx, and how to set it per problem or per control.
---

# Control Parameterization

The optimization variables live at discrete **nodes**, but the dynamics are
continuous, so OpenSCvx must decide how the control behaves *between* nodes. This
choice is the **control parameterization**, and it determines how the control
value on a segment is reconstructed from the node values during discretization.

## Zero-order versus first-order hold

Across a segment from node $k$ to node $k+1$, the control is interpolated as

$$
u(\tau) = u_k + \beta(\tau)\,\big(u_{k+1} - u_k\big),
$$

where $\beta$ ramps from $0$ to $1$ over the segment. The two standard choices
differ only in $\beta$:

- **Zero-order hold (ZOH):** $\beta = 0$, so $u(\tau) = u_k$ is held constant
  across the segment. Only the current node couples into the segment dynamics, so
  discretization produces a single control transition matrix $B_d$.
- **First-order hold (FOH):** $\beta$ blends linearly from $0$ to $1$, so the
  control is piecewise-linear and continuous across nodes. Both endpoints couple
  in, producing two transition matrices, $B_d$ (current node) and $C_d$ (next
  node).

FOH gives a smoother, continuous control profile at the cost of an extra
transition matrix per segment; ZOH is cheaper and matches hardware that applies a
piecewise-constant command.

## Choosing a parameterization

The default parameterization is **FOH**. Set it globally through the
discretizer:

```python
import openscvx as ox

problem = ox.Problem(
    dynamics,
    ...,
    discretizer={"dis_type": "ZOH"},  # or "FOH" (default)
)
```

`dis_type` also accepts a per-control sequence (e.g. `["FOH", "ZOH", "FOH"]`) to
mix parameterizations across control channels.

A per-control setting overrides the discretizer default, so you can request a
specific hold on the control that needs it:

```python
thrust = ox.Control("thrust", shape=(3,), parameterization="ZOH")
```

The time-dilation control added for [free-final-time problems](time_dilation.md)
is parameterized as ZOH.

## Related reading

- [Exact Discretization](discretization.md) — how the chosen hold enters the
  discretized transition matrices.
- [Time Dilation](time_dilation.md) — the dilation control and its
  parameterization.
