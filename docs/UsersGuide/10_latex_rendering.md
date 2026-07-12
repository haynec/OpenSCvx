# 10 LaTeX Rendering

Once you have built a problem, you often want to *see* it — as math, not as a
class-name tree dump. OpenSCvx can render any symbolic expression, or a whole
`Problem`, to a LaTeX string you can paste straight into a paper, a notebook, or
these docs.

This tutorial covers:

- Rendering a single expression with `ox.to_latex`
- Automatic math rendering in Jupyter
- Rendering a whole problem as a Mayer-form formulation with `problem.to_latex()`
- The `dynamics=` / `constraints=` detail levels

## Rendering an expression

`ox.to_latex` takes any symbolic expression (or a sequence of them) and returns
a LaTeX string:

```python
import openscvx as ox
from openscvx.symbolic.expr import Norm

position = ox.State("position", shape=(2,))

ox.to_latex(Norm(position) - 5.0)
# '\\left\\| x_{\\mathrm{position}} \\right\\| - 5'
```

States and controls are **role-prefixed** so every occurrence is grounded in the
skeleton's `f(x, u)`: a `State` renders as `x_{<sym>}` and a `Control` as
`u_{<sym>}`, where `<sym>` is the name rendered as a symbol — multi-letter names
as `\mathrm{...}`, Greek words as their commands (`theta` → `\theta`), and
`name_sub` as a subscript. A state literally named `x` (or a control named `u`)
renders bare, and time stays `t`. Element indexing comma-merges into the same
subscript group — `x_{\mathrm{position},0}`, never the invalid
`x_{\mathrm{position}}_{0}`.

Pass a list to render several expressions at once:

```python
ox.to_latex([Norm(position), ox.Sum(position)])
# ['\\left\\| x_{\\mathrm{position}} \\right\\|', '\\sum x_{\\mathrm{position}}']
```

The strings carry **no `$` delimiters** — you add your own, so the output drops
straight into any math environment.

!!! tip "Rendering a single dynamics equation"
    `problem.dynamics` exposes the user-authored `{state_name: expr}` dict, so
    you can render one equation on its own:

    ```python
    ox.to_latex(problem.dynamics["velocity"])
    ```

### In a Jupyter notebook

Expressions implement `_repr_latex_`, so they display as typeset math
automatically — just evaluate one as the last line of a cell:

```python
Norm(position) - 5.0   # renders as ‖position‖ − 5
```

## Rendering a whole problem

`problem.to_latex()` renders the entire problem as a classic **Mayer-form**
optimal control formulation: a minimize objective over a subject-to list of
dynamics, path constraints, box bounds, and boundary conditions.

```python
print(problem.to_latex())
```

For a small double-integrator problem — minimize final time, a CTCS bound on
position, a nodal control bound, a nodal terminal equality, box bounds on every
variable, and fixed initial/final positions — the default output is:

```latex
\begin{aligned}
\min_{x,\,u} \quad & \lambda_{t}\, t(t_f) \\
\text{s.t.} \quad & \dot{x} = f(x, u) \\
 & \left\| x_{\mathrm{position}} \right\| - 5 \le 0 \quad \forall t \\
 & t - 2 \le 0 \quad \forall t \\
 & 0 - t \le 0 \quad \forall t \\
 & \left\| u \right\| - 10 \le 0 \quad k = 4 \\
 & x_{\mathrm{velocity},0} - 0 = 0 \quad k = 4 \\
 & \begin{bmatrix} -10 \\ -10 \end{bmatrix} \le x_{\mathrm{position}} \le \begin{bmatrix} 10 \\ 10 \end{bmatrix} \\
 & \begin{bmatrix} -5 \\ -5 \end{bmatrix} \le x_{\mathrm{velocity}} \le \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & 0 \le t \le 2 \\
 & \begin{bmatrix} -1 \\ -1 \end{bmatrix} \le u \le \begin{bmatrix} 1 \\ 1 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_f) = \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & x_{\mathrm{velocity}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & t(t_0) = 0
\end{aligned}
```

which typesets as:

$$
\begin{aligned}
\min_{x,\,u} \quad & \lambda_{t}\, t(t_f) \\
\text{s.t.} \quad & \dot{x} = f(x, u) \\
 & \left\| x_{\mathrm{position}} \right\| - 5 \le 0 \quad \forall t \\
 & t - 2 \le 0 \quad \forall t \\
 & 0 - t \le 0 \quad \forall t \\
 & \left\| u \right\| - 10 \le 0 \quad k = 4 \\
 & x_{\mathrm{velocity},0} - 0 = 0 \quad k = 4 \\
 & \begin{bmatrix} -10 \\ -10 \end{bmatrix} \le x_{\mathrm{position}} \le \begin{bmatrix} 10 \\ 10 \end{bmatrix} \\
 & \begin{bmatrix} -5 \\ -5 \end{bmatrix} \le x_{\mathrm{velocity}} \le \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & 0 \le t \le 2 \\
 & \begin{bmatrix} -1 \\ -1 \end{bmatrix} \le u \le \begin{bmatrix} 1 \\ 1 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_f) = \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & x_{\mathrm{velocity}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & t(t_0) = 0
\end{aligned}
$$

Reading the rows:

- **Objective.** Built from the `minimize` / `maximize` boundary types: a
  minimized final time shows as `+\lambda_{t}\, t(t_f)`. By default the
  coefficient is a symbolic `\lambda` subscripted by the state's symbol and
  element (`\lambda_{t}`, `\lambda_{\mathrm{velocity},1}`) — the weight as
  notation, not a tuned number. Pass `weights="numeric"` to substitute the
  `lam_cost` values instead (`0.01\, t(t_f)`, omitted when the weight is `1`);
  `maximize` flips the sign either way.
- **Dynamics.** By default a single symbolic row `\dot{x} = f(x, u)`.
- **Path constraints.** Each CTCS / nodal / cross-node constraint, with its
  temporal annotation kept visible — `\forall t` for a full-horizon CTCS,
  `k = 4` (or `k \in \{...\}`, `\forall k`) for nodal constraints. The two
  `t` rows are the `min`/`max` you gave `ox.Time`: preprocessing enforces
  time bounds as continuous-time constraints, and they render like any
  other constraint on the problem.
- **Box bounds.** One `lb \le v \le ub` row per state and control with a finite
  bound; length-1 bounds collapse to bare scalars, vectors render as `bmatrix`.
- **Boundary conditions.** Fixed initial/final values only — free elements are
  omitted, and minimize/maximize elements already appear in the objective.

!!! note "It's the problem you wrote"
    The formulation is the **pre-augmentation, user-authored** problem. The
    solver augments the dynamics with a time-dilation control and per-CTCS
    penalty states, but that machinery never appears here — CTCS constraints
    render as the continuous-time path constraints they stand for, which is the
    whole point of the Mayer story.

## Detail levels

The `dynamics=` and `constraints=` keyword arguments independently set how much
of each section is expanded, at one of three levels:

| Level | Dynamics | Constraints |
|-------|----------|-------------|
| `"inline"` | one `\dot{x}_{i} = ...` row per equation | full constraint bodies with annotations |
| `"symbolic"` | one `\dot{x} = f(x, u)` placeholder | numbered `g_i(x,u) \le 0` / `h_j(x,u) = 0` references |
| `"separate"` | symbolic, with definitions appended in a `\text{where}` block | symbolic, with residual definitions appended |

The defaults are `dynamics="symbolic", constraints="inline"` — dynamics dicts
are where the bloat lives, while path constraints are usually one-liners worth
showing in place.

For a paper-style skeleton with everything defined below it, use
`"separate"` for both sections:

```python
print(problem.to_latex(dynamics="separate", constraints="separate"))
```

```latex
\begin{aligned}
\min_{x,\,u} \quad & \lambda_{t}\, t(t_f) \\
\text{s.t.} \quad & \dot{x} = f(x, u) \\
 & g_{1}(x, u) \le 0 \quad \forall t \\
 & g_{2}(x, u) \le 0 \quad \forall t \\
 & g_{3}(x, u) \le 0 \quad \forall t \\
 & g_{4}(x, u) \le 0 \quad k = 4 \\
 & h_{1}(x, u) = 0 \quad k = 4 \\
 & \begin{bmatrix} -10 \\ -10 \end{bmatrix} \le x_{\mathrm{position}} \le \begin{bmatrix} 10 \\ 10 \end{bmatrix} \\
 & \begin{bmatrix} -5 \\ -5 \end{bmatrix} \le x_{\mathrm{velocity}} \le \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & 0 \le t \le 2 \\
 & \begin{bmatrix} -1 \\ -1 \end{bmatrix} \le u \le \begin{bmatrix} 1 \\ 1 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & x_{\mathrm{position}}(t_f) = \begin{bmatrix} 5 \\ 5 \end{bmatrix} \\
 & x_{\mathrm{velocity}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \\
 & t(t_0) = 0
\end{aligned}
\\[1ex]
\text{where}\\[0.5ex]
\begin{aligned}
\dot{x}_{\mathrm{position}} &= x_{\mathrm{velocity}} \\
\dot{x}_{\mathrm{velocity}} &= u
\end{aligned}
\\[1ex]
\begin{aligned}
g_{1}(x, u) &= \left\| x_{\mathrm{position}} \right\| - 5 \\
g_{2}(x, u) &= t - 2 \\
g_{3}(x, u) &= 0 - t \\
g_{4}(x, u) &= \left\| u \right\| - 10 \\
h_{1}(x, u) &= x_{\mathrm{velocity},0} - 0
\end{aligned}
```

Inequalities are numbered `g_i` and equalities `h_j`, in bucket order; the same
label indexes both the formulation reference and its definition, so the two line
up. The whole thing is still one string — a reader who only wants the
constraints block copies it out of the output.

## A note on exotic nodes

The LaTeX backend ships visitors for arithmetic, linear algebra, elementwise
math, array construction, and the constraint nodes — enough for the vast
majority of problems. Nodes without a visitor yet (STL, Lie-group operators like
`SO3Exp`, spatial nodes) raise `NotImplementedError`:

```python
ox.to_latex(SO3Exp(omega))
# NotImplementedError: 'LatexLowerer' has no visitor for SO3Exp
```

In a notebook this is invisible — `_repr_latex_` catches the error and returns
`None`, so Jupyter simply falls back to the plain `__repr__`.
