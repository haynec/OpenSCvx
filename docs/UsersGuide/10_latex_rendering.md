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

Expression-level `ox.to_latex` output is **bare math** with no `$` delimiters —
you add your own, so it drops straight into any math environment. (A whole
problem is different: `problem.to_latex()` returns a complete display-math
fragment you paste as-is, covered [below](#rendering-a-whole-problem).)

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
\begin{subequations}
\begin{align}
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
\end{align}
\end{subequations}
\begin{align}
\dot{x}_{\mathrm{position}} &= x_{\mathrm{velocity}} \\
\dot{x}_{\mathrm{velocity}} &= u
\end{align}
```

The Mayer form keeps a clean `\dot{x} = f(x, u)` skeleton row, with the
per-state dynamics following as their own `align` block — that's the default
`dynamics="separate"`. The `subequations` + `align` wrapper makes each
formulation row number as `(4a)`, `(4b)`, … in a paper, and the whole string is
a complete display-math fragment — paste it in as-is, with **no `$$`
wrapping**. It typesets as (the docs and Jupyter render an `aligned` variant,
since MathJax doesn't implement `subequations`):

$$
\begin{gathered}
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
\\
\begin{aligned}
\dot{x}_{\mathrm{position}} &= x_{\mathrm{velocity}} \\
\dot{x}_{\mathrm{velocity}} &= u
\end{aligned}
\end{gathered}
$$

Reading the rows:

- **Objective.** Built from the `minimize` / `maximize` boundary types: a
  minimized final time shows as `+\lambda_{t}\, t(t_f)`. By default the
  coefficient is a symbolic `\lambda` subscripted by the state's symbol and
  element (`\lambda_{t}`, `\lambda_{\mathrm{velocity},1}`) — the weight as
  notation, not a tuned number. Pass `weights="numeric"` to substitute the
  `lam_cost` values instead (`0.01\, t(t_f)`, omitted when the weight is `1`);
  `maximize` flips the sign either way.
- **Dynamics.** By default the formulation carries the skeleton row
  `\dot{x} = f(x, u)` and the per-state definitions follow as their own
  block (`dynamics="separate"`); pass `dynamics="inline"` to put the
  `\dot{x}_{i} = \ldots` rows directly in the formulation.
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
of each section is expanded:

| Level | Dynamics | Constraints |
|-------|----------|-------------|
| `"inline"` | one `\dot{x}_{i} = ...` row per equation | full constraint bodies with annotations |
| `"symbolic"` | — | numbered `g_i(x,u) \le 0` / `h_j(x,u) = 0` references only |
| `"separate"` | `\dot{x} = f(x, u)` skeleton, definitions appended as a bare `align` block | numbered references, residual definitions appended |

The defaults are `dynamics="separate", constraints="inline"` — the default
output is complete (nothing about your problem is hidden), the Mayer skeleton
stays clean however large the dynamics get, and path constraints are usually
one-liners worth showing in place. If you want only the `\dot{x} = f(x, u)`
skeleton, delete the definition block from the output.

For a fully symbolic paper skeleton, separate the constraints too:

```python
print(problem.to_latex(constraints="separate"))
```

```latex
\begin{subequations}
\begin{align}
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
\end{align}
\end{subequations}
\begin{align}
\dot{x}_{\mathrm{position}} &= x_{\mathrm{velocity}} \\
\dot{x}_{\mathrm{velocity}} &= u
\end{align}
\begin{align}
g_{1}(x, u) &= \left\| x_{\mathrm{position}} \right\| - 5 \\
g_{2}(x, u) &= t - 2 \\
g_{3}(x, u) &= 0 - t \\
g_{4}(x, u) &= \left\| u \right\| - 10 \\
h_{1}(x, u) &= x_{\mathrm{velocity},0} - 0
\end{align}
```

Each separated section becomes its own bare `align` block, joined to the Mayer
form by a single newline and **no** connective text — no `\text{where}`, no
spacing glue. That editorial call (a "where", vertical space, prose) is left to
you. Inequalities are numbered `g_i` and equalities `h_j`, in bucket order; the
same label indexes both the formulation reference and its definition, so the two
line up. The whole thing is still one string — a reader who only wants the
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
