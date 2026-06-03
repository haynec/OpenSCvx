# CI-SCVX: 3D monoped contact-implicit locomotion (Frax + OpenSCvx)

**Status:** Draft — **integrations + example only**; open design items resolved except Euler index map on URDF.

**Schema:** [`plans/PLAN_SCHEMA.md`](PLAN_SCHEMA.md)

---

## Motivation

Contact-implicit locomotion in OpenSCvx using Frax, informed by CI-SCVX (arXiv:2604.09993), implemented as:

- **`openscvx/integrations/`** — contact kinematics, CITO dynamics BYOF, impulse map, complementarity helpers, dFOH indexing, attitude-limit wiring.
- **`examples/frax/monoped_3d_cito.py`** + **`assets/robots/monoped_3d/`** — URDF, `Problem` assembly, constraints, guesses, viz.

**Existing OpenSCvx features we consume (read-only — not modified):**

| Need | Where it lives today |
|------|---------------------|
| `FraxDynamics` baseline | `openscvx/integrations/frax.py` |
| CTCS + time dilation | `ox.ctcs(...)`, `symbolic/augmentation.py` |
| Per-control FOH / ZOH | `Control.parameterization`, `discretization/linearize_discretize.py` |
| Impulsive shooting | `dynamics_discrete`, `byof["dynamics_discrete"]`, `preprocessing.py` |
| Nodal / cross-node constraints | `byof` in `expert/byof.py` |
| Fused defect `A,B,C` + `E_d u_imp` | `solvers/cvxpy_ptr_solver.py` (used as-is) |

**Call site:** `examples/frax/monoped_3d_cito.py` + `assets/robots/monoped_3d/`.

---

## Approach

### 1. Robot model: 3-link 3D monoped

| Item | Specification |
|------|----------------|
| Kinematic tree | World → 6-DoF floating base (torso) → hip → thigh → knee → shank |
| `n_j = 8`, `n_a = 2`, demo `n_c = 1` | Formulas indexed `i = 1 … n_c` |
| URDF | `assets/robots/monoped_3d/monoped.urdf` |

Foot kinematics (Frax): `foot_transform`, `foot_jacobian` → `p^c`, `J^c = J[:3,:]`.

Flat ground: `sd(p) = p_z - z_ground`, constant `R^c`, `n^c`, `t^c`.

---

### 2. Frax contact forces (integrations-level)

```text
phi_eff = u[phi_zoh_slice] + u[phi_foh_slice]   # §3.3
f_world = R^c @ [phi^{c,t}; phi^{c,n}]
tau_contact = J^c(q)^T @ f_world
tau_full = pad(tau_act) + tau_contact
qdd = robot.forward_dynamics(q, qd, tau_full, fext=None)
```

Frax `fext` is `(num_joints, 6)` world wrenches per joint index; point contacts use **`J^T f`**, not raw `fext` (StanfordASL/frax `robot.py`).

---

### 3. State, controls, shooting (single knot)

#### 3.1 States

| State | Role |
|-------|------|
| `q`, `qd` | Mechanical (adapter) |
| `y_*` | Auxiliary integrators for integral cross-complementarity — extra `State`s in the **example**, `.y` rates from **integrations BYOF** |

No `x^-` / `x^+`. Fused shooting (existing):

```text
x_k ≈ A_{k-1} x_{k-1} + B_{k-1} u_{k-1} + C_{k-1} u_k + E_k u_imp_k + bias
```

- CT: contact in `cito_qdd_byof`; CTCS on `y` via example `ox.ctcs` + BYOF `y` dots.
- Impulse: `byof["dynamics_discrete"]["qd"]` at nodes; impulsive `Phi`, `Gamma`; impact complementarity via **nodal** FB only (§6.2), not cross-complementarity.

#### 3.2 `tau` — FOH

`Control("tau", parameterization="foh")`.

#### 3.3 dFOH — integrations only (no discretizer changes)

Two control channels per scalar; discretizer interpolates each; **sum in BYOF**:

| Channel | `parameterization` |
|---------|------------------|
| `phi^n_zoh`, `gamma_zoh`, … | `"zoh"` |
| `phi^n_foh`, `gamma_foh`, … | `"foh"` |

```text
phi_eff = u[zoh_slice] + u[foh_slice]
```

**`DfohControlSlices`** in `integrations/frax_cito.py` — slice/index helper only, not a discretizer module.

#### 3.4 Impulsive controls

`Phi`, `Gamma` with `parameterization="impulsive"`.

---

### 4. `openscvx/integrations` deliverables

New module **`openscvx/integrations/frax_cito.py`** (re-export from `integrations/__init__.py`):

| Component | Role |
|-----------|------|
| `ContactModelConfig` | `n_c`, `mu`, **`delta`** (FB tightness; constant for now, no homotopy), **`epsilon_c=0.1`** (restitution), attitude limits, foot frames |
| `DfohControlSlices` / `build_cito_controls` | Dual-channel dFOH layout + slices |
| `contact_kinematics(...)` | JAX: `p^c`, `J^c`, `sd`, `lambda^c`, `rho^c`, … |
| `cito_qdd_byof` | Continuous dynamics (`phi_eff` from dual channels) |
| `cito_aux_byof` | `y_*` integrands for cross-complementarity only |
| `cito_impact_byof` | `dynamics_discrete` for `qd` jump + restitution `epsilon_c` |
| `cito_continuous_complementarity_fns` | Nodal FB on `(phi^n, sd)`, `(phi^n, lambda)`, `(gamma, rho)` |
| `cito_cross_complementarity_fns` | `cross_nodal` FB on `Δy` integrals (paper eq. 9–11) |
| `cito_impact_complementarity_fns` | Nodal FB on impact pairs `(Phi^n, sd)`, etc. — **separate** from cross |
| `fischer_burmeister(a, b, delta)` | **Always** take `delta` as an argument; wire `config.delta` at call sites |
| `apply_base_attitude_limits` | Roll/pitch ±30° (parametric); yaw free |
| `CitoFraxDynamics` | **Separate** adapter (keep `FraxDynamics` lean for manipulator examples); merges BYOF via `_merge_byof` |

Attitude **path** limits: `ox.ctcs` in the **example**, not new symbolic types.

---

### 5. Example-only wiring

| DOF | Start | End |
|-----|-------|-----|
| `q[base]` | Fixed `q_base_i` | Fixed `q_base_f` |
| `q[leg]` | Free | Free |
| `qd` | `0` | `0` |

Cost: `ox.Time(final=ox.Minimize(...))`. Pass `ContactModelConfig.delta` into every FB call (constant over the solve; homotopy deferred).

```python
problem = ox.Problem(
    dynamics=...,
    dynamics_discrete=...,  # and/or byof["dynamics_discrete"]
    states=...,
    controls=...,
    constraints=[ox.ctcs(...), ...],
    byof=cito_build_byof(...),
    ...
)
```

---

### 6. Complementarity (consume existing BYOF / CTCS)

**Fischer–Burmeister:** single helper `fischer_burmeister(a, b, delta) <= 0` (sign convention per `expert/byof.py`). **`delta` is always a function/config parameter** — we do not hardcode it inside helpers even though we are **not** running δ-homotopy yet (`ContactModelConfig.delta` held fixed for the whole solve).

Three **separate** constraint families (do not mix impulsive with cross-complementarity):

#### 6.1 Continuous contact (nodal, per knot `k`)

Per contact `i`, at each node — `byof["nodal_constraints"]`:

| Pair | Role |
|------|------|
| `(phi^{c,n}_i, sd(p^c_i))` | Normal force vs gap |
| `(phi^{c,n}_i, lambda^c_i)` | Stationarity |
| `(gamma_i, rho^c_i)` | Friction cone |

Uses `phi_eff` from dFOH dual channels in `cito_qdd_byof`.

#### 6.2 Integral cross-complementarity (interval, not impulsive)

Per contact `i`, per interval `k` — auxiliary `y` states + `cito_aux_byof`, enforced by **`byof["cross_nodal_constraints"]`** only:

| Pair integrated | Cross FB on `Δy` |
|-----------------|------------------|
| `(phi^{c,n}, sd)`, `(phi^{c,n}, lambda)`, `(gamma, rho)` | `FB(Δy_a, Δy_b, delta)` |

This is the paper’s inter-node machinery; **orthogonal** to impact constraints at nodes.

#### 6.3 Impulsive / impact (nodal, per knot `k`)

At impact nodes — **`byof["nodal_constraints"]`** (separate factory from §6.1):

- `(Phi^{c,n}_i, sd(p^c_i))` and related impact stationarity / friction pairs (paper eq. 6b–6e).
- Restitution: `epsilon^c = 0.1` default in `ContactModelConfig` (normal velocity relation in impact map / constraints).

Same `fischer_burmeister(..., delta)`; **no** `cross_nodal` terms for impulsive complementarity.

#### 6.4 Path constraints

Joint/torque boxes, attitude: `ox.ctcs` in example.

---

### 7. Implementation phases

| Phase | Where | Deliverable |
|-------|--------|-------------|
| **A** | `assets/` + integrations | URDF, `contact_kinematics` |
| **B** | integrations | `cito_qdd_byof`, `cito_impact_byof`, `cito_aux_byof`, `CitoFraxDynamics` |
| **C** | integrations | `build_cito_controls`, complementarity BYOF factories |
| **D** | example | `monoped_3d_cito.py` |
| **E** | manual | Verification |

---

### 8. Verification

- [ ] `J^T R phi_eff` vs finite-diff
- [ ] dFOH two-channel sum under integrator interpolation
- [ ] Impulse + fused shooting (impulsive path confirmed on branch; cf. `examples/abstract/impulsive.py`)
- [ ] Min-time base pose hop

---

## Out of Scope

- **Any edits** to `openscvx/symbolic/`, `openscvx/discretization/`, `openscvx/solvers/`
- δ-homotopy (automatic δ updates); paper duplicate states; paper reproduction

---

## Open Questions

1. **Euler index map** — Which `q[3:6]` entries are roll / pitch / yaw for the monoped URDF? Resolve in Phase A from `robot.joint_names` before calling `apply_base_attitude_limits`.

---

## Decision Log

| Date | Decision | Rejected alternative |
|------|----------|---------------------|
| 2026-05-29 | Minimal coords + `J^T R phi` | Maximal + joint reactions |
| 2026-05-29 | Flat ground | General SDF |
| 2026-05-30 | 3D monoped; `n_c=1` demo; generic `n_c` | Paper 2D / HalfCheetah |
| 2026-05-30 | Impacts via existing impulsive + `dynamics_discrete` | Duplicate pre/post states |
| 2026-05-30 | Partial BVP; min time | Full BVP; torque cost |
| 2026-05-30 | dFOH ZOH+FOH; FOH `tau` | ZOH-only |
| 2026-05-30 | `delta` **parameterized** on every FB call; constant via config (no homotopy loop) | Hardcoded δ inside helpers |
| 2026-05-30 | Roll/pitch ±30° parametric; yaw free | Unbounded attitude |
| 2026-05-30 | Single knot + fused shooting | Paper pre/post pairs |
| 2026-05-30 | **`integrations` + example only** | `DfohControlLayout` under discretization; symbolic/discretization feature work |
| 2026-05-30 | dFOH = dual ZOH+FOH controls, sum in **integrations BYOF** | Discretizer extension |
| 2026-05-30 | Integral cross: `y` + **`cross_nodal` FB only** | Impulsive constraints in cross_nodal |
| 2026-05-30 | Impact complementarity: **nodal FB only** (same FB helper, separate factory) | Mixing impact with cross-complementarity |
| 2026-05-30 | **`CitoFraxDynamics`** separate from `FraxDynamics` | `FraxDynamics(contact=...)` bloating manipulator path |
| 2026-05-30 | Impulsive shooting **confirmed** on branch | Blocking plan on solver work |
| 2026-05-30 | **`epsilon_c = 0.1`** (restitution) | Inelastic `epsilon_c = 0` |
