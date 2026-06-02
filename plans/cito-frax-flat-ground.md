# CI-SCVX: 3D monoped contact-implicit locomotion (Frax + OpenSCvx)

**Status:** Draft — design locked including fused impulsive shooting, fixed complementarity `δ`, and parameterized base attitude limits; implementation not started.

**Schema:** [`plans/PLAN_SCHEMA.md`](PLAN_SCHEMA.md)

---

## Motivation

Implement a **contact-implicit trajectory optimization (CITO)** example in OpenSCvx that uses **Frax** for 3D minimal-coordinate multibody dynamics, informed by CI-SCVX (arXiv:2604.09993) but **not** aimed at reproducing the paper’s 2D monoped results.

Differentiators for this milestone:

- **Full 3D** floating-base monoped (torso + thigh + shank, hip + knee actuation) — a deliberate extension beyond the paper’s planar models.
- **Impacts at grid nodes** so `qd` can jump when contact initiates (physically necessary for impulsive hopping).
- **Minimum-time** objective.
- **Partial boundary value problem:** fix **base pose** at start and end; let the optimizer choose leg joint configurations.
- **dFOH** contact/KKT controls (ZOH + FOH superposition) for realistic force mode switches; **FOH** joint torques.

OpenSCvx hooks: `FraxDynamics` (`openscvx/integrations/frax.py:212-309`), CTCS + time dilation (`openscvx/symbolic/augmentation.py:490-687`), per-control FOH/ZOH (`openscvx/symbolic/expr/control.py:7-49`, `openscvx/discretization/base.py:73-104`), impulsive controls (`openscvx/symbolic/preprocessing.py:804-816`).

**Call site:** `examples/frax/monoped_3d_cito.py` + `assets/robots/monoped_3d/` (URDF + foot frame offset).

---

## Approach

### 1. Robot model: 3-link 3D monoped

| Item | Specification |
|------|----------------|
| Kinematic tree | World → **6-DoF floating base** (torso) → hip revolute → thigh → knee revolute → shank |
| Actuation | `n_a = 2` (hip, knee); base unactuated |
| Generalized coords | `n_j = 8` (`nq == nv` in Frax) |
| Contact | **`n_c = 1`** foot point on shank (URDF site / `Manipulator`-style frame offset) |
| Formulation generality | All contact formulas indexed by `i = 1 … n_c` |

**URDF / Frax loading**

- Add `assets/robots/monoped_3d/monoped.urdf` (and meshes if needed).
- Wrap as `frax.Manipulator` (or `Robot`) with `foot_offset` / `foot_parent_chain` analogous to `ee_offset` / `ee_parent_chain` in `frax/core/manipulator.py:59-98`.
- Foot world position: `p^c(q) = foot_transform(q)[:3, 3]`.
- Foot **linear** Jacobian: `J^c(q) = foot_jacobian(q)[:3, :]` (6×`n_j` spatial Jacobian; use top 3 rows per `frax/core/robot.py:383-384`).

**Flat ground (3D)**

- `sd(p) = p_z - z_ground`
- `n^c = e_z`, `t^c = [e_x, e_y]`, `R^c = [t^c | n^c]` (constant)

---

### 2. Frax force conventions (researched)

Source: StanfordASL/frax `frax/core/robot.py` (cloned for review; cite line numbers from upstream when implementing).

#### 2.1 `forward_dynamics(q, qd, tau, fext)`

| Argument | Shape | Frame / meaning |
|----------|-------|----------------|
| `q`, `qd` | `(num_joints,)` | Minimal coordinates |
| `tau` | `(num_joints,)` | **Joint torques / generalized actuation** on every DOF (floating-base entries included; OpenSCvx pads zeros for unactuated base — `openscvx/integrations/frax.py:199-204`) |
| `fext` | `(num_joints, 6)` or `None` | **Spatial wrench per joint index**, expressed in **world / root frame**. Each row is `[f_x, f_y, f_z, m_x, m_y, m_z]`. Applied in RNEA as `link_forces -= F_ext` (`robot.py:958-959`). |

Gravity is **always on** inside Frax FD (`g_accel = [0,0,9.81,0,0,0]` at `robot.py:1068`).

#### 2.2 Recommended contact wrench mapping (consistent with paper `J_c^T`)

Use the **positional Jacobian** (world linear velocity map):

```text
f_world_i = R^c_i(q) @ [phi^{c,t}_i; phi^{c,n}_i]     # 3-vector
tau_contact_i = J^c_i(q)^T @ f_world_i                 # (n_j,)
tau_full = tau_act_padded + sum_i tau_contact_i
qdd = robot.forward_dynamics(q, qd, tau_full, fext=None)
```

**Why not `fext` for point contacts?** `fext[k]` is a wrench on joint/link `k`’s spatial frame, not a world-space force at an offset contact point. A point foot force requires either an equivalent wrench with moment `(p^c - o^link) × f` on the shank joint row, or the **`J^T f`** form above. The latter matches the paper’s `J_c^T R_c phi` in minimal coordinates and avoids per-link moment bookkeeping.

**Optional equivalent (document only):** populate `fext[foot_joint_idx] = [f_world; cross(p^c - o_link, f_world)]` and pass `tau_full = tau_act_padded` only. Numerically must match `J^T f` in validation.

#### 2.3 Foot kinematics API

Reuse Frax `_frame_jacobian` pattern (`robot.py:372-423`):

- `foot_transform(q)` → 4×4 world pose
- `foot_jacobian(q)` → 6×`n_j`, `Jv = J[:3, :]`
- `v^c_t = t^c^T Jv @ qd`

---

### 3. State, controls, and indexing

#### 3.1 Mechanical state (single knot state — not paper pre/post)

OpenSCvx does **not** support duplicate pre/post states (`x^-`, `x^+`) as in the paper. Use **one** state vector per grid node `k`:

| Variable | Shape | Notes |
|----------|-------|-------|
| `q_k`, `qd_k` | `(n_j,)` each | Only nodal states; `q` is **continuous** across impacts |

**Impulse is folded into multiple shooting**, not a separate equality between two states at the same node. Per interval defect (linearized form in `openscvx/solvers/cvxpy_ptr_solver.py:601-614`):

```text
x_k ≈ A_{k-1} x_{k-1} + B_{k-1} u_{k-1} + C_{k-1} u_k + E_k u_imp_k + bias_{k-1}
```

where:

- `A_{k-1}, B_{k-1}, C_{k-1}` come from **CT propagation** of the augmented dynamics (contact forces inside the flow, time dilation, CTCS auxiliary `y`).
- `E_k u_imp_k` comes from the **impulsive map** at node `k` (primarily a jump in `qd`; `q` and `y` unchanged — consistent with CTCS identity on augmented states across impulses, `openscvx/symbolic/augmentation.py:592-617`).

Nominal propagation uses `x_prop_plus[k]` as the post-impulse target at node `k` when assembling `dyn_bias` (`cvxpy_ptr_solver.py:784-791`). Transcription must implement:

1. `f_CT`: integrate `(q, qd, y)` over `[t_{k-1}, t_k]` with dFOH contact controls and FOH `tau`.
2. `f_imp`: `x_k = f_imp(x_CT_end, Phi_k, Gamma_k)` with `f_imp` returning the same `x` stack but updated `qd` via paper eq. (6a) in minimal coordinates; `Phi`, `Gamma` on impulsive control slices.
3. **Combined Jacobian** for SCP: sensitivities of `x_k` w.r.t. `(x_{k-1}, u_{k-1}, u_k, u_imp_k)` — chain rule through CT sensitivities plus `D_d = ∂f_imp/∂x`, `E_d = ∂f_imp/∂u_imp` from `calculate_impulsive_discretization` (`openscvx/discretization/linearize_discretize.py:395-441`).

**Rejected (paper):** nodal pairs `(x^-, x^+)` and separate constraint `G̃(x^+, x^-, Phi) = 0`.

Augmented `y` for integral cross-complementarity + path CTCS: `n_y = 4 n_c + N_g` (paper eq. 12).

#### 3.2 Actuation control — FOH

| Control | `parameterization` | DOF per node |
|---------|-------------------|--------------|
| `tau` (hip, knee) | `"foh"` | `n_a = 2` |

Merged into discretizer FOH mask via `Control.parameterization` (`discretization/base.py:80-84`).

#### 3.3 Contact / KKT controls — dFOH (ZOH + FOH)

For **each** scalar component of `phi^c_i` and `gamma_i` (and impulsive `Phi`, `Gamma` if nodal), effective control on interval `k` (normalized time `s ∈ [0,1]`):

```text
u_eff(s) = u_zoh[k] + (1 - s) * u_foh[k] + s * u_foh[k+1]
```

- **`u_zoh[k]`:** piecewise-constant **jump** component between intervals (enables `phi` to go 0 → nonzero at a knot).
- **`u_foh[k]`, `u_foh[k+1]`:** nodal values for linear variation **during** contact (force redistribution for jumps).

At `s=0`: `u_eff = u_zoh[k] + u_foh[k]`. At `s=1`: `u_eff = u_zoh[k] + u_foh[k+1]`.

**Decision vector layout (per contact `i`, per scalar channel)**

| Block | Length | Role |
|-------|--------|------|
| `phi_zoh`, `gamma_zoh`, … | `N` intervals × dim | ZOH coefficients |
| `phi_foh`, `gamma_foh`, … | `(N) nodes × dim` | FOH nodal coefficients |

#### 3.4 `DfohControlLayout` tool (new)

**Location:** `openscvx/discretization/dfoh_layout.py` (or `openscvx/symbolic/controls/dfoh.py`).

**Responsibilities**

1. Register logical groups (`phi_c_t`, `phi_c_n`, `gamma`, …) each with `shape`, `n_c`, `N`.
2. Emit paired `Control` objects (or unified slices) for ZOH and FOH blocks.
3. Provide **`eval(k, s, u_unified) -> u_eff`** and **`jacobian_wrt_coeffs`** for transcription.
4. Expose **`slice_zoh` / `slice_foh`** into `UnifiedControl` for SCP linearization.
5. Support **`n_c` arbitrary** via group replication `for i in range(n_c)`.

**Discretizer integration:** extend `DiscretizeLinearize` (or wrapper) to reconstruct interval controls via `DfohControlLayout` before integrating augmented dynamics — parallel to existing FOH/ZOH mask (`discretize_linearize.py:28-38`).

#### 3.5 Consolidated sizes (`n_c` general, example `n_c = 1`)

| Block | Count |
|-------|-------|
| `n_x` mechanical | `2 n_j = 16` |
| `n_y` auxiliary | `4 n_c + N_g` |
| `tau` FOH nodal | `n_a × N` |
| Contact dFOH per contact | `n_c × (n_d + 1)` channels × `(N_interval + N_node)` coeffs |
| Impulses (nodal, `parameterization="impulsive"`) | `n_c × (n_d + 1)` × `N` for `Phi`, plus `Gamma`; enter shooting via `E_d`, not duplicate states |
| Time dilation | `+1` (CTCS) |

---

### 4. Boundary conditions and cost

#### 4.1 Two-point BVP (partial)

| DOF | Start | End |
|-----|-------|-----|
| Base pose `q[0:6]` | **Fixed** `q_base_i` | **Fixed** `q_base_f` (different pose → locomotion / hop) |
| Leg joints `q[6:8]` | **Free** | **Free** |
| `qd` (all) | `0` (rest) | `0` (rest) unless we later relax |

Initial **guess:** resting crouch: base poses interpolated, leg joints at nominal bent config (e.g. hip/knee slightly flexed), `qd = 0`, `tau` gravity-compensated (`panda_frax.py:69-77` pattern), `phi = 0`, `gamma = 0`.

#### 4.1b Floating-base attitude limits (parameters)

Frax emits `±1e6` “unlimited” sentinels on floating-base DOFs; `FraxDynamics` maps those to `±inf` (`openscvx/integrations/frax.py:67-83`). **Override** orientation components of `q` after constructing `dyn` with practical attitude caps; leave yaw unbounded.

Expose in the example (or a small `MonopedCitoConfig` dataclass) parameters such as:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `roll_limit_deg` | `30.0` | `q[roll_idx].min/max = ±roll_limit` |
| `pitch_limit_deg` | `30.0` | `q[pitch_idx].min/max = ±pitch_limit` |
| `yaw_limit_deg` | `None` | `None` → keep `±inf` (unconstrained yaw) |

**Implementation note:** Frax floating-base `q[0:6]` is `(x, y, z, …)` + three orientation scalars (Euler-style per URDF). **Verify `roll_idx`, `pitch_idx`, `_yaw_idx` from the monoped URDF / `robot.joint_names` at Phase A** — do not hard-code until confirmed. Translation bounds optional later (e.g. floor height on `z` only via path constraint / `sd`).

Example wiring after `dyn = ox.FraxDynamics(robot)`:

```python
q, qd = dyn.states
lim = np.deg2rad(config.pitch_limit_deg)
q.min[pitch_idx], q.max[pitch_idx] = -lim, lim
lim = np.deg2rad(config.roll_limit_deg)
q.min[roll_idx], q.max[roll_idx] = -lim, lim
# yaw_idx: do not tighten (remains ±inf from adapter)
```

Also enforce attitude limits **pathwise** via `ox.ctcs` if CTCS is active, so violations between nodes are penalized (consistent with rest of CI-SCVX path treatment).

#### 4.2 Objective

**Minimum maneuver time:**

```python
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(t_max_guess),
    ...
)
```

No torque-regularization primary cost for this example (differs from paper Section VI).

---

### 5. Dynamics, contact, and impacts

#### 5.1 Continuous dynamics

- `q̇ = qd`
- `qḋ = forward_dynamics(q, qd, tau_full(qd, tau, phi, gamma), fext=None)` with `tau_full` from §2.2.
- Friction stationarity `lambda^c`, `rho^c` as in paper (4) with smooth norms; `qd` replaces `v`.

#### 5.2 Complementarity (general `n_c`)

Per contact `i`, nodal + integral cross pairs (paper §III):

| Pair | Role |
|------|------|
| `(phi^{c,n}_i, sd(p^c_i))` | Normal force vs gap |
| `(phi^{c,n}_i, lambda^c_i)` | Active-contact stationarity |
| `(gamma_i, rho^c_i)` | Friction cone multiplier |

Relaxation: Fischer–Burmeister **`FB_δ(a, b) ≤ 0`** with a **fixed** `δ` for this milestone (no embedded homotopy; paper Algorithm 1 deferred).

| Parameter | Suggested initial value | Notes |
|-----------|-------------------------|-------|
| `delta_fb` | `1e-2` … `1e-1` | Tune once SCP runs; paper uses homotopy from `1` → `1e-3` |

Expose `delta_fb` on the example config / constraint factory; no iteration schedule.

#### 5.3 Impacts (fused multiple shooting)

Physical jump at node `k` (paper eq. 6a in minimal coords):

```text
M(q_k) (qd_k - qd_CT,k) = sum_i J^c_i(q_k)^T R^c_i Phi^c_{k,i}
```

where `qd_CT,k` is the velocity at the **end** of CT integration from node `k-1` (pre-impulse, held only inside the transcription — not a decision variable). The optimizer sees a **single** `qd_k` at the knot satisfying the composed map.

Plus impact complementarity on `(Phi, sd)`, restitution on normal velocity (`epsilon^c_i`, default `0` unless tuned).

**OpenSCvx encoding**

- Impulsive controls: `Phi^c`, `Gamma` with `parameterization="impulsive"`.
- `byof['dynamics_discrete']` / `dynamics_discrete` implementing `f_imp` and Jacobians (`preprocessing.py:804-816`).
- Discretizer emits `x_prop_plus`, `D_d`, `E_d` per node; PTR subproblem couples with `A_d`, `B_d`, `C_d` (`cvxpy_ptr_solver.py:601-614`, `784-791`).
- **Engineering gap:** `openscvx/solvers/cvxpy_ptr_solver.py:47` — finish impulsive PTR path so `E_d` is not dropped in production solves.

`q` and auxiliary `y` continuous across impact; only `qd` jumps, encoded in `f_imp` inside the shooting stack.

---

### 6. Implementation phases

| Phase | Deliverable |
|-------|-------------|
| **A** | `monoped.urdf` + Frax loader + `foot_*` kinematics (`p^c`, `Jv`) for general `n_c` sites |
| **B** | `tau_contact(q, phi)` + BYOF `qdd`; impact map `Δqd(q, Phi)` |
| **C** | `DfohControlLayout` + discretizer hook |
| **D** | Symbolic `Problem`: CTCS auxiliary, complementarity, fixed-`δ` FB |
| **E** | Fused CT + impulsive transcription (`A,B,C` + `x_prop_plus,D_d,E_d`) + PTR impulsive support |
| **F** | `examples/frax/monoped_3d_cito.py` + visualization |

---

### 7. Verification

- [ ] `tau_contact = J^T R phi` matches finite-diff power balance on flat ground
- [ ] Optional: `fext` wrench equivalent matches `J^T` path at foot link
- [ ] dFOH: interval with `u_zoh` step + linear `u_foh` reproduces intended knot discontinuity
- [ ] Impact: inactive → active contact produces `qd` jump with zero `phi` before, nonzero `Phi` at node
- [ ] Minimum-time solution moves base from `q_base_i` to `q_base_f` with feasible contact

---

## Out of Scope

- Paper 2D monoped / HalfCheetah / MJPC benchmarks
- `n_c > 1` for the **example** (formulation supports it; demo uses one foot)
- General terrain SDF
- CI-SCVX-CUSTOM / QOCO canonicalization
- Torque-minimization primary objective
- **δ-homotopy** (paper Algorithm 1) — fixed `δ` only for now
- Duplicate pre/post nodal states as in paper (17e)

---

## Open Questions

1. **Euler index map** — Which `q[3:6]` components are roll / pitch / yaw for the monoped URDF (needed to wire §4.1b defaults).

2. **Restitution** `epsilon^c` for foot — default `0` (paper) vs small elastic bounce for numerical stability.

---

## Decision Log

| Date | Decision | Rejected alternative |
|------|----------|---------------------|
| 2026-05-29 | Minimal coords via Frax + `J^T R phi` contact forces | Maximal coords + joint reactions |
| 2026-05-29 | Flat ground analytic `sd` | General SDF |
| 2026-05-29 | Reuse OpenSCvx CTCS + time dilation | Stand-alone JAX CI-SCVX stack |
| 2026-05-30 | **3D 3-link monoped** URDF (float + hip + knee), `n_c = 1` demo | Paper 2D monoped / HalfCheetah |
| 2026-05-30 | **Impacts required** in v1 | Continuous contact only |
| 2026-05-30 | Fix **base pose** only at start/end; free leg joints | Full `(q, qd)` BVP |
| 2026-05-30 | **Minimum time** cost | Squared torque cost |
| 2026-05-30 | **dFOH** (ZOH+FOH) for `phi`, `gamma`; **FOH** for `tau` | ZOH-only / paper C¹ polynomial coeffs |
| 2026-05-30 | Contact via **`tau_full = tau_act + J^T f_world`**, `fext=None` | Raw `fext` point forces without moment arms |
| 2026-05-30 | Write `n_c`-generic formulas; instantiate `n_c = 1` | Hard-coded single-contact algebra only |
| 2026-05-30 | **Single state per knot**; impulse in **fused shooting** (`A,B,C` + `E_d u_imp`) | Paper `(x^-, x^+)` pairs and separate impact equality |
| 2026-05-30 | **Fixed `δ`** for Fischer–Burmeister | Embedded δ-homotopy (Algorithm 1) |
| 2026-05-30 | **Parameterized** base roll/pitch ±30°, yaw free | Leave all base DOFs at Frax ±inf sentinel |
