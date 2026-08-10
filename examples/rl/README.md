# Reinforcement learning + OpenSCvx

Two complementary patterns for pairing JAX RL with successive convexification:

| Example | Learned artifact | OpenSCvx role |
|---------|------------------|---------------|
| [`rl_warmstart_obstacle.py`](rl_warmstart_obstacle.py) | PPO **policy** | Refines an RL rollout under hard CTCS constraints |
| [`rl_learned_dynamics.py`](rl_learned_dynamics.py) | Neural **dynamics** `a_θ(x,u)` from RL exploration | Plans on hybrid physics + learned model via BYOF |

Requires the optional RL extra (for `optax`):

```bash
pip install "openscvx[rl]"
# or from a checkout:
pip install -e ".[rl]"
```

## Example A — policy warm-start: `rl_warmstart_obstacle.py`

1. A PureJaxRL-style **PPO** policy (`_ppo_double_integrator.py`) is trained on an
   unconstrained planar double-integrator go-to-goal task.
2. The policy is rolled out to produce `(X, U)` guesses.
3. OpenSCvx re-solves the transfer with a **hard CTCS keep-out disk** the policy never saw.

```bash
python examples/rl/_ppo_double_integrator.py --updates 400   # regenerate checkpoint
python examples/rl/rl_warmstart_obstacle.py
python examples/rl/rl_warmstart_obstacle.py --retrain
```

## Example B — learned dynamics: `rl_learned_dynamics.py`

Model-based RL style:

1. PPO explores a **true nonlinear plant** (quadratic drag + position bias) that is
   *not* written as OpenSCvx symbolic dynamics.
2. Logged transitions fit an MLP acceleration model ``a_θ(x, u) ≈ v̇``.
3. OpenSCvx optimizes with hybrid dynamics: symbolic ``ṗ = v`` + BYOF ``v̇ = a_θ``.
4. CTCS keep-out is enforced on the learned model; controls are replayed on the
   true plant to show sim-to-real residual.

```bash
python examples/rl/_learned_dynamics.py --updates 300        # explore + fit
python examples/rl/rl_learned_dynamics.py
python examples/rl/rl_learned_dynamics.py --retrain
```

## JAX RL packages worth using

OpenSCvx is already JAX-native, so staying in the JAX ecosystem avoids CPU↔GPU
sync and keeps warm-start / batching ergonomic. Practical options:

| Package | Role | Fit with OpenSCvx |
|---------|------|-------------------|
| **[PureJaxRL](https://github.com/luchris429/purejaxrl)** | End-to-end PPO reference (single-file, `jit`/`vmap` training) | Best *recipe* for custom envs that mirror your OCP; these examples follow that style |
| **[Brax](https://github.com/google/brax)** | Differentiable physics + built-in PPO/SAC | Natural when MJX/Frax dynamics are too heavy; policy rollouts → SCP guesses |
| **[gymnax](https://github.com/RobertTLange/gymnax)** | Classic-control / bsuite / MinAtari envs in JAX | Fast prototyping of RL warm-starts on pendulum, cartpole, etc. |
| **[Flax](https://github.com/google/flax) + [Optax](https://github.com/google-deepmind/optax)** | NN modules + optimizers | Default stack for actor-critic / world models (`openscvx[rl]` pulls Optax) |
| **[RLax](https://github.com/google-deepmind/rlax)** | RL building blocks (losses, schedules) | Compose custom algorithms without a full framework |
| **[JaxMARL](https://github.com/FLAIROx/JaxMARL)** / **[Mava](https://github.com/instadeepai/Mava)** | Multi-agent RL | Pair with OpenSCvx `Vmap` multi-agent OCPs |
| **[Stoix](https://github.com/instadeepai/stoix)** | Modular single-agent JAX RL | Cleaner library API than single-file PPO when you outgrow inline training |

PyTorch-first stacks (Stable-Baselines3, CleanRL-PyTorch, rllib) work via
exported rollouts, but you lose the shared JAX toolchain that makes
`solve_batched` / `jax.export` attractive.

## Other RL ↔ SCP patterns to try next

- **Safety filter / shield**: RL action as a reference; OpenSCvx solves a short-horizon
  projection onto the CTCS-feasible set (related to the `examples/realtime/` MPC loop).
- **Terminal cost / value warm-start**: use a learned value function as a Mayer
  terminal cost weight schedule, or to seed `lam_cost`.
- **Residual policy**: SCP tracks a nominal; RL learns a residual on model mismatch.
- **Imitation / DAgger**: treat converged OpenSCvx solutions as expert labels for BC/PPO.
- **Full ``ẋ = f_θ`` world model**: learn every channel (not just acceleration) and
  expose it through a small `DynamicsAdapter` subclass.
