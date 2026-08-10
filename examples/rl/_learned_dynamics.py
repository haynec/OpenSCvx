"""Model-based RL helper: explore a plant with PPO, fit a neural ``v̇ = f_θ(x,u)``.

True plant
    Planar point mass with *unknown* quadratic drag and a position-dependent
    bias force. Kinematics ``ṗ = v`` are treated as known; only the acceleration
    channel is learned.

Pipeline
    1. PPO explores the true plant (goal-reaching reward, no obstacle).
    2. Transitions ``(x, u, x⁺)`` are logged during training.
    3. An MLP is fit by supervised regression on finite-difference accelerations
       ``a ≈ (v⁺ - v) / dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Shared geometry with the OpenSCvx example
X0 = np.array([-2.0, -2.0, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.0, 2.0], dtype=np.float32)
A_MAX = 3.0
DT = 0.1
HORIZON = 40

HIDDEN = 64
STATE_DIM = 4
ACT_DIM = 2
# Network input: [px, py, vx, vy, ux, uy]
IN_DIM = STATE_DIM + ACT_DIM
OUT_DIM = 2  # acceleration


# ── True plant (unknown to OpenSCvx) ─────────────────────────────────────────


def true_acceleration(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth ``v̇`` with quadratic drag + sinusoidal bias."""
    pos, vel = x[:2], x[2:]
    u = jnp.clip(u, -A_MAX, A_MAX)
    speed = jnp.linalg.norm(vel) + 1e-8
    drag = 0.35 * speed * vel
    bias = 0.45 * jnp.array([jnp.sin(pos[0]), jnp.cos(pos[1])])
    return u - drag + bias


def true_f(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time true dynamics ``ẋ = f(x,u)``."""
    vel = x[2:]
    return jnp.concatenate([vel, true_acceleration(x, u)])


def true_step(x: jnp.ndarray, u: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """Semi-implicit Euler step on the true plant."""
    u = jnp.clip(u, -A_MAX, A_MAX)
    vel = x[2:]
    acc = true_acceleration(x, u)
    vel_n = vel + dt * acc
    pos_n = x[:2] + dt * vel_n
    return jnp.concatenate([pos_n, vel_n])


# ── RL env on the true plant ─────────────────────────────────────────────────


@dataclass(frozen=True)
class EnvConfig:
    dt: float = DT
    a_max: float = A_MAX
    horizon: int = HORIZON
    w_goal: float = 5.0
    w_ctrl: float = 0.05
    w_vel: float = 0.25


class EnvState(NamedTuple):
    x: jnp.ndarray
    t: jnp.ndarray
    goal: jnp.ndarray


def _obs(state: EnvState) -> jnp.ndarray:
    return jnp.concatenate([state.x, state.goal])


def reset(key: jax.Array, cfg: EnvConfig) -> tuple[EnvState, jnp.ndarray]:
    key_pos, key_vel = jax.random.split(key)
    x0 = jnp.asarray(X0) + jnp.concatenate(
        [0.15 * jax.random.normal(key_pos, (2,)), 0.05 * jax.random.normal(key_vel, (2,))]
    )
    state = EnvState(x=x0, t=jnp.array(0, dtype=jnp.int32), goal=jnp.asarray(GOAL))
    return state, _obs(state)


def step(
    state: EnvState, action: jnp.ndarray, cfg: EnvConfig
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    action = jnp.clip(action, -cfg.a_max, cfg.a_max)
    x_next = true_step(state.x, action, cfg.dt)
    t_next = state.t + 1
    dist_sq = jnp.sum((x_next[:2] - state.goal) ** 2)
    reward = 0.1 * (
        -cfg.w_goal * dist_sq - cfg.w_ctrl * jnp.sum(action**2) - cfg.w_vel * jnp.sum(x_next[2:] ** 2)
    )
    done = t_next >= cfg.horizon
    next_state = EnvState(x=x_next, t=t_next, goal=state.goal)
    return next_state, _obs(next_state), reward, done


# ── Actor-critic (exploration policy) ────────────────────────────────────────


def _init_layer(key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.1):
    k_w, k_b = jax.random.split(key)
    return scale * jax.random.normal(k_w, (in_dim, out_dim)), jnp.zeros((out_dim,))


def init_policy(key: jax.Array) -> dict:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "w1": _init_layer(k1, 6, HIDDEN),
        "w2": _init_layer(k2, HIDDEN, HIDDEN),
        "mu": _init_layer(k3, HIDDEN, ACT_DIM, scale=0.01),
        "log_std": jnp.full((ACT_DIM,), -0.5),
        "v1": _init_layer(k4, 6, HIDDEN),
        "v2": _init_layer(k5, HIDDEN, 1, scale=0.01),
    }


def actor_mean_std(params, obs):
    w1, b1 = params["w1"]
    w2, b2 = params["w2"]
    wm, bm = params["mu"]
    h = jax.nn.tanh(jax.nn.tanh(obs @ w1 + b1) @ w2 + b2)
    mu = jnp.tanh(h @ wm + bm) * A_MAX
    std = jnp.exp(jnp.clip(params["log_std"], -3.0, 0.5))
    return mu, std


def value_fn(params, obs):
    w1, b1 = params["v1"]
    w2, b2 = params["v2"]
    return (jax.nn.tanh(obs @ w1 + b1) @ w2 + b2).squeeze(-1)


def actor_log_prob(params, obs, action):
    mu, std = actor_mean_std(params, obs)
    var = std**2
    return -0.5 * jnp.sum(
        ((action - mu) ** 2) / var + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi), axis=-1
    )


def actor_entropy(params, obs):
    _, std = actor_mean_std(params, obs)
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std**2))


def sample_action(params, obs, key):
    mu, std = actor_mean_std(params, obs)
    return jnp.clip(mu + std * jax.random.normal(key, mu.shape), -A_MAX, A_MAX)


def deterministic_action(params, obs):
    mu, _ = actor_mean_std(params, obs)
    return jnp.clip(mu, -A_MAX, A_MAX)


class PPOConfig(NamedTuple):
    num_envs: int = 128
    num_steps: int = HORIZON
    num_updates: int = 300
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 1e-3
    max_grad_norm: float = 0.5


def _gae(rewards, values, dones, gamma, gae_lambda, last_value):
    def body(carry, inputs):
        gae, next_value = carry
        reward, value, done = inputs
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    (_, _), advantages = jax.lax.scan(
        body,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages


def explore_with_ppo(
    seed: int = 0,
    env_cfg: EnvConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
) -> tuple[dict, dict]:
    """Run PPO on the true plant and return ``(policy_params, transition_batch)``.

    ``transition_batch`` has keys ``x``, ``u``, ``x_next`` with leading batch axis.
    """
    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()
    key = jax.random.PRNGKey(seed)
    key, k_params, k_reset = jax.random.split(key, 3)
    params = init_policy(k_params)
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_cfg.max_grad_norm), optax.adam(ppo_cfg.lr)
    )
    opt_state = tx.init(params)

    reset_v = jax.vmap(partial(reset, cfg=env_cfg))
    step_v = jax.vmap(partial(step, cfg=env_cfg))
    env_state, obs = reset_v(jax.random.split(k_reset, ppo_cfg.num_envs))

    # Accumulate a replay buffer of transitions across updates (host-side).
    xs, us, xns = [], [], []

    def update(runner_state, _):
        params, opt_state, env_state, obs, key = runner_state

        def env_step(carry, _):
            env_state, obs, key = carry
            key, k_act = jax.random.split(key)
            keys_act = jax.random.split(k_act, ppo_cfg.num_envs)
            action = jax.vmap(sample_action, in_axes=(None, 0, 0))(params, obs, keys_act)
            log_prob = jax.vmap(actor_log_prob, in_axes=(None, 0, 0))(params, obs, action)
            value = jax.vmap(value_fn, in_axes=(None, 0))(params, obs)
            x_before = env_state.x
            next_state, next_obs, reward, done = step_v(env_state, action)
            key, k_reset_local = jax.random.split(key)
            reset_state, reset_obs = reset_v(
                jax.random.split(k_reset_local, ppo_cfg.num_envs)
            )
            env_state = jax.tree.map(
                lambda a, b: jnp.where(done.reshape(-1, *([1] * (a.ndim - 1))), b, a),
                next_state,
                reset_state,
            )
            obs_out = jnp.where(done[:, None], reset_obs, next_obs)
            return (env_state, obs_out, key), (
                obs,
                action,
                log_prob,
                reward,
                done,
                value,
                x_before,
                next_state.x,
            )

        (env_state, obs, key), traj = jax.lax.scan(
            env_step, (env_state, obs, key), None, ppo_cfg.num_steps
        )
        obs_t, actions, log_probs, rewards, dones, values, x_before, x_after = traj
        last_values = jax.vmap(value_fn, in_axes=(None, 0))(params, obs)
        advantages = _gae(
            rewards, values, dones, ppo_cfg.gamma, ppo_cfg.gae_lambda, last_values
        )
        returns = advantages + values
        batch_size = ppo_cfg.num_steps * ppo_cfg.num_envs
        b_obs = obs_t.reshape(batch_size, -1)
        b_actions = actions.reshape(batch_size, -1)
        b_log_probs = log_probs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        # Flattened transitions for the world model (returned to host via jax)
        flat_x = x_before.reshape(batch_size, -1)
        flat_u = b_actions
        flat_xn = x_after.reshape(batch_size, -1)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch_size)
            mb = batch_size // ppo_cfg.num_minibatches

            def minibatch(carry, start):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (start,), (mb,))
                obs_b, act_b = b_obs[idx], b_actions[idx]
                old_lp, adv_b, ret_b = b_log_probs[idx], b_advantages[idx], b_returns[idx]
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                def loss_fn(p):
                    lp = jax.vmap(actor_log_prob, in_axes=(None, 0, 0))(p, obs_b, act_b)
                    ratio = jnp.exp(jnp.clip(lp - old_lp, -20.0, 20.0))
                    pg = -jnp.mean(
                        jnp.minimum(
                            ratio * adv_b,
                            jnp.clip(ratio, 1 - ppo_cfg.clip_eps, 1 + ppo_cfg.clip_eps) * adv_b,
                        )
                    )
                    v = jax.vmap(value_fn, in_axes=(None, 0))(p, obs_b)
                    ent = jnp.mean(jax.vmap(actor_entropy, in_axes=(None, 0))(p, obs_b))
                    return pg + ppo_cfg.vf_coef * jnp.mean((ret_b - v) ** 2) - ppo_cfg.ent_coef * ent

                _, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                minibatch, (params, opt_state), jnp.arange(0, batch_size, mb)
            )
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(
            epoch, (params, opt_state, key), None, ppo_cfg.update_epochs
        )
        return (params, opt_state, env_state, obs, key), (flat_x, flat_u, flat_xn)

    (params, *_), trajs = jax.lax.scan(
        update, (params, opt_state, env_state, obs, key), None, ppo_cfg.num_updates
    )
    # trajs: (num_updates, batch, dim) — keep the last half for on-policy coverage
    start = ppo_cfg.num_updates // 2
    batch = {
        "x": np.asarray(trajs[0][start:]).reshape(-1, STATE_DIM),
        "u": np.asarray(trajs[1][start:]).reshape(-1, ACT_DIM),
        "x_next": np.asarray(trajs[2][start:]).reshape(-1, STATE_DIM),
    }
    return params, batch


# ── Neural acceleration model ────────────────────────────────────────────────


def init_dynamics_net(key: jax.Array) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": _init_layer(k1, IN_DIM, HIDDEN),
        "w2": _init_layer(k2, HIDDEN, HIDDEN),
        "w3": _init_layer(k3, HIDDEN, OUT_DIM, scale=0.01),
    }


def dynamics_net_apply(params, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Predict acceleration ``a = f_θ(x, u)``."""
    inp = jnp.concatenate([x, u], axis=-1)
    w1, b1 = params["w1"]
    w2, b2 = params["w2"]
    w3, b3 = params["w3"]
    h = jax.nn.tanh(inp @ w1 + b1)
    h = jax.nn.tanh(h @ w2 + b2)
    return h @ w3 + b3


def fit_dynamics(
    batch: dict,
    seed: int = 0,
    num_epochs: int = 80,
    batch_size: int = 1024,
    lr: float = 3e-3,
    dt: float = DT,
) -> dict:
    """Supervised fit of ``a_θ(x,u) ≈ (v⁺ - v) / dt`` on RL transitions."""
    x = jnp.asarray(batch["x"], dtype=jnp.float32)
    u = jnp.asarray(batch["u"], dtype=jnp.float32)
    x_next = jnp.asarray(batch["x_next"], dtype=jnp.float32)
    target_a = (x_next[:, 2:] - x[:, 2:]) / dt

    key = jax.random.PRNGKey(seed)
    params = init_dynamics_net(key)
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    n = int(x.shape[0])

    @jax.jit
    def epoch_step(params, opt_state, key):
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n)
        mb = batch_size

        def minibatch(carry, start):
            params, opt_state = carry
            idx = jax.lax.dynamic_slice(perm, (start,), (mb,))

            def loss_fn(p):
                pred = jax.vmap(dynamics_net_apply, in_axes=(None, 0, 0))(p, x[idx], u[idx])
                return jnp.mean((pred - target_a[idx]) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        starts = jnp.arange(0, (n // mb) * mb, mb)
        (params, opt_state), losses = jax.lax.scan(minibatch, (params, opt_state), starts)
        return params, opt_state, key, jnp.mean(losses)

    for ep in range(num_epochs):
        params, opt_state, key, loss = epoch_step(params, opt_state, key)
        if ep % 20 == 0 or ep == num_epochs - 1:
            print(f"  dynamics fit epoch {ep:3d}: mse={float(loss):.6f}")
    return params


def acceleration_mse(params, batch: dict, dt: float = DT) -> float:
    x = jnp.asarray(batch["x"])
    u = jnp.asarray(batch["u"])
    x_next = jnp.asarray(batch["x_next"])
    target = (x_next[:, 2:] - x[:, 2:]) / dt
    pred = jax.vmap(dynamics_net_apply, in_axes=(None, 0, 0))(params, x, u)
    return float(jnp.mean((pred - target) ** 2))


def save_dynamics(params, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(params)
    np.savez(path, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})


def load_dynamics(path: Path) -> dict:
    template = init_dynamics_net(jax.random.PRNGKey(0))
    data = np.load(path)
    leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(len(jax.tree.leaves(template)))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def make_byof_acceleration(params) -> callable:
    """Return a BYOF ``(x, u, node, params) -> a`` closure over frozen NN weights."""

    def accel(x, u, node, params_dict):
        del node, params_dict
        # Unified vectors include time / dilation — use the plant state/control slices
        # via closure-captured indices set by the caller, OR assume leading layout
        # [px,py,vx,vy] / [ux,uy, s]. Prefer slices passed through params_dict.
        pos_vel = x[:4]
        ctrl = u[:2]
        return dynamics_net_apply(params, pos_vel, ctrl)

    return accel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "learned_accel.npz",
    )
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--fit-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"PPO exploration on true plant ({args.updates} updates)…")
    policy, batch = explore_with_ppo(
        seed=args.seed, ppo_cfg=PPOConfig(num_updates=args.updates)
    )
    print(f"Collected {batch['x'].shape[0]} transitions")
    print("Fitting neural acceleration model…")
    dyn_params = fit_dynamics(batch, seed=args.seed, num_epochs=args.fit_epochs)
    mse = acceleration_mse(dyn_params, batch)
    # Compare against true acceleration on a held-out slice
    hold = slice(-4096, None)
    x_h, u_h = batch["x"][hold], batch["u"][hold]
    true_a = np.stack(
        [np.asarray(true_acceleration(jnp.asarray(xi), jnp.asarray(ui))) for xi, ui in zip(x_h, u_h)]
    )
    pred_a = np.asarray(
        jax.vmap(dynamics_net_apply, in_axes=(None, 0, 0))(
            dyn_params, jnp.asarray(x_h), jnp.asarray(u_h)
        )
    )
    true_mse = float(np.mean((pred_a - true_a) ** 2))
    save_dynamics(dyn_params, args.out)
    print(f"Saved {args.out}")
    print(f"FD-target MSE={mse:.6f}  |  true-acceleration MSE={true_mse:.6f}")
    del policy
