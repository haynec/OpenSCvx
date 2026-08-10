"""Pure-JAX PPO for a planar double-integrator go-to-goal task.

Inspired by PureJaxRL (Lu et al.): the env, rollout, and PPO update are all
``jit``/``vmap``-friendly JAX. Dependencies stay within OpenSCvx's core stack
(``jax`` + ``optax``); no Flax/Brax import is required.

Observation: ``[px, py, vx, vy, gx, gy]``.
Action: continuous acceleration ``[ax, ay]`` in ``[-a_max, a_max]``.
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

# Match the OpenSCvx example geometry (start → goal; obstacle added only in SCP).
X0 = np.array([-2.0, -2.0, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.0, 2.0], dtype=np.float32)
A_MAX = 3.0
DT = 0.1
HORIZON = 40
GOAL_RADIUS = 0.15

HIDDEN = 64
OBS_DIM = 6
ACT_DIM = 2


@dataclass(frozen=True)
class EnvConfig:
    dt: float = DT
    a_max: float = A_MAX
    horizon: int = HORIZON
    goal_radius: float = GOAL_RADIUS
    # Soft LQR-style shaping only — hard obstacle avoidance is left to OpenSCvx.
    w_goal: float = 5.0
    w_ctrl: float = 0.05
    w_vel: float = 0.25
    success_bonus: float = 0.0


class EnvState(NamedTuple):
    x: jnp.ndarray  # (4,)
    t: jnp.ndarray  # scalar int
    goal: jnp.ndarray  # (2,)


def _obs(state: EnvState) -> jnp.ndarray:
    return jnp.concatenate([state.x, state.goal])


def reset(key: jax.Array, cfg: EnvConfig) -> tuple[EnvState, jnp.ndarray]:
    key_pos, key_vel = jax.random.split(key)
    pos_noise = 0.1 * jax.random.normal(key_pos, (2,))
    vel_noise = 0.05 * jax.random.normal(key_vel, (2,))
    x0 = jnp.asarray(X0) + jnp.concatenate([pos_noise, vel_noise])
    state = EnvState(x=x0, t=jnp.array(0, dtype=jnp.int32), goal=jnp.asarray(GOAL))
    return state, _obs(state)


def step(
    state: EnvState, action: jnp.ndarray, cfg: EnvConfig
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    action = jnp.clip(action, -cfg.a_max, cfg.a_max)
    pos = state.x[:2]
    vel = state.x[2:]
    vel_next = vel + cfg.dt * action
    pos_next = pos + cfg.dt * vel_next
    x_next = jnp.concatenate([pos_next, vel_next])
    t_next = state.t + 1

    dist_sq = jnp.sum((pos_next - state.goal) ** 2)
    reward = (
        -cfg.w_goal * dist_sq
        - cfg.w_ctrl * jnp.sum(action**2)
        - cfg.w_vel * jnp.sum(vel_next**2)
    )
    # Keep returns O(1)–O(10²) so Adam + GAE stay well-conditioned.
    reward = 0.1 * reward
    # Fixed-horizon episodes so the policy learns to arrive *and* brake.
    done = t_next >= cfg.horizon

    next_state = EnvState(x=x_next, t=t_next, goal=state.goal)
    return next_state, _obs(next_state), reward, done


def _init_layer(key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.1):
    k_w, k_b = jax.random.split(key)
    w = scale * jax.random.normal(k_w, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return w, b


def init_params(key: jax.Array) -> dict:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "w1": _init_layer(k1, OBS_DIM, HIDDEN),
        "w2": _init_layer(k2, HIDDEN, HIDDEN),
        "mu": _init_layer(k3, HIDDEN, ACT_DIM, scale=0.01),
        "log_std": jnp.full((ACT_DIM,), -0.5),
        "v1": _init_layer(k4, OBS_DIM, HIDDEN),
        "v2": _init_layer(k5, HIDDEN, 1, scale=0.01),
    }


def actor_mean_std(params, obs):
    w1, b1 = params["w1"]
    w2, b2 = params["w2"]
    wm, bm = params["mu"]
    h = jax.nn.tanh(obs @ w1 + b1)
    h = jax.nn.tanh(h @ w2 + b2)
    # tanh keeps the mean inside the actuator box without hard clipping in the density.
    mu = jnp.tanh(h @ wm + bm) * A_MAX
    std = jnp.exp(jnp.clip(params["log_std"], -3.0, 0.5))
    return mu, std


def value_fn(params, obs):
    w1, b1 = params["v1"]
    w2, b2 = params["v2"]
    h = jax.nn.tanh(obs @ w1 + b1)
    return (h @ w2 + b2).squeeze(-1)


def actor_log_prob(params, obs, action):
    mu, std = actor_mean_std(params, obs)
    var = std**2
    return -0.5 * jnp.sum(
        ((action - mu) ** 2) / var + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def actor_entropy(params, obs):
    _, std = actor_mean_std(params, obs)
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std**2))


def sample_action(params, obs, key):
    mu, std = actor_mean_std(params, obs)
    action = mu + std * jax.random.normal(key, mu.shape)
    return jnp.clip(action, -A_MAX, A_MAX)


def deterministic_action(params, obs):
    mu, _ = actor_mean_std(params, obs)
    return jnp.clip(mu, -A_MAX, A_MAX)


class PPOConfig(NamedTuple):
    num_envs: int = 128
    num_steps: int = HORIZON
    num_updates: int = 400
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


def train_ppo(
    seed: int = 0,
    env_cfg: EnvConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
) -> dict:
    """Train a PPO policy; returns actor-critic parameter pytree."""
    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()
    key = jax.random.PRNGKey(seed)
    key, k_params, k_reset = jax.random.split(key, 3)
    params = init_params(k_params)
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_cfg.max_grad_norm),
        optax.adam(ppo_cfg.lr),
    )
    opt_state = tx.init(params)

    reset_v = jax.vmap(partial(reset, cfg=env_cfg))
    step_v = jax.vmap(partial(step, cfg=env_cfg))

    keys = jax.random.split(k_reset, ppo_cfg.num_envs)
    env_state, obs = reset_v(keys)

    def update(runner_state, _):
        params, opt_state, env_state, obs, key = runner_state

        def env_step(carry, _):
            env_state, obs, key = carry
            key, k_act = jax.random.split(key)
            keys_act = jax.random.split(k_act, ppo_cfg.num_envs)
            action = jax.vmap(sample_action, in_axes=(None, 0, 0))(params, obs, keys_act)
            log_prob = jax.vmap(actor_log_prob, in_axes=(None, 0, 0))(params, obs, action)
            value = jax.vmap(value_fn, in_axes=(None, 0))(params, obs)
            next_state, next_obs, reward, done = step_v(env_state, action)
            key, k_reset_local = jax.random.split(key)
            reset_keys = jax.random.split(k_reset_local, ppo_cfg.num_envs)
            reset_state, reset_obs = reset_v(reset_keys)
            env_state = jax.tree.map(
                lambda a, b: jnp.where(done.reshape(-1, *([1] * (a.ndim - 1))), b, a),
                next_state,
                reset_state,
            )
            obs_out = jnp.where(done[:, None], reset_obs, next_obs)
            return (env_state, obs_out, key), (obs, action, log_prob, reward, done, value)

        (env_state, obs, key), traj = jax.lax.scan(
            env_step, (env_state, obs, key), None, ppo_cfg.num_steps
        )
        obs_t, actions, log_probs, rewards, dones, values = traj
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
                    pg1 = ratio * adv_b
                    pg2 = (
                        jnp.clip(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * adv_b
                    )
                    pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))
                    v = jax.vmap(value_fn, in_axes=(None, 0))(p, obs_b)
                    v_loss = jnp.mean((ret_b - v) ** 2)
                    ent = jnp.mean(jax.vmap(actor_entropy, in_axes=(None, 0))(p, obs_b))
                    return pg_loss + ppo_cfg.vf_coef * v_loss - ppo_cfg.ent_coef * ent

                _, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), None

            starts = jnp.arange(0, batch_size, mb)
            (params, opt_state), _ = jax.lax.scan(minibatch, (params, opt_state), starts)
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(
            epoch, (params, opt_state, key), None, ppo_cfg.update_epochs
        )
        return (params, opt_state, env_state, obs, key), None

    (params, *_), _ = jax.lax.scan(
        update, (params, opt_state, env_state, obs, key), None, ppo_cfg.num_updates
    )
    return params


def rollout_policy(
    params,
    x0: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    horizon: int = HORIZON,
    dt: float = DT,
    a_max: float = A_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic rollout → ``(X, U)`` with shapes ``(horizon, 4)`` / ``(horizon, 2)``."""
    x = jnp.asarray(x0 if x0 is not None else X0, dtype=jnp.float32)
    goal_j = jnp.asarray(goal if goal is not None else GOAL, dtype=jnp.float32)
    cfg = EnvConfig(dt=dt, a_max=a_max, horizon=horizon)

    xs = [np.asarray(x)]
    us: list[np.ndarray] = []
    for _ in range(horizon - 1):
        obs = jnp.concatenate([x, goal_j])
        u = deterministic_action(params, obs)
        us.append(np.asarray(u))
        state_next, _, _, _ = step(
            EnvState(x=x, t=jnp.asarray(0, dtype=jnp.int32), goal=goal_j), u, cfg
        )
        x = state_next.x
        xs.append(np.asarray(x))
    obs = jnp.concatenate([x, goal_j])
    us.append(np.asarray(deterministic_action(params, obs)))
    return np.stack(xs, axis=0).astype(np.float64), np.stack(us, axis=0).astype(np.float64)


def save_params(params, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(params)
    np.savez(path, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})


def load_params(path: Path, template: dict | None = None) -> dict:
    template = template or init_params(jax.random.PRNGKey(0))
    data = np.load(path)
    leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(len(jax.tree.leaves(template)))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "ppo_di_policy.npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=400)
    args = parser.parse_args()

    print(f"Training PPO ({args.updates} updates)…")
    params = train_ppo(seed=args.seed, ppo_cfg=PPOConfig(num_updates=args.updates))
    save_params(params, args.out)
    X, U = rollout_policy(params)
    dist = float(np.linalg.norm(X[-1, :2] - GOAL))
    print(f"Saved {args.out}")
    print(f"Rollout final pos {X[-1, :2]}, dist-to-goal={dist:.4f}")
    print(f"|U|_max={np.abs(U).max():.3f}")
