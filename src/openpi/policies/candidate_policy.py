"""Sample N candidate action chunks per request, with a flow-matching noise temperature.

Tactile steering scores several candidate chunks from the same observation and executes one
(tactile_steering/docs/HANDOFF.md section 7). This wrapper lets the policy server return all of
them from one batched call, and exposes the only knob that controls how far apart they are.

Protocol (backward compatible -- an unmodified client sees no difference at the defaults):

  request   the observation dict may carry two optional control keys. They are removed
            before the input transforms, so the model never sees them.
              "num_candidates"     int >= 1    default: server --num-candidates (1)
              "noise_temperature"  float >= 0  default: server --noise-temperature (1.0)
  response  "actions"             (H, A)     candidate 0 -- what an unmodified client executes
            "actions_candidates"  (N, H, A)  every candidate, through the same output
                                             transforms as "actions"
            "candidate_noise"     (N, H, D)  the initial noise each candidate started from
                                             (D = the model's padded action dim)
            "num_candidates", "noise_temperature"

  With N == 1 and T == 1.0 the request goes straight to Policy.infer, unchanged: same rng
  stream, same output, and none of the extra response keys.

Temperature. pi0 / pi0.5 start the flow from x_1 ~ N(0, I) (models/pi0.py sample_actions);
this passes x_1 = T * eps instead. T = 1 is the model's own sampler. T > 1 spreads the
candidates; T = 0 puts every candidate on the one deterministic flow from zero noise.
  - T != 1 samples from a different distribution than the policy's. Best-of-N selection is
    unaffected, but importance-weighted resampling needs a pi / pi_T correction.
  - Temperature scales the candidate cloud; it does not create new modes.

Cost. Every candidate shares the observation, so by default (JAX pi0 / pi0.5) the VLM prefix runs
ONCE and its KV cache is tiled over the N candidates; only the action-expert denoising runs at
batch N (_shared_prefix_sample_actions, same candidates as batching the prefix).
`CandidatePolicy(share_prefix=False)` restores the old path, where the prefix is recomputed N times
in parallel. That old path, measured with sharpa_egg egg_70ep_b64/15120 on an A100-80GB, 10 steps
(tactile_steering/dp2/out/dp2b_candidates.md): N=1 92 ms, N=4 193 ms, N=8 310 ms, N=16 558 ms.
The JIT compiles once per new N -- tens of seconds -- so warm up with the N you will use
before the robot is enabled.

This reads Policy's private fields on purpose: keeping the code in its own module avoids
touching an upstream file (this repo is a fork that gets rebased). candidate_policy_test.py
exercises every field it reads, so an upstream rename fails the tests instead of the robot.
"""

# Reading Policy's private fields is deliberate -- see the last paragraph of the docstring.
# ruff: noqa: SLF001

from __future__ import annotations

import math
import time
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy as _policy

NUM_CANDIDATES_KEY = "num_candidates"
TEMPERATURE_KEY = "noise_temperature"
# Stochastic sampler (models/pi0.py flow_ddim_eta_step): 0 = the deterministic Euler sampler, 1 = ancestral.
# It samples the SAME distribution as eta=0 when the model is exact -- it changes how candidates are drawn,
# not how far apart they can be. JAX models only.
SDE_ETA_KEY = "sde_eta"


def _validate(num_candidates: int, noise_temperature: float, sde_eta: float = 0.0) -> None:
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if not math.isfinite(noise_temperature) or noise_temperature < 0:
        raise ValueError(f"noise_temperature must be finite and >= 0, got {noise_temperature}")
    if not math.isfinite(sde_eta) or not 0.0 <= sde_eta <= 1.0:
        raise ValueError(f"sde_eta must be in [0, 1], got {sde_eta}")


class CandidatePolicy(_base_policy.BasePolicy):
    """Wraps a `Policy` so one request can return N candidates at a chosen noise temperature."""

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        num_candidates: int = 1,
        noise_temperature: float = 1.0,
        sde_eta: float = 0.0,
        share_prefix: bool = True,
    ):
        _validate(num_candidates, noise_temperature, sde_eta)
        self._policy = policy
        self._default_n = int(num_candidates)
        self._default_t = float(noise_temperature)
        self._default_eta = float(sde_eta)
        self._share_prefix = bool(share_prefix)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        n = int(obs.pop(NUM_CANDIDATES_KEY, self._default_n))
        t = float(obs.pop(TEMPERATURE_KEY, self._default_t))
        eta = float(obs.pop(SDE_ETA_KEY, self._default_eta))
        _validate(n, t, eta)
        if n == 1 and t == 1.0 and eta == 0.0:
            return self._policy.infer(obs)
        return _sample_candidates(self._policy, obs, n, t, eta, share_prefix=self._share_prefix)


def _sample_candidates(
    policy: _policy.Policy, obs: dict, n: int, t: float, eta: float = 0.0, *, share_prefix: bool = True
) -> dict:
    single = policy._input_transform(jax.tree.map(lambda x: x, obs))
    shared = share_prefix and not policy._is_pytorch_model and isinstance(policy._model, _pi0.Pi0)
    if shared:
        # One observation; the prefix runs once and its KV cache is tiled (_shared_prefix_sample_actions).
        inputs = jax.tree.map(lambda x: np.asarray(x)[np.newaxis, ...], single)
    else:
        # Same batching as Policy.infer, but N copies of the one observation instead of one.
        inputs = jax.tree.map(lambda x: np.repeat(np.asarray(x)[np.newaxis, ...], n, axis=0), single)

    model = policy._model
    shape = (n, model.action_horizon, model.action_dim)
    sample_kwargs = {k: v for k, v in policy._sample_kwargs.items() if k not in ("noise", "sde_eta")}
    if eta > 0.0:
        # Only passed when on: sde_eta=None is what keeps the model on its unchanged Euler path.
        sample_kwargs["sde_eta"] = eta

    if policy._is_pytorch_model:
        device = policy._pytorch_device
        inputs = jax.tree.map(lambda x: torch.from_numpy(x).to(device), inputs)
        noise = t * torch.randn(shape, device=device, dtype=torch.float32)
        rng_or_device = device
    else:
        inputs = jax.tree.map(jnp.asarray, inputs)
        # Advance the policy's own rng exactly as Policy.infer does, so interleaving plain and
        # candidate requests never replays a key.
        policy._rng, key = jax.random.split(policy._rng)
        rng_or_device, noise_key = jax.random.split(key)
        noise = t * jax.random.normal(noise_key, shape, dtype=jnp.float32)

    observation = _model.Observation.from_dict(inputs)

    start = time.monotonic()
    if shared:
        actions = _shared_prefix_fn(policy)(rng_or_device, observation, noise, **sample_kwargs)
    else:
        actions = policy._sample_actions(rng_or_device, observation, noise=noise, **sample_kwargs)
    model_ms = (time.monotonic() - start) * 1000

    to_np = (lambda x: x.detach().float().cpu().numpy()) if policy._is_pytorch_model else np.asarray
    actions, state, noise = to_np(actions), to_np(inputs["state"]), to_np(noise)
    if shared:
        state = np.repeat(state, n, axis=0)

    # Output transforms are written for one sample (Policy.infer strips the batch dim first),
    # so apply them per candidate rather than trusting each one to broadcast over N.
    per = [policy._output_transform({"state": state[i], "actions": actions[i]}) for i in range(n)]
    candidates = np.stack([p["actions"] for p in per])

    out = dict(per[0])
    out.update(
        {
            "actions": candidates[0],
            "actions_candidates": candidates,
            "candidate_noise": noise.astype(np.float32),
            NUM_CANDIDATES_KEY: n,
            TEMPERATURE_KEY: t,
            SDE_ETA_KEY: eta,
            "candidate_sampler": "shared_prefix" if shared else "batched",
            "policy_timing": {"infer_ms": model_ms},
        }
    )
    return out


def _shared_prefix_fn(policy: _policy.Policy):
    """`_shared_prefix_sample_actions` jitted against the policy's model, built once per policy.

    The same freeze-the-module trick as nnx_utils.module_jit (which only accepts bound methods).
    Recompiles once per new N, exactly like the batched path.
    """
    fn = getattr(policy, "_candidate_shared_prefix_fn", None)
    if fn is None:
        graphdef, state = nnx.split(policy._model)

        def run(state, rng, observation, noise, kwargs):
            return _shared_prefix_sample_actions(nnx.merge(graphdef, state), rng, observation, noise, **kwargs)

        jitted = jax.jit(run)

        def fn(rng, observation, noise, **kwargs):
            return jitted(state, rng, observation, noise, kwargs)

        policy._candidate_shared_prefix_fn = fn
    return fn


def _shared_prefix_sample_actions(
    model: _pi0.Pi0,
    rng: jax.Array,
    observation: _model.Observation,
    noise: jax.Array,
    *,
    num_steps: int = 10,
    sde_eta: float | None = None,
) -> jax.Array:
    """models/pi0.py Pi0.sample_actions for N candidates of ONE observation, prefix computed once.

    `observation` has batch 1 and `noise` has batch N. The N candidates of one request share every
    prefix token (images + prompt), so the prefix KV cache is identical across them: compute it
    at batch 1 and tile it, instead of running the 2B VLM N times. Only the action-expert denoising
    loop runs at batch N. The loop body is Pi0.sample_actions' own, line for line; with the same
    noise and rng it gives the batched candidates (candidate_shared_prefix_test.py pins that).
    """
    n = noise.shape[0]
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    # The layers are nn.scan'ed, so every cache leaf is (layers, batch=1, tokens, kv_heads, head_dim).
    kv_cache = jax.tree.map(lambda x: jnp.repeat(x, n, axis=1), kv_cache)
    prefix_mask = jnp.repeat(prefix_mask, n, axis=0)
    # embed_suffix reads observation.state (pi0's state token); cheap to tile for pi0.5 too.
    observation = jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), observation)
    batch_size = n

    def step(carry):
        x_t, time = carry[0], carry[1]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

        if sde_eta is None:
            return x_t + dt * v_t, time + dt
        key, sub = jax.random.split(carry[2])
        xi = jax.random.normal(sub, x_t.shape, x_t.dtype)
        x_next = _pi0.flow_ddim_eta_step(x_t, v_t, time, jnp.maximum(time + dt, 0.0), sde_eta, xi)
        return x_next.astype(x_t.dtype), time + dt, key

    def cond(carry):
        return carry[1] >= -dt / 2

    init = (noise, 1.0) if sde_eta is None else (noise, 1.0, jax.random.fold_in(rng, 1))
    return jax.lax.while_loop(cond, step, init)[0]
