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

Cost. The candidates are one batch of size N, so the VLM prefix is recomputed N times in
parallel. Measured with sharpa_egg egg_70ep_b64/15120 on an A100-80GB, 10 steps
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

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.models import model as _model
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
    ):
        _validate(num_candidates, noise_temperature, sde_eta)
        self._policy = policy
        self._default_n = int(num_candidates)
        self._default_t = float(noise_temperature)
        self._default_eta = float(sde_eta)

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
        return _sample_candidates(self._policy, obs, n, t, eta)


def _sample_candidates(policy: _policy.Policy, obs: dict, n: int, t: float, eta: float = 0.0) -> dict:
    inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
    # Same batching as Policy.infer, but N copies of the one observation instead of one.
    inputs = jax.tree.map(lambda x: np.repeat(np.asarray(x)[np.newaxis, ...], n, axis=0), inputs)

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
    actions = policy._sample_actions(rng_or_device, observation, noise=noise, **sample_kwargs)
    model_ms = (time.monotonic() - start) * 1000

    to_np = (lambda x: x.detach().float().cpu().numpy()) if policy._is_pytorch_model else np.asarray
    actions, state, noise = to_np(actions), to_np(inputs["state"]), to_np(noise)

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
            "policy_timing": {"infer_ms": model_ms},
        }
    )
    return out
