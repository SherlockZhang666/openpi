# ruff: noqa: SLF001
# This in-package adapter intentionally reuses Policy transforms and its frozen model.
"""Noise-aware F6/proprioceptive guidance for the existing JAX pi0.5 policy.
Weights remain separate from the policy checkpoint. No hardware execution here.
"""

import inspect
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_api
from openpi.shared.nnx_utils import module_jit


def tower(p, x, prefix):
    scale = p[prefix + ".net.0.weight"]
    bias = p[prefix + ".net.0.bias"]
    mean = jnp.mean(x, -1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, -1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(variance + 1e-5) * scale + bias
    x = x @ p[prefix + ".net.1.weight"].T + p[prefix + ".net.1.bias"]
    x = jax.nn.gelu(x, approximate=False)
    x = x @ p[prefix + ".net.3.weight"].T + p[prefix + ".net.3.bias"]
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def score(p, actions, time_value, context, *, no_touch=False):
    """All action coordinates are in the base policy's normalization convention."""
    f = (context["f6"].reshape(-1, 480) - jnp.tile(p["f6_mean"], 16)) / jnp.tile(p["f6_std"], 16)
    state = (context["state"] - p["state_mean"]) / p["state_std"]
    if no_touch:
        f = jnp.zeros_like(f)
    obs = tower(p, jnp.concatenate([f, state], -1), "obs")
    t = jnp.broadcast_to(time_value, (actions.shape[0],))
    phase = t[:, None] * jnp.array([1.0, 2.0, 4.0, 8.0]) * jnp.pi
    ac = jnp.concatenate([actions[..., :28].reshape(actions.shape[0], 840), jnp.sin(phase), jnp.cos(phase)], -1)
    act = tower(p, ac, "act")
    return jnp.sum(obs * act, -1)


def guidance_step(p, x, time_value, base_next, context, *, scale, max_time, max_step_rms, num_steps, no_touch=False):
    """Ascent step consistent with t=1 noise -> 0 action, dt=-1/num_steps.

    Modify v to v-scale*t/(1-t)*grad(score), then take negative dt.
    Clip the additional normalized-action RMS per step. Never change padding.
    """
    active = (time_value <= max_time) & (time_value > 0) & (scale > 0)

    def guided(_):
        grad = jax.grad(lambda a: score(p, a, time_value, context, no_touch=no_touch).sum())(x)
        weight = time_value / jnp.maximum(1.0 - time_value, 1e-3)
        delta = (scale / num_steps) * weight * grad
        delta = delta * (jnp.arange(x.shape[-1]) < 28)
        rms = jnp.sqrt(jnp.mean(delta[..., :28] ** 2, axis=(-2, -1), keepdims=True))
        delta = delta * jnp.minimum(1.0, max_step_rms / jnp.maximum(rms, 1e-12))
        return base_next + delta

    return jax.lax.cond(active, guided, lambda _: base_next, operand=None)


class GuidedSampler(nnx.Module):
    def __init__(self, base, archive):
        self.base = base
        with np.load(archive, allow_pickle=False) as z:
            self.no_touch = bool(z["no_touch"])
            self.guide_weights = nnx.Variable({k: jnp.asarray(z[k]) for k in z.files if k != "no_touch"})

    def sample_actions(
        self, rng, observation, *, guide_context, scale=1.0, max_time=0.4, max_step_rms=0.03, num_steps=10, noise=None
    ):
        def update(x, t, v, base_next):
            return guidance_step(
                self.guide_weights.value,
                x,
                t,
                base_next,
                guide_context,
                scale=scale,
                max_time=max_time,
                max_step_rms=max_step_rms,
                num_steps=num_steps,
                no_touch=self.no_touch,
            )

        return self.base.sample_actions(rng, observation, num_steps=num_steps, noise=noise, guidance_fn=update)


class GuidedPolicy:
    """openpi-client compatible wrapper; touch enters only the external guide.

    Required extra key: guidance/f6_history [16,5,6], bias-subtracted N/Nm in
    the rig cache's finger/frame order. raw state is the standard 29D state.
    No auto zero padding of missing tactile history, no automatic actuation.
    """

    def __init__(self, base, archive, *, scale=1.0, max_time=0.4, max_step_rms=0.03, num_steps=10, seed=0):
        if base._is_pytorch_model:
            raise ValueError("This wrapper requires the existing JAX checkpoint")
        if "guidance_fn" not in inspect.signature(base._model.sample_actions).parameters:
            raise RuntimeError("This OpenPI model does not expose the optional guidance_fn sampler hook")
        if (
            not np.isfinite([scale, max_time, max_step_rms]).all()
            or scale < 0
            or not 0 < max_time <= 0.4
            or max_step_rms <= 0
        ):
            raise ValueError("Invalid guidance configuration")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if base._model.action_horizon != 30 or base._model.action_dim != 32:
            raise ValueError("Egg guide requires a policy with action_horizon=30 and native action_dim=32")
        self.base = base
        self.sampler = GuidedSampler(base._model, archive)
        self.sample = module_jit(self.sampler.sample_actions)
        self.scale = scale
        self.max_time = max_time
        self.max_step_rms = max_step_rms
        self.num_steps = num_steps
        self.rng = jax.random.key(seed)
        self.archive = str(archive)
        # Check the actual deployed inverse action transform against guide training.
        params = self.sampler.guide_weights.value
        for value in (-1.0, 0.0, 1.0):
            probe = {"state": np.zeros(32, np.float32), "actions": np.full((30, 32), value, np.float32)}
            physical = self.base._output_transform(probe)["actions"]
            expected = np.asarray(params["policy_offset"]) + value * np.asarray(params["policy_scale"])
            np.testing.assert_allclose(
                physical,
                np.broadcast_to(expected, (30, 28)),
                atol=3e-6,
                rtol=3e-6,
                err_msg="Guide and base-policy action normalization differ",
            )

    @property
    def metadata(self):
        return {
            **self.base.metadata,
            "tactile_guidance": True,
            "guide_archive": self.archive,
            "guidance_scale": self.scale,
            "guidance_max_time": self.max_time,
            "guidance_max_step_rms": self.max_step_rms,
            "guidance_scope": "egg F6/proprioception compatibility adaptation; no physical benefit established",
        }

    def infer(self, obs, *, noise=None):
        if self.scale == 0:
            raw = {k: v for k, v in obs.items() if not k.startswith("guidance/")}
            outputs = self.base.infer(raw, noise=noise)
            outputs["guidance"] = {"scale": 0.0, "enabled": False, "path": "original base sampler"}
            return outputs
        return self._infer_guided(obs, noise=noise)

    def _infer_guided(self, obs, *, noise=None, scale=None):
        # Internal dynamic-scale entry provides a SAME-COMPILED-GRAPH zero reference
        # for evaluation. Public scale=0 uses the exact original inference path.
        scale = self.scale if scale is None else scale
        f6 = np.asarray(obs["guidance/f6_history"], np.float32)
        state = np.asarray(obs["state"], np.float32)
        if f6.shape != (16, 5, 6) or state.shape != (29,):
            raise ValueError("Expected causal tactile history [16,5,6] and raw state [29]")
        if not np.isfinite(f6).all() or not np.isfinite(state).all():
            raise ValueError("Non-finite guidance inputs")
        raw = {k: v for k, v in obs.items() if not k.startswith("guidance/")}
        inputs = self.base._input_transform(jax.tree.map(lambda x: x, raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        context = {"f6": jnp.asarray(f6)[None], "state": jnp.asarray(state)[None]}
        self.rng, key = jax.random.split(self.rng)
        kwargs = {}
        if noise is not None:
            noise = np.asarray(noise, np.float32)
            if noise.shape == (30, 32):
                noise = noise[None]
            if noise.shape != (1, 30, 32) or not np.isfinite(noise).all():
                raise ValueError("Expected finite noise [30,32] or [1,30,32]")
            kwargs["noise"] = jnp.asarray(noise)
        start = time.monotonic()
        actions = self.sample(
            key,
            model_api.Observation.from_dict(inputs),
            guide_context=context,
            scale=scale,
            max_time=self.max_time,
            max_step_rms=self.max_step_rms,
            num_steps=self.num_steps,
            **kwargs,
        )
        actions.block_until_ready()
        if not np.isfinite(np.asarray(actions)).all():
            raise RuntimeError("Non-finite guided sample; do not execute")
        outputs = {"state": np.asarray(inputs["state"][0]), "actions": np.asarray(actions[0])}
        outputs = self.base._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": 1000 * (time.monotonic() - start)}
        outputs["guidance"] = {
            "scale": float(scale),
            "late_time_threshold": self.max_time,
            "normalized_step_rms_cap": self.max_step_rms,
        }
        return outputs
