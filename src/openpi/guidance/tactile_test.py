"""Regression checks for action-gradient math and causal tactile input contracts."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.guidance.compatibility import Guide
from openpi.guidance.tactile import GuidedPolicy
from openpi.guidance.tactile import guidance_step
from openpi.guidance.tactile import score
from openpi.guidance.tactile_history import TactileHistory


@pytest.fixture
def weights():
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.manual_seed(9)
    model = Guide()
    params = {k: jnp.asarray(v.detach().numpy()) for k, v in model.state_dict().items()}
    params.update(f6_mean=jnp.zeros(30), f6_std=jnp.ones(30), state_mean=jnp.zeros(29), state_std=jnp.ones(29))
    return model, params


def test_score_and_action_gradient_match_torch(weights):
    model, params = weights
    rng = np.random.default_rng(7)
    f6 = rng.normal(size=(2, 16, 5, 6)).astype("f4")
    state = rng.normal(size=(2, 29)).astype("f4")
    action = rng.normal(size=(2, 30, 28)).astype("f4")
    context = {"f6": jnp.asarray(f6), "state": jnp.asarray(state)}
    obs = torch.from_numpy(np.concatenate([f6.reshape(2, 480), state], axis=-1))
    a = torch.tensor(action, requires_grad=True)
    expected = model.score(obs, a, torch.full((2,), 0.2))
    expected_grad = torch.autograd.grad(expected.sum(), a)[0]
    native = jnp.pad(jnp.asarray(action), ((0, 0), (0, 0), (0, 4)))
    actual = score(params, native, 0.2, context)
    gradient = jax.grad(lambda x: score(params, x, 0.2, context).sum())(native)
    np.testing.assert_allclose(actual, expected.detach(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gradient[..., :28], expected_grad, rtol=3e-4, atol=2e-6)
    np.testing.assert_array_equal(gradient[..., 28:], 0)


def test_guidance_direction_bound_and_disabled_times(weights):
    _, params = weights
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.normal(size=(2, 30, 32)).astype("f4"))
    context = {"f6": jnp.asarray(rng.normal(size=(2, 16, 5, 6)).astype("f4")), "state": jnp.zeros((2, 29))}

    def step(scale, t):
        return guidance_step(params, x, t, x, context, scale=scale, max_time=0.4, max_step_rms=0.0001, num_steps=10)

    np.testing.assert_array_equal(step(0.0, 0.2), x)
    np.testing.assert_array_equal(step(1.0, 0.8), x)
    np.testing.assert_array_equal(step(1.0, 0.0), x)
    y = step(1.0, 0.2)
    np.testing.assert_array_equal(y[..., 28:], x[..., 28:])
    rms = np.sqrt(np.mean(np.asarray(y - x)[..., :28] ** 2, axis=(1, 2)))
    assert np.max(rms) <= 0.000101
    gain = score(params, y, 0.2, context) - score(params, x, 0.2, context)
    assert np.min(gain) >= -1e-6
    assert np.max(gain) > 0
    touch_gradient = jax.grad(
        lambda f: score(params, x, 0.2, {"f6": f, "state": context["state"]}, no_touch=True).sum()
    )(context["f6"])
    np.testing.assert_array_equal(touch_gradient, 0)


def test_history_requires_fresh_control_rate_data_and_reset():
    history = TactileHistory()
    with pytest.raises(ValueError, match="16 measured"):
        history.attach({}, now=0.0)
    for i in range(16):
        history.append(np.full((5, 6), i), timestamp=i / 30.0)
    obs = history.attach({"state": np.zeros(29)}, now=0.51)
    np.testing.assert_array_equal(obs["guidance/f6_history"][:, 0, 0], np.arange(16))
    with pytest.raises(ValueError, match="stale"):
        history.attach({}, now=1.0)
    with pytest.raises(ValueError, match="increase"):
        history.append(np.zeros((5, 6)), timestamp=0.1)
    history.reset()
    with pytest.raises(ValueError, match="16 measured"):
        history.attach({}, now=0.0)
    history.append(np.zeros((5, 6)), timestamp=0.0)


@pytest.mark.parametrize(
    ("scale", "max_time", "cap", "steps"),
    [
        (float("nan"), 0.4, 0.03, 10),
        (1.0, 0.4, float("inf"), 10),
        (-1.0, 0.4, 0.03, 10),
        (1.0, 0.5, 0.03, 10),
        (1.0, 0.4, 0.03, 0),
    ],
)
def test_invalid_guidance_settings_rejected_before_sampling(scale, max_time, cap, steps):
    def sample_actions(*, guidance_fn=None):
        return guidance_fn

    base = SimpleNamespace(_is_pytorch_model=False, _model=SimpleNamespace(sample_actions=sample_actions))
    with pytest.raises(ValueError, match="Invalid guidance configuration|positive integer"):
        GuidedPolicy(base, Path("not-loaded.npz"), scale=scale, max_time=max_time, max_step_rms=cap, num_steps=steps)
