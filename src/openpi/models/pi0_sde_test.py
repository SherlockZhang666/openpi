"""Tests for the stochastic DDIM-eta flow sampler (models/pi0.py flow_ddim_eta_step, sde_eta).

The Gaussian test uses the exact velocity field of a 1-D Gaussian data distribution on this file's path
x_t = t * eps + (1 - t) * x_0, so it checks the update rule itself, independent of any trained model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.shared import nnx_utils
from openpi.training import config as _config


def _exact_velocity(x, t, mu, s0):
    """v = E[eps | x_t] - E[x_0 | x_t] for x_0 ~ N(mu, s0^2)."""
    var = (1 - t) ** 2 * s0**2 + t**2
    d = x - (1 - t) * mu
    return (t / var) * d - (mu + (1 - t) * s0**2 / var * d)


def test_eta_zero_is_exactly_the_euler_step():
    rng = np.random.default_rng(0)
    x, v, xi = (jnp.asarray(rng.standard_normal((4, 8)), jnp.float32) for _ in range(3))
    out = pi0.flow_ddim_eta_step(x, v, 0.7, 0.6, 0.0, xi)
    np.testing.assert_allclose(out, x + (0.6 - 0.7) * v, atol=1e-6)


def test_last_step_returns_the_clean_estimate_whatever_the_noise():
    rng = np.random.default_rng(1)
    x, v = (jnp.asarray(rng.standard_normal(16), jnp.float32) for _ in range(2))
    for eta in (0.0, 0.5, 1.0):
        out = pi0.flow_ddim_eta_step(x, v, 0.1, 0.0, eta, 1e3 * jnp.ones_like(x))
        np.testing.assert_allclose(out, x - 0.1 * v, atol=1e-5)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_samples_the_right_gaussian_with_enough_steps(eta):
    """Every eta must converge to the data distribution. (With 10 steps all of them under-disperse, and
    more so for larger eta: std 0.247 / 0.242 / 0.219 for eta 0 / 0.5 / 1 against 0.3.)"""
    mu, s0, n, steps = 1.5, 0.3, 50_000, 400
    key = jax.random.key(0)
    key, sub = jax.random.split(key)
    x = jax.random.normal(sub, (n,))
    t = 1.0
    for _ in range(steps):
        t_next = max(t - 1.0 / steps, 0.0)
        key, sub = jax.random.split(key)
        x = pi0.flow_ddim_eta_step(x, _exact_velocity(x, t, mu, s0), t, t_next, eta, jax.random.normal(sub, (n,)))
        t -= 1.0 / steps
    assert abs(float(x.mean()) - mu) < 0.01
    assert abs(float(x.std()) - s0) / s0 < 0.03


@pytest.fixture(scope="module")
def dummy_model():
    cfg = _config.get_config("debug_pi05").model
    model = cfg.create(jax.random.key(0))
    return cfg, model, nnx_utils.module_jit(model.sample_actions)


def test_model_eta_zero_matches_the_default_sampler(dummy_model):
    cfg, _, sample = dummy_model
    obs = cfg.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(3), (2, cfg.action_horizon, cfg.action_dim))
    a = sample(jax.random.key(4), obs, noise=noise)
    b = sample(jax.random.key(4), obs, noise=noise, sde_eta=0.0)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_model_eta_one_is_stochastic_given_the_same_initial_noise(dummy_model):
    cfg, _, sample = dummy_model
    obs = cfg.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(3), (2, cfg.action_horizon, cfg.action_dim))
    a = np.asarray(sample(jax.random.key(5), obs, noise=noise, sde_eta=1.0))
    b = np.asarray(sample(jax.random.key(6), obs, noise=noise, sde_eta=1.0))
    c = np.asarray(sample(jax.random.key(5), obs, noise=noise, sde_eta=1.0))
    assert np.isfinite(a).all()
    assert not np.allclose(a, b)  # different per-step noise
    np.testing.assert_array_equal(a, c)  # same key -> same sample
