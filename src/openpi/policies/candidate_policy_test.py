"""Tests for CandidatePolicy (N candidates per request + noise temperature).

Runs on CPU with the randomly initialised `debug_pi05` model, so no checkpoint is needed.
Every private Policy field that candidate_policy.py reads is exercised here: an upstream rename
fails these tests instead of the robot.
"""

import jax
import numpy as np
import pytest

from openpi.policies import candidate_policy as cp
from openpi.policies import policy as _policy
from openpi.training import config as _config


@pytest.fixture(scope="module")
def setup():
    model_config = _config.get_config("debug_pi05").model
    model = model_config.create(jax.random.key(0))
    obs = jax.tree.map(lambda x: np.asarray(x[0]), model_config.fake_obs(batch_size=1).to_dict())

    def make(seed=0, **kwargs):
        # Not Policy(rng=...): Policy.__init__ does `rng or jax.random.key(0)`, and truth-testing a
        # typed PRNG key raises "len() of unsized object". Production never passes rng
        # (create_trained_policy doesn't), so set the seed afterwards instead.
        p = _policy.Policy(model, **kwargs)
        p._rng = jax.random.key(seed)  # noqa: SLF001
        return p

    return make, obs, model_config


def _copy(obs):
    return jax.tree.map(lambda x: np.array(x), obs)


def test_defaults_are_byte_identical_to_plain_infer(setup):
    make, obs, _ = setup
    plain = make(seed=3).infer(_copy(obs))
    wrapped = cp.CandidatePolicy(make(seed=3)).infer(_copy(obs))
    np.testing.assert_array_equal(wrapped["actions"], plain["actions"])
    assert "actions_candidates" not in wrapped


def test_shapes_and_candidate_zero_is_actions(setup):
    make, obs, cfg = setup
    out = cp.CandidatePolicy(make(), num_candidates=4).infer(_copy(obs))
    assert out["actions_candidates"].shape == (4, cfg.action_horizon, cfg.action_dim)
    assert out["candidate_noise"].shape == (4, cfg.action_horizon, cfg.action_dim)
    np.testing.assert_array_equal(out["actions"], out["actions_candidates"][0])
    assert out[cp.NUM_CANDIDATES_KEY] == 4
    assert out[cp.TEMPERATURE_KEY] == 1.0


def test_candidates_differ_at_default_temperature(setup):
    make, obs, _ = setup
    c = cp.CandidatePolicy(make(), num_candidates=3).infer(_copy(obs))["actions_candidates"]
    assert not np.allclose(c[0], c[1])
    assert not np.allclose(c[1], c[2])


def test_zero_temperature_collapses_all_candidates(setup):
    make, obs, _ = setup
    out = cp.CandidatePolicy(make(), num_candidates=3, noise_temperature=0.0).infer(_copy(obs))
    assert np.all(out["candidate_noise"] == 0)
    np.testing.assert_allclose(out["actions_candidates"][1], out["actions_candidates"][0], atol=1e-5)
    np.testing.assert_allclose(out["actions_candidates"][2], out["actions_candidates"][0], atol=1e-5)


def test_each_candidate_equals_single_infer_from_its_noise(setup):
    """Batching must not change what a candidate is: re-running Policy.infer with candidate i's
    starting noise reproduces candidate i."""
    make, obs, _ = setup
    out = cp.CandidatePolicy(make(), num_candidates=3, noise_temperature=1.7).infer(_copy(obs))
    single = make(seed=99)
    for i in range(3):
        again = single.infer(_copy(obs), noise=out["candidate_noise"][i])["actions"]
        np.testing.assert_allclose(again, out["actions_candidates"][i], atol=1e-4)


def test_temperature_scales_the_noise_exactly(setup):
    make, obs, _ = setup
    n1 = cp.CandidatePolicy(make(seed=5), num_candidates=2, noise_temperature=1.0 + 1e-9)
    n2 = cp.CandidatePolicy(make(seed=5), num_candidates=2, noise_temperature=2.0)
    a = n1.infer(_copy(obs))["candidate_noise"]
    b = n2.infer(_copy(obs))["candidate_noise"]
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-6)


class _RejectControlKeys:
    def __call__(self, data):
        assert cp.NUM_CANDIDATES_KEY not in data, "control key leaked into the input transforms"
        assert cp.TEMPERATURE_KEY not in data, "control key leaked into the input transforms"
        return data


def test_request_keys_override_defaults_and_never_reach_transforms(setup):
    make, obs, _ = setup
    policy = cp.CandidatePolicy(make(transforms=[_RejectControlKeys()]))
    req = _copy(obs)
    req[cp.NUM_CANDIDATES_KEY] = 2
    req[cp.TEMPERATURE_KEY] = 0.5
    out = policy.infer(req)
    assert out["actions_candidates"].shape[0] == 2
    assert out[cp.TEMPERATURE_KEY] == 0.5
    # an explicit 1 / 1.0 in the request also takes the plain path
    req = _copy(obs)
    req[cp.NUM_CANDIDATES_KEY] = 1
    req[cp.TEMPERATURE_KEY] = 1.0
    assert "actions_candidates" not in policy.infer(req)


def test_sde_eta_changes_candidates_and_is_reported(setup):
    make, obs, _ = setup
    ode = cp.CandidatePolicy(make(seed=7), num_candidates=3).infer(_copy(obs))
    sde = cp.CandidatePolicy(make(seed=7), num_candidates=3, sde_eta=1.0).infer(_copy(obs))
    assert ode[cp.SDE_ETA_KEY] == 0.0
    assert sde[cp.SDE_ETA_KEY] == 1.0
    # same seed -> same initial noise; only the per-step noise differs
    np.testing.assert_array_equal(ode["candidate_noise"], sde["candidate_noise"])
    assert not np.allclose(ode["actions_candidates"], sde["actions_candidates"])


def test_sde_eta_alone_takes_the_candidate_path(setup):
    make, obs, _ = setup
    req = _copy(obs)
    req[cp.SDE_ETA_KEY] = 0.5
    out = cp.CandidatePolicy(make()).infer(req)
    assert out["actions_candidates"].shape[0] == 1
    assert out[cp.SDE_ETA_KEY] == 0.5


@pytest.mark.parametrize("eta", [-0.1, 1.5, float("nan")])
def test_invalid_sde_eta_raises(setup, eta):
    make, _, _ = setup
    with pytest.raises(ValueError, match="sde_eta"):
        cp.CandidatePolicy(make(), sde_eta=eta)


@pytest.mark.parametrize(("n", "t"), [(0, 1.0), (2, -0.1), (2, float("nan")), (2, float("inf"))])
def test_invalid_arguments_raise(setup, n, t):
    make, obs, _ = setup
    with pytest.raises(ValueError, match="num_candidates|noise_temperature"):
        cp.CandidatePolicy(make(), num_candidates=n, noise_temperature=t)
    req = _copy(obs)
    req[cp.NUM_CANDIDATES_KEY] = n
    req[cp.TEMPERATURE_KEY] = t
    with pytest.raises(ValueError, match="num_candidates|noise_temperature"):
        cp.CandidatePolicy(make()).infer(req)
