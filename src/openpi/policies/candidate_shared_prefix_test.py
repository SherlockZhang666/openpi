"""The shared-prefix candidate sampler must be the batched one, only faster.

Runs on CPU with the randomly initialised `debug_pi05` model. Same seed, same noise: computing the
VLM prefix once and tiling its KV cache over N candidates has to give the candidates that N
independent copies of the prefix give -- for the Euler sampler and for the stochastic one.
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

    def make(seed):
        p = _policy.Policy(model)
        p._rng = jax.random.key(seed)  # noqa: SLF001
        return p

    return make, obs


def _copy(obs):
    return jax.tree.map(lambda x: np.array(x), obs)


@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_shared_prefix_gives_the_batched_candidates(setup, eta):
    make, obs = setup
    shared = cp._sample_candidates(make(5), _copy(obs), 3, 1.3, eta, share_prefix=True)  # noqa: SLF001
    batched = cp._sample_candidates(make(5), _copy(obs), 3, 1.3, eta, share_prefix=False)  # noqa: SLF001
    np.testing.assert_array_equal(shared["candidate_noise"], batched["candidate_noise"])
    np.testing.assert_allclose(shared["actions_candidates"], batched["actions_candidates"], atol=1e-5)
    np.testing.assert_array_equal(shared["actions"], shared["actions_candidates"][0])
    assert shared["candidate_sampler"] == "shared_prefix"
    assert batched["candidate_sampler"] == "batched"


def test_candidates_still_differ_with_a_shared_prefix(setup):
    make, obs = setup
    out = cp._sample_candidates(make(1), _copy(obs), 4, 1.0, share_prefix=True)  # noqa: SLF001
    assert not np.allclose(out["actions_candidates"][0], out["actions_candidates"][1])


def test_the_wrapper_shares_the_prefix_by_default(setup):
    make, obs = setup
    request = {**_copy(obs), "num_candidates": 2}
    assert cp.CandidatePolicy(make(0)).infer(request)["candidate_sampler"] == "shared_prefix"
    off = cp.CandidatePolicy(make(0), share_prefix=False).infer({**_copy(obs), "num_candidates": 2})
    assert off["candidate_sampler"] == "batched"
