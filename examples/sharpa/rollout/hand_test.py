"""Tests for the hand command envelope. No hardware -- `SdkHand` is never constructed.

    /path/to/python -m pytest examples/sharpa/rollout/hand_test.py -q
"""

import json

import hand
import numpy as np
import pytest

N = hand.N_HAND


def write_stats(tmp_path, q01, q99):
    """A minimal norm_stats.json in the layout the checkpoint's assets/ actually uses."""
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {"actions": {
        "q01": list(q01) + [-0.01] * 6, "q99": list(q99) + [0.01] * 6,
        "mean": [0.0] * 28, "std": [1.0] * 28}}}))
    return path


def test_limits_come_from_the_action_quantiles_plus_a_margin(tmp_path):
    q01 = np.linspace(-0.4, 0.0, N)
    q99 = np.linspace(0.1, 1.0, N)
    lo, hi = hand.limits_from_norm_stats(write_stats(tmp_path, q01, q99), margin=0.15)
    assert np.allclose(lo, q01 - 0.15)
    assert np.allclose(hi, q99 + 0.15)


def test_limits_can_only_tighten_the_firmware_clamp(tmp_path):
    """A joint that saturated at pi/2 during teleop must not widen past it once a margin
    is added -- middle_PIP and ring_PIP really do sit at 1.5705 in this checkpoint."""
    q01 = np.full(N, -3.0)
    q99 = np.full(N, np.pi / 2)
    lo, hi = hand.limits_from_norm_stats(write_stats(tmp_path, q01, q99), margin=0.5)
    assert np.allclose(hi, hand.DEFAULT_CLIP[1])
    assert np.allclose(lo, hand.DEFAULT_CLIP[0])


def test_limits_reject_a_truncated_stats_file(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"norm_stats": {"actions": {"q01": [0.0] * 10,
                                                           "q99": [1.0] * 10}}}))
    with pytest.raises(ValueError, match="action dims"):
        hand.limits_from_norm_stats(path)


def test_slew_accepts_per_joint_bounds_and_clips_each_one():
    lo = np.full(N, -0.1)
    hi = np.full(N, 0.1)
    lo[6], hi[6] = -0.28, 0.20          # index_MCP_AA, the joint a scalar clip would miss
    slew = hand.Slew(max_rate=1e6, clip=(lo, hi), dt=1.0)
    out = slew(np.full(N, 1.5))
    assert out[6] == pytest.approx(0.20)
    assert out[0] == pytest.approx(0.10)


def test_slew_rate_limits_the_first_command_away_from_the_seed():
    slew = hand.Slew(max_rate=6.0, clip=hand.DEFAULT_CLIP, dt=1.0 / 30.0)
    slew.seed(np.zeros(N))
    out = slew(np.full(N, 1.5))
    assert np.allclose(out, 6.0 / 30.0)          # one step of slew, not the full jump
    for _ in range(10):
        out = slew(np.full(N, 1.5))
    assert np.all(out > 6.0 / 30.0)


def test_slew_without_a_seed_takes_the_first_target_as_given():
    """No readback available is not a reason to slew up from zero -- that would itself be
    a commanded motion the policy never asked for."""
    slew = hand.Slew(max_rate=6.0, clip=hand.DEFAULT_CLIP, dt=1.0 / 30.0)
    out = slew(np.full(N, 0.4))
    assert np.allclose(out, 0.4)


def test_slew_refuses_non_finite_commands():
    slew = hand.Slew(max_rate=6.0, clip=hand.DEFAULT_CLIP, dt=1.0 / 30.0)
    bad = np.zeros(N)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        slew(bad)


def test_slew_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lo > hi"):
        hand.Slew(max_rate=1.0, clip=(np.full(N, 1.0), np.full(N, -1.0)), dt=0.03)


def test_null_hand_passes_the_target_through_untouched():
    sink = hand.NullHand()
    target = np.linspace(-0.4, 1.4, N)
    assert np.allclose(sink.send(target), target)
    assert sink.n == 1
