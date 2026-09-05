"""Tests for the OpenArm(7) + Sharpa Wave left-hand(22) transforms.

See tactile_steering/docs/sharpa-pi05-plan.md Task 2.

Note that PI05 is the model type we actually train with (Pi0Config.model_type returns
PI05 whenever pi05=True), so it gets its own coverage here -- testing only PI0 and
PI0_FAST would leave the live branch untested.
"""

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import sharpa_policy


def _example():
    rng = np.random.default_rng(0)
    return {
        "base": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "state": rng.random(sharpa_policy.SHARPA_STATE_DIM).astype(np.float32),
        "prompt": "pick up the egg",
    }


def test_inputs_maps_wrist_to_left_wrist_slot():
    """The wrist camera must land in left_wrist_0_rgb and the head camera in
    base_0_rgb. Mixing them up is silent, hence this test."""
    ex = _example()
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], ex["base"])
    np.testing.assert_array_equal(out["image"]["left_wrist_0_rgb"], ex["wrist"])
    assert np.all(out["image"]["right_wrist_0_rgb"] == 0)


def test_inputs_masks_missing_right_wrist_for_pi05():
    """PI05 is what we actually train. The masking branch keys off PI0_FAST, so PI05
    takes the same path as PI0 (right wrist masked out) -- which is what we want, but
    it has to be pinned: rewriting the branch as `!= ModelType.PI0` would silently
    flip it for PI05."""
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(_example())
    assert bool(out["image_mask"]["base_0_rgb"]) is True
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False


def test_inputs_masks_missing_right_wrist_for_pi0():
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI0)(_example())
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False


def test_inputs_keeps_right_wrist_unmasked_for_pi0_fast():
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI0_FAST)(_example())
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is True


def test_inputs_converts_float_chw_images():
    """LeRobot hands back float32 (C,H,W) during training; inference hands back uint8
    (H,W,C). Both have to come out as uint8 (H,W,C)."""
    ex = _example()
    ex["base"] = (np.arange(3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224) % 255) / 255.0
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)
    assert out["image"]["base_0_rgb"].dtype == np.uint8
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)


def test_inputs_rejects_wrong_state_dim():
    """State is [left arm 7 | left hand 22] = 29. A wrong width would otherwise be
    zero-padded to 32 downstream by PadStatesAndActions and train silently wrong."""
    ex = _example()
    ex["state"] = np.zeros(23, dtype=np.float32)
    with pytest.raises(ValueError, match="29"):
        sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)


def test_inputs_passes_actions_through_only_when_present():
    """Actions exist during training and not at inference time."""
    ex = _example()
    assert "actions" not in sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)
    ex["actions"] = np.zeros((30, sharpa_policy.SHARPA_ACTION_DIM), dtype=np.float32)
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)
    assert out["actions"].shape == (30, sharpa_policy.SHARPA_ACTION_DIM)


def test_inputs_rejects_wrong_action_dim():
    """The 28-d layout is a hard contract shared with tactile_steering's palm_centric.
    A dataset carrying some other width must not be zero-padded into place silently."""
    ex = _example()
    ex["actions"] = np.zeros((30, 26), dtype=np.float32)
    with pytest.raises(ValueError, match="28"):
        sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(ex)


def test_outputs_slices_to_28():
    """The model emits 32 dims (28 real + 4 zero padding). Only the first 28 are ours."""
    rng = np.random.default_rng(1)
    padded = rng.random((30, 32)).astype(np.float32)
    out = sharpa_policy.SharpaOutputs()({"actions": padded})
    assert out["actions"].shape == (30, sharpa_policy.SHARPA_ACTION_DIM)
    np.testing.assert_allclose(out["actions"], padded[:, : sharpa_policy.SHARPA_ACTION_DIM])


def test_action_dim_fits_in_native_32():
    """28 <= 32, which is the whole reason no checkpoint action-dim surgery is needed.
    If this ever stops holding, the fix is a conversion script, not a wider slice."""
    assert sharpa_policy.SHARPA_ACTION_DIM <= 32
    assert sharpa_policy.SHARPA_STATE_DIM <= 32


def test_make_sharpa_example_roundtrips_through_inputs():
    out = sharpa_policy.SharpaInputs(model_type=_model.ModelType.PI05)(
        sharpa_policy.make_sharpa_example()
    )
    assert out["state"].shape == (sharpa_policy.SHARPA_STATE_DIM,)
    assert out["prompt"]
