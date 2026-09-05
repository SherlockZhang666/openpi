"""Transforms for the OpenArm(7) + Sharpa Wave left-hand(22) single-hand rig.

Actions are 28-d palm-centric:
    [0:22]  left-hand joints, ABSOLUTE angles, in SHARPA_HAND_JOINT_ORDER
    [22:25] palm-frame translation delta (m)
    [25:28] palm-frame rotation delta (axis-angle, rad)

The asymmetry -- fingers absolute, wrist incremental -- is deliberate; do not
"unify" it. Deltas compose in the *previous* palm frame:
    p_t = p_{t-1} + R_{t-1} @ dp,   R_t = R_{t-1} @ exp(dr)
The definition and its inverse live in tactile_steering/data/actions.py
(`palm_centric` / `integrate_palm_centric`). That package cannot be imported from
here -- openpi and tactile_steering are separate packages in separate venvs -- so
the dimensions are restated below and pinned by tests on both sides.

28 <= 32, so the native pi0.5 action_dim is used with zero padding. No checkpoint
action-dim conversion is needed; do not run any convert_to_action_dim_* script.

State is 29-d: [0:7] left arm joints, [7:29] left hand joints. Note that state is
joint angles while the action's wrist part is a pose delta -- the two are NOT
isomorphic and must never be assigned to each other.

Camera slots: wrist -> left_wrist_0_rgb, head -> base_0_rgb. Both are fed to the
model; the right wrist slot is zero-padded and masked out. Which of the two views
actually carries the task signal is an open question -- the D1 ablation (wrist-only /
head-only / both) hangs off this layer.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

SHARPA_ACTION_DIM = 28
SHARPA_STATE_DIM = 29  # left arm 7 + left hand 22


def make_sharpa_example() -> dict:
    """A random example, for policy_config / server warm-up."""
    return {
        "base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(SHARPA_STATE_DIM).astype(np.float32),
        "prompt": "pick up the egg",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SharpaInputs(transforms.DataTransformFn):
    """Dataset/env keys -> the keys the model expects. Used in training and inference."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != SHARPA_STATE_DIM:
            raise ValueError(
                f"expected state dim {SHARPA_STATE_DIM} (arm 7 + hand 22), got {state.shape[-1]}"
            )

        head_image = _parse_image(data["base"])
        wrist_image = _parse_image(data["wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": wrist_image,
                # This rig has no right wrist camera; pad with zeros and mask it out.
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are only masked for the flow-matching models (PI0/PI05),
                # not for PI0_FAST. Do not change this.
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != SHARPA_ACTION_DIM:
                raise ValueError(
                    f"expected action dim {SHARPA_ACTION_DIM} "
                    f"(hand 22 + wrist dp 3 + wrist drot 3), got {actions.shape[-1]}"
                )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SharpaOutputs(transforms.DataTransformFn):
    """Model output (padded to the native 32 dims) -> this rig's 28 dims. Inference only."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :SHARPA_ACTION_DIM])}
