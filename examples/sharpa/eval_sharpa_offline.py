"""Offline check of a trained sharpa_egg checkpoint, before it goes near the robot.

What this DOES verify -- the whole serving path, end to end:
  checkpoint load -> norm stats from the checkpoint -> SharpaInputs -> model -> Unnormalize
  -> SharpaOutputs -> 28-d action -> integrate_palm_centric -> wrist trajectory.
Every one of those can be silently wrong (wrong asset_id, wrong quantiles, transposed image,
swapped camera slots) and a wrong one shows up here as a large error rather than as a crash.

What this does NOT tell you: generalisation. All 38 episodes went into training, so the numbers
below are TRAINING error. They bound how well the policy fits, not how it will behave on the
robot. Treat a small error as "the plumbing is right", never as "the policy works".

Usage:
    uv run examples/sharpa/eval_sharpa_offline.py \
        --checkpoint-dir /n/netscratch/.../ckpt/sharpa_pi05/sharpa_egg/egg_70ep_b64/15120
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import tyro

ACTION_DIM = 28


def _chunk_to_wrist(action28: np.ndarray, pos0: np.ndarray, rot0: np.ndarray):
    """Integrate a chunk the way the robot will: step 0's delta IS applied.

    Mirrors tactile_steering.dp2.sharpa_env_bridge.chunk_to_targets -- the sentinel frame is what
    stops the first step of every chunk from being silently dropped.
    """
    pos, rot = pos0.astype(np.float64), rot0.astype(np.float64)
    out_p, out_r = [], []
    for dp, dr in zip(action28[:, 22:25], action28[:, 25:28], strict=True):
        pos = pos + rot @ dp
        rot = rot @ Rotation.from_rotvec(dr).as_matrix()
        out_p.append(pos)
        out_r.append(rot)
    return np.stack(out_p), np.stack(out_r)


def main(
    checkpoint_dir: pathlib.Path,
    *,
    config_name: str = "sharpa_egg",
    num_samples: int = 24,
    seed: int = 0,
) -> None:
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    train_config = _config.get_config(config_name)
    print(f"loading policy from {checkpoint_dir}")
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    # The RAW LeRobot dataset -- no transforms. create_trained_policy already owns the transform
    # stack, so feeding it transformed data would apply everything twice.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
    )
    print(f"dataset: {len(dataset)} frames, action_horizon={train_config.model.action_horizon}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    err_finger, err_dp, err_drot = [], [], []
    err_wrist_mm, err_wrist_deg = [], []
    for n, i in enumerate(idx):
        item = dataset[int(i)]
        obs = {
            "base": np.asarray(item["observation.images.head"]),
            "wrist": np.asarray(item["observation.images.wrist"]),
            "state": np.asarray(item["observation.state"], dtype=np.float32),
            "prompt": item["task"],
        }
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
        truth = np.asarray(item["action"], dtype=np.float64)
        if pred.shape != truth.shape:
            raise SystemExit(f"shape mismatch: predicted {pred.shape}, dataset {truth.shape}")

        err_finger.append(np.abs(pred[:, 0:22] - truth[:, 0:22]).mean())
        err_dp.append(np.abs(pred[:, 22:25] - truth[:, 22:25]).mean())
        err_drot.append(np.abs(pred[:, 25:28] - truth[:, 25:28]).mean())

        # Where the wrist actually ends up after executing the whole chunk. This is the number
        # that matters on the robot: per-step delta errors compound over the 30 steps.
        p_hat, r_hat = _chunk_to_wrist(pred, np.zeros(3), np.eye(3))
        p_ref, r_ref = _chunk_to_wrist(truth, np.zeros(3), np.eye(3))
        err_wrist_mm.append(np.linalg.norm(p_hat[-1] - p_ref[-1]) * 1e3)
        err_wrist_deg.append(
            np.rad2deg(np.linalg.norm(Rotation.from_matrix(r_ref[-1].T @ r_hat[-1]).as_rotvec()))
        )
        if n < 5:
            print(f"  sample {n}: finger {err_finger[-1]:.4f} rad, "
                  f"wrist end {err_wrist_mm[-1]:.2f} mm / {err_wrist_deg[-1]:.2f} deg")

    def rep(name, v, unit):
        v = np.asarray(v)
        print(f"  {name:28s} mean {v.mean():8.4f}  p50 {np.median(v):8.4f}  max {v.max():8.4f}  {unit}")

    print(f"\n== {len(idx)} samples (TRAINING data -- fit, not generalisation) ==")
    rep("finger MAE [0:22]", err_finger, "rad")
    rep("wrist dp MAE [22:25]", err_dp, "m")
    rep("wrist drot MAE [25:28]", err_drot, "rad")
    rep("wrist end-of-chunk pos err", err_wrist_mm, "mm")
    rep("wrist end-of-chunk rot err", err_wrist_deg, "deg")


if __name__ == "__main__":
    tyro.cli(main)
