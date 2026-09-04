"""Check a converted Sharpa LeRobot dataset before spending GPU hours on it.

Four things, in increasing order of how expensive they are to discover later:

  1. schema      -- the features and shapes SharpaDataConfig's repack transform expects
  2. round-trip  -- integrating action[:, 22:28] back reproduces the raw HDF5 wrist trajectory
  3. pipeline    -- openpi's own data loader produces a batch of the right shapes
  4. norm stats  -- q01/q99 spans and dead dims (only if norm_stats.json exists yet)

(2) is the one that cannot be recovered from later: if the palm-centric encoding is wrong, the
policy trains fine, converges fine, and drives the wrist to the wrong place at deployment,
because `integrate_palm_centric` on the robot is the exact inverse of what the converter did.

Usage:
    uv run examples/sharpa/verify_sharpa_dataset.py \
        --root /n/netscratch/ydu_lab/Lab/hangxing/data/sharpa_lerobot/pick_up_the_egg
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import tyro

STATE_DIM = 29
ACTION_DIM = 28

EXPECTED_FEATURES = {
    "observation.images.head": None,  # shape is a conversion choice, not pinned here
    "observation.images.wrist": None,
    "observation.state": (STATE_DIM,),
    "action": (ACTION_DIM,),
    "observation.tactile_force": (30,),
}


def check_schema(root: pathlib.Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    print(f"codebase_version={info['codebase_version']}  fps={info['fps']}  "
          f"episodes={info['total_episodes']}  frames={info['total_frames']}")

    ok = True
    for key, want in EXPECTED_FEATURES.items():
        if key not in features:
            print(f"  MISSING  {key}")
            ok = False
            continue
        got = tuple(features[key]["shape"])
        flag = "ok " if (want is None or got == want) else "BAD"
        ok &= flag == "ok "
        print(f"  {flag} {key:38s} {features[key]['dtype']:6s} {got}")
    if not ok:
        raise SystemExit("schema does not match what SharpaDataConfig repacks; fix the converter")
    return info


def integrate(action28: np.ndarray, t0_pos: np.ndarray, t0_rot: np.ndarray):
    """The deployment-side inverse: 28-d chunk + start pose -> absolute wrist trajectory.

    Mirrors tactile_steering.data.actions.integrate_palm_centric. Frame 0's delta is ignored by
    construction (the converter writes zeros there), so poses[0] == the start pose.
    """
    pos = [t0_pos.astype(np.float64)]
    rot = [t0_rot.astype(np.float64)]
    for dp, dr in zip(action28[1:, 22:25], action28[1:, 25:28], strict=True):
        pos.append(pos[-1] + rot[-1] @ dp)
        rot.append(rot[-1] @ Rotation.from_rotvec(dr).as_matrix())
    return np.stack(pos), np.stack(rot)


def check_round_trip(root: pathlib.Path, raw_root: pathlib.Path, n_episodes: int) -> None:
    """Re-derive the wrist trajectory from the stored actions and compare against the raw HDF5."""
    import h5py
    import pandas as pd

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import convert_sharpa_data_to_lerobot as conv  # noqa: PLC0415 -- sibling module

    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    # The converter drops defective episodes, so LeRobot's episode_index is NOT the position in
    # a listing of the raw directory. meta/sharpa_conversion.json records the actual mapping.
    sidecar = json.loads((root / "meta" / "sharpa_conversion.json").read_text())
    episode_paths = [
        raw_root / rel / f"{pathlib.Path(rel).name}.hdf5" for rel in sidecar["episodes"]
    ]
    print(f"  {len(sidecar['episodes'])} episodes kept, {len(sidecar['excluded'])} excluded")

    worst_pos = worst_rot = 0.0
    for idx, hdf5_path in enumerate(episode_paths[:n_episodes]):
        parquet = root / f"data/chunk-000/episode_{idx:06d}.parquet"
        actions = np.stack(pd.read_parquet(parquet)["action"].to_numpy()).astype(np.float64)

        ep = conv.read_episode(hdf5_path, fps)
        if len(actions) != len(ep.action):
            raise SystemExit(f"{parquet}: {len(actions)} rows but the converter derives {len(ep.action)}")

        with h5py.File(hdf5_path, "r") as f:
            li = json.loads(f.attrs["side_order"]).index("left")
            pose = conv._take(f["action/wrist_pose_b"], ep.head_rows)[:, li].astype(np.float64)
        truth_pos = pose[:, :3]
        truth_rot = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()

        got_pos, got_rot = integrate(actions, truth_pos[0], truth_rot[0])
        pos_err = float(np.abs(got_pos - truth_pos).max())
        rot_err = float(
            np.abs(Rotation.from_matrix(np.einsum("nji,njk->nik", truth_rot, got_rot)).as_rotvec()).max()
        )
        worst_pos = max(worst_pos, pos_err)
        worst_rot = max(worst_rot, rot_err)
        print(f"  ep{idx:03d} {len(actions):4d} frames   pos err {pos_err * 1e3:7.4f} mm   "
              f"rot err {np.rad2deg(rot_err):7.4f} deg")

    print(f"  worst over {min(n_episodes, len(episode_paths))} episodes: "
          f"{worst_pos * 1e3:.4f} mm / {np.rad2deg(worst_rot):.4f} deg")
    # float32 storage of the deltas is the only lossy step; anything above this is a real bug.
    if worst_pos > 1e-4 or worst_rot > 1e-3:
        raise SystemExit("round-trip error is too large -- the palm-centric encoding is wrong")


def check_pipeline(config_name: str) -> None:
    """Pull one batch through openpi's real training data pipeline."""
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    item = dataset[0]
    print(f"  dataset length: {len(dataset)}")
    for key, value in sorted(item.items()):
        if isinstance(value, dict):
            for sub, sub_value in sorted(value.items()):
                print(f"    {key}.{sub:20s} {np.shape(sub_value)} {np.asarray(sub_value).dtype}")
        else:
            print(f"    {key:24s} {np.shape(value)} {np.asarray(value).dtype}")


def check_norm_stats(assets_dir: pathlib.Path) -> None:
    """q01/q99 spans. PI05 forces use_quantile_norm=True, so mean/std never reach the model."""
    paths = sorted(assets_dir.rglob("norm_stats.json"))
    if not paths:
        print(f"  no norm_stats.json under {assets_dir} yet -- run scripts/compute_norm_stats.py first")
        return
    stats = json.loads(paths[-1].read_text())["norm_stats"]["actions"]
    span = np.asarray(stats["q99"]) - np.asarray(stats["q01"])
    print(f"  {paths[-1]}")
    print(f"  fingers [0:22] span  min {span[:22].min():.5f}  max {span[:22].max():.5f}")
    print(f"  wrist dp [22:25] span {np.round(span[22:25], 5)}  (m)")
    print(f"  wrist dr [25:28] span {np.round(span[25:28], 5)}  (rad)")
    dead = [i for i, w in enumerate(span[:ACTION_DIM]) if w < 1e-6]
    print(f"  DEAD dims (q99-q01 < 1e-6, normalise to a constant -1): {dead}")


def main(
    root: pathlib.Path = pathlib.Path(
        "/n/netscratch/ydu_lab/Lab/hangxing/data/sharpa_lerobot/pick_up_the_egg"
    ),
    raw_root: pathlib.Path = pathlib.Path(
        "/n/netscratch/ydu_lab/Lab/hangxing/data/tactile-steering-data/pick_up_the_egg"
    ),
    *,
    config_name: str = "sharpa_egg",
    round_trip_episodes: int = 5,
    skip_pipeline: bool = False,
) -> None:
    print("== 1. schema ==")
    check_schema(root)

    print("\n== 2. wrist round-trip (encode -> integrate -> raw HDF5) ==")
    check_round_trip(root, raw_root, round_trip_episodes)

    if not skip_pipeline:
        print("\n== 3. openpi data pipeline ==")
        check_pipeline(config_name)

        print("\n== 4. norm stats ==")
        import openpi.training.config as _config

        check_norm_stats(_config.get_config(config_name).assets_dirs)

    print("\nall checks passed")


if __name__ == "__main__":
    tyro.cli(main)
