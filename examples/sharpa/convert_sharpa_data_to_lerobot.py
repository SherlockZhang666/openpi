"""Raw OpenArm + Sharpa Wave teleop HDF5 -> LeRobot v2.1 datasets.

Reads the collection rig's schema-v1 episodes (`<session>/ep_XXXX/ep_XXXX.hdf5` plus the
two sibling mkvs) and writes TWO LeRobot datasets:

  <out_root>/<task_dir>            training set:  head + wrist video, state(29), action(28),
                                                  observation.tactile_force(30)
  <out_root>/<task_dir>_tactile    the 5 per-finger deform video streams, same frames in the
                                   same order, for the tactile-steering phase

They are split on purpose. `LeRobotDataset.__getitem__` decodes EVERY video key on every
sample, so folding the 5 deform streams into the training set would make the pi0.5 dataloader
decode 7 videos per item instead of 2, for data the base policy never sees. The naming follows
the T-Rex dataset (`observation.images.tactile_left_deform_<finger>`, `observation.tactile_force`)
so the two sources stay interchangeable downstream.

The raw HDF5 is never modified; it stays the archival source of truth (video encoding here is
lossy and the images are downscaled).

What this script decides, and why:

* **Only the engaged segment is kept.** `teleop/engaged[:, left]` is exactly one contiguous run
  in all 100 pick_up_the_egg episodes (checked). The frames outside it are homing / hold, where
  the arm moves on its own -- training on them teaches the policy to drive itself back to home.

* **Resampled onto an exact 30 Hz grid** by nearest neighbour on `time/t_mono`. The rig runs at
  30 Hz with jitter (dt max 0.049s, i.e. the occasional dropped frame), and LeRobot requires
  timestamps that match the declared fps to within 1e-4 s. Nearest neighbour, not interpolation:
  the row carries a quaternion, a video frame index and a tactile image that must all come from
  the same instant.

* **Actions are the teleop command, state is the measurement.** action[0:22] = `action/hand_joint_pos`
  (what the hand was told to do), state[7:29] = `hand/joint_pos` (what it did).

* **Incomplete recordings are excluded, not repaired.** Of the 100 pick_up_the_egg episodes only
  38 have both streams intact; see `UnusableEpisode` for what is wrong with the other 62 and
  which sessions they are. Confirmed with the operator on 2026-09-03 as real collection faults,
  so they are dropped rather than filled in. `meta/sharpa_conversion.json` records every
  exclusion and its reason, and `--expect-episodes` pins the surviving count.

* **Palm-centric wrist deltas are computed AFTER resampling**, between consecutive output frames,
  so they compose to the trajectory the policy will actually be rolled out at.

Usage:
    uv run examples/sharpa/convert_sharpa_data_to_lerobot.py \
        --raw-root /n/netscratch/ydu_lab/Lab/hangxing/data/tactile-steering-data/pick_up_the_egg \
        --out-root /n/netscratch/ydu_lab/Lab/hangxing/data/sharpa_lerobot \
        --task-dir pick_up_the_egg
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import shutil
import time
from typing import Literal

import av
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot.common.datasets.lerobot_dataset as _lrd
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

# Kept in sync with openpi.policies.sharpa_policy; restated rather than imported so this
# converter can run without the training deps on the path.
STATE_DIM = 29  # left arm 7 + left hand 22
ACTION_DIM = 28  # hand 22 + wrist dp 3 + wrist drot 3
N_HAND = 22
N_TACTILE_CH = 5
N_F6 = 6

ROBOT_TYPE = "openarm_bimanual_v10_sharpa_left"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# episode reading
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Episode:
    """One resampled, engaged-only episode, ready to be written frame by frame."""

    task: str
    state: np.ndarray  # (N, 29) float32
    action: np.ndarray  # (N, 28) float32
    tactile_force: np.ndarray  # (N, 30) float32
    head_rows: np.ndarray  # (N,) int -- frame indices into the scene mkv
    wrist_rows: np.ndarray  # (N,) int -- frame indices into the wrist mkv
    deform_rows: np.ndarray  # (N,) int -- row indices into tactile/deform
    hdf5_path: pathlib.Path
    resample_err_s: float


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Inclusive [start, end] of the longest contiguous True run. Raises if there is none."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("no True frames")
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _resample_rows(t: np.ndarray, lo: int, hi: int, fps: int) -> tuple[np.ndarray, float]:
    """Nearest-neighbour rows of `t[lo:hi+1]` on an exact `fps` grid. Returns (rows, max error)."""
    ts = t[lo : hi + 1]
    n = int(np.floor((ts[-1] - ts[0]) * fps)) + 1
    grid = ts[0] + np.arange(n) / fps
    j = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    take_left = (grid - ts[j - 1]) <= (ts[j] - grid)
    rows = np.where(take_left, j - 1, j)
    err = float(np.max(np.abs(ts[rows] - grid)))
    return rows + lo, err


def _take(dset, rows: np.ndarray) -> np.ndarray:
    """`dset[rows]` for possibly-repeating `rows`.

    h5py fancy indexing demands strictly increasing indices, but the 30 Hz grid legitimately
    lands on the same source row twice wherever the rig dropped a frame. Read each distinct row
    once, then fan out in numpy.
    """
    uniq, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dset[uniq])[inverse]


def _left_arm_columns(joint_names: list[str]) -> np.ndarray:
    """Column indices of the 7 left-arm joints. Derived from names, never assumed to be 7:14."""
    cols = [i for i, n in enumerate(joint_names) if "left" in n]
    if len(cols) != 7:
        raise ValueError(f"expected 7 left-arm joints, found {len(cols)} in {joint_names}")
    return np.asarray(cols)


def _palm_centric_wrist(pose7: np.ndarray) -> np.ndarray:
    """(N,7) [x,y,z,qw,qx,qy,qz] in base frame -> (N,6) deltas in the PREVIOUS palm frame.

    p_t = p_{t-1} + R_{t-1} @ dp_t,  R_t = R_{t-1} @ exp(dr_t).  Frame 0 is zero (there is no
    previous frame); this matches tactile_steering.data.actions.palm_centric and is what makes
    integrate_palm_centric an exact inverse.
    """
    pos = pose7[:, :3].astype(np.float64)
    # The rig writes wxyz (DATA_SCHEMA_v1.md / collect_core._pose7_wxyz); scipy wants xyzw.
    rot = Rotation.from_quat(pose7[:, [4, 5, 6, 3]].astype(np.float64))
    mats = rot.as_matrix()

    out = np.zeros((len(pose7), 6), dtype=np.float32)
    r_prev = mats[:-1]
    out[1:, 0:3] = np.einsum("nji,nj->ni", r_prev, pos[1:] - pos[:-1])
    out[1:, 3:6] = Rotation.from_matrix(np.einsum("nji,njk->nik", r_prev, mats[1:])).as_rotvec()
    return out


class UnusableEpisode(Exception):
    """This episode has a defect in the RAW DATA and must be left out of the dataset.

    Distinct from every other exception the converter can raise: those mean the converter is
    wrong and must stop the run, this one means the recording is incomplete and the episode is
    skipped and reported. Never widen it to paper over a converter bug.

    The two defects that actually occur in pick_up_the_egg (surveyed over all 100 episodes,
    2026-09-03), both session-wide and non-overlapping:

      * 40 episodes have no `action/hand_joint_pos` at all (`with_hand_action=False`): the WAVE
        glove's command stream was never written. The hand was still teleoperated, so
        `hand/joint_pos` exists -- but substituting the measurement for the command would make
        the action distribution inhomogeneous (measured vs commanded differ by the hand's
        tracking error: 0.5 deg median per joint, 2-3 deg on the worst joints).
        Sessions: 20260902_232453 (19), 20260902_234155 (16), 20260902_235550 (5).

      * 22 episodes have `thumb_CMC_FE` NaN for every frame in `hand/joint_pos` -- and in
        joint_vel and joint_torque too, i.e. that joint reported nothing at all, on the same
        hand (serial C5549233C555) that reports it fine elsewhere. It is state dim 7.
        The command side is clean, so only the observation is missing.
        Sessions: 20260902_204619 (1), 20260902_205056 (21).

    That leaves 38 fully clean episodes, which is what we train on (decision 2026-09-03).
    ⚠️ Do NOT read `attrs["hand_state_columns"]` as evidence about any of this -- the collection
    script writes the same "position only; vel/torque unverified -> NaN" string into every
    episode regardless of what was actually recorded.
    """


def read_episode(hdf5_path: pathlib.Path, fps: int) -> Episode:
    with h5py.File(hdf5_path, "r") as f:
        a = f.attrs
        side_order = json.loads(a["side_order"])
        li = side_order.index("left")

        if "action/hand_joint_pos" not in f:
            raise UnusableEpisode("no action/hand_joint_pos (with_hand_action=False)")

        engaged = f["teleop/engaged"][:, li]
        lo, hi = _longest_true_run(engaged)
        rows, err = _resample_rows(f["time/t_mono"][:], lo, hi, fps)

        for key in ("obs/valid", "hand/valid", "cam_wrist/valid", "action/hand_valid"):
            if not _take(f[key], rows).all():
                raise UnusableEpisode(f"{key} is False somewhere in the engaged segment")
        if not _take(f["action/valid"], rows)[:, li].all():
            raise UnusableEpisode("action/valid is False for the left side")

        arm_cols = _left_arm_columns(json.loads(a["joint_names"]))
        arm = _take(f["obs/joint_pos"], rows)[:, arm_cols]
        hand = _take(f["hand/joint_pos"], rows)
        hand_cmd = _take(f["action/hand_joint_pos"], rows)
        wrist_pose = _take(f["action/wrist_pose_b"], rows)[:, li]

        # Every numeric input is checked. NaN here is not hypothetical: thumb_CMC_FE is all-NaN
        # in 22 episodes, and before this check it flowed through to norm_stats.json as a JSON
        # `null`, which came back as None and blew up inside Normalize -- at the first training
        # batch, not here.
        hand_names = json.loads(a["hand_joint_names"])
        for label, arr, names in (
            ("obs/joint_pos[left arm]", arm, [f"arm{i}" for i in range(len(arm_cols))]),
            ("hand/joint_pos", hand, hand_names),
            ("action/hand_joint_pos", hand_cmd, hand_names),
        ):
            bad = [n for n, ok in zip(names, np.isfinite(arr).all(0), strict=True) if not ok]
            if bad:
                raise UnusableEpisode(f"{label} is non-finite for {bad}")
        if not np.isfinite(wrist_pose).all():
            raise UnusableEpisode("action/wrist_pose_b is non-finite for the left side")

        state = np.concatenate([arm, hand], axis=1).astype(np.float32)
        action = np.concatenate([hand_cmd, _palm_centric_wrist(wrist_pose)], axis=1).astype(np.float32)

        tactile_force = _take(f["tactile/f6"], rows).reshape(len(rows), N_TACTILE_CH * N_F6).astype(np.float32)
        wrist_rows = _take(f["cam_wrist/frame_index"], rows)

        if int(a["video_frames"]) != int(a["num_frames"]):
            raise ValueError(f"{hdf5_path}: scene mkv is not 1:1 with the step timeline")
        if wrist_rows.max() >= int(a["video_wrist_frames"]):
            raise ValueError(f"{hdf5_path}: cam_wrist/frame_index runs past the wrist mkv")

        ep = Episode(
            task=str(a["task"]),
            state=state,
            action=action,
            tactile_force=tactile_force,
            head_rows=rows,
            wrist_rows=np.asarray(wrist_rows),
            deform_rows=rows,
            hdf5_path=hdf5_path,
            resample_err_s=err,
        )

    if ep.state.shape[1] != STATE_DIM or ep.action.shape[1] != ACTION_DIM:
        raise ValueError(f"{hdf5_path}: got state {ep.state.shape} action {ep.action.shape}")
    if err > 1.0 / fps:
        raise ValueError(f"{hdf5_path}: resampling gap {err:.4f}s exceeds one {fps} Hz period")
    return ep


# --------------------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------------------


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def decode_frames(path: pathlib.Path, wanted: np.ndarray, short_side: int) -> list[np.ndarray]:
    """Decode `path` sequentially and return the requested frames, downscaled.

    The rig writes MJPEG, so every frame is a keyframe and a single linear pass is the cheapest
    way to gather scattered indices. Frames are resized on the way out: keeping 800x1280 in
    memory for a whole episode costs > 1 GB and pi0.5 sees 224x224 anyway.
    """
    need = set(int(i) for i in wanted)
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i not in need:
                continue
            h, w = frame.height, frame.width
            scale = short_side / min(h, w)
            out[i] = frame.reformat(
                width=_even(w * scale), height=_even(h * scale), format="rgb24"
            ).to_ndarray()
            if len(out) == len(need):
                break
    missing = need - out.keys()
    if missing:
        raise ValueError(f"{path}: could not decode frames {sorted(missing)[:5]}...")
    return [out[int(i)] for i in wanted]


def probe_size(path: pathlib.Path, short_side: int) -> tuple[int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        h, w = stream.codec_context.height, stream.codec_context.width
    scale = short_side / min(h, w)
    return _even(h * scale), _even(w * scale)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def use_h264() -> None:
    """Encode with h264 instead of LeRobot's libsvtav1 default.

    Measured on one episode of this data (594 frames, 416x256): svtav1 crf30 10.3s / 2.2MB,
    h264 crf23 4.7s / 1.9MB. The bigger win is at read time -- 50 random seeks take 0.36s on
    the AV1 file and 0.10s on the h264 one, and random seeks are exactly what the training
    dataloader does. `g=2` (LeRobot's default) is kept: it is what makes those seeks cheap.

    The tactile deform streams are encoded LOSSLESSLY instead; see `_encode_video_frames`.
    """
    _lrd.encode_video_frames = _encode_video_frames


# crf23 is fine for a camera and wrong for a deform map. A deform map is 93-97% flat at the
# resting code 2 with the signal in 3-7% of the pixels, so a perceptual encoder spends its
# bitrate on exactly the wrong thing. Measured round trip on one real index channel (524
# frames of 193151/ep_0005, |F| up to 14.7 N):
#
#            encoding                 size   code err mean/max   mm err max   uniform frames kept
#   h264 crf23 yuv420p (was used)   0.04 MB      0.072 / 50        1080 um         69 / 96
#   h264 crf0  yuv444p (now used)   0.34 MB      0.004 /  1          30 um         96 / 96
#
# Two things break at crf23. The max error is 1.08 mm against a peak deformation of 2.7 mm,
# i.e. 40% of the signal locally. And 28% of the no-contact frames stop being uniform planes
# once the encoder puts noise into them, which destroys the one clean bit the map carries --
# "this pad reports no deformation at all". That is the control group a dead-band study
# counts, and `collect/tactile.py` goes out of its way to keep it (the 2026-08-19 egg
# sessions lost 83.5% of their (frame, finger) samples to an earlier version of this bug).
#
# Aggregates survive crf23 -- corr(sum of deform, |F|) only moves +0.8031 -> +0.8025 -- so
# this only matters if the policy looks at the map per pixel. It does.
#
# yuv444p, not yuv420p: chroma subsampling would average the map with its own neighbours.
# Full-range signalling (`-color_range pc`) was tried and is worse, not better: the decoder
# does not honour the tag and re-applies the tv->pc expansion, costing a flat +16 codes.
# The residual +-1 at crf0 is RGB->YUV444 rounding, not the codec; it is 0.005 mm below
# code 100 and 0.03 mm above, against a 1080 um error at crf23.
#
# ffv1 would be bit exact and about the same size, but LeRobot's `encode_video_frames`
# rejects any vcodec outside {h264, hevc, libsvtav1} and the container is .mp4.
_TACTILE_ENCODING = {"vcodec": "h264", "crf": 0, "pix_fmt": "yuv444p"}
_CAMERA_ENCODING = {"vcodec": "h264", "crf": 23}


# Bound at import, before `use_h264()` swaps the module attribute, so the wrapper below
# always calls the real encoder and not itself.
_lrd_encode_video_frames = _lrd.encode_video_frames


def _encode_video_frames(imgs_dir, video_path, fps, **kwargs):
    """Dispatch on the video key: lossless for the deform maps, crf23 for the cameras.

    LeRobot lays videos out as `videos/chunk-NNN/<video_key>/episode_NNNNNN.mp4`, so the
    parent directory name is the feature key.
    """
    key = pathlib.Path(video_path).parent.name
    preset = _TACTILE_ENCODING if "tactile" in key else _CAMERA_ENCODING
    return _lrd_encode_video_frames(imgs_dir, video_path, fps, **{**preset, **kwargs})


# `tactile/deform` is uint8 and the SDK's `deform_map_value()` maps it to millimetres. The
# mapping is piecewise linear with a knee at code 100, so the codes are NOT linear in mm --
# above the knee one code is 6x the deformation it is below it. Normalising the raw uint8
# to [0, 1] therefore hands the policy a kinked signal that no longer lines up with
# `observation.tactile_force`; convert first, normalise after.
#
# Extracted from libsharpa-wave-sdk.so (symbol _ZN6sharpa7tactile5Touch16deform_map_valueEh)
# and checked against all 256 codes, max deviation 4.2e-07 (float32 noise). Restated here so
# the training side does not need the SDK loaded.
#
#   code   0 -> 0.000 mm      code 2 (pad at rest) -> 0.010 mm
#   code 100 -> 0.500 mm      code 255             -> 5.150 mm
DEFORM_MM_KNEE_CODE = 100
DEFORM_MM_PER_CODE_LOW = 0.005
DEFORM_MM_PER_CODE_HIGH = 0.030


def deform_code_to_mm(code: np.ndarray) -> np.ndarray:
    """uint8 deform codes -> millimetres, matching the SDK's `deform_map_value()`."""
    c = np.asarray(code, dtype=np.float32)
    return np.where(
        c <= DEFORM_MM_KNEE_CODE,
        c * DEFORM_MM_PER_CODE_LOW,
        DEFORM_MM_KNEE_CODE * DEFORM_MM_PER_CODE_LOW
        + (c - DEFORM_MM_KNEE_CODE) * DEFORM_MM_PER_CODE_HIGH,
    )


def verify_episode_videos(dataset: LeRobotDataset, episode_index: int, expected: int) -> None:
    """Every video stream of this episode must have exactly `expected` frames.

    Not paranoia: `lerobot.common.datasets.image_writer.write_image` catches its own exceptions
    and only prints them, so a failed PNG write silently drops a frame. `encode_video_frames`
    then globs whatever PNGs exist and produces a SHORT video, while the parquet still has the
    full row count -- the video and the low-dim columns would be silently off by one from that
    frame onward. Reading the container header is cheap; do it.
    """
    for key in dataset.meta.video_keys:
        path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        with av.open(str(path)) as container:
            got = container.streams.video[0].frames
        if got != expected:
            raise ValueError(
                f"{path}: encoded {got} frames but the episode has {expected}. A PNG write "
                f"probably failed silently -- check the log for 'Error writing image'."
            )


def _video_feature(height: int, width: int) -> dict:
    return {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]}


def _vector_feature(dim: int, name: str) -> dict:
    return {"dtype": "float32", "shape": (dim,), "names": [name]}


def list_episodes(raw_root: pathlib.Path) -> list[pathlib.Path]:
    """Episodes marked usable in each session's index.jsonl, in a stable order."""
    paths: list[pathlib.Path] = []
    for index_path in sorted(raw_root.glob("*/index.jsonl")):
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("discarded") or not rec.get("success"):
                logger.info("skipping %s/%s (success=%s discarded=%s)", index_path.parent.name,
                            rec["path"], rec.get("success"), rec.get("discarded"))
                continue
            paths.append(index_path.parent / rec["path"])
    return paths


def main(
    raw_root: pathlib.Path = pathlib.Path(
        "/n/netscratch/ydu_lab/Lab/hangxing/data/tactile-steering-data/pick_up_the_egg"
    ),
    out_root: pathlib.Path = pathlib.Path("/n/netscratch/ydu_lab/Lab/hangxing/data/sharpa_lerobot"),
    task_dir: str = "pick_up_the_egg",
    *,
    only: Literal["both", "train", "tactile"] = "both",
    fps: int = 30,
    image_short_side: int = 256,
    # Stop after this many episodes have been WRITTEN (not scanned). Counting written ones is
    # what makes a small smoke run useful: the first three episodes on disk are all defective,
    # so slicing the input list would convert nothing.
    limit_episodes: int | None = None,
    # Hard assertion on how many episodes survive the data-completeness checks. Pass it for
    # real runs: silently converting fewer episodes than intended is the failure mode this
    # whole exclusion path could otherwise hide.
    expect_episodes: int | None = None,
    overwrite: bool = False,
    image_writer_processes: int = 8,
    image_writer_threads: int = 4,
) -> None:
    """Convert the raw teleop episodes under `raw_root` into LeRobot datasets under `out_root`.

    `--only` splits the work so the two datasets can be built as two concurrent jobs; they read
    the same HDF5 and produce identical episode/frame ordering either way, because the segment
    and the resampling grid are a pure function of the file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_h264()

    want_train = only in ("both", "train")
    want_tactile = only in ("both", "tactile")

    episode_paths = list_episodes(raw_root)
    if not episode_paths:
        raise SystemExit(f"no usable episodes under {raw_root}")
    logger.info("found %d episodes; building %s", len(episode_paths), only)

    probe = episode_paths[0]
    with h5py.File(probe, "r") as f:
        finger_order = json.loads(f.attrs["tactile_finger_order"])
        hand_joint_names = json.loads(f.attrs["hand_joint_names"])
    if len(hand_joint_names) != N_HAND:
        raise ValueError(f"expected {N_HAND} hand joints, got {len(hand_joint_names)}")

    train_root = out_root / task_dir
    tactile_root = out_root / f"{task_dir}_tactile"
    targets = ([train_root] if want_train else []) + ([tactile_root] if want_tactile else [])
    for path in targets:
        if path.exists():
            if not overwrite:
                raise SystemExit(f"{path} already exists; pass --overwrite to replace it")
            shutil.rmtree(path)

    writer_kwargs = {
        "image_writer_processes": image_writer_processes,
        "image_writer_threads": image_writer_threads,
        # LeRobot defaults to torchcodec whenever the package imports; it does not load on this
        # cluster (no ffmpeg shared libs). Pin the working fallback. See DataConfig.video_backend.
        "video_backend": "pyav",
    }

    train = None
    if want_train:
        head_hw = probe_size(probe.with_name(probe.stem + "_scene.mkv"), image_short_side)
        wrist_hw = probe_size(probe.with_name(probe.stem + "_wrist.mkv"), image_short_side)
        logger.info("head video %s, wrist video %s", head_hw, wrist_hw)
        train = LeRobotDataset.create(
            repo_id="local_repo",
            root=train_root,
            robot_type=ROBOT_TYPE,
            fps=fps,
            features={
                "observation.images.head": _video_feature(*head_hw),
                "observation.images.wrist": _video_feature(*wrist_hw),
                "observation.state": _vector_feature(STATE_DIM, "state"),
                "action": _vector_feature(ACTION_DIM, "action"),
                "observation.tactile_force": _vector_feature(N_TACTILE_CH * N_F6, "tactile_force"),
            },
            **writer_kwargs,
        )

    tactile = None
    if want_tactile:
        tactile = LeRobotDataset.create(
            repo_id="local_repo_tactile",
            root=tactile_root,
            robot_type=ROBOT_TYPE,
            fps=fps,
            features={
                **{
                    f"observation.images.tactile_left_deform_{finger}": _video_feature(240, 240)
                    for finger in finger_order
                },
                "observation.state": _vector_feature(STATE_DIM, "state"),
                "action": _vector_feature(ACTION_DIM, "action"),
                "observation.tactile_force": _vector_feature(N_TACTILE_CH * N_F6, "tactile_force"),
            },
            **writer_kwargs,
        )

    total_frames = 0
    worst_err = 0.0
    written = 0
    excluded: list[dict] = []
    included: list[str] = []
    for n, hdf5_path in enumerate(episode_paths):
        started = time.time()
        rel = f"{hdf5_path.parent.parent.name}/{hdf5_path.parent.name}"
        try:
            ep = read_episode(hdf5_path, fps)
        except UnusableEpisode as exc:
            # A defect in the recording, not in this script: leave the episode out and say so.
            # Any other exception is a converter bug and is deliberately left to propagate.
            excluded.append({"episode": rel, "reason": str(exc)})
            logger.warning("[%d/%d] SKIP %s -- %s", n + 1, len(episode_paths), rel, exc)
            continue
        n_frames = len(ep.state)

        head = wrist = deform = None
        if train is not None:
            head = decode_frames(hdf5_path.with_name(hdf5_path.stem + "_scene.mkv"), ep.head_rows, image_short_side)
            wrist = decode_frames(hdf5_path.with_name(hdf5_path.stem + "_wrist.mkv"), ep.wrist_rows, image_short_side)
        if tactile is not None:
            with h5py.File(hdf5_path, "r") as f:
                if not _take(f["tactile/deform_valid"], ep.deform_rows).all():
                    raise ValueError(f"{hdf5_path}: tactile/deform_valid is False inside the segment")
                deform = _take(f["tactile/deform"], ep.deform_rows)  # (N, 5, 240, 240) uint8

        for i in range(n_frames):
            shared = {
                "observation.state": ep.state[i],
                "action": ep.action[i],
                "observation.tactile_force": ep.tactile_force[i],
                "task": ep.task,
            }
            if train is not None:
                train.add_frame(
                    {
                        "observation.images.head": head[i],
                        "observation.images.wrist": wrist[i],
                        **shared,
                    }
                )
            if tactile is not None:
                tactile.add_frame(
                    {
                        **{
                            f"observation.images.tactile_left_deform_{finger}": np.repeat(
                                deform[i, c][:, :, None], 3, axis=2
                            )
                            for c, finger in enumerate(finger_order)
                        },
                        **shared,
                    }
                )

        for dataset in (d for d in (train, tactile) if d is not None):
            dataset.save_episode()
            # `written`, not `n`: LeRobot numbers episodes consecutively from 0, so skipping an
            # episode shifts every later index. Verifying against `n` would read the wrong file.
            verify_episode_videos(dataset, written, n_frames)

        included.append(rel)
        written += 1
        total_frames += n_frames
        worst_err = max(worst_err, ep.resample_err_s)
        logger.info(
            "[%d/%d] %s -> episode %d, %d frames (resample err %.4fs, %.0fs)",
            n + 1, len(episode_paths), rel, written - 1,
            n_frames, ep.resample_err_s, time.time() - started,
        )
        if limit_episodes is not None and written >= limit_episodes:
            logger.info("stopping after %d written episodes (--limit-episodes)", written)
            break

    by_reason: dict[str, int] = {}
    for item in excluded:
        by_reason[item["reason"]] = by_reason.get(item["reason"], 0) + 1
    logger.info("kept %d of %d episodes; excluded %d", written, len(episode_paths), len(excluded))
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        logger.info("  excluded %3d x %s", count, reason)
    if written == 0:
        raise SystemExit("every episode was excluded -- nothing was written")
    if expect_episodes is not None and written != expect_episodes:
        raise SystemExit(f"expected {expect_episodes} usable episodes, kept {written}")

    sidecar = {
        "raw_root": str(raw_root),
        "fps": fps,
        "image_short_side": image_short_side,
        "engaged_only": True,
        "hand_joint_names": hand_joint_names,
        "tactile_finger_order": finger_order,
        "video_encoding": {"cameras": _CAMERA_ENCODING, "tactile_deform": _TACTILE_ENCODING},
        # The deform videos carry the SDK's raw uint8 codes, not millimetres: requantising mm
        # back into uint8 would throw away the fine 0.005 mm step below the knee. Convert on
        # the training side with `deform_code_to_mm` (or this formula) BEFORE normalising --
        # the codes are piecewise linear in mm, not linear.
        "deform_code_to_mm": {
            "formula": "mm = code*0.005 if code <= 100 else 0.5 + (code-100)*0.030",
            "knee_code": DEFORM_MM_KNEE_CODE,
            "mm_per_code_low": DEFORM_MM_PER_CODE_LOW,
            "mm_per_code_high": DEFORM_MM_PER_CODE_HIGH,
            "resting_code": 2,
            "source": "libsharpa-wave-sdk.so deform_map_value(), all 256 codes, max dev 4.2e-07",
        },
        "state_layout": "[0:7] left arm joints (obs/joint_pos), [7:29] left hand joints (hand/joint_pos)",
        "action_layout": (
            "[0:22] left hand command (action/hand_joint_pos), [22:25] palm-frame dp (m), "
            "[25:28] palm-frame drot (axis-angle, rad)"
        ),
        "num_episodes": written,
        "num_frames": total_frames,
        "worst_resample_error_s": worst_err,
        # The dataset's episode_index i corresponds to episodes[i]; consumers that need to go
        # back to the raw HDF5 must use this, not the position in a fresh directory listing.
        "episodes": included,
        "excluded": excluded,
    }
    for root in targets:
        (root / "meta" / "sharpa_conversion.json").write_text(json.dumps(sidecar, indent=2))
        logger.info("wrote %d episodes / %d frames to %s", written, total_frames, root)


if __name__ == "__main__":
    tyro.cli(main)
