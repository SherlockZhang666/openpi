"""Decoding the 28-d Sharpa action into a commanded wrist pose, and nothing else.

No hardware, no ROS, no GStreamer: this module is pure numpy so it can be tested in the
openpi venv while the rest of the client only runs in the robot venv. It is also the one
place where a sign or a frame convention can silently ruin a rollout, so it is small on
purpose.

THE ACTION
----------
    [0:22]  left-hand joint angles, ABSOLUTE, radians, SHARPA joint order
    [22:25] palm-frame translation delta, metres
    [25:28] palm-frame rotation delta, axis-angle, radians

`palm frame` means the PREVIOUS commanded wrist pose, so the deltas compose as

    p_k = p_{k-1} + R_{k-1} @ dp_k
    R_k = R_{k-1} @ exp(dr_k)

which is the exact inverse of `_palm_centric_wrist` in
examples/sharpa/convert_sharpa_data_to_lerobot.py. `actions_test.py` pins that: it runs a
random pose sequence through the converter's formula and back through `PalmChain`.

DO NOT ADD A SENTINEL SKIP HERE
-------------------------------
`step` integrates EVERY delta it is handed. That differs from the training-side inverse
(`tactile_steering`'s `integrate_palm_centric`), which ignores index 0 -- correctly, because
`palm_centric` FABRICATES a zero at index 0 of each episode, there being no previous frame
to difference against. A policy chunk has no such sentinel: all 30 of its actions are real
increments. Calling a function with the training-side convention on a chunk, or "tidying"
this one to match it, silently drops the first action of every chunk. Nothing errors; the
arm just accumulates a small deficit at every boundary and drifts. `loop_test.py` and
`schedule_test.py` both assert N control steps produce exactly N integrations.

WHICH POSE THE CHAIN TRACKS
---------------------------
`p, R` here are the same quantity the dataset differenced: `action/wrist_pose_b`, which is
the teleop node's `target_filt` -- the Cartesian target its IK was solving for. It is NOT
the measured wrist pose and NOT `obs/wrist_pose_b`. The distinction matters because the
two differ by the servo's tracking error (8.2 mm median, per the rig's SCHEMA.md), and a
chain seeded from the measurement would accumulate that error into the command.

So the chain is seeded from the node's own telemetry at engage, and then integrated OPEN
LOOP -- never re-seeded from a measurement mid-episode, for the same reason the node never
re-seeds `q_iter`. Re-seeding would make the policy's deltas relative to something it was
never trained against.

FEEDING IT BACK TO THE NODE
---------------------------
openarm_vr_teleop.py does not accept an absolute pose. It accepts a delta since engage and
applies it with `apply_base_delta`, which ADDS the translation (times --scale) and
LEFT-MULTIPLIES the rotation:

    target.t = anchor.t + scale * d.t
    target.R = d.R @ anchor.R

so `base_delta()` inverts exactly that. The `--scale` division is not decoration: the rig
collected with `--scale 0.5` (collect_up.sh), and `action/wrist_pose_b` recorded the
target AFTER that scaling, i.e. already in robot space. Replaying those deltas through a
node still set to 0.5 would halve every translation.
"""

from __future__ import annotations

import dataclasses

import numpy as np

ACTION_DIM = 28
N_HAND = 22
HAND = slice(0, 22)
DP = slice(22, 25)
DR = slice(25, 28)


def split_action(action) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(28,) -> (hand 22 rad, wrist dp 3 m, wrist drot 3 rad)."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-d action, got {a.shape[0]}")
    return a[HAND].copy(), a[DP].copy(), a[DR].copy()


# ------------------------------------------------------------------ SO(3) helpers
# Hand-rolled rather than scipy/pinocchio: this module must import in both venvs, and
# neither dependency is present in both. Cross-checked against scipy in actions_test.py.


def exp_so3(r) -> np.ndarray:
    """Rodrigues. Axis-angle (3,) -> rotation matrix (3,3)."""
    r = np.asarray(r, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        # Second order is enough here and avoids a 0/0: for theta this small the
        # first-order term already differs from the exact value by less than 1e-24.
        k = _skew(r)
        return np.eye(3) + k + 0.5 * (k @ k)
    k = _skew(r / theta)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def log_so3(rot) -> np.ndarray:
    """Rotation matrix (3,3) -> axis-angle (3,). Inverse of `exp_so3`."""
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos = (np.trace(rot) - 1.0) / 2.0
    cos = min(1.0, max(-1.0, cos))
    theta = float(np.arccos(cos))
    if theta < 1e-8:
        return np.array([rot[2, 1] - rot[1, 2],
                         rot[0, 2] - rot[2, 0],
                         rot[1, 0] - rot[0, 1]]) / 2.0
    if np.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I, whose
        # columns are all parallel to it, and take the best-conditioned one.
        m = (rot + np.eye(3)) / 2.0
        axis = m[:, int(np.argmax(np.diag(m)))]
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    v = np.array([rot[2, 1] - rot[1, 2],
                  rot[0, 2] - rot[2, 0],
                  rot[1, 0] - rot[0, 1]])
    return v * (theta / (2.0 * np.sin(theta)))


def _skew(v) -> np.ndarray:
    x, y, z = (float(t) for t in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """[qw,qx,qy,qz] -> (3,3). W FIRST, matching the rig's wire format everywhere."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(rot) -> np.ndarray:
    """(3,3) -> [qw,qx,qy,qz], w >= 0 so the sign is deterministic across calls."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def orthonormalise(rot) -> np.ndarray:
    """Nearest rotation matrix, so integrating thousands of deltas cannot drift off SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64).reshape(3, 3))
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


# ------------------------------------------------------------------ the chain


@dataclasses.dataclass
class PalmChain:
    """The commanded wrist pose, integrated from palm-centric deltas.

    `reset` seeds both the running pose and the engage anchor from the node's telemetry;
    `step` advances the running pose by one action; `base_delta` renders the running pose
    into the delta-since-engage the node consumes.
    """

    pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    anchor_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    anchor_rot: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    n_steps: int = 0

    @classmethod
    def from_pose7(cls, pose7) -> PalmChain:
        chain = cls()
        chain.reset(pose7)
        return chain

    def reset(self, pose7) -> None:
        """Seed from [x,y,z,qw,qx,qy,qz] -- the engage pose, in the ROBOT BASE frame."""
        p = np.asarray(pose7, dtype=np.float64).reshape(7)
        if not np.isfinite(p).all():
            raise ValueError(f"non-finite engage pose {p}; the node has no target yet")
        self.pos = p[:3].copy()
        self.rot = quat_wxyz_to_matrix(p[3:])
        self.anchor_pos = self.pos.copy()
        self.anchor_rot = self.rot.copy()
        self.n_steps = 0

    def step(self, dp, dr) -> None:
        dp = np.asarray(dp, dtype=np.float64).reshape(3)
        dr = np.asarray(dr, dtype=np.float64).reshape(3)
        if not (np.isfinite(dp).all() and np.isfinite(dr).all()):
            raise ValueError(f"non-finite wrist delta dp={dp} dr={dr}")
        self.pos = self.pos + self.rot @ dp
        self.rot = orthonormalise(self.rot @ exp_so3(dr))
        self.n_steps += 1

    @property
    def pose7(self) -> np.ndarray:
        return np.concatenate([self.pos, matrix_to_quat_wxyz(self.rot)])

    def base_delta(self, scale: float = 1.0) -> np.ndarray:
        """Delta since engage, in the convention `apply_base_delta` expects.

        Translation is pre-divided by the node's --scale so the node's own multiplication
        gives back the pose this chain actually holds. Rotation is NOT scaled -- the node
        does not scale it either.
        """
        if not np.isfinite(scale) or abs(scale) < 1e-6:
            raise ValueError(f"node scale {scale} is not usable")
        d_t = (self.pos - self.anchor_pos) / float(scale)
        d_r = self.rot @ self.anchor_rot.T
        return np.concatenate([d_t, matrix_to_quat_wxyz(d_r)])

    def travel(self) -> tuple[float, float]:
        """(metres, radians) moved from the engage anchor. For the safety envelope."""
        lin = float(np.linalg.norm(self.pos - self.anchor_pos))
        ang = float(np.linalg.norm(log_so3(self.rot @ self.anchor_rot.T)))
        return lin, ang
