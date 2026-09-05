"""Pins the rollout's action decoding against the converter's action definition.

Run in the openpi venv (needs scipy for the independent rotation path):

    uv run pytest examples/sharpa/rollout/actions_test.py -q

The important test is `test_round_trip_against_converter_formula`: it recomputes the
deltas exactly as convert_sharpa_data_to_lerobot._palm_centric_wrist does -- via scipy,
an independent implementation from actions.py's hand-rolled Rodrigues -- and checks that
PalmChain integrates them back to the poses they came from. If that ever fails, the
policy is being replayed in a frame it was not trained in, and no amount of tuning on the
robot will fix it.
"""

import actions
import numpy as np
import pytest

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def _random_pose7_sequence(n: int, seed: int = 0) -> np.ndarray:
    """(n,7) [x,y,z,qw,qx,qy,qz] with plausible per-step motion for a 30 Hz rig."""
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(scale=0.004, size=(n, 3)), axis=0) + np.array([0.4, 0.1, -0.3])
    rot = Rotation.identity()
    quats = []
    for _ in range(n):
        rot = rot * Rotation.from_rotvec(rng.normal(scale=0.02, size=3))
        x, y, z, w = rot.as_quat()
        quats.append([w, x, y, z])
    return np.concatenate([pos, np.asarray(quats)], axis=1)


def _palm_centric_wrist(pose7: np.ndarray) -> np.ndarray:
    """Verbatim from examples/sharpa/convert_sharpa_data_to_lerobot.py.

    Copied rather than imported: the converter pulls in av, h5py and lerobot at module
    level, none of which this test needs. Any edit there must be mirrored here, and this
    test is what will catch it if it is not.
    """
    pos = pose7[:, :3].astype(np.float64)
    rot = Rotation.from_quat(pose7[:, [4, 5, 6, 3]].astype(np.float64))
    mats = rot.as_matrix()
    out = np.zeros((len(pose7), 6), dtype=np.float32)
    r_prev = mats[:-1]
    out[1:, 0:3] = np.einsum("nji,nj->ni", r_prev, pos[1:] - pos[:-1])
    out[1:, 3:6] = Rotation.from_matrix(np.einsum("nji,njk->nik", r_prev, mats[1:])).as_rotvec()
    return out


def test_round_trip_against_converter_formula():
    poses = _random_pose7_sequence(200, seed=7)
    deltas = _palm_centric_wrist(poses).astype(np.float64)

    chain = actions.PalmChain.from_pose7(poses[0])
    for k in range(1, len(poses)):
        chain.step(deltas[k, 0:3], deltas[k, 3:6])
        assert np.allclose(chain.pos, poses[k, :3], atol=1e-9), f"position drifted at step {k}"
        got = actions.matrix_to_quat_wxyz(chain.rot)
        want = poses[k, 3:] * np.sign(poses[k, 3] or 1.0)
        # Quaternion double cover: compare the rotations, not the four numbers.
        assert min(np.linalg.norm(got - want), np.linalg.norm(got + want)) < 1e-8, (
            f"rotation drifted at step {k}")


def test_base_delta_inverts_apply_base_delta():
    """`base_delta` must be the exact inverse of openarm_vr_teleop.apply_base_delta."""
    poses = _random_pose7_sequence(50, seed=3)
    deltas = _palm_centric_wrist(poses).astype(np.float64)
    chain = actions.PalmChain.from_pose7(poses[0])
    anchor_p = poses[0, :3]
    anchor_r = actions.quat_wxyz_to_matrix(poses[0, 3:])

    for scale in (1.0, 0.5):
        chain.reset(poses[0])
        for k in range(1, len(poses)):
            chain.step(deltas[k, 0:3], deltas[k, 3:6])
            d = chain.base_delta(scale=scale)
            # What the node computes from what we sent.
            node_t = anchor_p + scale * d[:3]
            node_r = actions.quat_wxyz_to_matrix(d[3:]) @ anchor_r
            assert np.allclose(node_t, poses[k, :3], atol=1e-9)
            assert np.allclose(node_r, actions.quat_wxyz_to_matrix(poses[k, 3:]), atol=1e-9)


def test_zero_delta_holds_the_anchor():
    """A frame-0 action (all-zero wrist delta) must not move the arm.

    The converter writes zeros into row 0 of every episode, so the policy has seen them
    and will emit them; if they moved the arm, every chunk boundary would nudge it.
    """
    pose = np.array([0.3, -0.1, 0.5, 0.7071, 0.0, 0.7071, 0.0])
    chain = actions.PalmChain.from_pose7(pose)
    for _ in range(10):
        chain.step(np.zeros(3), np.zeros(3))
    d = chain.base_delta()
    assert np.allclose(d[:3], 0.0, atol=1e-12)
    assert np.allclose(d[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    lin, ang = chain.travel()
    assert lin < 1e-12
    assert ang < 1e-12


def test_exp_log_so3_agree_with_scipy():
    rng = np.random.default_rng(11)
    for _ in range(200):
        r = rng.normal(scale=1.2, size=3)
        assert np.allclose(actions.exp_so3(r), Rotation.from_rotvec(r).as_matrix(), atol=1e-12)
        assert np.allclose(actions.log_so3(actions.exp_so3(r)), Rotation.from_rotvec(r).as_rotvec(), atol=1e-9)
    # The two branches that are not exercised by random draws.
    assert np.allclose(actions.exp_so3(np.zeros(3)), np.eye(3))
    axis = np.array([0.0, 0.0, 1.0])
    near_pi = actions.log_so3(Rotation.from_rotvec(axis * (np.pi - 1e-9)).as_matrix())
    assert np.allclose(np.abs(near_pi), axis * np.pi, atol=1e-4)


def test_quaternion_round_trip_is_sign_stable():
    rng = np.random.default_rng(5)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        rot = actions.quat_wxyz_to_matrix(q)
        back = actions.matrix_to_quat_wxyz(rot)
        assert back[0] >= 0.0
        assert np.allclose(actions.quat_wxyz_to_matrix(back), rot, atol=1e-12)


def test_orthonormalise_stops_drift_over_a_long_episode():
    """30 Hz x 60 s of rotation deltas must not leave SO(3)."""
    rng = np.random.default_rng(2)
    chain = actions.PalmChain.from_pose7(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    for _ in range(1800):
        chain.step(np.zeros(3), rng.normal(scale=0.02, size=3))
    assert np.allclose(chain.rot @ chain.rot.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(chain.rot) - 1.0) < 1e-12


def test_split_action_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="28-d action"):
        actions.split_action(np.zeros(32))
    hand, dp, dr = actions.split_action(np.arange(28, dtype=np.float32))
    assert hand.shape == (22,)
    assert dp.shape == (3,)
    assert dr.shape == (3,)
    assert dp[0] == 22.0
    assert dr[0] == 25.0


def test_step_rejects_nan():
    """A NaN action must stop the rollout, not integrate into the command chain."""
    chain = actions.PalmChain.from_pose7(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        chain.step(np.array([np.nan, 0.0, 0.0]), np.zeros(3))
