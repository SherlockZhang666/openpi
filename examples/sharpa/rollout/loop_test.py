"""Runs the real control loop against fake hardware.

    /path/to/python -m pytest examples/sharpa/rollout/loop_test.py -q

No ROS, no GStreamer, no SDK, no policy server: the loop, the scheduler, the arm bridge
(socket disabled), the null hand sink, the flight recorder and the real action decoding
are all the production objects. Only the four leaves that touch hardware are faked.

This is the test that catches wiring mistakes -- an action slice used in the wrong place,
a chunk index off by one, a delta that stops being sent -- which the unit tests around
`actions.py` and `ChunkSchedule` cannot see because they never meet each other there.
"""

import time
import types

import actions
import arm as arm_mod
import hand as hand_mod
import main
import numpy as np
import pytest

FPS = 200.0  # run the loop fast; nothing here depends on wall-clock realism
ANCHOR = np.array([0.35, 0.12, -0.28, 1.0, 0.0, 0.0, 0.0])


class FakeSources:
    """Observations that change every step, so a frozen read would be visible."""

    def __init__(self):
        self.n = 0

    def read(self, side):
        self.n += 1
        state = np.full(main.STATE_DIM, self.n * 1e-3, dtype=np.float32)
        img = np.full((8, 8, 3), self.n % 256, dtype=np.uint8)
        return main.Observation(state=state, head=img, wrist=img, t=time.time())


class FakeClient:
    """A policy server that returns a known chunk and records what it was asked."""

    def __init__(self, chunk):
        self.chunk = np.asarray(chunk, dtype=np.float64)
        self.seen = []

    def infer(self, payload):
        assert set(payload) == {"state", "base", "wrist", "prompt"}, payload.keys()
        assert payload["state"].shape == (main.STATE_DIM,)
        self.seen.append(payload["state"][0])
        return {"actions": self.chunk}


def make_args(**over):
    a = types.SimpleNamespace(
        side="left", fps=FPS, chunk_steps=8, infer_lead=4, max_steps=40,
        max_lin=0.45, max_ang=1.8)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run_loop(chunk, **over):
    a = make_args(**over)
    src = FakeSources()
    client = FakeClient(chunk)
    inference = main.Inference(client, "pick up the egg")
    hand_sink = hand_mod.NullHand()
    recorder = main.Recorder(None)
    bridge = arm_mod.ArmBridge(enabled=False, send_hz=1000.0,
                              max_lin=a.max_lin, max_ang=a.max_ang).start()
    bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
    stats = {"steps": 0}
    try:
        main.control_loop(a, src, bridge, hand_sink, inference, recorder, 1.0 / a.fps, stats)
    finally:
        bridge.stop()
    return a, bridge, hand_sink, client, stats


def test_runs_to_max_steps_and_commands_every_step():
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    a, bridge, hand_sink, client, stats = run_loop(chunk)
    assert stats["steps"] == a.max_steps
    assert hand_sink.n == a.max_steps
    # 40 steps / 8 per chunk = 5 chunks, plus the prefetch fired in the last cycle.
    assert len(client.seen) in (5, 6)


def test_wrist_deltas_integrate_once_per_step():
    """1 mm of +x per action for 40 steps must be 40 mm at the wrist, not 60."""
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 22] = 0.001
    _, bridge, _, _, stats = run_loop(chunk)
    assert bridge.chain.pos[0] - ANCHOR[0] == pytest.approx(0.001 * stats["steps"], abs=1e-12)
    lin, _ = bridge.chain.travel()
    assert lin == pytest.approx(0.001 * stats["steps"], abs=1e-12)


def test_the_published_delta_tracks_the_chain():
    """What the UDP thread would send must always be the current commanded pose."""
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 23] = 0.002  # +y in the palm frame
    _, bridge, _, _, _ = run_loop(chunk)
    published = bridge.published_delta
    expected = bridge.chain.base_delta(scale=1.0)
    assert np.allclose(published, expected)
    # And it inverts the node's own composition back to the commanded pose.
    assert np.allclose(ANCHOR[:3] + published[:3], bridge.chain.pos)


def test_hand_targets_are_the_first_22_dims_verbatim():
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, :22] = np.linspace(0.0, 0.5, 22)
    _, _, hand_sink, _, _ = run_loop(chunk)
    assert np.allclose(hand_sink.last, np.linspace(0.0, 0.5, 22))


def test_one_wild_action_trips_the_per_step_guard_immediately():
    """5 cm in a single 1/30 s step is ten times anything in the training data.

    It must stop on the step that produced it, not after it has accumulated into the
    travel envelope -- at these magnitudes the cumulative check would take ten steps, and
    at in-distribution magnitudes it would take a hundred.
    """
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 22] = 0.05
    with pytest.raises(arm_mod.WristEnvelopeError, match="single action"):
        run_loop(chunk, max_steps=200)


def test_a_slow_persistent_drift_trips_the_travel_envelope():
    """4 mm per step is INSIDE the per-step bound -- every action looks normal.

    Only the cumulative envelope catches this one, which is why both exist.
    """
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 22] = 0.004
    with pytest.raises(arm_mod.WristEnvelopeError, match="left the envelope"):
        run_loop(chunk, max_steps=1000)


def test_a_rejected_step_never_reaches_the_published_delta():
    a = make_args(max_steps=1000)
    bridge = arm_mod.ArmBridge(enabled=False, send_hz=1000.0,
                               max_lin=a.max_lin, max_ang=a.max_ang).start()
    bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 22] = 0.004
    try:
        with pytest.raises(arm_mod.WristEnvelopeError):
            main.control_loop(a, FakeSources(), bridge, hand_mod.NullHand(),
                              main.Inference(FakeClient(chunk), "p"), main.Recorder(None),
                              1.0 / a.fps, {"steps": 0})
        # The chain advanced past the limit; the payload the node is repeating did not.
        assert bridge.chain.travel()[0] > a.max_lin
        assert np.linalg.norm(bridge.published_delta[:3]) <= a.max_lin
    finally:
        bridge.stop()


def test_the_per_step_guard_leaves_the_chain_untouched():
    """A rejected step must not be integrated -- otherwise a stop still moves the target."""
    bridge = arm_mod.ArmBridge(enabled=False, send_hz=1000.0).start()
    bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
    try:
        with pytest.raises(arm_mod.WristEnvelopeError, match="single action"):
            bridge.step(np.array([0.05, 0.0, 0.0]), np.zeros(3))
        assert bridge.chain.n_steps == 0
        assert np.allclose(bridge.chain.pos, ANCHOR[:3])
    finally:
        bridge.stop()


def test_in_distribution_deltas_pass_the_per_step_guard():
    """The bound must not bite on real data. 4.6 mm / 1.3 deg is the training maximum."""
    bridge = arm_mod.ArmBridge(enabled=False, send_hz=1000.0).start()
    bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
    try:
        for _ in range(50):
            bridge.step(np.array([0.0046, 0.0, 0.0]), np.array([0.0, 0.0, 0.023]))
        assert bridge.chain.n_steps == 50
    finally:
        bridge.stop()


def test_a_non_finite_action_stops_the_rollout():
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[3, 24] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        run_loop(chunk)


def test_recorder_captures_one_row_per_step():
    chunk = np.zeros((main.ACTION_HORIZON, 28))
    chunk[:, 22] = 0.001
    a = make_args()
    src, client = FakeSources(), FakeClient(chunk)
    rec = main.Recorder("/dev/null")  # path only enables capture; save() is not called
    bridge = arm_mod.ArmBridge(enabled=False, send_hz=1000.0).start()
    bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
    try:
        main.control_loop(a, src, bridge, hand_mod.NullHand(), main.Inference(client, "p"), rec,
                1.0 / a.fps, {"steps": 0})
    finally:
        bridge.stop()
    assert len(rec.state) == len(rec.action) == len(rec.wrist_cmd) == a.max_steps
    assert np.asarray(rec.wrist_cmd)[-1][0] == pytest.approx(0.001 * a.max_steps)


def test_disabled_bridge_never_opens_a_socket():
    bridge = arm_mod.ArmBridge(enabled=False).start()
    try:
        assert not bridge.socket_open
        bridge.chain = actions.PalmChain.from_pose7(ANCHOR)
        bridge.step(np.array([0.01, 0.0, 0.0]), np.zeros(3))
        assert bridge.n_sent == 0
    finally:
        bridge.stop()


def test_step_before_engage_is_refused():
    bridge = arm_mod.ArmBridge(enabled=False)
    with pytest.raises(RuntimeError, match="before engage"):
        bridge.step(np.zeros(3), np.zeros(3))
