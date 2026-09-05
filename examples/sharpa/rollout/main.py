#!/usr/bin/env python3
"""pi0.5 rollout on the OpenArm + Sharpa Wave left-hand rig.

    # policy server, in the openpi venv, on a machine with a GPU:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=sharpa_egg --policy.dir=<ckpt>/<step>

    # this client, in the rig venv, with ROS sourced:
    source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
    /home/yiming/openarm/openarm_track/vr/.venv/bin/python \
        examples/sharpa/rollout/main.py --prompt "pick up the egg"

NOTHING MOVES BY DEFAULT. The hand sink is `NullHand` and the arm bridge does not open a
socket until `--enable-hand` / `--enable-arm` are passed. A dry run exercises the whole
pipeline at full rate -- cameras, state, inference, action decoding, the safety envelope
-- and prints what it would have commanded. Do that first, every time.

READ THE ORDER OF `main()` BEFORE EDITING IT. The rclpy node is constructed before
GStreamer is initialised, because `collect/openarm_collect.py` records that h5py +
Gst.init() + Node construction is a three-way segfault and any two of them are fine.

CHUNK SCHEDULING
----------------
The model returns 30 actions at 30 Hz -- a second of motion -- and executing all of them
open loop is a second without feedback. This client executes `--chunk-steps` of them and
re-infers, with the next inference fired `--infer-lead` steps before the current chunk
runs out so the 30 Hz cadence is not interrupted by the forward pass.

That lead is also why the new chunk is consumed from index `--infer-lead` rather than 0.
The observation was captured `lead` steps before the new chunk starts executing, and
action[j] of a chunk means "the action `j` steps after the observation". Starting at 0
would replay `lead` steps of stale motion at every chunk boundary -- on this rig that is
not a small error, because the wrist part of the action is an INCREMENT: replaying it
would integrate that motion into the command chain twice.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rig

STATE_DIM = 29
N_ARM = 7
N_HAND = 22
ACTION_HORIZON = 30

logger = logging.getLogger("sharpa-rollout")


# ====================================================================== args
def arm_defaults():
    """The per-step wrist bounds, read from arm.py so the help text cannot drift from it.

    Imported lazily: `parse_args` runs before `rig.add_paths`, and arm.py is only importable
    from this directory (which it is, as the script's own directory) -- but keeping it out
    of the module body also keeps `--help` working with nothing else set up.
    """
    import arm

    return arm.DEFAULT_MAX_STEP_LIN, arm.DEFAULT_MAX_STEP_ANG


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True,
                   help="the task string, VERBATIM as the LeRobot dataset recorded it "
                        "(the dataset was built with prompt_from_task=True), e.g. "
                        "'pick up the egg'")
    p.add_argument("--host", default="127.0.0.1", help="policy server host")
    p.add_argument("--port", type=int, default=8000, help="policy server port")
    p.add_argument("--rig-root", default=None,
                   help="openarm_track/vr checkout (default $OPENARM_RIG_ROOT)")

    g = p.add_argument_group("actuation -- everything here is off by default")
    g.add_argument("--enable-arm", action="store_true",
                   help="actually send UDP to the teleop node. THE ARM WILL MOVE.")
    g.add_argument("--enable-hand", action="store_true",
                   help="actually command the hand over the SDK. THE FINGERS WILL MOVE, "
                        "and Pilot's control source is taken away from the glove.")
    g.add_argument("--hand-units", choices=("rad", "deg"), default=None,
                   help="units set_joint_position expects. Required with --enable-hand; "
                        "measure it once with probe_hand_units.py.")

    g = p.add_argument_group("teleop node")
    g.add_argument("--side", default="left", choices=("left", "right"))
    g.add_argument("--udp-host", default="127.0.0.1")
    g.add_argument("--udp-port", type=int, default=9873,
                   help="the node's --udp-port; collect_up.sh uses 9873 for the left arm")
    g.add_argument("--node-scale", type=float, default=1.0,
                   help="the node's --scale. Launch the node with --scale 1.0 and leave "
                        "this at 1.0: the dataset's wrist deltas are already in robot "
                        "space. If the node is still at the collection value of 0.5, pass "
                        "0.5 here and the translation is pre-divided to compensate.")
    g.add_argument("--send-hz", type=float, default=100.0)

    g = p.add_argument_group("cadence")
    g.add_argument("--fps", type=float, default=30.0, help="the rate the data was recorded at")
    # Measured with ping_policy.py on an RTX 5090 Laptop: p50 183 ms, p95 228 ms per
    # forward pass. A lead of 8 steps buys 267 ms, which covers p95 with room; the old
    # default of 4 (133 ms) stalled the loop at every single boundary. Re-measure on the
    # serving machine you actually use -- ping_policy.py prints the value to pass.
    g.add_argument("--chunk-steps", type=int, default=15,
                   help="actions executed per inference. Larger means less frequent "
                        "re-planning (15 at 30 Hz is half a second of open loop) but "
                        "leaves the GPU idle between passes.")
    g.add_argument("--infer-lead", type=int, default=8,
                   help="steps before the chunk ends at which the next inference starts, "
                        "and the index the returned chunk is consumed from. Must cover the "
                        "forward pass: lead/fps seconds, 267 ms at the defaults.")
    g.add_argument("--max-steps", type=int, default=0, help="0 = until 'q'")

    g = p.add_argument_group("safety envelope")
    g.add_argument("--max-lin", type=float, default=0.45,
                   help="metres the commanded wrist may travel from the engage pose")
    g.add_argument("--max-ang", type=float, default=1.8,
                   help="radians the commanded wrist may rotate from the engage pose")
    g.add_argument("--max-step-lin", type=float, default=arm_defaults()[0],
                   help="metres a SINGLE action may move the wrist. The training data "
                        "never exceeded 4.6 mm; the default is ~4x that.")
    g.add_argument("--max-step-ang", type=float, default=arm_defaults()[1],
                   help="radians a SINGLE action may rotate the wrist. The training data "
                        "never exceeded 0.023 rad; the default is ~4x that.")
    g.add_argument("--hand-max-rate", type=float, default=6.0, help="rad/s per finger joint")
    g.add_argument("--hand-clip", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="scalar radian envelope for every finger joint. Prefer "
                        "--hand-norm-stats, which is per joint and much tighter.")
    g.add_argument("--hand-norm-stats", default=None, metavar="PATH",
                   help="the checkpoint's assets/local_repo/norm_stats.json. Derives a "
                        "PER-JOINT command envelope from the action q01/q99 the policy was "
                        "trained on -- e.g. index_MCP_AA becomes [-0.28, +0.20] rad instead "
                        "of the firmware's +-pi/2. Strongly recommended.")
    g.add_argument("--hand-clip-margin", type=float, default=0.15,
                   help="radians added either side of the q01/q99 envelope")
    g.add_argument("--max-age", type=float, default=0.25,
                   help="seconds; any observation source staler than this stops the rollout")

    g = p.add_argument_group("cameras and hand")
    g.add_argument("--image-short-side", type=int, default=256,
                   help="downscale to this short side, as the LeRobot converter did. "
                        "0 keeps the sensor resolution.")
    g.add_argument("--hand-serial", default=None, help="default: the rig's left hand")
    g.add_argument("--hand-speed-coeff", type=float, default=0.3)
    g.add_argument("--hand-current-coeff", type=float, default=0.6)

    p.add_argument("--record", default=None, metavar="PATH",
                   help="write observations, actions and commands to this .npz on exit")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    if a.enable_hand and a.hand_units is None:
        p.error("--enable-hand requires --hand-units; run probe_hand_units.py first. "
                "Guessing is a factor of 57.3.")
    if a.infer_lead >= a.chunk_steps:
        p.error("--infer-lead must be smaller than --chunk-steps")
    if a.infer_lead + a.chunk_steps > ACTION_HORIZON:
        p.error(f"--infer-lead + --chunk-steps must fit in the {ACTION_HORIZON}-step horizon")
    return a


# ====================================================================== plumbing
@dataclasses.dataclass
class Observation:
    state: np.ndarray
    head: np.ndarray
    wrist: np.ndarray
    t: float


class Sources:
    """Everything the policy observes, plus the staleness rule that stops the rollout.

    Zero-order hold: each field is the newest sample that had arrived, which is exactly
    how the collector built a training row (one row per scene-camera frame, everything
    else sampled and held onto it). The staleness rule in `read` is the other half of it:
    the collector marked over-age rows invalid after the fact, and here an over-age row
    must not reach the policy at all, because there is nothing after the fact to fix.
    """

    def __init__(self, *, ros, hand_state, head_cam, wrist_cam, max_age: float):
        self.ros = ros
        self.hand_state = hand_state
        self.head_cam = head_cam
        self.wrist_cam = wrist_cam
        self.max_age = float(max_age)

    def ages(self) -> dict[str, float]:
        return {"joint_states": self.ros.joints.age(),
                "hand_state": self.hand_state.age(),
                "head_cam": self.head_cam.age(),
                "wrist_cam": self.wrist_cam.age()}

    def wait(self, timeout: float = 15.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            ages = self.ages()
            if all(a <= self.max_age for a in ages.values()):
                return
            time.sleep(0.05)
        bad = {k: v for k, v in self.ages().items() if v > self.max_age}
        raise RuntimeError(f"observation sources not live after {timeout:.0f}s: {bad}")

    def read(self, side: str) -> Observation:
        ages = self.ages()
        stale = {k: round(v, 3) for k, v in ages.items() if v > self.max_age}
        if stale:
            raise RuntimeError(f"stale observation (max_age={self.max_age}s): {stale}")

        joints = self.ros.joints.value
        # sources.JOINT_NAMES is openarm_{left,right}_joint1..7 in that order, so the left
        # arm is the first seven columns -- the same columns the converter selected with
        # `"left" in name` when it built `state[0:7]`.
        offset = 0 if side == "left" else N_ARM
        arm = np.asarray(joints["pos"], dtype=np.float64)[offset:offset + N_ARM]

        hand, _ = self.hand_state.latest()
        head, _ = self.head_cam.latest()
        wrist, _ = self.wrist_cam.latest()
        if hand is None or head is None or wrist is None:
            raise RuntimeError("an observation source went empty between the age check and the read")
        if not np.isfinite(arm).all():
            raise RuntimeError(f"/joint_states has NaN for the {side} arm: {arm}")

        state = np.concatenate([arm, hand[:N_HAND]]).astype(np.float32)
        if state.shape[0] != STATE_DIM:
            raise RuntimeError(f"state is {state.shape[0]}-d, expected {STATE_DIM}")
        return Observation(state=state, head=head, wrist=wrist, t=time.time())


class Inference:
    """One outstanding forward pass, run off the control thread.

    The 30 Hz loop must not block on the network or on the model. `submit` starts a pass,
    `take` collects it; if it is not done yet `take` blocks and says so, which is the
    signal to raise --infer-lead rather than something to tune away silently.
    """

    def __init__(self, client, prompt: str):
        self.client = client
        self.prompt = prompt
        self._thread: threading.Thread | None = None
        self._result: np.ndarray | None = None
        self._error: BaseException | None = None
        self._t0 = 0.0
        self.last_latency = float("nan")
        self.n_stalls = 0

    def submit(self, obs: Observation) -> None:
        if self._thread is not None:
            raise RuntimeError("an inference is already in flight")
        self._result = None
        self._error = None
        self._t0 = time.monotonic()
        payload = {"state": obs.state, "base": obs.head, "wrist": obs.wrist,
                   "prompt": self.prompt}
        self._thread = threading.Thread(target=self._run, args=(payload,),
                                        name="infer", daemon=True)
        self._thread.start()

    def _run(self, payload) -> None:
        try:
            out = self.client.infer(payload)
            self._result = np.asarray(out["actions"], dtype=np.float64)
        except BaseException as e:
            self._error = e

    @property
    def in_flight(self) -> bool:
        return self._thread is not None

    def take(self) -> np.ndarray:
        if self._thread is None:
            raise RuntimeError("take() with nothing in flight")
        if self._thread.is_alive():
            self.n_stalls += 1
        self._thread.join()
        self._thread = None
        self.last_latency = time.monotonic() - self._t0
        if self._error is not None:
            raise self._error
        chunk = self._result
        if chunk is None or chunk.ndim != 2 or chunk.shape[1] != 28:
            raise RuntimeError(f"policy returned {None if chunk is None else chunk.shape}, "
                               f"expected (horizon, 28)")
        if not np.isfinite(chunk).all():
            raise RuntimeError("policy returned a non-finite action chunk")
        return chunk


@dataclasses.dataclass
class ChunkSchedule:
    """Which index of the current chunk to execute, and when to fire the next inference.

    Pulled out of the loop because the offset is the part that is easy to get silently
    wrong, and wrong here means every chunk boundary replays `lead` steps of stale wrist
    INCREMENTS -- motion that is integrated into the command chain a second time. See
    `schedule_test.py`, which pins the index sequence rather than trusting the reasoning.

    The first chunk is consumed from 0 (its observation is current). Every later chunk was
    requested `lead` steps before it starts executing, so it is consumed from `lead`.
    Either way exactly `chunk_steps` actions are executed per inference.
    """

    chunk_steps: int
    lead: int
    idx: int = 0
    end: int = 0
    started: bool = False

    def at_boundary(self) -> bool:
        """True when a new chunk must be taken before anything can be executed."""
        return not self.started or self.idx >= self.end

    def on_new_chunk(self) -> None:
        if not self.started:
            self.idx, self.end, self.started = 0, self.chunk_steps, True
        else:
            self.idx, self.end = self.lead, self.lead + self.chunk_steps

    def submit_now(self) -> bool:
        """Fire the next inference `lead` steps before the current chunk runs out."""
        return self.lead > 0 and self.idx == self.end - self.lead

    def advance(self) -> None:
        self.idx += 1


class Keys:
    """Non-blocking single keypresses, when there is a terminal to read them from."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old = None

    def __enter__(self):
        if self.enabled:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def get(self) -> str | None:
        if not self.enabled:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Recorder:
    """Optional flight recorder. Debugging a rollout without one is guesswork."""

    def __init__(self, path: str | None):
        self.path = pathlib.Path(path) if path else None
        self.state, self.action, self.hand_cmd, self.wrist_cmd, self.t = [], [], [], [], []

    def add(self, obs, action, hand_cmd, wrist_delta) -> None:
        if self.path is None:
            return
        self.state.append(obs.state)
        self.action.append(action)
        self.hand_cmd.append(hand_cmd)
        self.wrist_cmd.append(wrist_delta)
        self.t.append(obs.t)

    def save(self) -> None:
        if self.path is None or not self.t:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path,
                            state=np.asarray(self.state, dtype=np.float32),
                            action=np.asarray(self.action, dtype=np.float32),
                            hand_cmd=np.asarray(self.hand_cmd, dtype=np.float32),
                            wrist_delta=np.asarray(self.wrist_cmd, dtype=np.float32),
                            t_wall=np.asarray(self.t, dtype=np.float64))
        logger.info("recorded %d steps -> %s", len(self.t), self.path)


# ====================================================================== main
def main(argv=None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    rig.add_paths(a.rig_root)
    rig.require_ros()
    rig.check_client_deps()
    if a.enable_hand:
        rig.require_hand_sdk()

    import actions
    import arm as arm_mod
    import cameras as cam_mod
    import hand as hand_mod
    from openpi_client import websocket_client_policy
    import rclpy
    from sources import RosSources

    if not (a.enable_arm and a.enable_hand):
        logger.warning("DRY RUN: arm=%s hand=%s -- nothing will be commanded",
                       "on" if a.enable_arm else "OFF",
                       "on" if a.enable_hand else "OFF")

    dt = 1.0 / a.fps
    short_side = a.image_short_side or None

    # ORDER: rclpy node first, GStreamer second. See the module docstring.
    rclpy.init()
    node = rclpy.create_node("sharpa_rollout")
    ros = RosSources(node, max_age=a.max_age, sides=(a.side,))
    spinning = threading.Event()

    def spin():
        while not spinning.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)

    spin_thread = threading.Thread(target=spin, name="rclpy-spin", daemon=True)
    spin_thread.start()

    head_cam = wrist_cam = None
    hand_state = None
    hand_sink: hand_mod.HandCommander = hand_mod.NullHand()
    bridge = None
    recorder = Recorder(a.record)
    # Mutable so the step count survives an exception out of the loop -- the number of
    # steps that ran before a safety stop is the first thing you want to know.
    stats = {"steps": 0}
    try:
        logger.info("opening cameras ...")
        head_cam = cam_mod.head_grabber(short_side=short_side).start()
        wrist_cam = cam_mod.wrist_grabber(short_side=short_side).start()
        logger.info("head  %s -> %dx%d", head_cam.device, head_cam.out_width, head_cam.out_height)
        logger.info("wrist %s -> %dx%d", wrist_cam.device, wrist_cam.out_width, wrist_cam.out_height)

        hand_state = hand_mod.HandState(serial=a.hand_serial or hand_mod.DEFAULT_LEFT_SERIAL).start()
        hand_state.wait()

        src = Sources(ros=ros, hand_state=hand_state, head_cam=head_cam,
                      wrist_cam=wrist_cam, max_age=a.max_age)
        src.wait()
        logger.info("all observation sources live: %s",
                    {k: round(v, 3) for k, v in src.ages().items()})

        clip = hand_mod.DEFAULT_CLIP
        if a.hand_norm_stats:
            clip = hand_mod.limits_from_norm_stats(a.hand_norm_stats, a.hand_clip_margin)
            logger.info("hand envelope from %s: %.3f..%.3f rad across the 22 joints",
                        a.hand_norm_stats, float(clip[0].min()), float(clip[1].max()))
        elif a.hand_clip:
            clip = tuple(a.hand_clip)
        elif a.enable_hand:
            logger.warning("no --hand-norm-stats: falling back to a scalar +-pi/2 envelope, "
                           "which is the firmware clamp rather than anything this policy "
                           "was trained inside")

        if a.enable_hand:
            seed, _ = hand_state.latest()
            hand_sink = hand_mod.SdkHand(
                serial=a.hand_serial or hand_mod.DEFAULT_LEFT_SERIAL,
                units=a.hand_units, dt=dt, seed=seed,
                speed_coeff=a.hand_speed_coeff, current_coeff=a.hand_current_coeff,
                max_rate=a.hand_max_rate, clip=clip)

        client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
        logger.info("policy server metadata: %s", client.get_server_metadata())

        bridge = arm_mod.ArmBridge(side=a.side, host=a.udp_host, port=a.udp_port,
                                   node_scale=a.node_scale, send_hz=a.send_hz,
                                   enabled=a.enable_arm, max_lin=a.max_lin,
                                   max_ang=a.max_ang, max_step_lin=a.max_step_lin,
                                   max_step_ang=a.max_step_ang).start()
        if a.enable_arm:
            bridge.engage(ros.telemetry[a.side])
        else:
            offset = 0 if a.side == "left" else N_ARM
            q = np.asarray(ros.joints.value["pos"], float)[offset:offset + N_ARM]
            bridge.chain = actions.PalmChain.from_pose7(dry_run_anchor(a.side, q))
            logger.info("dry run: anchor FK(q_meas) = %s",
                        np.array2string(bridge.chain.pose7, precision=4, suppress_small=True))

        inference = Inference(client, a.prompt)
        control_loop(a, src, bridge, hand_sink, inference, recorder, dt, stats)
        return 0
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except arm_mod.WristEnvelopeError as e:
        logger.error("SAFETY STOP: %s", e)
        logger.error("The node holds the last valid target; nothing past the envelope was "
                     "ever sent. Inspect the --record file before re-running.")
        return 1
    finally:
        logger.info("stopping after %d steps", stats["steps"])
        if bridge is not None:
            bridge.stop()
        hand_sink.close()
        for cam in (head_cam, wrist_cam):
            if cam is not None:
                cam.stop()
        if hand_state is not None:
            hand_state.stop()
        recorder.save()
        spinning.set()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
        if a.enable_arm:
            logger.warning("the arm is in HOLD, not home. Put it away with "
                           "<rig>/scripts/after_teleop_home.sh")


def dry_run_anchor(side: str, q_meas: np.ndarray) -> np.ndarray:
    """The pose the node WOULD anchor at, computed here so a dry run needs no engagement.

    The node publishes `target_pose` as NaN until something engages it -- `target_filt`
    does not exist before then -- so a dry run cannot read the anchor off the telemetry.
    It can compute it, because the node's own engage is
    `anchor_pose = FK(q_iter)` with `q_iter` just re-seeded from `q_meas`. So FK of the
    measured joint state IS the anchor, and it comes from the rig's own `arm_model`, the
    same kinematics the node uses.

    Getting this right matters more than it looks: the anchor's ROTATION is what maps the
    policy's palm-frame translation deltas into base axes. A placeholder identity anchor
    would make every dry-run number point in the wrong direction while looking plausible.
    """
    from arm_model import ArmModel
    import pinocchio as pin

    pose = ArmModel(side).fk(np.asarray(q_meas, float))
    quat = pin.Quaternion(pose.rotation)
    return np.array([*pose.translation, quat.w, quat.x, quat.y, quat.z], dtype=np.float64)


def control_loop(a, src, bridge, hand_sink, inference, recorder, dt, stats) -> None:
    """The 30 Hz control loop. Counts executed steps into `stats`."""
    import actions

    chunk = None
    sched = ChunkSchedule(chunk_steps=a.chunk_steps, lead=a.infer_lead)
    steps = 0
    paused = False
    next_t = time.monotonic()

    with Keys() as keys:
        logger.info("running -- 'q' quit, 'p' pause/resume")
        while True:
            key = keys.get()
            if key == "q":
                logger.info("quit")
                break
            if key == "p":
                paused = not paused
                if paused:
                    logger.info("PAUSED -- the arm holds its last target")
                else:
                    # Throw away anything computed before the pause and start a fresh
                    # chunk. An in-flight forward pass is conditioned on an observation
                    # from before the pause, and its wrist deltas would be executed as if
                    # no time had passed -- on a pause of any length that is the scene
                    # having moved on without the policy.
                    if inference.in_flight:
                        inference.take()
                    sched = ChunkSchedule(chunk_steps=a.chunk_steps, lead=a.infer_lead)
                    chunk = None
                    logger.info("resumed -- re-inferring from a current observation")
            if paused:
                # Do not advance the chain; the UDP repeater keeps the last delta alive,
                # which is what holds the arm rather than dropping the engagement.
                time.sleep(0.05)
                next_t = time.monotonic()
                continue

            obs = src.read(a.side)

            if sched.at_boundary():
                first = not sched.started
                stalls = inference.n_stalls
                if not inference.in_flight:
                    # The first chunk, --infer-lead 0, or a prefetch that never fired.
                    # Synchronous: the loop will overrun by one inference, and the overrun
                    # warning below is the honest place for that to show up.
                    inference.submit(obs)
                chunk = inference.take()
                sched.on_new_chunk()
                if first:
                    logger.info("first chunk in %.0f ms", inference.last_latency * 1e3)
                elif inference.n_stalls > stalls:
                    logger.warning("inference was not ready at the chunk boundary "
                                   "(%.0f ms, %d so far) -- raise --infer-lead",
                                   inference.last_latency * 1e3, inference.n_stalls)

            if sched.submit_now() and not inference.in_flight:
                inference.submit(obs)

            action = chunk[sched.idx]
            hand_target, dp, dr = actions.split_action(action)
            hand_cmd = hand_sink.send(hand_target)
            wrist_delta = bridge.step(dp, dr)
            recorder.add(obs, action, hand_cmd, wrist_delta)

            sched.advance()
            steps += 1
            stats["steps"] = steps
            if steps % max(1, int(a.fps)) == 0:
                lin, ang = bridge.chain.travel()
                logger.info("step %5d | wrist %+.3f %+.3f %+.3f m from anchor "
                            "(|d|=%.3f m, %.1f deg) | infer %.0f ms",
                            steps, *(bridge.chain.pos - bridge.chain.anchor_pos),
                            lin, np.rad2deg(ang), inference.last_latency * 1e3)
            if a.max_steps and steps >= a.max_steps:
                logger.info("reached --max-steps")
                break

            next_t += dt
            slack = next_t - time.monotonic()
            if slack < -dt:
                logger.warning("control loop overran by %.0f ms", -slack * 1e3)
                next_t = time.monotonic()
            else:
                time.sleep(max(0.0, slack))


if __name__ == "__main__":
    sys.exit(main())
