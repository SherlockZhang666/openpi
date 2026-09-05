"""Driving the left arm by impersonating the Quest bridge.

The policy does NOT talk to the controller. It talks to `openarm_vr_teleop.py`, over the
same UDP/JSON the Quest bridge uses, and that node keeps doing everything it does during
teleoperation: differential IK with nullspace posture, the soft joint-limit envelope, the
open-loop `q_iter` integration, the gravity feed-forward, the per-joint rate limit, the
deviation trip and its RAMP_BACK. Publishing to `<side>_forward_position_controller`
directly would mean reimplementing all eight of those on live hardware, and the reason
they exist is that each one of them has already caught something.

So this file is small on purpose: it owns the wire, the engage handshake and the safety
envelope on top of the policy's own output. The kinematics belong to the node.

THE WIRE
--------
    {"left_wrist": [x,y,z, qw,qx,qy,qz], "engaged": true, "seq": n}

sent to 127.0.0.1:9873 (the node's --udp-port for the left arm in collect_up.sh). The
pose is the delta SINCE ENGAGE in base axes, not an absolute pose -- see actions.py for
how `PalmChain.base_delta` inverts the node's `apply_base_delta`. The quaternion is
w-first, matching the rest of the rig.

The datagram is repeated by a background thread at `send_hz`, independently of how fast
the policy produces actions. Two reasons: the node runs its loop at 200 Hz and disengages
after `--stale-drop` seconds without a frame (default 1.0), and inference between chunks
takes long enough that a send-on-new-action-only design would starve it. Repeating the
last delta means an inference stall holds the arm at the last commanded target instead of
dropping the engagement.

THE ENGAGE HANDSHAKE
--------------------
The node latches its anchor itself: on the rising edge of `engaged` it sets
`anchor_pose = FK(q_iter)` and `target_filt = anchor_pose`. So the client must not guess
the anchor -- it sends `engaged` with a zero delta (which commands exactly the anchor,
i.e. no motion), waits for the telemetry to report state ENGAGED, and seeds the chain
from the `target_pose` the node publishes. Anything else and the policy's deltas are
measured from a pose the node is not holding.

The node can also REFUSE to engage, when the arm is more than `--engage-gap` from its hold
anchor. That shows up as telemetry `engaged`=1 with `state` != ENGAGED, and `engage()`
reports it rather than timing out silently.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time

from actions import PalmChain
import numpy as np

# openarm_vr_teleop.STATE_CODE. Mirrored rather than imported: that module pulls in
# rclpy and pinocchio at import time, and `collect/sources.py` already keeps its own copy
# of the same table for the same reason.
STATE_CODE = {"PRE_SWITCH": 0, "HOLD": 1, "ENGAGED": 2, "RAMP_BACK": 3, "HOMING": 4}
ENGAGED = STATE_CODE["ENGAGED"]

logger = logging.getLogger(__name__)

# Per-step bounds on a single wrist increment, at roughly 4x what the training data ever
# contained. From the checkpoint's own action q01/q99 (assets/local_repo/norm_stats.json):
# the largest per-step translation is 4.6 mm (13.7 cm/s at 30 Hz) and the largest rotation
# is 1.3 deg (39 deg/s). Anything several times that is out of distribution, and catching
# it on the step that produces it is worth a lot more than catching it after it has
# accumulated into the travel envelope: the cumulative check needs ~100 bad steps at these
# magnitudes, which at 30 Hz is three seconds of the arm already going the wrong way.
DEFAULT_MAX_STEP_LIN = 0.02    # m per action
DEFAULT_MAX_STEP_ANG = 0.10    # rad per action


class WristEnvelopeError(RuntimeError):
    """The integrated command left the allowed envelope around the engage pose."""


class ArmBridge:
    """Integrates the policy's wrist deltas and streams them to the teleop node.

    `enabled=False` runs everything -- the chain, the envelope, the packet -- but never
    opens the socket. That is the default in `main.py`: a dry run tells you exactly what
    would have been commanded, at full rate, with the robot untouched.
    """

    def __init__(self, *, side: str = "left", host: str = "127.0.0.1", port: int = 9873,
                 node_scale: float = 1.0, send_hz: float = 100.0, enabled: bool = False,
                 max_lin: float = 0.45, max_ang: float = 1.8,
                 max_step_lin: float = DEFAULT_MAX_STEP_LIN,
                 max_step_ang: float = DEFAULT_MAX_STEP_ANG):
        self.side = side
        self.key = f"{side}_wrist"
        self.addr = (host, int(port))
        self.node_scale = float(node_scale)
        self.send_hz = float(send_hz)
        self.enabled = bool(enabled)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.max_step_lin = float(max_step_lin)
        self.max_step_ang = float(max_step_ang)

        self.chain: PalmChain | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._delta = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._engaged = False
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_sent = 0

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> ArmBridge:
        if self.enabled:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(target=self._run, name="arm-udp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.disengage()
        # One last beat at the send rate so the disengage actually reaches the node
        # before the socket closes; it is what puts the node into HOLD.
        time.sleep(max(3.0 / self.send_hz, 0.05))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _run(self) -> None:
        period = 1.0 / self.send_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                payload = {self.key: [float(v) for v in self._delta],
                           "engaged": bool(self._engaged),
                           "seq": self._seq}
                self._seq += 1
            if self._sock is not None:
                try:
                    self._sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
                    self.n_sent += 1
                except OSError as e:
                    logger.warning("UDP send failed: %s", e)
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    # ---------------------------------------------------------------- engage
    def engage(self, telemetry, *, timeout: float = 5.0) -> np.ndarray:
        """Request engagement and seed the chain from the node's own target pose.

        `telemetry` is `sources.RosSources.telemetry[side]`; the caller must be spinning
        the rclpy node, otherwise nothing ever arrives and this reports a stale source
        rather than a refusal.
        """
        with self._lock:
            self._delta = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self._engaged = True

        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < timeout:
            tel = telemetry.value if telemetry.valid else None
            if tel is not None:
                last = tel
                if int(round(tel["state"])) == ENGAGED:
                    pose = np.asarray(tel["target_pose"], dtype=np.float64)
                    self.chain = PalmChain.from_pose7(pose)
                    logger.info("engaged; anchor %s",
                                np.array2string(pose, precision=4, suppress_small=True))
                    return pose
            time.sleep(0.02)

        self.disengage()
        if last is None:
            raise RuntimeError(
                f"no telemetry on /openarm_teleop/{self.side}/state -- is the node running "
                f"with --telemetry, and is this process spinning rclpy?")
        name = next((k for k, v in STATE_CODE.items() if v == int(round(last["state"]))),
                    str(last["state"]))
        raise RuntimeError(
            f"the node did not engage within {timeout:.0f}s; it is in {name}. It refuses "
            f"to engage when the arm is further than --engage-gap from its hold anchor "
            f"(check the teleop log), and it cannot engage at all from PRE_SWITCH.")

    def disengage(self) -> None:
        with self._lock:
            self._engaged = False

    # ---------------------------------------------------------------- stepping
    def step(self, dp, dr) -> np.ndarray:
        """Advance the commanded pose by one action and publish it. Returns the delta."""
        if self.chain is None:
            raise RuntimeError("step() before engage(): there is no anchor to integrate from")

        step_lin = float(np.linalg.norm(np.asarray(dp, dtype=np.float64)))
        step_ang = float(np.linalg.norm(np.asarray(dr, dtype=np.float64)))
        if step_lin > self.max_step_lin or step_ang > self.max_step_ang:
            raise WristEnvelopeError(
                f"single action moves the wrist {step_lin * 1000:.1f} mm / "
                f"{np.rad2deg(step_ang):.1f} deg in one 1/30 s step (limits "
                f"{self.max_step_lin * 1000:.0f} mm / {np.rad2deg(self.max_step_ang):.0f} deg). "
                f"The training data never exceeded 4.6 mm / 1.3 deg, so this is the policy "
                f"out of distribution, not fast motion.")

        self.chain.step(dp, dr)
        lin, ang = self.chain.travel()
        if lin > self.max_lin or ang > self.max_ang:
            raise WristEnvelopeError(
                f"commanded wrist left the envelope: {lin * 100:.1f} cm / {np.rad2deg(ang):.0f} deg "
                f"from the engage pose (limits {self.max_lin * 100:.0f} cm / "
                f"{np.rad2deg(self.max_ang):.0f} deg)")
        delta = self.chain.base_delta(scale=self.node_scale)
        with self._lock:
            self._delta = delta
        return delta

    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._engaged

    @property
    def published_delta(self) -> np.ndarray:
        """The delta the repeater thread is currently sending. Never a rejected one: a
        step that trips the envelope raises before it assigns, so the node keeps holding
        the last target that passed."""
        with self._lock:
            return self._delta.copy()

    @property
    def socket_open(self) -> bool:
        return self._sock is not None
