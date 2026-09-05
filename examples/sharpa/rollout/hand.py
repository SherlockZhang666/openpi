"""Commanding the Sharpa Wave left hand's 22 joints, and reading them back.

READBACK comes from the hand's own 500 Hz broadcast on UDP :50000, via the rig's
`collect/sharpa_state.py`. That is the exact source of `hand/joint_pos` in the training
data, it is in radians, and binding a broadcast port takes nothing away from Sharpa-pilot.
Nothing here can disturb it.

COMMANDING is the part that needs care, for three separate reasons.

1. THE CONTROL SOURCE HAS TO CHANGE. Data was collected with Pilot's Control Source set to
   `GLOVE`: the retargeter's WAVE packets on :50020 drive the hand. A policy that simply
   calls the SDK while Pilot is still in GLOVE mode will be ignored -- the SDK header says
   the control source "is used to filter udp packet by content". So `SdkHand` sets
   ControlSource.SDK, and that is a mode change on live hardware: the glove stops driving
   the hand the moment it happens.

2. THE UNITS ARE NOT SETTLED, AND THE ERROR IS A FACTOR OF 57.3. The dataset is radians
   (the rig's SCHEMA.md, and `sharpa_command.decide_units` measured the command wire as
   radians on 2026-08-31). But the vendor's own `sharpa_wave_example.py` drives
   `set_joint_position` with an ANGLE_RANGES table written in degrees. Those cannot both
   describe the same call. Rather than pick one, `units` has NO DEFAULT: run
   `probe_hand_units.py` once on the hardware and pass what it measured. Guessing here
   either does nothing at all (rad values read as degrees) or drives every joint to its
   limit at once (degree values read as radians).

3. THE FIRST COMMAND IS THE DANGEROUS ONE. Whatever the fingers are doing when a rollout
   starts, the policy's first action is a step change. `max_rate` slews toward the target
   instead, so a bad chunk cannot slam the hand, and `clip` bounds the absolute command
   regardless of what the network emitted.

The joint order needs no remapping: the dataset's 22 (rig SCHEMA.md) and the SDK's
JOINT_NAMES are the same list -- thumb CMC_FE, CMC_AA, MCP_FE, MCP_AA, DIP, then index,
middle, ring, pinky. Checked name by name, not assumed.
"""

from __future__ import annotations

import abc
import logging
import pathlib
import time

import numpy as np

N_HAND = 22
# The left hand on this rig; `sharpa_command._build_wave` carries the same serial as its
# packer default, and it is the one whose thumb_CMC_FE dropped out in 22 episodes.
DEFAULT_LEFT_SERIAL = "C5549233C555"
# Radian envelope on any commanded joint, NOT per-joint limits. It is set to the
# firmware's own clamp (limit_joint_angles, +-pi/2) so that a diverged chunk is stopped
# here rather than relying on the far end to stop it. The real joint ranges are much
# narrower -- the vendor's ANGLE_RANGES table tops out around 50 deg -- so tightening
# this with --hand-clip once the training data's actual per-joint spans are to hand is
# a strict improvement. The rate limit below is what protects the first command.
DEFAULT_CLIP = (-np.pi / 2, np.pi / 2)
# rad/s. A finger flex is ~1.5 rad and the rig ran at 30 Hz, so real motion is well under
# this; it only bites on a step change.
DEFAULT_MAX_RATE = 6.0

logger = logging.getLogger(__name__)


class HandCommander(abc.ABC):
    """Sink for 22 joint targets in radians."""

    @abc.abstractmethod
    def send(self, angles_rad: np.ndarray) -> np.ndarray:
        """Command the hand; returns what was actually sent, in radians."""

    def close(self) -> None:
        return


class NullHand(HandCommander):
    """Accepts commands and sends nothing. The default, so a mistake costs a log line."""

    def __init__(self):
        self.last: np.ndarray | None = None
        self.n = 0

    def send(self, angles_rad: np.ndarray) -> np.ndarray:
        a = np.asarray(angles_rad, dtype=np.float64).reshape(N_HAND)
        self.last = a
        self.n += 1
        return a


def limits_from_norm_stats(path, margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint command bounds, read off the checkpoint's own action statistics.

    `assets/local_repo/norm_stats.json` inside the checkpoint carries q01/q99 of the 28-d
    action over the training set, and the first 22 of those ARE the finger commands. So
    the policy's own training distribution defines a far tighter envelope than the
    firmware clamp: index_MCP_AA, for instance, spans [-0.13, +0.04] rad in the data while
    a scalar +-pi/2 clip would let a diverged chunk drive it to a hard abduction.

    `margin` widens each side, because q01/q99 are percentiles rather than extremes -- the
    envelope is there to stop divergence, not to censor the tails of normal behaviour. The
    result is still intersected with the firmware clamp, so this can only ever tighten.
    """
    import json

    stats = json.loads(pathlib.Path(path).read_text())["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)[:N_HAND]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[:N_HAND]
    if q01.shape != (N_HAND,):
        raise ValueError(f"{path}: expected at least {N_HAND} action dims, got {q01.shape}")
    lo = np.maximum(q01 - margin, DEFAULT_CLIP[0])
    hi = np.minimum(q99 + margin, DEFAULT_CLIP[1])
    return lo, hi


class Slew:
    """Rate limit and clip, shared by every real sink. Public so it can be tested alone.

    `clip` is a (lo, hi) pair; each side is a scalar or a per-joint array of 22.
    """

    def __init__(self, *, max_rate: float, clip, dt: float):
        self.step = float(max_rate) * float(dt)
        self.lo = np.broadcast_to(np.asarray(clip[0], dtype=np.float64), (N_HAND,)).copy()
        self.hi = np.broadcast_to(np.asarray(clip[1], dtype=np.float64), (N_HAND,)).copy()
        if (self.lo > self.hi).any():
            raise ValueError("hand clip has lo > hi on some joint")
        self.current: np.ndarray | None = None

    def seed(self, angles_rad) -> None:
        self.current = np.clip(np.asarray(angles_rad, dtype=np.float64).reshape(N_HAND),
                               self.lo, self.hi)

    def __call__(self, target) -> np.ndarray:
        t = np.asarray(target, dtype=np.float64).reshape(N_HAND)
        if not np.isfinite(t).all():
            raise ValueError(f"non-finite hand command: {t}")
        t = np.clip(t, self.lo, self.hi)
        if self.current is None:
            self.current = t
            return t
        self.current = self.current + np.clip(t - self.current, -self.step, self.step)
        return self.current.copy()


class SdkHand(HandCommander):
    """The real thing, over the vendor SDK.

    `units` is required and has no default -- see the module docstring. `seed` should be
    the hand's current readback so the first command is a hold, not a jump.
    """

    def __init__(self, *, serial: str = DEFAULT_LEFT_SERIAL, units: str,
                 dt: float, seed: np.ndarray | None = None,
                 speed_coeff: float = 0.3, current_coeff: float = 0.6,
                 interpolate: bool = True,
                 max_rate: float = DEFAULT_MAX_RATE,
                 clip=DEFAULT_CLIP):
        if units not in ("rad", "deg"):
            raise ValueError(
                "hand units must be 'rad' or 'deg', measured with probe_hand_units.py. "
                "There is no safe default: see hand.py's module docstring.")
        from sharpa import ControlMode
        from sharpa import ControlSource
        from sharpa import SharpaWaveManager

        self.units = units
        self.interpolate = bool(interpolate)
        self._slew = Slew(max_rate=max_rate, clip=clip, dt=dt)
        if seed is not None:
            self._slew.seed(seed)

        manager = SharpaWaveManager.get_instance()
        time.sleep(1.0)  # device discovery; the vendor examples do the same
        devices = list(manager.get_all_device_sn())
        if serial not in devices:
            raise RuntimeError(f"hand {serial} not among the discovered devices {devices}")
        self.hand = manager.connect(serial)
        self.serial = serial
        self._manager = manager

        for label, call in (
            ("control mode", lambda: self.hand.set_control_mode(ControlMode.POSITION)),
            ("speed coeff", lambda: self.hand.set_speed_coeff(speed_coeff)),
            ("current coeff", lambda: self.hand.set_current_coeff(current_coeff)),
            # LAST, deliberately: it is the step that takes the hand away from the glove,
            # so everything else is already configured when it happens.
            ("control source", lambda: self.hand.set_control_source(ControlSource.SDK)),
        ):
            err = call()
            if getattr(err, "code", 0) != 0:
                raise RuntimeError(f"failed to set {label}: {getattr(err, 'message', err)}")
        logger.info("hand %s: POSITION / ControlSource.SDK, units=%s", serial, units)

    def send(self, angles_rad: np.ndarray) -> np.ndarray:
        cmd = self._slew(angles_rad)
        wire = np.rad2deg(cmd) if self.units == "deg" else cmd
        err = self.hand.set_joint_position([float(v) for v in wire], self.interpolate)
        if getattr(err, "code", 0) != 0:
            raise RuntimeError(f"set_joint_position failed: {getattr(err, 'message', err)}")
        return cmd

    def close(self) -> None:
        """Stops commanding. Deliberately does NOT drive the fingers anywhere.

        Sending zeros here -- which the T-Rex client does at shutdown -- would open the
        hand and drop whatever it is holding, at the least predictable moment. Leaving the
        last command in place holds the grasp; putting the hand away is the operator's
        step, along with putting the arm away.
        """
        try:
            self._manager.disconnect_all()
        except Exception:
            logger.exception("hand disconnect failed")


class HandState:
    """The 22-DoF readback, in radians, from the hand's own broadcast."""

    def __init__(self, serial: str | None = DEFAULT_LEFT_SERIAL, port: int = 50000):
        from sharpa_state import StateReceiver  # rig module; see rig.add_paths

        self._rx = StateReceiver(port=port, serial=serial)

    def start(self) -> HandState:
        self._rx.start()
        return self

    def wait(self, timeout: float = 5.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._rx.n_packets:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"no hand state on UDP :{self._rx.port} after {timeout:.0f}s "
            f"(saw serial {self._rx.serial_seen!r}, {self._rx.n_rejected} rejected). "
            f"Is the hand powered and on 192.168.10.10?")

    def latest(self) -> tuple[np.ndarray | None, float]:
        got, t = self._rx.latest()
        return (None, t) if got is None else (np.asarray(got[0], dtype=np.float64), t)

    def age(self) -> float:
        _, t = self._rx.latest()
        return float("inf") if not np.isfinite(t) else time.time() - t

    def stop(self) -> None:
        self._rx.stop()
