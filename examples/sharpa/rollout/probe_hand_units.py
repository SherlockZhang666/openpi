#!/usr/bin/env python3
"""Measure whether `SharpaWave.set_joint_position` wants radians or degrees.

    <rig>/.venv/bin/python examples/sharpa/rollout/probe_hand_units.py --read-only
    <rig>/.venv/bin/python examples/sharpa/rollout/probe_hand_units.py --i-am-watching

WHY THIS EXISTS
---------------
The two available sources disagree, and the disagreement is a factor of 57.3:

  * the training data is radians -- the rig's SCHEMA.md, and `sharpa_command.decide_units`
    measured the COMMAND wire as radians on 2026-08-31 from a live capture;
  * the vendor's own `sharpa_wave_example.py` drives `set_joint_position` from an
    ANGLE_RANGES table written in degrees.

Both can be true: the broadcast wire and the SDK call are different interfaces. Which one
this rollout has to convert for cannot be settled by reading either file, so it is
measured, once, on the hardware.

WHY THE PROBE IS SAFE IN BOTH DIRECTIONS
----------------------------------------
It commands ONE joint (index MCP flexion by default) to 0.30. That number is small under
either reading -- 0.30 rad is 17 degrees, within this joint's ~20 degree range, and 0.30
degrees is 0.005 rad, i.e. nothing -- so the probe cannot drive anything to a limit no
matter which way the answer comes out. That asymmetry is the whole measurement: the
readback (which is definitely radians) either follows to ~0.30 or does not move.

The reverse experiment -- commanding 17 to see if it is degrees -- is the one to avoid:
under the radian reading 17 rad is ten full turns, and it would be the firmware's clamp
standing between the probe and the hardware.

WHAT IT CHANGES ON THE ROBOT
----------------------------
Taking control means `set_control_source(ControlSource.SDK)`, which stops Pilot's GLOVE
source driving the hand. `--restore-source glove` puts it back on the way out; the probe
also drives the joint back to where it found it before releasing.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rig

# Index MCP flexion/extension. Position 5 of the 22 in both the rig's SCHEMA.md order and
# the SDK's JOINT_NAMES -- they are the same list. A single-DoF flexion of one finger is
# the least entangled motion available.
DEFAULT_JOINT = 5
DEFAULT_TARGET = 0.30
# set_joint_position's second argument. The vendor examples pass it positionally;
# named here so the call site says what the flag is.
INTERPOLATE = True
# Radians of readback movement. A radian command lands near 0.30, a degree command near
# 0.005, so the bands are an order of magnitude apart and the gap between them is an
# honest refusal rather than a coin flip.
RAD_MIN = 0.15
DEG_MAX = 0.05


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--read-only", action="store_true",
                   help="print the readback and exit; commands nothing, takes nothing")
    p.add_argument("--i-am-watching", action="store_true",
                   help="required to command: THE HAND WILL MOVE ONE FINGER")
    p.add_argument("--joint", type=int, default=DEFAULT_JOINT)
    p.add_argument("--target", type=float, default=DEFAULT_TARGET,
                   help="the value passed to set_joint_position, in whatever unit it "
                        "turns out to want. Must be small under BOTH readings.")
    p.add_argument("--settle", type=float, default=1.5, help="seconds to wait per command")
    p.add_argument("--serial", default=None)
    p.add_argument("--restore-source", choices=("glove", "none"), default="glove")
    p.add_argument("--rig-root", default=None)
    return p.parse_args(argv)


def read_hand(hand_state, label: str) -> np.ndarray:
    pos, _ = hand_state.latest()
    if pos is None:
        raise RuntimeError("no hand state")
    print(f"  {label}: " + " ".join(f"{v:+.3f}" for v in pos))
    return pos


def main(argv=None) -> int:
    a = parse_args(argv)
    rig.add_paths(a.rig_root)
    if not a.read_only:
        rig.require_hand_sdk()

    import hand as hand_mod

    serial = a.serial or hand_mod.DEFAULT_LEFT_SERIAL
    hand_state = hand_mod.HandState(serial=serial).start()
    hand_state.wait()
    print(f"hand {serial}: state broadcast live (radians, from the device itself)")
    baseline = read_hand(hand_state, "baseline")

    if a.read_only:
        return 0
    if not a.i_am_watching:
        print("\nrefusing to command without --i-am-watching. This moves a finger and "
              "takes the hand away from the glove.", file=sys.stderr)
        return 2
    if not (abs(a.target) <= 0.5):
        print(f"\nrefusing: --target {a.target} is not small under both readings. "
              f"Under the radian reading anything past ~0.5 starts to matter, and the "
              f"whole point of the probe is that it cannot hurt either way.", file=sys.stderr)
        return 2

    from sharpa import ControlMode
    from sharpa import ControlSource
    from sharpa import SharpaWaveManager

    manager = SharpaWaveManager.get_instance()
    time.sleep(1.0)
    devices = list(manager.get_all_device_sn())
    if serial not in devices:
        print(f"hand {serial} not among {devices}", file=sys.stderr)
        return 1
    hand = manager.connect(serial)

    def check(label, err):
        if getattr(err, "code", 0) != 0:
            raise RuntimeError(f"{label}: {getattr(err, 'message', err)}")

    check("control mode", hand.set_control_mode(ControlMode.POSITION))
    check("speed coeff", hand.set_speed_coeff(0.2))
    check("current coeff", hand.set_current_coeff(0.4))
    print("taking control: Pilot's GLOVE source stops driving the hand now")
    check("control source", hand.set_control_source(ControlSource.SDK))

    try:
        # Hold everything where it already is, then move only the probe joint. Commanding
        # a zero vector instead would open the whole hand, which is a much bigger motion
        # than this measurement needs.
        cmd = baseline.astype(float).copy()
        check("hold", hand.set_joint_position([float(v) for v in cmd], INTERPOLATE))
        time.sleep(a.settle)
        before = read_hand(hand_state, "holding ")

        cmd[a.joint] = float(a.target)
        print(f"commanding joint {a.joint} to {a.target}")
        check("probe", hand.set_joint_position([float(v) for v in cmd], INTERPOLATE))
        time.sleep(a.settle)
        after = read_hand(hand_state, "probed  ")

        moved = float(after[a.joint] - before[a.joint])
        others = float(np.max(np.abs(np.delete(after - before, a.joint))))
        print(f"\njoint {a.joint} moved {moved:+.4f} rad "
              f"(largest other joint {others:+.4f} rad)")

        if abs(moved) >= RAD_MIN:
            verdict = "rad"
            why = (f"the readback followed the command almost one-for-one "
                   f"({moved:+.3f} rad for a command of {a.target}), so the SDK is "
                   f"reading it as radians")
        elif abs(moved) <= DEG_MAX:
            verdict = "deg"
            why = (f"the readback barely moved ({moved:+.4f} rad), which is what "
                   f"{a.target} DEGREES = {np.deg2rad(a.target):.4f} rad looks like")
        else:
            verdict = None
            why = (f"{moved:+.4f} rad falls between the two bands "
                   f"(rad >= {RAD_MIN}, deg <= {DEG_MAX}). The joint may be blocked, or "
                   f"the speed coefficient may be too low for the settle time. Retry with "
                   f"a longer --settle before believing either answer.")
        print(("VERDICT: --hand-units " + verdict) if verdict else "VERDICT: inconclusive")
        print("  " + why)

        print("returning the joint to where it was found")
        check("restore", hand.set_joint_position([float(v) for v in baseline], INTERPOLATE))
        time.sleep(a.settle)
        read_hand(hand_state, "restored")
        return 0 if verdict else 3
    finally:
        if a.restore_source == "glove":
            print("restoring ControlSource.GLOVE")
            try:
                hand.set_control_source(ControlSource.GLOVE)
            except Exception as e:
                print(f"could not restore the control source: {e}", file=sys.stderr)
        manager.disconnect_all()
        hand_state.stop()


if __name__ == "__main__":
    sys.exit(main())
