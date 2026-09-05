"""Makes the OpenArm rig's own modules importable, instead of reimplementing them.

Everything this client needs on the observation side already exists in the collection
stack and was used to record the training data:

    collect/sources.py       RosSources  -- /joint_states and the teleop telemetry
    collect/sharpa_state.py  StateReceiver -- the hand's 22-DoF broadcast, radians
    collect/camera.py        resolve_device / Camera -- which /dev/videoN, and the
                             exact v4l2 control string the episodes were recorded with

Reusing them is not laziness. `obs/joint_pos` and `hand/joint_pos` in the dataset ARE the
output of these two readers, so importing them is what guarantees the rollout's state
vector is assembled the same way as the training rows. A second implementation would be a
second chance to get the joint order, the units or the ZOH semantics wrong.

The rig lives outside this repository, so its location is a setting:

    OPENARM_RIG_ROOT   default /home/yiming/openarm/openarm_track/vr

`sources.py` and `sharpa_state.py` import their siblings by bare name (`from
episode_writer import ...`), so `<rig>/collect` goes on the path, not just `<rig>`.

openpi-client is added from this repository's own `packages/` rather than pip-installed.
The robot venv is a working real-hardware environment carrying rclpy, pinocchio and
GStreamer bindings; openpi-client declares `numpy<2.0` and that venv runs numpy 2.5, so a
plain `pip install` would try to downgrade numpy underneath the control stack. The three
modules actually used here -- websocket_client_policy, msgpack_numpy, base_policy -- need
only numpy, msgpack and websockets, and are numpy-2 clean.

Two of those three do have to come from somewhere, though, and `check_client_deps` is
where that is caught. The rig venv inherits websockets 10.4 from the system packages, and
`websockets.sync` -- the blocking client openpi-client uses -- only exists from 12.0. That
surfaces as a bare `ModuleNotFoundError: websockets.sync` well after the cameras are open,
so it is checked up front and answered with the exact command rather than installed
behind the operator's back. Installing into the venv shadows the system copy inside it and
touches nothing else: no other module in the rig checkout imports websockets.
"""

from __future__ import annotations

import os
import pathlib
import sys

DEFAULT_RIG_ROOT = "/home/yiming/openarm/openarm_track/vr"
# The vendor SDK, installed by `dpkg -i sharpa-wave-sdk_*.deb`. It is NOT importable from
# the rig venv on its own: nothing puts it on sys.path, and without this the hand sink dies
# with a bare `ModuleNotFoundError: No module named 'sharpa'` -- after the cameras are
# already open, and only on the runs that actually command the hand.
#
# The rig's own docs export LD_LIBRARY_PATH alongside PYTHONPATH for this, but that is not
# needed here: the extension modules carry `RUNPATH $ORIGIN/../../lib`, so the loader finds
# libSharpaWaveSDK itself. A path entry is enough, which means this can be done in-process
# rather than being one more thing the operator has to remember to export.
DEFAULT_SDK_ROOT = "/opt/sharpa-wave-sdk/python"

# .../openpi/examples/sharpa/rollout/rig.py -> .../openpi
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLIENT_SRC = _REPO_ROOT / "packages" / "openpi-client" / "src"


def rig_root(override: str | os.PathLike | None = None) -> pathlib.Path:
    return pathlib.Path(override or os.environ.get("OPENARM_RIG_ROOT", DEFAULT_RIG_ROOT))


def sdk_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SHARPA_SDK_ROOT", DEFAULT_SDK_ROOT))


def add_paths(override: str | os.PathLike | None = None) -> pathlib.Path:
    """Put the rig and openpi-client on sys.path. Returns the resolved rig root.

    Raises with the actual paths in the message rather than letting the caller meet a
    bare ModuleNotFoundError three imports later.
    """
    root = rig_root(override)
    collect = root / "collect"
    for name, path in (("rig root", root), ("rig collect/", collect)):
        if not path.is_dir():
            raise SystemExit(
                f"{name} not found at {path}\n"
                f"Point OPENARM_RIG_ROOT (or --rig-root) at the openarm_track/vr checkout.")
    if not (_CLIENT_SRC / "openpi_client").is_dir():
        raise SystemExit(f"openpi-client sources not found at {_CLIENT_SRC}")

    paths = [str(collect), str(root), str(_CLIENT_SRC)]
    # Appended, not inserted: the SDK ships a package literally named `sharpa`, and the
    # rig checkout is where every other import here comes from. Last position keeps it from
    # shadowing anything.
    sdk = sdk_root()
    if (sdk / "sharpa").is_dir():
        paths.append(str(sdk))
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def require_hand_sdk() -> None:
    """Fail before the cameras are opened if the hand SDK is not importable.

    Only the runs that command the hand need it, but they need it several seconds in, by
    which point two camera pipelines are running and the failure reads as unrelated.
    """
    try:
        import sharpa  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"the Sharpa Wave SDK is not importable (looked in {sdk_root()}).\n"
            f"It is installed by `dpkg -i sharpa-wave-sdk_*.deb`; point SHARPA_SDK_ROOT at "
            f"its `python/` directory if it lives somewhere else.\n"
            f"Note this needs no LD_LIBRARY_PATH -- the extension carries its own RUNPATH.") from e


def require_ros() -> None:
    """Fail early, and say the actual fix, if ROS was not sourced.

    The robot venv is --system-site-packages, so rclpy is only importable once
    /opt/ros/jazzy and ~/ros2_ws are on the environment. Without this check the failure
    surfaces as a ModuleNotFoundError inside RosSources' constructor, several seconds
    after the cameras have already been opened.
    """
    try:
        import rclpy  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "rclpy is not importable -- source ROS before running:\n"
            "    source /opt/ros/jazzy/setup.bash\n"
            "    source ~/ros2_ws/install/setup.bash\n"
            "and use the rig's interpreter: <rig>/.venv/bin/python") from e


# openpi-client's blocking client is `websockets.sync.client`, added in websockets 12.0.
# The rig venv inherits 10.4 from the system packages, so this one does have to be
# installed; nothing else in the rig checkout imports websockets.
CLIENT_REQUIREMENTS = ("websockets>=12.0",)

# DO NOT ALSO UPGRADE msgpack HERE. openpi-client's metadata asks for >= 1.0.5, but the
# only parts of it this client uses -- `msgpack.Packer(default=...)` and
# `msgpack.unpackb(..., object_hook=..., raw=False)` -- have been stable since 1.0, and
# the system's 1.0.3 round-trips the real observation payload correctly (checked with the
# (29,) state and both camera arrays at their deployed sizes). Meanwhile `python-can
# 4.3.1`, in those same system packages, requires `msgpack~=1.0.0`. Installing 1.2 into
# the venv satisfies a bound nothing here actually needs while breaking one the CAN stack
# declares. The bound below is what the CODE requires, not what the metadata claims.
MIN_MSGPACK = (1, 0)


def check_client_deps() -> None:
    """Verify what openpi-client actually needs, and say exactly how to get it."""
    missing = []
    try:
        import websockets.sync.client
    except ImportError:
        import websockets

        missing.append(f"websockets>=12.0 (found {getattr(websockets, '__version__', '?')}, "
                       f"which has no .sync)")
    try:
        import msgpack

        if tuple(msgpack.version)[:2] < MIN_MSGPACK:
            missing.append(f"msgpack>=1.0 (found {'.'.join(map(str, msgpack.version))})")
    except ImportError:
        missing.append("msgpack>=1.0 (not installed)")

    if missing:
        raise SystemExit(
            "the policy client needs:\n  " + "\n  ".join(missing) + "\n\n"
            "Install INTO THE RIG VENV -- pure Python, shadows the system copy only "
            "inside the venv, and nothing else in the rig imports it:\n"
            f"    {sys.executable} -m pip install "
            + " ".join(f"'{r}'" for r in CLIENT_REQUIREMENTS))
