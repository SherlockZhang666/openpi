#!/usr/bin/env python3
"""Turn a rollout's stored frames into MP4s. Run it whenever, on any run directory.

    <rig>/.venv/bin/python openarm_track/rollout/make_video.py <run-dir-or-npz>
    <rig>/.venv/bin/python openarm_track/rollout/make_video.py <run-dir> --fps 10 --cam head

The rollout appends the observation frames to `<camera>.jpgs` as it goes and encodes
nothing -- see record.py. This is the other half: it decodes them and pipes them to
ffmpeg. Nothing here runs anywhere near the robot, so it can be slow, rerun, and thrown
away.

A `.jpgs` file is `<uint32 little-endian length><jpeg bytes>` repeated. A torn tail from a
killed process costs the last frame and nothing else, so this reads until the records stop
making sense rather than trusting a count.

Frame N of the MP4 is step N of the arrays, because one frame was stored per recorded
control step. That is what makes it possible to pause on a frame and go read the numbers
for that step.

`--fps` only changes playback speed; it does not resample. The default is the rate the
frames were captured at, which is stored in the file. `mpv`, `vlc` and any browser play
the result.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys

import numpy as np


def run_dir(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_dir() else path.parent


def cameras(d: pathlib.Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.jpgs"))


def frames(path: pathlib.Path):
    """Yield decoded RGB frames from one .jpgs stream, stopping at a torn tail."""
    import struct

    from PIL import Image

    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if len(head) < 4:
                return
            (n,) = struct.unpack("<I", head)
            blob = fh.read(n)
            if len(blob) < n:
                print(f"  {path.name}: last record is short "
                      f"({len(blob)} of {n} bytes) -- stopping there")
                return
            yield np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the rollout's run directory, or its .npz")
    ap.add_argument("--cam", action="append", default=None,
                    help="only this camera; repeatable. Default: all of them")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback rate. Default: the capture rate stored in the file")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="default: alongside the npz")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import video

    if not video.ffmpeg_available():
        raise SystemExit("ffmpeg is not on PATH; it is what does the encoding here")

    src = run_dir(pathlib.Path(a.run).expanduser())
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else src
    have = cameras(src)
    if not have:
        # There was a brief window on 2026-09-08 when frames went into the npz instead.
        # Say so, rather than let someone conclude the run has no frames at all.
        legacy = [n for f in src.glob("*.npz")
                  for n in np.load(f).files if n.startswith("frames_")]
        if legacy:
            raise SystemExit(
                f"{src} stores its frames inside the npz, which only runs from "
                f"2026-09-08 21:14-21:45 do. Extract them with numpy directly: the blob "
                f"is frames_<cam> and the index is frames_<cam>_offsets.")
        raise SystemExit(
            f"{src} has no .jpgs streams. Rollouts record them by default; this one was "
            f"run with --no-frames, or predates frame recording.")
    want = a.cam or have
    unknown = sorted(set(want) - set(have))
    if unknown:
        raise SystemExit(f"no such camera {unknown} -- this run has {have}")

    fps = a.fps
    if fps is None:
        npz = sorted(src.glob("*.npz"))
        d = np.load(npz[0]) if npz else {}
        fps = float(d["fps"]) if (npz and "fps" in d.files) else 30.0
    for cam in want:
        w = video.VideoWriter(out_dir / f"{cam}.mp4", fps=fps)
        for frame in frames(src / f"{cam}.jpgs"):
            w.add(frame)
        w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
