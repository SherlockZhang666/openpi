"""An MP4 writer that takes frames one at a time. Used offline, by make_video.py.

NOTHING HERE RUNS DURING A ROLLOUT. The rollout stores JPEG frames in the npz and encodes
nothing -- see record.py. This turns them into video afterwards, on a machine that is not
holding a robot, where being slow costs nothing.

WHY A SUBPROCESS AND NOT A LIBRARY
----------------------------------
The rig venv has no cv2, no imageio and no av, and it is not a venv to casually add binary
wheels to: it is the interpreter that drives live hardware, carrying pinocchio, rclpy and
GStreamer bindings that a bad resolve would disturb. `/usr/bin/ffmpeg` is already on the
machine, so raw frames go down a pipe to it instead.

FRAME N IS STEP N
-----------------
One frame per recorded control step, in order, never dropped -- that correspondence is
what lets you pause on a frame and go read the numbers for that step. The queue exists so
`add` does not block on the encoder; it is not allowed to discard.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading

import numpy as np

logger = logging.getLogger(__name__)

# 4 seconds of slack at 30 Hz. Bounded on purpose: an unbounded queue turns a stalled
# encoder into unbounded memory growth on the machine holding the arm.
QUEUE_DEPTH = 120


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class VideoWriter:
    """One MP4, written by ffmpeg on the other end of a pipe.

    Opened lazily on the first frame, because the frame is what tells us the resolution --
    the two cameras have different aspect ratios and only the sensor knows.
    """

    def __init__(self, path, *, fps: float = 30.0):
        self.path = path
        self.fps = float(fps)
        self.n_frames = 0
        self.n_blocked = 0
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._shape: tuple[int, int] | None = None

    def _start(self, h: int, w: int) -> None:
        self._shape = (h, w)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # yuv420p and the scale filter are both for compatibility rather than quality:
        # h264 needs even dimensions and the wrist camera at short-side 256 comes out
        # 455 wide, which is odd. Without the filter ffmpeg exits and the recording is
        # simply absent, discovered hours later.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{self.fps:g}",
            "-i", "-", "-an",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._pump, name=f"vid-{self.path.stem}",
                                        daemon=True)
        self._thread.start()
        logger.info("recording %dx%d -> %s", w, h, self.path)

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                self._proc.stdin.write(frame)
        except BaseException as e:      # noqa: BLE001 -- surfaced on close()
            self._error = e
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

    def add(self, frame) -> None:
        """Queue one frame. Never drops: frame N must stay step N."""
        if self._error is not None:
            return
        a = np.ascontiguousarray(frame, dtype=np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8, got {a.shape}")
        if self._proc is None:
            self._start(a.shape[0], a.shape[1])
        elif a.shape[:2] != self._shape:
            # ffmpeg was told a fixed frame size; a different one would be silently
            # reinterpreted as garbage rather than rejected.
            raise ValueError(f"frame size changed {self._shape} -> {a.shape[:2]}")
        if self._q.full():
            self.n_blocked += 1
        self._q.put(a.tobytes())
        self.n_frames += 1

    def close(self) -> None:
        if self._proc is None:
            return
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        rc = self._proc.wait(timeout=30.0)
        err = self._proc.stderr.read().decode("utf-8", "replace").strip()
        self._proc = None
        if self.n_blocked:
            logger.debug("%s: the encoder queue filled %d times; every frame was still "
                         "written, the producer just waited", self.path.name, self.n_blocked)
        if rc != 0:
            logger.error("ffmpeg exited %d for %s: %s", rc, self.path, err or "(no stderr)")
        else:
            logger.info("wrote %d frames -> %s", self.n_frames, self.path)
        if self._error is not None:
            logger.error("video writer thread failed for %s: %s", self.path, self._error)
