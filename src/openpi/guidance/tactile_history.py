"""Client-side causal F6 history, filled at CONTROL rate, not request rate."""

from collections import deque
import time

import numpy as np


class TactileHistory:
    def __init__(self):
        self.frames = deque(maxlen=16)
        self.times = deque(maxlen=16)

    def reset(self):
        self.frames.clear()
        self.times.clear()

    def append(self, calibrated_f6, *, timestamp=None):
        value = np.asarray(calibrated_f6, np.float32)
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if value.shape != (5, 6) or not np.isfinite(value).all() or not np.isfinite(stamp):
            raise ValueError("Expected finite calibrated F6 [5,6]")
        if self.times and stamp <= self.times[-1]:
            raise ValueError("Tactile timestamps must increase")
        self.frames.append(value.copy())
        self.times.append(stamp)

    def attach(self, observation, *, now=None):
        stamp = time.monotonic() if now is None else float(now)
        if len(self.frames) != 16:
            raise ValueError("Need 16 measured control-rate frames before guided inference")
        ts = np.array(self.times)
        if (
            not np.isfinite(stamp)
            or stamp < ts[-1]
            or stamp - ts[-1] > 0.2
            or np.diff(ts).max() > 0.1
            or not 0.25 <= ts[-1] - ts[0] <= 0.8
        ):
            raise ValueError("History is stale, discontinuous or inconsistent with 30Hz control")
        return {**observation, "guidance/f6_history": np.stack(self.frames)}
