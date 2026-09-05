"""Live RGB frames from the rig's two cameras, configured exactly as they were recorded.

WHY THIS FILE EXISTS AT ALL
---------------------------
`collect/camera.py`'s `Camera` is a recorder: MJPG in, MKV out, and its `on_frame`
callback hands you a `FrameTick` -- a frame index and two clocks, no pixels. That is
deliberate on the collection side (the `identity` element sits inside the muxer's own
branch, which is what makes "MKV frame N == HDF5 row N" true by topology). It leaves no
way to get an ndarray, so inference needs its own pipeline.

What it does NOT need is its own idea of how the cameras are set up. Both the device node
and the v4l2 control string come from a real `Camera` instance, so the sensor is
configured byte-identically to the sessions the policy was trained on. That matters more
than it looks: `resolve_device` exists because the Gemini 336L enumerates as eight
/dev/videoN nodes and the obvious "first node with this name" lands on the DEPTH node,
and the control string carries the exposure, gain, white balance and 50 Hz flicker
setting that decide what the images look like.

CONSTRUCTION ORDER IS LOAD-BEARING
----------------------------------
`Camera.__init__` calls `Gst.init()`, and `collect/openarm_collect.py` records that
h5py + `Gst.init()` + rclpy Node construction is a three-way segfault, any two of which
are fine. So: create the rclpy node FIRST, then build these grabbers. `main.py` does that
and says so; do not reorder it.

WHAT STILL DIFFERS FROM TRAINING, AND WHY IT IS LEFT ALONE
----------------------------------------------------------
Training frames made one extra generation: sensor MJPG (or wrist YUYV -> jpegenc q85) ->
MKV -> h264 crf23 in the LeRobot converter. Here they are decoded straight to RGB. Adding
a matching compress/decompress round trip would be emulating an artefact rather than
removing a difference, and at crf23 it is far below the resize the model applies anyway.
The resize, on the other hand, IS reproduced: `--image-short-side` defaults to the 256 the
converter used, done here in `videoscale` so the network sees the same pixel scale it was
trained at rather than a differently-downsampled version of the same view.
"""

from __future__ import annotations

import threading
import time

import numpy as np

# appsink delivers frames through the `new-sample` signal rather than its Python pull
# methods; see FrameGrabber.start for why.
EMIT_SIGNALS = True


def _even(x: float) -> int:
    """Mirrors convert_sharpa_data_to_lerobot._even, so the eval resize lands on the
    same dimensions the training frames were written at."""
    return max(2, int(round(x / 2)) * 2)


def scaled_size(width: int, height: int, short_side: int | None) -> tuple[int, int]:
    if not short_side:
        return int(width), int(height)
    scale = short_side / min(width, height)
    return _even(width * scale), _even(height * scale)


class FrameGrabber:
    """Newest-frame holder over a v4l2 -> RGB appsink pipeline.

    `drop=true max-buffers=1` on the appsink means a slow consumer gets the newest frame
    rather than a queue of stale ones -- the same zero-order-hold the collector applied
    when it sampled every non-camera stream onto the scene camera's clock.
    """

    def __init__(self, *, name: str, device: str, controls: str, caps: str,
                 out_size: tuple[int, int], decode: str):
        self.name = name
        self.device = device
        self.out_width, self.out_height = out_size
        self._desc = (
            f'v4l2src device={device} extra-controls="{controls}" '
            f"! {caps} "
            f"{decode}"
            f"! videoconvert ! videoscale "
            f"! video/x-raw,format=RGB,width={self.out_width},height={self.out_height} "
            f"! appsink name=out max-buffers=1 drop=true sync=false"
        )
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t = float("nan")
        self._n = 0
        self._pipeline = None
        self._sink = None
        self._error: str | None = None

    @property
    def description(self) -> str:
        return self._desc

    def start(self, timeout: float = 10.0) -> FrameGrabber:
        from camera import _load_gst  # rig module; see rig.add_paths

        _load_gst()
        from gi.repository import Gst

        Gst.init(None)
        self._pipeline = Gst.parse_launch(self._desc)
        self._sink = self._pipeline.get_by_name("out")
        # `new-sample` + the `pull-sample` ACTION SIGNAL, rather than appsink's Python
        # methods. `try_pull_sample`/`pull_sample` need the GstApp typelib
        # (gir1.2-gst-plugins-base-1.0), which is not installed on the rig and would be an
        # apt dependency for a client that otherwise needs none. Signals are plain GObject
        # and go through the generic GstElement wrapper `parse_launch` returns, so this
        # works with what is already there. Same data, no extra package.
        #
        # The callback runs on the STREAMING THREAD, so it does the least it can: convert
        # and store. No GLib main loop is involved -- `new-sample` is emitted directly,
        # not dispatched through one.
        self._sink.set_property("emit-signals", EMIT_SIGNALS)
        self._sink.connect("new-sample", self._on_sample)
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"{self.name}: pipeline refused to start\n  {self._desc}")

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._error:
                self.stop()
                raise RuntimeError(f"{self.name}: {self._error}")
            if self._n:
                return self
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(
            f"{self.name}: no frame within {timeout:.0f}s from {self.device}\n"
            f"  is another process holding the camera? (collect_up.sh down)\n"
            f"  {self._desc}")

    def _on_sample(self, sink):
        from gi.repository import Gst

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        try:
            frame = self._to_rgb(sample)
        except Exception as e:  # reported, never raised on this thread
            self._error = f"{type(e).__name__}: {e}"
            return Gst.FlowReturn.ERROR
        with self._lock:
            self._frame = frame
            self._t = time.time()
            self._n += 1
        return Gst.FlowReturn.OK

    def _to_rgb(self, sample) -> np.ndarray:
        from gi.repository import Gst

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map the GStreamer buffer")
        try:
            raw = np.frombuffer(info.data, dtype=np.uint8)
            # GStreamer pads each row out to a 4-byte boundary, so the buffer is not
            # necessarily height*width*3. Recover the stride from the actual size and
            # slice the padding off, rather than assuming a packed layout.
            stride = raw.size // height
            if stride < width * 3:
                raise RuntimeError(f"buffer too small: {raw.size} B for {width}x{height} RGB")
            return raw[: height * stride].reshape(height, stride)[:, : width * 3].reshape(
                height, width, 3).copy()
        finally:
            buf.unmap(info)

    def latest(self) -> tuple[np.ndarray | None, float]:
        """(RGB uint8 HxWx3, wall clock of arrival) or (None, nan)."""
        with self._lock:
            return self._frame, self._t

    def age(self) -> float:
        with self._lock:
            return float("inf") if np.isnan(self._t) else time.time() - self._t

    @property
    def n_frames(self) -> int:
        return self._n

    def stop(self) -> None:
        if self._pipeline is not None:
            from gi.repository import Gst

            # NULL first: it tears down the streaming thread, so no callback can be
            # running by the time the sink reference goes away.
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._sink = None


def head_grabber(*, short_side: int | None = 256, device: str | None = None,
                 exposure="auto", gain: int | None = None, wb="auto") -> FrameGrabber:
    """Orbbec Gemini 336L colour, 1280x800 MJPG @30 -- the rig's scene camera.

    Defaults match `collect_up.sh`'s: auto exposure (with exposure_dynamic_framerate
    pinned at 0 so the frame period stays put) and asserted auto white balance.
    """
    import camera as rig_cam

    cam = rig_cam.Camera(device=device, name=rig_cam.DEVICE_NAME,
                         width=rig_cam.DEFAULT_W, height=rig_cam.DEFAULT_H,
                         fps=rig_cam.DEFAULT_FPS, exposure=exposure,
                         gain=rig_cam.DEFAULT_GAIN if gain is None else gain,
                         wb=wb, label="scene")
    caps = (f"image/jpeg,width={rig_cam.DEFAULT_W},height={rig_cam.DEFAULT_H},"
            f"framerate={rig_cam.DEFAULT_FPS}/1")
    return FrameGrabber(
        name="head", device=cam.device, controls=cam.controls, caps=caps,
        decode="! jpegdec ",
        out_size=scaled_size(rig_cam.DEFAULT_W, rig_cam.DEFAULT_H, short_side))


def wrist_grabber(*, short_side: int | None = 256, device: str | None = None,
                  exposure: int | None = None, gain: int | None = None,
                  wb="auto") -> FrameGrabber:
    """RealSense D435i, 960x540 YUYV @60 -- bolted to the left arm.

    This is the rig's PRIMARY view for the policy (it maps to `left_wrist_0_rgb`), so its
    exposure default is the collection one: 56, i.e. 5.6 ms, chosen for motion blur rather
    than brightness because this camera rides the arm.
    """
    import camera as rig_cam

    cam = rig_cam.wrist_camera(
        device=device, name=rig_cam.WRIST_DEVICE_NAME,
        width=rig_cam.WRIST_W, height=rig_cam.WRIST_H, fps=rig_cam.WRIST_FPS,
        exposure=rig_cam.WRIST_EXPOSURE if exposure is None else exposure,
        gain=rig_cam.WRIST_GAIN if gain is None else gain, wb=wb)
    caps = (f"video/x-raw,format={rig_cam.WRIST_RAW_FORMAT},width={rig_cam.WRIST_W},"
            f"height={rig_cam.WRIST_H},framerate={rig_cam.WRIST_FPS}/1")
    return FrameGrabber(
        name="wrist", device=cam.device, controls=cam.controls, caps=caps, decode="",
        out_size=scaled_size(rig_cam.WRIST_W, rig_cam.WRIST_H, short_side))
