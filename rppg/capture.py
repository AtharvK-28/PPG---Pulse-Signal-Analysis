"""Stage 0 — frame sources that timestamp every frame.

The timestamp is not bookkeeping. `cv2.VideoCapture` does not deliver frames at
a constant interval — dropped frames, exposure-dependent readout and OS
scheduling all jitter it — and the FFT downstream assumes uniform sampling. A
nominal 30 fps that is really a 28.4 fps mean puts a systematic scale error on
the whole frequency axis. So: record `perf_counter()` per frame, resample later.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float  #: seconds, zero at the first frame
    image: np.ndarray  #: BGR, as OpenCV delivers it


class VideoFileSource:
    """Recorded clip. Timestamps come from the container, not the wall clock."""

    def __init__(self, path, use_container_pts: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_container_pts = use_container_pts

    def __iter__(self):
        idx, t0, prev = 0, None, None
        while True:
            pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) if self.use_container_pts else 0.0
            ok, img = self.cap.read()
            if not ok:
                break
            t = pts_ms / 1000.0 if self.use_container_pts else idx / self.fps
            # Containers lie in two ways: some report 0.0 for every frame, and
            # some repeat the first PTS. Either produces a zero interval, which
            # is a duplicate timestamp the resampler would have to discard —
            # silently throwing away a frame of real data. Fall back to the
            # nominal grid whenever the reported PTS does not advance.
            if t0 is None:
                t0 = t
            elif t <= prev:
                t = prev + 1.0 / self.fps
            prev = t
            yield Frame(index=idx, timestamp=t - t0, image=img)
            idx += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def list_cameras(max_index: int = 5, backend: int | None = None) -> list[int]:
    """Indices that actually open *and* deliver a frame.

    `isOpened()` alone is not enough — some backends report success for an
    index that then yields nothing, so each candidate is probed with a real
    read. Probing is a second or two per index; cache the result rather than
    calling this in a loop.
    """
    working = []
    try:  # the probe walks past the end of the device list and OpenCV is loud about it
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass
    try:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            try:
                if cap.isOpened():
                    ok, img = cap.read()
                    if ok and img is not None:
                        working.append(idx)
            finally:
                cap.release()
    finally:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
        except AttributeError:
            pass
    return working


class WebcamSource:
    """Live capture with per-frame `perf_counter()` timestamps.

    Auto-exposure and auto-white-balance are disabled by default. Both are
    closed loops driven by scene brightness, i.e. by exactly the sub-percent
    intensity changes we are trying to measure — leaving them on lets the camera
    fight the pulse signal, and the hunting behaviour can beat against mains
    flicker. Record one clip with them on to show the damage; run everything
    else with them off.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        fix_exposure: bool = True,
        backend: int | None = None,
    ):
        self.cap = cv2.VideoCapture(device, backend) if backend else cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.cap.release()
            # "could not open camera 1" on its own leaves you guessing between a
            # wrong index, a camera in use, and a blocked permission. Say which.
            available = list_cameras()
            if available:
                hint = f"Available camera indices on this machine: {available}."
            else:
                hint = (
                    "No cameras found at all. Check that no other application is "
                    "holding the webcam, and that camera access is enabled in "
                    "Windows Settings > Privacy & security > Camera."
                )
            raise RuntimeError(f"Could not open camera {device}. {hint}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        self.exposure_locked = False
        self.exposure_revert_reason = ""
        if fix_exposure:
            self.exposure_locked = self._lock_exposure()

    def _measure_fps(self, seconds: float = 1.0) -> float:
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            if self.cap.read()[0]:
                n += 1
        return n / seconds

    def _lock_exposure(self) -> bool:
        """Lock exposure only if it does not cost the frame rate.

        Locking is theoretically right: AGC is a feedback loop that regulates
        mean brightness, and the pulse *is* a mean-brightness modulation, so
        auto-exposure actively attenuates the thing being measured.

        In practice it is backend-dependent and frequently lies. Measured here:
        `set()` returned True and the readback was -1.0 ("unsupported") on
        DirectShow, yet the camera still switched to a long manual exposure,
        halving the rate 30.1 -> 15.9 fps and the light 168 -> 80. Trusting the
        return value cost half the Nyquist headroom.

        So: measure the rate before and after, and revert if it collapsed. That
        generalises to hardware this was never tested on.
        """
        before = self._measure_fps()
        ok = False
        # 0.25 = manual on V4L2; 0.75 = manual on DirectShow/MSMF.
        for value in (0.25, 0.75, 1):
            if self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
                ok = True
                break
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if not ok:
            return False

        after = self._measure_fps()
        if before > 0 and after < 0.75 * before:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # back to auto
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.exposure_revert_reason = (
                f"locking exposure dropped {before:.1f} -> {after:.1f} fps; "
                "reverted to auto (a halved frame rate costs more than AGC does)"
            )
            return False
        return True

    def __iter__(self):
        idx, t0 = 0, None
        while True:
            ok, img = self.cap.read()
            # Timestamp immediately after read() returns, before any processing,
            # so downstream work does not contaminate the sampling instants.
            t = time.perf_counter()
            if not ok:
                break
            if t0 is None:
                t0 = t
            yield Frame(index=idx, timestamp=t - t0, image=img)
            idx += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ThreadedWebcamSource(WebcamSource):
    """Grab frames on a dedicated thread so `read()` overlaps with processing.

    Measured on this machine: `cap.read()` costs 13.8 ms and the landmark stage
    18.5 ms, which serialised is 32.3 ms against a 33.3 ms budget. At ~97 %
    utilisation the loop has no slack, so any hiccup — a plot, a rerun, a GC —
    turns into a burst of missed frames; a recorded session showed p99 = 438 ms
    intervals while the camera itself was delivering p99 = 49 ms.

    Overlapping the two stages puts the ceiling at max(13.8, 18.5) rather than
    their sum. Timestamps are taken in the grabber the instant `read()` returns,
    so queueing delay never reaches the sampling grid.

    The queue is deliberately shallow. When the consumer falls behind, the
    oldest frame is discarded so the stream stays near-real-time and the losses
    are spread thinly instead of arriving as one long stall. `dropped` counts
    them, because a dropped frame is a hole in the sampling grid and the
    estimator is entitled to know.
    """

    def __init__(self, *args, queue_size: int = 4, **kw):
        super().__init__(*args, **kw)
        self._q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._eof = threading.Event()
        self.dropped = 0
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        idx, t0 = 0, None
        try:
            while not self._stop.is_set():
                ok, img = self.cap.read()
                t = time.perf_counter()
                if not ok:
                    break
                if t0 is None:
                    t0 = t
                frame = Frame(index=idx, timestamp=t - t0, image=img)
                idx += 1
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(frame)
                    except queue.Full:
                        self.dropped += 1
        finally:
            self._eof.set()

    def __iter__(self):
        while True:
            try:
                yield self._q.get(timeout=1.0)
            except queue.Empty:
                if self._eof.is_set() or self._stop.is_set():
                    break

    def close(self):
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            # The grabber can be parked inside a blocking read(); joining with a
            # timeout and releasing anyway is safer than hanging the caller.
            thread.join(timeout=2.0)
        super().close()


def open_source(source, **kw):
    """`0` / `"1"` -> webcam; a path -> file."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        threaded = kw.pop("threaded", False)
        cls = ThreadedWebcamSource if threaded else WebcamSource
        return cls(int(source), **kw)
    kw.pop("threaded", None)
    return VideoFileSource(source, **kw)
