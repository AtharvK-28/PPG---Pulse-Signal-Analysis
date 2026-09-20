"""Sample the ROI on a dedicated thread, so rendering cannot perforate the grid.

The sampling instants are the measurement. Everything downstream — the FFT, the
frequency axis, every BPM — assumes they are regular. So the loop that produces
them must not share a thread with anything whose cost it does not control.

It did, and the cost was visible: a live session reported 21.6 fps processed
against a camera delivering 30, with a worst sampling gap of 495 ms, because
each Streamlit redraw blocked the capture loop for as long as it took. Moving
`read()` to its own thread was not enough; the landmark stage was still queued
behind the UI.

Here the producer does read + ROI and nothing else. Measured on this machine
that is 13.8 + ~6.4 ms amortised = ~20 ms against a 33 ms arrival interval, so
it keeps up with roughly a third to spare. The consumer drains whatever has
accumulated whenever it gets around to it, so a 500 ms render turns into 15
buffered samples rather than a 500 ms hole. Latency is recoverable; a hole in
the sampling grid is not.

The queue holds slim samples — timestamp, mean RGB, pixel count — not images.
Full `RoiSample`s carry a full-frame mask, and 20 s of those would be ~180 MB.
The preview needs an image, so the newest frame and its mask are kept in a
single slot that the consumer reads at its own pace.
"""

from __future__ import annotations

import queue
import threading

from .roi import RoiSample


class ThreadedSampler:
    """Run `source -> roi` on a producer thread; drain samples from anywhere.

    Parameters
    ----------
    source : iterable of Frame
    roi : callable(Frame) -> RoiSample
    queue_size : int
        Samples buffered before the oldest is discarded. The default holds
        ~20 s at 30 fps, far longer than any plausible render stall, so in
        practice nothing is dropped.
    drop_when_full : bool
        True for a live camera: the world does not wait, so falling behind
        must cost the oldest sample rather than desynchronise the timestamps
        from real time. False for a file or synthetic source, which produces
        frames as fast as the CPU allows and will happily wait — dropping
        there would silently discard the ground truth being validated against.
    """

    def __init__(self, source, roi, queue_size: int = 600, drop_when_full: bool = True):
        self.source = source
        self.roi = roi
        self.drop_when_full = drop_when_full
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._done = threading.Event()
        self._latest = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                sample = self.roi(frame)
                # Slim copy: the mask and polygons stay out of the queue.
                slim = RoiSample(
                    sample.timestamp, sample.rgb, sample.n_pixels, sample.ok
                )
                if self.drop_when_full:
                    try:
                        self._q.put_nowait(slim)
                    except queue.Full:
                        try:
                            self._q.get_nowait()
                            self.dropped += 1
                        except queue.Empty:
                            pass
                        try:
                            self._q.put_nowait(slim)
                        except queue.Full:
                            self.dropped += 1
                else:
                    while not self._stop.is_set():
                        try:
                            self._q.put(slim, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                with self._lock:
                    self._latest = (frame, sample)
        except BaseException as exc:  # noqa: BLE001
            # A producer that dies silently looks exactly like a camera that
            # stopped delivering. Keep it and re-raise on the consumer side.
            self._error = exc
        finally:
            self._done.set()

    def drain(self, timeout: float = 0.2):
        """Every sample available now, oldest first. Blocks briefly if empty."""
        out = []
        try:
            out.append(self._q.get(timeout=timeout))
        except queue.Empty:
            return out
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def latest(self):
        """Newest (Frame, RoiSample) for the preview, or None."""
        with self._lock:
            return self._latest

    @property
    def running(self) -> bool:
        return not self._done.is_set()

    @property
    def error(self):
        return self._error

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            # The producer can be parked in a blocking read(); joining with a
            # timeout beats hanging the caller.
            self._thread.join(timeout=2.0)
