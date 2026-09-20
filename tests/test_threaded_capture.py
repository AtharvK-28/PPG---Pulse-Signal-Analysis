"""ThreadedWebcamSource: overlap read() with processing without losing the grid.

The failure this exists to prevent: a saturated synchronous loop stalls in
bursts. A recorded session showed p99 = 438 ms sampling intervals while the
camera itself was delivering p99 = 49 ms.
"""

import time

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.capture import ThreadedWebcamSource


class _FakeCap:
    """Stands in for cv2.VideoCapture, delivering frames at a fixed rate."""

    def __init__(self, n=60, interval=0.005):
        self.n, self.interval, self.i = n, interval, 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def read(self):
        if self.i >= self.n:
            return False, None
        time.sleep(self.interval)
        img = np.full((8, 8, 3), self.i % 256, np.uint8)
        self.i += 1
        return True, img

    def release(self):
        self.released = True


@pytest.fixture
def threaded(monkeypatch):
    def _make(queue_size=4, **kw):
        cap = _FakeCap(**kw)
        monkeypatch.setattr("cv2.VideoCapture", lambda *a, **k: cap)
        src = ThreadedWebcamSource(0, fix_exposure=False, queue_size=queue_size)
        return src, cap

    return _make


def test_delivers_every_frame_when_consumer_keeps_up(threaded):
    src, _ = threaded(n=40)
    frames = list(src)
    src.close()
    assert len(frames) == 40
    assert src.dropped == 0
    assert [f.index for f in frames] == list(range(40))


def test_timestamps_are_monotonic_and_start_at_zero(threaded):
    src, _ = threaded(n=25)
    ts = [f.timestamp for f in src]
    src.close()
    assert ts[0] == 0.0
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_timestamps_reflect_grab_time_not_consumption_time(threaded):
    """The whole point: a slow consumer must not distort the sampling grid.

    Measured with a queue deep enough to absorb the lag. A shallow one drops
    frames by design, so its intervals legitimately contain jumps and say
    nothing about whether timestamps track grab time.
    """
    src, _ = threaded(n=40, interval=0.004, queue_size=200)
    ts = []
    for f in src:
        ts.append(f.timestamp)
        time.sleep(0.02)  # consumer 5x slower than the camera
    src.close()
    assert src.dropped == 0, "deep queue should absorb the lag rather than drop"
    dt = np.diff(ts)
    # Every interval reflects the camera's ~4 ms cadence, not the 20 ms the
    # consumer spent. If timestamps were taken at consumption they would all
    # be ~20 ms.
    assert np.median(dt) < 0.012, f"median interval {np.median(dt) * 1000:.1f} ms"
    assert max(ts) < 40 / 30.0 + 1e-6, "timestamps must come from the source"


def test_slow_consumer_drops_frames_and_says_so(threaded):
    src, _ = threaded(n=60, interval=0.002)
    for _ in src:
        time.sleep(0.03)
    dropped = src.dropped
    src.close()
    # Losses are reported rather than silently corrupting the grid.
    assert dropped > 0


def test_close_releases_the_capture_and_stops_the_thread(threaded):
    src, cap = threaded(n=10_000, interval=0.001)
    next(iter(src))
    src.close()
    assert cap.released
    assert not src._thread.is_alive()


def test_iteration_ends_when_the_camera_stops(threaded):
    src, _ = threaded(n=5)
    frames = list(src)
    src.close()
    assert len(frames) == 5
