"""ThreadedSampler: rendering must not be able to perforate the sampling grid.

The bug this exists to prevent: ROI extraction sharing a thread with Streamlit,
so every redraw became a hole. A live session showed 21.6 fps processed against
a camera delivering 30, with a worst gap of 495 ms.
"""

import time

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.capture import Frame
from rppg.roi import RoiSample
from rppg.sampler import ThreadedSampler


def _frames(n, interval=0.0):
    for i in range(n):
        if interval:
            time.sleep(interval)
        yield Frame(index=i, timestamp=i / 30.0, image=np.zeros((4, 4, 3), np.uint8))


def _roi(frame):
    return RoiSample(
        timestamp=frame.timestamp,
        rgb=np.array([100.0, 90.0, 80.0]),
        n_pixels=1000,
        ok=True,
    )


def _drain_all(sampler, limit=2000):
    out = []
    while len(out) < limit:
        batch = sampler.drain(timeout=0.15)
        out.extend(batch)
        if not batch and not sampler.running:
            break
    return out


def test_delivers_every_sample_to_a_prompt_consumer():
    s = ThreadedSampler(_frames(120), _roi, drop_when_full=False)
    got = _drain_all(s)
    s.close()
    assert len(got) == 120
    assert [round(x.timestamp, 6) for x in got] == [round(i / 30.0, 6) for i in range(120)]


def test_slow_consumer_loses_nothing_when_the_source_can_wait():
    """A file or synthetic source must never be silently decimated."""
    s = ThreadedSampler(_frames(60), _roi, queue_size=8, drop_when_full=False)
    got = []
    while True:
        batch = s.drain(timeout=0.2)
        got.extend(batch)
        time.sleep(0.02)  # a slow "render"
        if not batch and not s.running:
            break
    s.close()
    assert len(got) == 60
    assert s.dropped == 0


def test_live_source_drops_rather_than_falling_behind_real_time():
    s = ThreadedSampler(_frames(400), _roi, queue_size=4, drop_when_full=True)
    time.sleep(0.4)  # let the producer outrun the consumer
    got = _drain_all(s, limit=400)
    s.close()
    assert s.dropped > 0, "a live camera must shed samples rather than lag"
    assert len(got) < 400


def test_timestamps_stay_ordered_across_batches():
    s = ThreadedSampler(_frames(200), _roi, drop_when_full=False)
    got = _drain_all(s)
    s.close()
    ts = [x.timestamp for x in got]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_queued_samples_carry_no_image_data():
    """20 s of full-frame masks would be ~180 MB; the queue must stay slim."""
    s = ThreadedSampler(_frames(10), _roi, drop_when_full=False)
    got = _drain_all(s)
    s.close()
    assert got
    for x in got:
        assert x.mask is None or getattr(x.mask, "size", 0) == 0
        assert not x.polygons


def test_latest_exposes_a_frame_for_the_preview():
    s = ThreadedSampler(_frames(30), _roi, drop_when_full=False)
    _drain_all(s)
    newest = s.latest()
    s.close()
    assert newest is not None
    frame, sample = newest
    assert frame.image is not None
    assert sample.ok


def test_producer_exception_is_reported_not_swallowed():
    def _boom(frame):
        raise ValueError("roi exploded")

    s = ThreadedSampler(_frames(5), _boom)
    _drain_all(s)
    s.close()
    assert isinstance(s.error, ValueError)


def test_running_goes_false_when_the_source_is_exhausted():
    s = ThreadedSampler(_frames(5), _roi, drop_when_full=False)
    _drain_all(s)
    time.sleep(0.05)
    assert not s.running
    s.close()


def test_close_is_safe_to_call_twice():
    s = ThreadedSampler(_frames(10_000, interval=0.001), _roi)
    s.drain()
    s.close()
    s.close()
    assert not s._thread.is_alive()
