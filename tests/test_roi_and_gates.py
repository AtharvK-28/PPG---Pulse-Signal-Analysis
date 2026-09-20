"""ROI geometry and the window-rejection gates.

Both of these exist because of a real failure: a live webcam run reported
47.9 / 48.6 / 62.3 / 49.3 BPM across the four arms, with every SNR marginal.
The ROI was a sliver of a few thousand pixels, and shared transients from
splining across dropped frames dominated every projection.
"""

import numpy as np
import pytest

from rppg.pipeline import PipelineConfig, WindowEstimate, agreement_spread, gate
from rppg.resample import find_gaps, max_gap_in_span
from rppg.roi import landmarks_to_polygons
from rppg.spectral import estimate_bpm

# A synthetic face: 200 px wide, 260 px tall, eyes 65 px apart.
FACE_W, FACE_H, IOD = 200.0, 260.0, 65.0


def _landmarks(cx=320.0, cy=240.0, angle_deg=0.0):
    """Plausible pixel landmarks for the indices roi.py actually reads."""
    pts = np.zeros((478, 2))
    named = {
        10: (cx, cy - 130),  # hairline
        152: (cx, cy + 130),  # chin
        234: (cx - 100, cy),  # face left edge
        454: (cx + 100, cy),  # face right edge
        33: (cx - 50, cy - 10),  # left eye outer/inner
        133: (cx - 15, cy - 10),
        362: (cx + 15, cy - 10),  # right eye inner/outer
        263: (cx + 50, cy - 10),
        50: (cx - 45, cy + 45),  # cheeks
        280: (cx + 45, cy + 45),
    }
    for idx, (x, y) in named.items():
        pts[idx] = (x, y)
    if angle_deg:
        th = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        centre = np.array([cx, cy])
        for idx in named:
            pts[idx] = centre + rot @ (pts[idx] - centre)
    return pts


def _area(poly):
    """Shoelace area of a quad."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_roi_covers_enough_of_the_face_to_beat_the_noise_floor():
    """The regression guard for the sliver bug.

    Spatial averaging cuts noise as 1/sqrt(N) and the modulation is under 1% of
    intensity, so ROI area is not cosmetic — an 8x smaller region gives away a
    factor of ~3 in SNR, which is the whole margin.
    """
    polys = landmarks_to_polygons(_landmarks())
    assert len(polys) == 3  # forehead + two cheeks

    total = sum(_area(p) for p in polys)
    face_bbox = FACE_W * FACE_H
    assert 0.15 * face_bbox < total < 0.45 * face_bbox, (
        f"ROI is {total:.0f} px^2 = {total / face_bbox:.1%} of the face bbox"
    )
    # In absolute terms, on a face this size, comfortably past the floor.
    assert total > 8000


def test_forehead_region_is_not_a_sliver():
    """The exact shape of the original bug: a correct-width, near-zero-height box."""
    forehead = landmarks_to_polygons(_landmarks())[0]
    p = np.asarray(forehead)
    width = np.linalg.norm(p[1] - p[0])
    height = np.linalg.norm(p[3] - p[0])
    assert height > 0.4 * width, f"forehead {width:.0f}x{height:.0f} px is a sliver"
    assert height > 0.2 * FACE_H


def test_roi_rotates_with_a_head_tilt():
    """Built from landmark vectors, so a tilt must move the box, not just the face."""
    upright = landmarks_to_polygons(_landmarks(angle_deg=0))[0]
    tilted = landmarks_to_polygons(_landmarks(angle_deg=25))[0]

    # Same area (rigid rotation), different orientation.
    assert _area(tilted) == pytest.approx(_area(upright), rel=0.02)
    edge_up = upright[1] - upright[0]
    edge_tilt = tilted[1] - tilted[0]
    cos = np.dot(edge_up, edge_tilt) / (np.linalg.norm(edge_up) * np.linalg.norm(edge_tilt))
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) == pytest.approx(25.0, abs=1.0)


def test_roi_scales_with_distance_from_camera():
    """Everything is in units of interocular distance, so it must scale cleanly."""
    near = landmarks_to_polygons(_landmarks())
    pts_far = _landmarks()
    pts_far = 320.0 + (pts_far - 320.0) * 0.5  # subject twice as far away
    far = landmarks_to_polygons(pts_far)
    assert sum(_area(p) for p in far) == pytest.approx(
        0.25 * sum(_area(p) for p in near), rel=0.05
    )


def test_degenerate_landmarks_yield_no_polygons():
    assert landmarks_to_polygons(np.zeros((478, 2))) == []


# --------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------


def test_find_gaps_locates_dropped_stretches():
    t = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(7, 10, 1 / 30)])
    gaps = find_gaps(t, max_gap=0.5)
    assert len(gaps) == 1
    assert gaps[0][0] == pytest.approx(4.97, abs=0.05)
    assert gaps[0][1] == pytest.approx(7.0, abs=0.05)


def test_max_gap_in_span_only_counts_overlapping_intervals():
    t = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(7, 12, 1 / 30)])
    assert max_gap_in_span(t, 0.0, 4.0) < 0.1  # before the hole
    assert max_gap_in_span(t, 8.0, 11.0) < 0.1  # after it
    assert max_gap_in_span(t, 3.0, 8.0) == pytest.approx(2.0, abs=0.1)


def test_gate_rejects_a_gap_even_when_the_snr_looks_excellent():
    """A spline overshoot is narrowband, so SNR alone cannot catch it.

    This is the failure mode the SNR gate structurally cannot see: interpolated
    fiction that scores well precisely because it is smooth.
    """
    fs = 30.0
    t = np.arange(int(15 * fs)) / fs
    clean = estimate_bpm(np.sin(2 * np.pi * 1.2 * t), fs)
    assert clean.snr > 5.0

    cfg = PipelineConfig(fs=fs, max_gap_sec=0.5)
    ok, reason = gate(clean, max_gap=0.03, config=cfg)
    assert ok and reason == ""

    ok, reason = gate(clean, max_gap=1.8, config=cfg)
    assert not ok and "gap" in reason


def test_gate_rejects_low_snr():
    fs = 30.0
    t = np.arange(int(15 * fs)) / fs
    res = estimate_bpm(np.sin(2 * np.pi * 1.2 * t), fs)
    cfg = PipelineConfig(fs=fs, snr_threshold=99.0)
    ok, reason = gate(res, max_gap=0.0, config=cfg)
    assert not ok and "SNR" in reason


# --------------------------------------------------------------------------
# Cross-method agreement
# --------------------------------------------------------------------------


def _est(method, bpm, accepted=True):
    return WindowEstimate(
        t_start=0.0, t_end=15.0, t_center=7.5, method=method, bpm=bpm,
        bpm_smoothed=bpm, snr=1.0, accepted=accepted, resolution_bpm=4.0,
    )


def test_agreement_spread_flags_the_disagreeing_case():
    """The live-webcam numbers that started this: 14 BPM apart, all 'usable'."""
    bad = {m: _est(m, v) for m, v in
           zip(("green", "ica", "chrom", "pos"), (47.9, 48.6, 62.3, 49.3))}
    assert agreement_spread(bad) == pytest.approx(14.4, abs=0.1)

    good = {m: _est(m, v) for m, v in
            zip(("green", "ica", "chrom", "pos"), (72.0, 72.0, 72.0, 71.9))}
    assert agreement_spread(good) < 1.0


def test_agreement_spread_ignores_rejected_arms_and_degenerate_input():
    mixed = {
        "green": _est("green", 72.0),
        "ica": _est("ica", 150.0, accepted=False),
        "chrom": _est("chrom", 72.5),
    }
    assert agreement_spread(mixed) == pytest.approx(0.5, abs=0.01)
    assert np.isnan(agreement_spread({"green": _est("green", 72.0)}))


# --------------------------------------------------------------------------
# Dropout recovery
# --------------------------------------------------------------------------


def test_estimator_discards_the_buffer_on_a_dropout():
    """A 1.8 s occlusion must not leave a spline artefact in the next window."""
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    fs = 30.0
    cfg = PipelineConfig(fs=fs, window_sec=15.0, max_gap_sec=0.5)
    est = SlidingWindowEstimator(cfg)

    clip = synth_rgb(duration=14.0, fs=fs, bpm=72.0, rng=0)
    for t, rgb in zip(clip.t, clip.rgb):
        est.push(t, rgb)
    assert est.buffered_sec > 13.0
    assert est.gaps_seen == 0

    # Face lost for 1.8 s, then reacquired. The caller reports each missing
    # frame — a bare timestamp jump is deliberately NOT enough, since a slow
    # loop produces the same jump without any data actually going missing.
    for k in range(1, 54):
        est.mark_lost(clip.t[-1] + k / fs)
    est.push(clip.t[-1] + 1.8, clip.rgb[-1])
    assert est.gaps_seen == 1
    assert est.last_gap == pytest.approx(1.8, abs=0.05)
    assert est.buffered_sec < 0.1, "pre-gap samples must be gone, not splined across"
    assert not est.ready
    assert est.refill_remaining == pytest.approx(15.0, abs=0.2)


def test_estimator_recovers_a_correct_bpm_after_a_dropout():
    """And the estimate that eventually arrives is clean, not gap-contaminated."""
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    fs = 30.0
    # An explicit permissive gate: this test is about the buffer recovering a
    # clean window after a dropout, not about where the SNR bar sits. Inheriting
    # the production -3 dB default made it fail when that default was
    # recalibrated, even though the recovered BPM was correct to 0.7 BPM.
    cfg = PipelineConfig(fs=fs, window_sec=15.0, max_gap_sec=0.5, snr_threshold=-10.0)
    est = SlidingWindowEstimator(cfg)

    before = synth_rgb(duration=10.0, fs=fs, bpm=72.0, rng=0)
    for t, rgb in zip(before.t, before.rgb):
        est.push(t, rgb)

    after = synth_rgb(duration=16.0, fs=fs, bpm=72.0, rng=1)
    offset = before.t[-1] + 2.0  # a 2 s dropout, reported frame by frame
    for k in range(1, 60):
        est.mark_lost(before.t[-1] + k / fs)
    for t, rgb in zip(after.t, after.rgb):
        est.push(offset + t, rgb)

    assert est.gaps_seen == 1
    assert est.ready
    out = est.estimate()
    for method, e in out.items():
        assert e.accepted, f"{method} rejected: {e.reject_reason}"
        assert e.bpm == pytest.approx(72.0, abs=2.0)
    assert agreement_spread(out) < 3.0


def test_dropout_clears_stale_estimates_so_none_are_shown_as_live():
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    fs = 30.0
    cfg = PipelineConfig(fs=fs, window_sec=15.0, max_gap_sec=0.5)
    est = SlidingWindowEstimator(cfg)
    clip = synth_rgb(duration=16.0, fs=fs, bpm=72.0, rng=0)
    for t, rgb in zip(clip.t, clip.rgb):
        est.push(t, rgb)
    est.estimate()
    assert est.latest

    for k in range(1, 90):
        est.mark_lost(clip.t[-1] + k / fs)
    est.push(clip.t[-1] + 3.0, clip.rgb[-1])
    assert est.latest == {}, "a pre-dropout BPM must not survive as a live reading"


def test_cheek_regions_sit_below_the_eyes():
    """They used to land on the lenses of a pair of glasses."""
    pts = _landmarks()
    eye_y = pts[33][1]
    forehead, cheek_l, cheek_r = landmarks_to_polygons(pts)

    for cheek in (cheek_l, cheek_r):
        top_y = np.asarray(cheek)[:, 1].min()
        assert top_y > eye_y + 0.15 * FACE_H, "cheek box reaches up into the eye/glasses region"
    # And still below the forehead, which must stay above the eyes.
    assert np.asarray(forehead)[:, 1].max() < eye_y


def test_slow_processing_is_not_treated_as_a_dropout():
    """The bug that made the app blank out while the user sat perfectly still.

    A loop too slow to keep up produces the same long inter-sample intervals as
    a face dropout. Inferring "dropout" from the interval reset the buffer on
    every frame, so it never filled, so the estimate never arrived — and the
    display stayed blank forever with nobody having moved.
    """
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    cfg = PipelineConfig(fs=30.0, window_sec=15.0, max_gap_sec=0.5)
    est = SlidingWindowEstimator(cfg)

    # 4 fps: every interval is 0.25 s, and bursts of 0.6 s — well over
    # max_gap_sec — but the face was tracked the entire time.
    clip = synth_rgb(duration=20.0, fs=30.0, bpm=72.0, rng=0)
    t = 0.0
    for i, rgb in enumerate(clip.rgb[::7]):
        t += 0.6 if i % 5 == 0 else 0.25
        est.push(t, rgb)

    assert est.gaps_seen == 0, "no face loss was reported, so nothing may reset"
    assert est.ready, "the buffer must still fill when the loop is merely slow"
    assert est.too_slow and est.effective_fps < 10.0


def test_a_long_stall_resets_even_without_a_reported_loss():
    """Belt and braces: if the whole app froze, the buffer is stale regardless."""
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    cfg = PipelineConfig(fs=30.0, window_sec=15.0, max_gap_sec=0.5, stall_sec=3.0)
    est = SlidingWindowEstimator(cfg)
    clip = synth_rgb(duration=10.0, fs=30.0, bpm=72.0, rng=0)
    for t, rgb in zip(clip.t, clip.rgb):
        est.push(t, rgb)

    est.push(clip.t[-1] + 8.0, clip.rgb[-1])  # nothing reported, but 8 s passed
    assert est.gaps_seen == 1
    assert est.buffered_sec < 0.1


def test_effective_fps_measures_the_loop_not_the_camera():
    from rppg.pipeline import SlidingWindowEstimator
    from rppg.synthetic import synth_rgb

    cfg = PipelineConfig(fs=30.0, window_sec=15.0)
    est = SlidingWindowEstimator(cfg)
    clip = synth_rgb(duration=12.0, fs=30.0, rng=0)
    for i, rgb in enumerate(clip.rgb[::3]):  # a third of the frames make it
        est.push(i / 10.0, rgb)
    assert est.effective_fps == pytest.approx(10.0, rel=0.05)
    assert not est.too_slow  # exactly at the threshold is acceptable


def test_gap_threshold_scales_with_window_length():
    """0.5 s is 3% of a 15 s window but 10% of a 5 s one — very different costs."""
    assert PipelineConfig(window_sec=5.0).max_gap_sec == pytest.approx(0.25)
    assert PipelineConfig(window_sec=15.0).max_gap_sec == pytest.approx(0.75)
    assert PipelineConfig(window_sec=30.0).max_gap_sec == pytest.approx(1.50)
    # An explicit value still wins, so experiments can pin it.
    assert PipelineConfig(window_sec=15.0, max_gap_sec=0.4).max_gap_sec == pytest.approx(0.4)


def test_gap_threshold_matches_the_measured_cost():
    """The threshold is set from measurement, and this pins the measurement.

    Injecting one contiguous gap into a 15 s window and scoring all four arms:

        gap      % of window   MAE (BPM)
        0 ms         0%          2.49     baseline
        500 ms       3.3%        2.22     no measurable cost
        750 ms       5.0%        4.30     marginal
        1000 ms      6.7%        7.77     clearly harmful

    A gap costing nothing must be accepted; one costing 3x baseline must not.
    """
    from rppg.resample import resample_uniform
    from rppg.spectral import estimate_bpm
    from rppg.synthetic import nonuniform_timestamps, synth_rgb

    fs, win_sec = 30.0, 15.0
    cfg = PipelineConfig(fs=fs, window_sec=win_sec)

    def err_with_gap(gap_s):
        errs = []
        for s in range(4):
            ts = nonuniform_timestamps(win_sec + 6, fs, jitter_ms=3.0, rng=500 + s)
            clip = synth_rgb(timestamps=ts, bpm=float(58 + 7 * s), rng=500 + s)
            t, rgb = clip.t, clip.rgb
            if gap_s > 0:
                mid = t[0] + (t[-1] - t[0]) / 2
                keep = ~((t > mid - gap_s / 2) & (t < mid + gap_s / 2))
                t, rgb = t[keep], rgb[keep]
            t_u, rgb_u = resample_uniform(t, rgb, fs=fs)
            n = int(win_sec * fs)
            seg = rgb_u[:n]
            ref = clip.mean_bpm_over(t_u[0], t_u[n - 1])
            for m in cfg.methods:
                from rppg.pipeline import project_and_filter

                errs.append(abs(estimate_bpm(project_and_filter(seg, fs, m, cfg), fs).bpm - ref))
        return float(np.mean(errs))

    baseline = err_with_gap(0.0)
    at_threshold = err_with_gap(cfg.max_gap_sec * 0.7)  # comfortably accepted
    far_over = err_with_gap(cfg.max_gap_sec * 2.5)  # comfortably rejected

    assert at_threshold < baseline + 1.5, (
        f"a gap we accept costs too much: {at_threshold:.2f} vs baseline {baseline:.2f}"
    )
    assert far_over > baseline + 2.0, (
        f"a gap we reject should actually hurt: {far_over:.2f} vs baseline {baseline:.2f}"
    )


# --------------------------------------------------------------------------
# Landmark rate decoupled from sampling rate
# --------------------------------------------------------------------------


class _CountingROI:
    """MediaPipeROI's redetect logic with the detector stubbed out."""

    def __init__(self, redetect_every, detections):
        from rppg.roi import MediaPipeROI

        self.redetect_every = max(1, int(redetect_every))
        self._polys = None
        self._since_detect = 1 << 30
        self.calls = 0
        self._script = list(detections)
        self._impl = MediaPipeROI.__call__

    def _detect(self, frame):
        poly = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return poly

    use_skin_mask = False

    def __call__(self, frame):
        return self._impl(self, frame)


def _frame(i, value=100):
    from rppg.capture import Frame

    img = np.full((60, 60, 3), value, np.uint8)
    return Frame(index=i, timestamp=i / 30.0, image=img)


POLY = [np.array([[10, 10], [50, 10], [50, 50], [10, 50]], np.int32)]


def test_redetect_every_runs_detector_once_per_k_frames():
    roi = _CountingROI(3, [POLY] * 10)
    for i in range(9):
        roi(_frame(i))
    # Frames 0, 3, 6 detect; the rest reuse. Cutting the 18.5 ms landmark stage
    # to a third is what buys back the frame budget.
    assert roi.calls == 3


def test_reused_polygon_still_produces_a_sample_every_frame():
    roi = _CountingROI(3, [POLY] * 10)
    samples = [roi(_frame(i)) for i in range(9)]
    assert all(s.ok for s in samples), "non-detect frames must still be measured"
    assert all(s.n_pixels > 0 for s in samples)


def test_failed_detection_drops_the_stale_polygon():
    # Detect once (found), then a detect that fails. The frames in between must
    # not keep averaging over wherever the face used to be.
    roi = _CountingROI(3, [POLY, None])
    assert roi(_frame(0)).ok
    assert roi(_frame(1)).ok  # reused
    assert roi(_frame(2)).ok  # reused
    assert not roi(_frame(3)).ok  # detect fails -> lost
    assert not roi(_frame(4)).ok, "must not resurrect the stale polygon"


def test_redetect_every_one_detects_on_every_frame():
    roi = _CountingROI(1, [POLY] * 10)
    for i in range(5):
        roi(_frame(i))
    assert roi.calls == 5
