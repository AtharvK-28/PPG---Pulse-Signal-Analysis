"""Offline record-and-analyse path.

The point of recording is that nothing in the measurement depends on keeping
up with the camera. These tests pin the two things that would silently undo
that: a lossy codec, and timestamps that describe delivery rather than
exposure.
"""

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.offline import analyse, regularise_timestamps, summarise
from rppg.record import CaptureReport, burst_score, describe

FS = 30.0


def _report(**kw):
    base = dict(
        frames=1800, duration_sec=60.0, fps=30.0, dt_median_ms=33.3, dt_p99_ms=49.0,
        dt_max_ms=55.0, late_frames=3, dropped=0, max_queue=12, width=640,
        height=480, fourcc="YUY2", mean_luma=120.0, burst_fraction=0.0,
        burst_pairs=0, backend="dshow",
    )
    base.update(kw)
    return CaptureReport(**base)


# --------------------------------------------------------------------------
# Capture quality verdict
# --------------------------------------------------------------------------


def test_clean_recording_is_accepted():
    ok, why = _report().verdict()
    assert ok, why


def test_stalled_capture_is_rejected():
    """The exact profile of the session that failed: p99 438 ms, median 32."""
    ok, why = _report(fps=18.2, dt_median_ms=31.8, dt_p99_ms=438.0).verdict()
    assert not ok
    assert "stalled" in why


def test_low_frame_rate_is_rejected_for_aliasing():
    ok, why = _report(fps=6.0).verdict()
    assert not ok
    assert "aliased" in why


def test_dark_and_clipped_recordings_are_rejected():
    assert not _report(mean_luma=2.0).verdict()[0]
    assert not _report(mean_luma=250.0).verdict()[0]


def test_dropped_frames_are_reported():
    ok, why = _report(dropped=17).verdict()
    assert not ok and "17 frames dropped" in why


def test_bursty_delivery_is_flagged():
    ok, why = _report(burst_fraction=0.67, burst_pairs=600).verdict()
    assert not ok
    assert "bursty" in why and "transport" in why


def test_describe_mentions_delivery_evenness():
    assert "(even)" in describe(_report())
    assert "bursty" in describe(_report(burst_fraction=0.67, burst_pairs=600))


# --------------------------------------------------------------------------
# Burst detection
# --------------------------------------------------------------------------


def test_burst_score_is_zero_for_even_delivery():
    frac, pairs = burst_score(np.full(200, 33.3))
    assert frac == 0.0 and pairs == 0


def test_burst_score_detects_paired_delivery():
    """Measured MSMF signature: 16 ms then 50 ms, repeating."""
    dt = np.tile([16.0, 50.0], 100)
    frac, pairs = burst_score(dt)
    assert frac > 0.4
    assert pairs > 80


def test_burst_score_tolerates_mild_jitter():
    rng = np.random.default_rng(0)
    frac, pairs = burst_score(33.3 + rng.normal(0, 2.0, 300))
    assert pairs == 0


# --------------------------------------------------------------------------
# Timestamp reconstruction
# --------------------------------------------------------------------------


def _bursty(n=600, period=1 / 30.0):
    """Uniform exposure, paired delivery: the MSMF failure mode."""
    true_t = np.arange(n) * period
    delivered = true_t.copy()
    delivered[0::2] = true_t[1::2] - 0.017  # first of each pair arrives late
    return np.sort(delivered), true_t


def test_bursty_but_lossless_delivery_is_reconstructed():
    t, _ = _bursty()
    out, changed, reason = regularise_timestamps(t, nominal_fps=FS)
    assert changed, reason
    assert np.allclose(np.diff(out), np.median(np.diff(out)))


def test_even_delivery_is_left_alone():
    t = np.arange(600) / FS
    out, changed, reason = regularise_timestamps(t, nominal_fps=FS)
    assert not changed
    assert np.allclose(out, t)


def test_real_gaps_are_never_reconstructed_away():
    """A genuine dropout must survive: reconstructing it would hide lost data."""
    t = np.arange(900) / FS
    keep = np.ones(900, dtype=bool)
    keep[300:360] = False  # a real 2 s dropout
    out, changed, reason = regularise_timestamps(t[keep], nominal_fps=FS)
    assert not changed
    assert "genuinely lost" in reason
    assert np.allclose(out, t[keep])


def test_reconstruction_uses_the_measured_period_not_the_nominal_one():
    """A camera running at 29.6 fps must not be forced onto a 30.000 grid."""
    true_fps = 29.6
    t, _ = _bursty(n=600, period=1 / true_fps)
    out, changed, _ = regularise_timestamps(t, nominal_fps=FS)
    assert changed
    assert 1.0 / np.median(np.diff(out)) == pytest.approx(true_fps, abs=0.2)


def test_too_few_frames_is_refused_rather_than_guessed():
    out, changed, reason = regularise_timestamps(np.arange(5) / FS)
    assert not changed and "too few" in reason


def _score(times, rgb):
    from rppg.preprocess import bandpass, detrend
    from rppg.resample import resample_uniform
    from rppg.spectral import estimate_bpm

    order = np.argsort(times)
    times, rgb = times[order], rgb[order]
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    tu, xu = resample_uniform(times[keep], rgb[keep], FS)
    est = estimate_bpm(bandpass(detrend(xu[:, 1], fs=FS), FS), FS)
    return est.bpm, est.snr


def _tone(true_t, bpm=72.0):
    dc = np.array([120.0, 95.0, 88.0])
    return dc * (1 + 0.006 * np.sin(2 * np.pi * bpm / 60 * true_t)[:, None])


def test_strictly_alternating_delivery_costs_nothing():
    """Measured, and it is why the burst heuristic was not the right trigger.

    A two-frame burst puts the timing error at Nyquist, where the bandpass
    removes it: +42.18 dB against +42.17 for perfect delivery. The MSMF
    signature on this hardware is exactly this pattern.
    """
    delivered, true_t = _bursty(n=900)
    rgb = _tone(true_t)
    _, snr_true = _score(true_t, rgb)
    _, snr_burst = _score(delivered, rgb)
    assert abs(snr_burst - snr_true) < 1.0


def test_reconstruction_recovers_irregular_transport_jitter():
    """The case that justifies reconstructing: -6.4 dB -> +42.2 dB, measured."""
    rng = np.random.default_rng(0)
    true_t = np.arange(900) / FS
    rgb = _tone(true_t)
    jittered = true_t + rng.normal(0, 0.017, len(true_t))

    _, snr_true = _score(true_t, rgb)
    _, snr_jitter = _score(jittered, rgb)
    fixed, changed, _ = regularise_timestamps(np.sort(jittered), nominal_fps=FS)
    _, snr_fixed = _score(fixed, rgb)

    assert changed
    assert snr_jitter < snr_true - 20.0, (
        "17 ms of transport jitter should be devastating, got %.1f vs %.1f dB"
        % (snr_jitter, snr_true)
    )
    assert snr_fixed > snr_jitter + 20.0, (
        "reconstruction should recover it: %.1f -> %.1f dB" % (snr_jitter, snr_fixed)
    )


def test_reconstruction_does_not_help_when_frames_are_truly_missing():
    """And must not be attempted: the gap is data, not noise."""
    true_t = np.arange(900) / FS
    rgb = _tone(true_t)
    keep = np.ones(900, dtype=bool)
    keep[300:360] = False
    _, changed, _ = regularise_timestamps(true_t[keep], nominal_fps=FS)
    assert not changed
    # And the dropout genuinely costs SNR, so the refusal is not free.
    _, snr_full = _score(true_t, rgb)
    _, snr_gap = _score(true_t[keep], rgb[keep])
    assert snr_gap < snr_full


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _clip(bpm=72.0, dur=60.0, fs=FS, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    p = np.sin(2 * np.pi * bpm / 60 * t)
    dc = np.array([120.0, 95.0, 88.0])
    rgb = dc * (1 + 0.005 * np.array([0.45, 1.0, 0.35]) * p[:, None])
    return t, rgb + rng.normal(0, 0.0005 * dc, (len(t), 3))


def test_analyse_recovers_a_known_rate_on_every_arm():
    t, rgb = _clip(bpm=72.0)
    results, _, _ = analyse(t, rgb, fs=FS, window_sec=30.0)
    for method, arr in results.items():
        assert len(arr) > 0, method
        assert abs(np.median(arr[:, 1]) - 72.0) < 3.0, method


def test_summarise_reports_error_only_when_a_reference_exists():
    t, rgb = _clip()
    results, _, _ = analyse(t, rgb, fs=FS, window_sec=30.0)
    assert "mae" not in summarise(results)[0]
    assert "mae" in summarise(results, reference=72.0)[0]


def test_longer_windows_give_finer_resolution():
    """60/T, and the reason the offline path defaults to 30 s rather than 15."""
    from rppg.spectral import bpm_resolution

    assert bpm_resolution(15.0) == pytest.approx(4.0)
    assert bpm_resolution(30.0) == pytest.approx(2.0)


def test_analyse_returns_empty_rather_than_raising_on_a_short_clip():
    t, rgb = _clip(dur=5.0)
    results, _, _ = analyse(t, rgb, fs=FS, window_sec=30.0)
    assert all(len(v) == 0 for v in results.values())


# --------------------------------------------------------------------------
# The recorder's codec
# --------------------------------------------------------------------------


def test_recorder_codec_round_trips_exactly(tmp_path):
    """The recording is worthless if the codec discards the modulation.

    A lossy codec is *designed* to throw away sub-percent uniform brightness
    changes as imperceptible, which is precisely what the pulse is. Written
    against the recorder's own CODEC constant so changing it fails here.
    """
    import cv2

    from rppg.record import CODEC, CONTAINER

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (64, 80, 3), dtype=np.uint8) for _ in range(12)]
    path = tmp_path / ("clip" + CONTAINER)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*CODEC), 30.0, (80, 64))
    if not writer.isOpened():
        pytest.skip("no %s encoder in this OpenCV build" % CODEC)
    for f in frames:
        writer.write(f)
    writer.release()

    cap = cv2.VideoCapture(str(path))
    read_back = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        read_back.append(img)
    cap.release()

    assert len(read_back) == len(frames)
    for i, (a, b) in enumerate(zip(frames, read_back)):
        assert np.array_equal(a, b), "frame %d changed on the round trip" % i
