"""Time-domain pulse analysis: beat detection, intervals, HRV.

Each test targets a failure actually observed on BIDMC against ECG ground
truth, so a regression shows up as a failing test rather than as a
plausible-looking number.
"""

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.beats import beat_rate, clean_intervals, detect_beats, intervals, slope_sum
from rppg.ecg import match_beats, pan_tompkins
from rppg.hrv import frequency_domain, summary, tachogram, time_domain

FS = 125.0


def synth_ppg(bpm=75.0, dur=60.0, fs=FS, notch=0.35, hrv_ms=0.0, seed=0):
    """PPG-like waveform: steep upstroke, rounded peak, dicrotic notch.

    The notch is the point. A waveform without one cannot demonstrate that the
    detector avoids counting it, which is the failure that halved precision on
    bidmc47.
    """
    rng = np.random.default_rng(seed)
    beat_times, t_now = [], 0.0
    base = 60.0 / bpm
    while t_now < dur:
        beat_times.append(t_now)
        t_now += base + (rng.normal(0, hrv_ms / 1000.0) if hrv_ms else 0.0)
    t = np.arange(0, dur, 1 / fs)
    x = np.zeros_like(t)
    for bt in beat_times:
        u = (t - bt) / base
        m = (u >= 0) & (u < 1)
        uu = u[m]
        # Systolic: fast rise, slower fall. Dicrotic: a smaller bump at ~40%.
        systole = np.exp(-((uu - 0.15) ** 2) / (2 * 0.055**2))
        dicrotic = notch * np.exp(-((uu - 0.40) ** 2) / (2 * 0.06**2))
        x[m] += systole + dicrotic
    return x + rng.normal(0, 0.01, len(t)), np.asarray(beat_times)


def synth_ecg(bpm=75.0, dur=60.0, fs=FS, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    x = np.zeros_like(t)
    beats = np.arange(0, dur, 60.0 / bpm)
    for b in beats:
        x = x + 1.00 * np.exp(-((t - b) ** 2) / (2 * 0.010**2))          # R
        x = x - 0.15 * np.exp(-((t - b + 0.03) ** 2) / (2 * 0.012**2))   # Q
        x = x + 0.25 * np.exp(-((t - b - 0.25) ** 2) / (2 * 0.040**2))   # T
    return x + rng.normal(0, 0.02, len(t)), beats


# --------------------------------------------------------------------------
# Beat detection
# --------------------------------------------------------------------------


def test_slope_sum_responds_to_rises_only():
    """A pure fall must contribute nothing, or diastole competes with systole."""
    fs = 100.0
    rising = np.linspace(0, 1, 200)
    assert slope_sum(rising, fs).max() > 0
    assert np.allclose(slope_sum(rising[::-1], fs), 0.0)


def test_slope_sum_looks_backwards_not_forwards():
    """A forward-looking window puts every beat early by half its length."""
    fs = 100.0
    x = np.zeros(300)
    x[150:160] = np.linspace(0, 1, 10)
    peak = int(np.argmax(slope_sum(x, fs, win_sec=0.2)))
    assert peak >= 159, "SSF peak at %d should not precede the end of the rise" % peak


def test_detects_the_right_number_of_beats():
    x, beats = synth_ppg(bpm=75.0, dur=60.0)
    found = detect_beats(x, FS)
    assert abs(len(found) - len(beats)) <= 2


def test_does_not_count_the_dicrotic_notch():
    """The bidmc47 failure: 1306 detections against 671 R waves, 51% precision."""
    x, beats = synth_ppg(bpm=90.0, dur=60.0, notch=0.6)
    found = detect_beats(x, FS)
    assert len(found) < 1.3 * len(beats), (
        "%d detections for %d beats - the notch is being counted"
        % (len(found), len(beats))
    )


def test_notch_rejection_holds_at_a_high_rate():
    """At 60 BPM a fixed 300 ms refractory hides the bug; at 120 it does not."""
    x, beats = synth_ppg(bpm=120.0, dur=60.0, notch=0.6)
    found = detect_beats(x, FS)
    assert len(found) < 1.3 * len(beats)


def test_beat_times_land_on_the_upstroke_consistently():
    """Jitter here goes straight into RMSSD; measured 2-3x inflation on BIDMC."""
    x, beats = synth_ppg(bpm=75.0, dur=60.0)
    found = detect_beats(x, FS) / FS
    offs = [found[np.argmin(np.abs(found - b))] - b for b in beats[2:-2]]
    assert np.std(offs) * 1000 < 15.0, "fiducial jitter %.1f ms" % (np.std(offs) * 1000)


def test_beat_rate_matches_the_synthesised_rate():
    for bpm in (55.0, 75.0, 110.0):
        x, _ = synth_ppg(bpm=bpm, dur=60.0)
        assert abs(beat_rate(detect_beats(x, FS), FS) - bpm) < 2.0


def test_short_signal_returns_no_beats_rather_than_raising():
    assert len(detect_beats(np.zeros(50), FS)) == 0


# --------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------


def test_intervals_are_milliseconds():
    peaks = np.array([0, 125, 250, 375])  # 1 s apart at 125 Hz
    assert np.allclose(intervals(peaks, FS), 1000.0)


def test_clean_intervals_removes_a_merged_interval_from_a_missed_beat():
    """One missed beat doubles an interval and inflates RMSSD roughly twofold."""
    ibi = np.full(40, 800.0)
    ibi[20] = 1600.0
    kept, mask = clean_intervals(ibi)
    assert 1600.0 not in kept
    assert mask.sum() == 39


def test_clean_intervals_rejects_impossible_values():
    ibi = np.array([800.0, 100.0, 810.0, 5000.0, 795.0])
    kept, _ = clean_intervals(ibi)
    assert kept.min() >= 300 and kept.max() <= 2000


def test_clean_intervals_keeps_a_healthy_series_intact():
    rng = np.random.default_rng(0)
    ibi = 800 + rng.normal(0, 20, 100)
    kept, _ = clean_intervals(ibi)
    assert len(kept) == 100


# --------------------------------------------------------------------------
# ECG reference
# --------------------------------------------------------------------------


def test_pan_tompkins_finds_r_peaks():
    x, beats = synth_ecg(bpm=75.0, dur=60.0)
    found = pan_tompkins(x, FS)
    assert abs(len(found) - len(beats)) <= 2


def test_pan_tompkins_is_not_fooled_by_the_t_wave():
    """The T wave is the ECG's equivalent of the PPG's dicrotic notch."""
    x, beats = synth_ecg(bpm=100.0, dur=60.0)
    found = pan_tompkins(x, FS)
    assert len(found) < 1.3 * len(beats)


def test_match_beats_scores_a_perfect_match():
    ref = np.arange(0, 1000, 100)
    tp, fn, fp, off = match_beats(ref, ref.copy(), FS)
    assert (tp, fn, fp) == (len(ref), 0, 0)
    assert np.allclose(off, 0.0)


def test_match_beats_reports_a_constant_lag_as_offset_not_error():
    """PPG lags ECG by the pulse transit time; that is physiology, not a miss."""
    ref = np.arange(0, 2000, 125).astype(float)
    test = ref + 25  # 200 ms at 125 Hz
    tp, fn, fp, off = match_beats(ref, test, FS, tolerance=0.35)
    assert fn == 0 and fp == 0
    assert np.allclose(off, 200.0)


def test_match_beats_counts_extra_detections_as_false_positives():
    ref = np.arange(0, 1000, 100).astype(float)
    test = np.sort(np.concatenate([ref, ref + 30]))
    tp, fn, fp, _ = match_beats(ref, test, FS, tolerance=0.1)
    assert fp > 0


# --------------------------------------------------------------------------
# HRV
# --------------------------------------------------------------------------


def test_time_domain_on_a_constant_series_has_zero_variability():
    td = time_domain(np.full(50, 800.0))
    assert td.sdnn == pytest.approx(0.0)
    assert td.rmssd == pytest.approx(0.0)
    assert td.pnn50 == pytest.approx(0.0)
    assert td.mean_hr == pytest.approx(75.0)


def test_rmssd_ignores_slow_drift_but_sdnn_does_not():
    """The distinction that makes RMSSD vagal and SDNN a total-variability measure."""
    n = 300
    drift = 800 + 60 * np.sin(np.linspace(0, 2 * np.pi, n))
    flat = np.full(n, 800.0)
    assert time_domain(drift).sdnn > 10 * time_domain(flat).sdnn
    assert time_domain(drift).rmssd < 5.0


def test_pnn50_counts_successive_differences_over_50ms():
    ibi = np.array([800.0, 800, 900, 900, 800])  # diffs 0, 100, 0, -100
    assert time_domain(ibi).pnn50 == pytest.approx(50.0)


def test_tachogram_uses_beat_time_not_beat_number():
    """Indexing by beat number distorts the frequency axis wherever HR changed."""
    ibi = np.concatenate([np.full(50, 1000.0), np.full(50, 500.0)])
    grid, series = tachogram(ibi)
    assert grid[-1] == pytest.approx(74.5, abs=1.5)  # 50*1 s + 50*0.5 s
    # Two thirds of the elapsed time is spent in the slow half.
    assert np.mean(series[: len(series) // 2]) > 900


def test_frequency_domain_finds_respiratory_sinus_arrhythmia():
    """A 0.25 Hz modulation of the intervals must land in the HF band."""
    n = 600
    beat_t = np.cumsum(np.full(n, 0.8))
    ibi = 800 + 40 * np.sin(2 * np.pi * 0.25 * beat_t)
    fd = frequency_domain(ibi)
    assert fd.hf > 5 * fd.lf, "HF %.1f should dominate LF %.1f" % (fd.hf, fd.lf)
    assert fd.reliable


def test_frequency_domain_finds_a_low_frequency_oscillation():
    n = 600
    beat_t = np.cumsum(np.full(n, 0.8))
    ibi = 800 + 40 * np.sin(2 * np.pi * 0.10 * beat_t)
    fd = frequency_domain(ibi)
    assert fd.lf > 5 * fd.hf
    assert fd.lf_nu > 80


def test_short_record_is_flagged_unreliable():
    """LF needs ~2 minutes; a 30 s number is arithmetic, not a measurement."""
    ibi = np.full(40, 800.0)  # 32 s
    assert not frequency_domain(ibi).reliable


def test_summary_reports_every_metric():
    rng = np.random.default_rng(0)
    ibi = 800 + rng.normal(0, 30, 400)
    s = summary(ibi)
    for key in ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct",
                "lf_hf", "lf_nu", "hf_nu", "hrv_reliable"):
        assert key in s
    assert s["mean_hr_bpm"] == pytest.approx(75.0, abs=1.0)


def test_hrv_survives_a_series_too_short_to_analyse():
    assert np.isnan(time_domain(np.array([800.0])).sdnn)
    assert not frequency_domain(np.array([800.0, 810.0])).reliable


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_ppg_and_ecg_agree_on_heart_rate():
    """The BIDMC result in miniature: 0.04 BPM agreement on clean records."""
    ppg, _ = synth_ppg(bpm=72.0, dur=90.0)
    ecg, _ = synth_ecg(bpm=72.0, dur=90.0)
    hr_ppg = beat_rate(detect_beats(ppg, FS), FS)
    hr_ecg = beat_rate(pan_tompkins(ecg, FS), FS)
    assert abs(hr_ppg - hr_ecg) < 1.0


def test_known_hrv_is_recovered_from_the_waveform():
    ppg, _ = synth_ppg(bpm=75.0, dur=180.0, hrv_ms=40.0, seed=3)
    ibi, _ = clean_intervals(intervals(detect_beats(ppg, FS), FS))
    s = summary(ibi)
    assert s["mean_hr_bpm"] == pytest.approx(75.0, abs=2.0)
    assert 15 < s["sdnn_ms"] < 90, "SDNN %.1f outside plausible range" % s["sdnn_ms"]
