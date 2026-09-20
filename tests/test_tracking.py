"""Harmonic peak selection and continuity-constrained HR tracking.

These exist because a more realistic signal model exposed a failure the old one
structurally could not: with a sine-wave "pulse" there are no harmonics, so an
estimator can never mistake one for the fundamental. Give the synthetic pulse a
real PPG's harmonic structure and CHROM starts reporting double the true rate.
"""

import numpy as np
import pytest

from rppg.methods.chrom import chrom
from rppg.pipeline import PipelineConfig, analyse_signal, project_and_filter
from rppg.preprocess import bandpass
from rppg.resample import resample_uniform
from rppg.spectral import estimate_bpm, harmonic_score, periodogram
from rppg.synthetic import nonuniform_timestamps, synth_rgb
from rppg.tracking import score_gram, track_bpm, viterbi_track

FS = 30.0


def _harmonic_rich(f0_hz, duration=15.0, fund_gain=0.65, fs=FS):
    """A signal whose second harmonic is taller than its fundamental.

    This is what CHROM's internal normalisation produces at low heart rates,
    and it is precisely the case a tallest-bin search gets wrong.
    """
    t = np.arange(int(duration * fs)) / fs
    return (
        fund_gain * np.sin(2 * np.pi * f0_hz * t)
        + 1.0 * np.sin(2 * np.pi * 2 * f0_hz * t + 0.7)
        + 0.5 * np.sin(2 * np.pi * 3 * f0_hz * t + 1.9)
    )


def test_peak_selector_reports_double_when_the_harmonic_is_taller():
    """The failure being fixed, demonstrated first."""
    x = _harmonic_rich(1.0)  # true rate 60 BPM
    res = estimate_bpm(x, FS, selector="peak")
    assert res.bpm == pytest.approx(120.0, abs=3.0), "expected the doubling failure"
    # And it is confident about it, which is why SNR cannot catch this.
    assert res.snr > 0.0


def test_harmonic_selector_recovers_the_fundamental():
    x = _harmonic_rich(1.0)
    res = estimate_bpm(x, FS, selector="harmonic")
    assert res.bpm == pytest.approx(60.0, abs=3.0)


def test_harmonic_score_prefers_a_series_over_an_isolated_line():
    """A lone tone gets no harmonic support; a harmonic series does."""
    t = np.arange(int(15 * FS)) / FS
    series = (0.80 * np.sin(2 * np.pi * 1.0 * t)
              + 0.70 * np.sin(2 * np.pi * 2.0 * t)
              + 0.50 * np.sin(2 * np.pi * 3.0 * t))
    both = series + 0.90 * np.sin(2 * np.pi * 1.45 * t)  # taller, but no harmonics

    freqs, power = periodogram(both, FS)
    # The isolated tone is the tallest single line...
    assert freqs[np.argmax(power)] == pytest.approx(1.45, abs=0.05)
    # ...but the harmonic series wins on total evidence.
    cand, score = harmonic_score(freqs, power)
    assert cand[int(np.argmax(score))] == pytest.approx(1.0, abs=0.05)


def test_chrom_per_window_bandpass_tilts_the_spectrum():
    """Why the default is a global bandpass.

    A 1.6 s window is 48 samples at 30 fps. A 0.97 Hz fundamental has a
    31-sample period — barely one cycle — while its second harmonic gets three.
    Filtering inside that window attenuates the fundamental far more than the
    harmonic, and the estimator then reports double.
    """
    ts = nonuniform_timestamps(60.0, FS, jitter_ms=3.0, rng=2000)
    clip = synth_rgb(timestamps=ts, bpm=58.0, rng=2000)
    _, rgb_u = resample_uniform(clip.t, clip.rgb, FS)

    def ratio(scope):
        sig = bandpass(chrom(rgb_u, FS, bandpass_scope=scope), FS)
        freqs, power = periodogram(sig[: int(15 * FS)], FS)
        at = lambda hz: power[np.argmin(np.abs(freqs - hz))]  # noqa: E731
        return at(2 * 58 / 60) / at(58 / 60)

    assert ratio("window") > ratio("global"), "global filtering must reduce the tilt"
    # The absolute ratio moves with the clip; the ordering is the claim.
    assert ratio("window") / ratio("global") > 1.2


# --------------------------------------------------------------------------
# Viterbi tracking
# --------------------------------------------------------------------------


def test_viterbi_rejects_an_isolated_octave_jump():
    """A median filter cannot fix this; continuity can.

    Doubling is not a statistical outlier — it is a confident, repeatable wrong
    answer that a median happily passes through if it persists for a couple of
    windows. Only the cost of *moving there and back* rules it out.
    """
    grid = np.arange(42.0, 180.0 + 1e-9, 0.5)
    n_win = 20
    scores = np.full((n_win, grid.size), -30.0)
    true_idx = int(np.argmin(np.abs(grid - 70.0)))
    octave_idx = int(np.argmin(np.abs(grid - 140.0)))
    for i in range(n_win):
        scores[i, true_idx] = 0.0
    # Two consecutive windows where the octave looks better.
    for i in (8, 9):
        scores[i, true_idx] = -4.0
        scores[i, octave_idx] = 0.0

    times = np.arange(n_win) * 1.0
    path = viterbi_track(times, grid, scores, max_slew_bpm_per_sec=12.0)
    assert np.allclose(grid[path], 70.0), "the octave excursion must be outvoted"

    # Without continuity, the naive per-window argmax takes the bait.
    naive = grid[np.argmax(scores, axis=1)]
    assert naive[8] == pytest.approx(140.0)


def test_viterbi_follows_a_real_change_within_the_slew_limit():
    """It must not be so stiff that a genuine change is suppressed."""
    grid = np.arange(42.0, 180.0 + 1e-9, 0.5)
    n_win = 40
    target = np.linspace(70.0, 100.0, n_win)  # 0.77 BPM/s, physiological
    scores = np.full((n_win, grid.size), -30.0)
    for i, bpm in enumerate(target):
        scores[i, int(np.argmin(np.abs(grid - bpm)))] = 0.0

    path = viterbi_track(np.arange(n_win) * 1.0, grid, scores, max_slew_bpm_per_sec=12.0)
    assert np.max(np.abs(grid[path] - target)) < 2.0


def test_viterbi_slew_limit_is_enforced():
    """A physically impossible jump must be refused even if the evidence favours it."""
    grid = np.arange(42.0, 180.0 + 1e-9, 0.5)
    scores = np.full((2, grid.size), -50.0)
    scores[0, int(np.argmin(np.abs(grid - 60.0)))] = 0.0
    scores[1, int(np.argmin(np.abs(grid - 170.0)))] = 0.0  # +110 BPM in 1 s

    path = viterbi_track(np.array([0.0, 1.0]), grid, scores, max_slew_bpm_per_sec=12.0)
    assert abs(grid[path[1]] - grid[path[0]]) <= 12.0 + 1e-6


def test_track_bpm_end_to_end_on_a_clean_signal():
    t = np.arange(int(60 * FS)) / FS
    x = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.4 * t)
    track = track_bpm(x, FS, window_sec=15.0, hop_sec=1.0)
    assert track.bpm.size > 30
    assert np.median(track.bpm) == pytest.approx(72.0, abs=1.0)


def test_score_gram_rows_are_normalised():
    """Absolute spectral power varies far more between windows than HR does."""
    t = np.arange(int(45 * FS)) / FS
    x = np.sin(2 * np.pi * 1.2 * t) * np.linspace(1.0, 20.0, t.size)
    _, _, scores = score_gram(x, FS, window_sec=15.0, hop_sec=2.0)
    assert np.allclose(scores.max(axis=1), 0.0), "each row must peak at 0 dB"


def test_tracking_beats_median_on_an_octave_prone_arm():
    """The end-to-end claim, on the arm that actually suffers."""
    ts = nonuniform_timestamps(90.0, FS, jitter_ms=3.0, rng=2000)
    clip = synth_rgb(timestamps=ts, bpm=58.0, rng=2000)
    cfg = PipelineConfig(fs=FS, window_sec=15.0, hop_sec=2.0)
    df = analyse_signal(clip.t, clip.rgb, cfg)
    sub = df[df.method == "chrom"]
    ref = np.array([clip.mean_bpm_over(a, b) for a, b in zip(sub.t_start, sub.t_end)])

    err_median = np.nanmean(np.abs(sub.bpm_smoothed.to_numpy() - ref))
    err_track = np.nanmean(np.abs(sub.bpm_tracked.to_numpy() - ref))
    assert err_track < err_median, f"tracked {err_track:.2f} vs median {err_median:.2f}"
