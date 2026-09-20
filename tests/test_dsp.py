"""Stage-by-stage DSP tests.

The headline one is `test_all_arms_recover_known_bpm`: a 1.2 Hz sinusoid plus
noise plus a 0.3 Hz drift must come back as 72 BPM. If that fails, no amount of
real video will help.
"""

import numpy as np
import pytest

from rppg.metrics import bland_altman, mae, mape, pearson, rmse
from rppg.methods import PROJECTIONS, ica_full
from rppg.pipeline import PipelineConfig, analyse_signal, estimate_window
from rppg.preprocess import bandpass, detrend_moving_average, detrend_smoothness_priors
from rppg.quality import snr_db
from rppg.resample import frame_interval_stats, resample_uniform
from rppg.spectral import bpm_resolution, estimate_bpm, estimate_bpm_welch, periodogram
from rppg.synthetic import nonuniform_timestamps, synth_rgb

FS = 30.0


# --------------------------------------------------------------------------
# Stage 2 — resampling
# --------------------------------------------------------------------------


def test_resample_recovers_frequency_from_jittered_grid():
    t = nonuniform_timestamps(20.0, FS, jitter_ms=5.0, drop_prob=0.02, rng=0)
    x = np.sin(2 * np.pi * 1.2 * t)
    t_u, x_u = resample_uniform(t, x, fs=FS)

    assert np.allclose(np.diff(t_u), 1.0 / FS)
    assert estimate_bpm(x_u, FS).bpm == pytest.approx(72.0, abs=0.5)


def test_assuming_uniform_sampling_biases_the_frequency_axis():
    """The failure the resampling stage exists to prevent.

    A camera whose true mean rate is 27 fps, processed as if it were 30, scales
    every frequency by 30/27 — a 72 BPM pulse reads as 80.
    """
    true_fs, assumed_fs = 27.0, 30.0
    t = np.arange(int(20 * true_fs)) / true_fs
    x = np.sin(2 * np.pi * 1.2 * t)

    naive = estimate_bpm(x, assumed_fs).bpm
    assert naive == pytest.approx(72.0 * assumed_fs / true_fs, rel=0.02)

    _, x_u = resample_uniform(t, x, fs=assumed_fs)
    assert estimate_bpm(x_u, assumed_fs).bpm == pytest.approx(72.0, abs=0.5)


def test_resample_handles_duplicate_timestamps():
    t = np.arange(200) / FS
    t[50] = t[49]  # a backend that reused a clock tick
    x = np.sin(2 * np.pi * 1.2 * t)
    t_u, x_u = resample_uniform(t, x, fs=FS)
    assert np.all(np.diff(t_u) > 0)


def test_jitter_stats_report_drops():
    t = nonuniform_timestamps(10.0, FS, jitter_ms=2.0, drop_prob=0.05, rng=1)
    stats = frame_interval_stats(t)
    assert 25.0 < stats.mean_fps < 31.0
    assert stats.dropped_estimate > 0
    assert stats.std_dt > 0


# --------------------------------------------------------------------------
# Stage 3 — detrending
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [detrend_smoothness_priors, None])
def test_detrend_removes_respiration_baseline_keeps_pulse(fn):
    t = np.arange(int(20 * FS)) / FS
    pulse = np.sin(2 * np.pi * 1.2 * t)
    drift = 8.0 * np.sin(2 * np.pi * 0.3 * t) + 0.5 * t
    x = pulse + drift

    y = fn(x) if fn else detrend_moving_average(x, FS, win_sec=1.0)

    # The trend must go, and the pulse must survive it.
    assert np.std(y) < np.std(x)
    assert estimate_bpm(y, FS).bpm == pytest.approx(72.0, abs=1.0)
    p_before = periodogram(x, FS)[1]
    freqs, p_after = periodogram(y, FS)
    low = freqs < 0.5
    assert p_after[low].sum() < 0.05 * p_before[low].sum()


# --------------------------------------------------------------------------
# Stage 4 — bandpass
# --------------------------------------------------------------------------


def test_bandpass_rejects_out_of_band_and_is_zero_phase():
    t = np.arange(int(20 * FS)) / FS
    in_band = np.sin(2 * np.pi * 1.2 * t)
    out_of_band = np.sin(2 * np.pi * 6.0 * t) + np.sin(2 * np.pi * 0.2 * t)

    y = bandpass(in_band + out_of_band, FS)
    core = slice(int(2 * FS), -int(2 * FS))  # ignore filtfilt edge transients
    assert np.corrcoef(y[core], in_band[core])[0, 1] > 0.99

    # Zero phase: the peak of a symmetric burst must not move.
    burst = np.exp(-((t - 10.0) ** 2) / 0.5) * np.sin(2 * np.pi * 1.2 * (t - 10.0))
    filtered = bandpass(burst, FS)
    lag = np.argmax(np.abs(filtered)) - np.argmax(np.abs(burst))
    assert abs(lag) <= 1


def test_bandpass_refuses_windows_too_short_for_filtfilt():
    with pytest.raises(ValueError, match="too short"):
        bandpass(np.random.default_rng(0).normal(size=10), FS)


# --------------------------------------------------------------------------
# Stage 5 — spectral estimation
# --------------------------------------------------------------------------


def test_parabolic_interpolation_beats_raw_bin_peak():
    """A frequency deliberately placed between bins."""
    t = np.arange(int(10 * FS)) / FS  # 10 s => 6 BPM raw resolution
    x = np.sin(2 * np.pi * 1.19 * t)  # 71.4 BPM

    raw = estimate_bpm(x, FS, zero_pad=1, interpolate=False).bpm
    interp = estimate_bpm(x, FS, zero_pad=8, interpolate=True).bpm

    assert abs(interp - 71.4) < abs(raw - 71.4)
    assert interp == pytest.approx(71.4, abs=0.5)


def test_zero_padding_interpolates_but_does_not_add_resolution():
    """The distinction examiners probe for, asserted rather than asserted-in-prose.

    Two tones 3 BPM apart inside a 10 s window (6 BPM resolution) stay merged
    into a single peak no matter how much zero-padding is applied, while the
    reported resolution figure is unchanged by padding.
    """
    t = np.arange(int(10 * FS)) / FS
    x = np.sin(2 * np.pi * 1.15 * t) + np.sin(2 * np.pi * 1.20 * t)  # 69 and 72 BPM

    r1 = estimate_bpm(x, FS, zero_pad=1)
    r64 = estimate_bpm(x, FS, zero_pad=64)
    assert r1.resolution_bpm == pytest.approx(r64.resolution_bpm)
    assert r64.resolution_bpm == pytest.approx(6.0, abs=0.01)

    # Still one peak: padding did not separate them.
    band = (r64.freqs >= 1.0) & (r64.freqs <= 1.4)
    p = r64.power[band]
    interior = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > 0.2 * p.max()))
    assert len(interior) == 1

    # Doubling the window to 20 s does resolve them.
    t2 = np.arange(int(20 * FS)) / FS
    x2 = np.sin(2 * np.pi * 1.15 * t2) + np.sin(2 * np.pi * 1.20 * t2)
    r2 = estimate_bpm(x2, FS, zero_pad=64)
    band2 = (r2.freqs >= 1.0) & (r2.freqs <= 1.4)
    p2 = r2.power[band2]
    peaks2 = np.flatnonzero((p2[1:-1] > p2[:-2]) & (p2[1:-1] > p2[2:]) & (p2[1:-1] > 0.2 * p2.max()))
    assert len(peaks2) == 2


def test_bpm_resolution_matches_60_over_T():
    assert bpm_resolution(10.0) == pytest.approx(6.0)
    assert bpm_resolution(15.0) == pytest.approx(4.0)
    t = np.arange(int(15 * FS)) / FS
    assert estimate_bpm(np.sin(2 * np.pi * 1.2 * t), FS).resolution_bpm == pytest.approx(4.0)


def test_hann_window_suppresses_leakage_from_a_strong_trend():
    """Windowing is not cosmetic — without it this clip reads 44 BPM.

    A tiny pulse riding a large ramp: the rectangular window's -13 dB sidelobes
    smear the ramp's energy across the whole cardiac band and the peak lands on
    leakage. Hann's -31 dB sidelobes leave the real peak standing.
    """
    t = np.arange(int(15 * FS)) / FS
    x = 0.02 * np.sin(2 * np.pi * 1.2 * t) + t / t.max()

    freqs, p_rect = periodogram(x, FS, window=None)
    _, p_hann = periodogram(x, FS, window="hann")
    band = np.flatnonzero((freqs >= 0.7) & (freqs <= 4.0))

    assert p_hann[band].sum() < 0.15 * p_rect[band].sum()
    assert estimate_bpm(x, FS, window=None).bpm < 60.0  # leakage wins
    assert estimate_bpm(x, FS, window="hann").bpm == pytest.approx(72.0, abs=1.0)


def test_welch_trades_resolution_for_variance():
    t = np.arange(int(15 * FS)) / FS
    x = np.sin(2 * np.pi * 1.2 * t)
    w = estimate_bpm_welch(x, FS, seg_sec=4.0)
    assert w.bpm == pytest.approx(72.0, abs=2.0)
    assert w.resolution_bpm > estimate_bpm(x, FS).resolution_bpm


# --------------------------------------------------------------------------
# Stage 7 — quality
# --------------------------------------------------------------------------


def test_snr_high_for_clean_pulse_low_for_noise():
    t = np.arange(int(15 * FS)) / FS
    rng = np.random.default_rng(0)
    clean = estimate_bpm(np.sin(2 * np.pi * 1.2 * t), FS)
    noise = estimate_bpm(bandpass(rng.normal(size=t.size), FS), FS)
    assert clean.snr > 5.0
    assert noise.snr < clean.snr


def test_snr_counts_the_first_harmonic_as_signal():
    freqs = np.linspace(0, 5, 1000)
    power = np.zeros_like(freqs)
    power[np.argmin(np.abs(freqs - 1.2))] = 1.0
    power[np.argmin(np.abs(freqs - 2.4))] = 0.5  # harmonic, must not count as noise
    power += 1e-6
    assert snr_db(freqs, power, 1.2) > snr_db(freqs, power, 1.9)


# --------------------------------------------------------------------------
# The four arms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(PROJECTIONS))
def test_all_arms_recover_known_bpm(method):
    """The week-3 sanity check: 1.2 Hz + noise + 0.3 Hz drift must give 72 BPM."""
    clip = synth_rgb(duration=20.0, fs=FS, bpm=72.0, respiration_bpm=18.0, rng=42)
    cfg = PipelineConfig(fs=FS, window_sec=15.0)
    res = estimate_window(clip.rgb[: int(15 * FS)], FS, method, cfg)
    assert res.bpm == pytest.approx(72.0, abs=2.0)


@pytest.mark.parametrize("method", ["green", "ica", "pos"])
def test_all_arms_track_elevated_heart_rate(method):
    """Condition D. A method that always outputs ~72 looks great and means nothing."""
    clip = synth_rgb(duration=20.0, fs=FS, bpm=128.0, rng=7)
    res = estimate_window(clip.rgb[: int(15 * FS)], FS, method, PipelineConfig(fs=FS))
    assert res.bpm == pytest.approx(128.0, abs=4.0)


def test_chrom_is_octave_sensitive_at_elevated_heart_rate():
    """A known, measured limitation — pinned so it cannot silently worsen.

    At 128 BPM the pulse's harmonics sit at 256 and 384 BPM, outside the
    0.7-4.0 Hz passband, so the correct candidate gets no harmonic support
    while a subharmonic near 64 BPM can borrow the true line's power. CHROM is
    worst affected because its own normalisation already suppresses the
    fundamental. GREEN, ICA and POS are covered strictly above; this documents
    where CHROM is not trustworthy rather than pretending otherwise.
    """
    clip = synth_rgb(duration=20.0, fs=FS, bpm=128.0, rng=7)
    res = estimate_window(clip.rgb[: int(15 * FS)], FS, "chrom", PipelineConfig(fs=FS))
    ratio = res.bpm / 128.0
    assert 0.4 < ratio < 1.15, f"unexpected regime: chrom read {res.bpm:.1f} BPM"


def test_model_based_arms_survive_motion_that_breaks_green_and_ica():
    """The central claim of the project, as a test rather than an assertion.

    Motion enters through the shared multiplicative I(t) — a correlated,
    nonlinear distortion, not the independent additive source ICA assumes — so
    GREEN and ICA lock onto the motion frequency while CHROM and POS hold.
    """
    clip = synth_rgb(
        duration=25.0, fs=FS, bpm=72.0, motion_amplitude=0.05, motion_bpm=48.0, rng=3
    )
    cfg = PipelineConfig(fs=FS, window_sec=15.0)
    seg = clip.rgb[: int(15 * FS)]
    err = {m: abs(estimate_window(seg, FS, m, cfg).bpm - 72.0) for m in PROJECTIONS}

    assert err["chrom"] < 3.0
    assert err["pos"] < 3.0
    assert err["green"] > err["chrom"]
    assert err["ica"] > err["chrom"]


def test_ica_permutation_ambiguity_is_real_and_the_selector_handles_it():
    """"Component 3 is the pulse" is not something ICA guarantees.

    Same data, same clip, only the FastICA initialisation changed: the pulse
    lands on component 0, 1 or 2. The ordering is an artefact of the algorithm,
    not a property of the signal — which is precisely why the spectral selector
    is mandatory rather than a convenience.
    """
    clip = synth_rgb(duration=20.0, fs=FS, bpm=72.0, rng=0)
    seg = clip.rgb[: int(15 * FS)]

    chosen = set()
    for random_state in range(8):
        res = ica_full(seg, FS, random_state=random_state)
        assert res.scores[res.chosen] == res.scores.max()
        # Whichever index it landed on, the selected component is the pulse.
        assert estimate_bpm(bandpass(res.signal, FS), FS).bpm == pytest.approx(72.0, abs=2.0)
        chosen.add(res.chosen)

    assert len(chosen) > 1, "identical data must not always yield the same ordering"


def test_ica_degrades_but_still_returns_a_number_on_a_degenerate_window():
    """A constant ROI (face lost, saturated frame) must not crash the pipeline."""
    rgb = np.tile([120.0, 90.0, 70.0], (int(15 * FS), 1))
    rgb += np.random.default_rng(0).normal(0, 1e-9, rgb.shape)
    res = ica_full(rgb, FS)
    assert np.isfinite(res.signal).all()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_analyse_signal_produces_one_row_per_window_per_method():
    t = nonuniform_timestamps(40.0, FS, jitter_ms=3.0, rng=0)
    clip = synth_rgb(bpm=72.0, timestamps=t, rng=0)
    cfg = PipelineConfig(fs=FS, window_sec=15.0, hop_sec=1.0)
    df = analyse_signal(clip.t, clip.rgb, cfg)

    assert set(df.method.unique()) == set(cfg.methods)
    n_windows = df.groupby("method").size()
    assert n_windows.nunique() == 1 and n_windows.iloc[0] > 20

    for method in cfg.methods:
        sub = df[df.method == method]
        assert np.nanmedian(sub.bpm_smoothed) == pytest.approx(72.0, abs=2.0)


def test_snr_gate_rejects_rather_than_emitting_a_bad_number():
    rng = np.random.default_rng(0)
    rgb = 128.0 + rng.normal(0, 3.0, size=(int(40 * FS), 3))  # no pulse at all
    t = np.arange(rgb.shape[0]) / FS
    df = analyse_signal(t, rgb, PipelineConfig(fs=FS, snr_threshold=0.0))
    assert not df.accepted.all(), "a pure-noise clip must trip the SNR gate"


def test_analyse_signal_refuses_a_clip_shorter_than_the_window():
    clip = synth_rgb(duration=5.0, fs=FS, rng=0)
    with pytest.raises(ValueError, match="shorter than"):
        analyse_signal(clip.t, clip.rgb, PipelineConfig(fs=FS, window_sec=15.0))


def test_live_and_batch_paths_agree():
    """The number on stage must be produced by the same DSP as the table.

    This asserts *agreement*, not accuracy — accuracy is covered elsewhere. If
    the two paths ever diverge, a demo could show something the benchmark
    cannot reproduce, which is the failure this guards against.
    """
    from rppg.pipeline import SlidingWindowEstimator

    t = nonuniform_timestamps(20.0, FS, jitter_ms=3.0, rng=0)
    clip = synth_rgb(bpm=72.0, timestamps=t, rng=0)
    cfg = PipelineConfig(fs=FS, window_sec=15.0, hop_sec=1.0, track=False)

    live = SlidingWindowEstimator(cfg)
    for ti, row in zip(clip.t, clip.rgb):
        live.push(ti, row)
    out = live.estimate()

    batch = analyse_signal(clip.t, clip.rgb, cfg)
    last = batch[batch.t_end == batch.t_end.max()].set_index("method")

    for method in cfg.methods:
        assert out[method].bpm == pytest.approx(last.loc[method, "bpm"], abs=0.75), (
            f"{method}: live {out[method].bpm:.2f} vs batch {last.loc[method, 'bpm']:.2f}"
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_on_known_errors():
    ref = np.array([60.0, 70.0, 80.0, 90.0])
    pred = ref + np.array([1.0, -1.0, 2.0, -2.0])

    assert mae(pred, ref) == pytest.approx(1.5)
    assert rmse(pred, ref) == pytest.approx(np.sqrt(10 / 4))
    assert mape(pred, ref) == pytest.approx(
        np.mean([1 / 60, 1 / 70, 2 / 80, 2 / 90]) * 100
    )
    assert pearson(pred, ref) > 0.99


def test_metrics_ignore_nans_from_rejected_windows():
    ref = np.array([60.0, 70.0, 80.0])
    pred = np.array([61.0, np.nan, 79.0])
    assert mae(pred, ref) == pytest.approx(1.0)


def test_bland_altman_bias_and_limits():
    rng = np.random.default_rng(0)
    ref = rng.uniform(55, 130, 500)
    pred = ref + 2.0 + rng.normal(0, 3.0, 500)
    ba = bland_altman(pred, ref)
    assert ba.bias == pytest.approx(2.0, abs=0.4)
    assert ba.loa_upper - ba.loa_lower == pytest.approx(2 * 1.96 * 3.0, rel=0.15)


# --------------------------------------------------------------------------
# Environment guard
# --------------------------------------------------------------------------


def test_pandas_survives_the_crashing_import_order():
    """Regression guard for the native-library conflict documented in _compat.py.

    Run in a subprocess: without the guard this does not raise, it takes the
    whole interpreter down with an access violation.
    """
    import subprocess
    import sys as _sys

    script = (
        "import matplotlib; matplotlib.use('Agg');import matplotlib.pyplot as plt\n"
        "import scipy.signal\n"
        "import rppg\n"
        "import pandas as pd\n"
        "print(pd.DataFrame([{'a': 1.0, 'method': 'green'}] * 200).shape)\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"interpreter died: {proc.returncode}\n{proc.stderr}"
    assert "(200, 2)" in proc.stdout


# --------------------------------------------------------------------------
# Motion-transient repair
# --------------------------------------------------------------------------


def _pulse_with_spikes(n_spikes, amp=8.0, dur=30.0, bpm=86.0, fs=30.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    p = np.sin(2 * np.pi * bpm / 60 * t)
    dc = np.array([120.0, 95.0, 88.0])
    x = dc * (1 + 0.004 * np.array([0.45, 1.0, 0.35]) * p[:, None])
    x = x + rng.normal(0, 0.0004 * dc, x.shape)
    for i in rng.choice(np.arange(20, len(t) - 20), size=n_spikes, replace=False):
        x[i : i + 2] += amp * rng.choice([-1, 1])
    return x


def test_despike_restores_snr_destroyed_by_transients():
    from rppg.preprocess import bandpass, despike, detrend
    from rppg.methods import green
    from rppg.spectral import estimate_bpm

    fs = 30.0
    x = _pulse_with_spikes(3)
    before = estimate_bpm(bandpass(detrend(green(x, fs), fs=fs), fs), fs).snr
    after = estimate_bpm(bandpass(detrend(green(despike(x), fs), fs=fs), fs), fs).snr
    # Measured: three spikes take a clean +25.7 dB pulse down to -1.6 dB.
    assert before < 5.0, f"expected the spikes to wreck SNR, got {before:.1f} dB"
    assert after > 15.0, f"expected repair to restore it, got {after:.1f} dB"


def test_despike_leaves_a_clean_signal_alone():
    from rppg.preprocess import despike

    x = _pulse_with_spikes(0)
    assert np.allclose(despike(x), x), "must not touch a signal with no transients"


def test_despike_repairs_all_channels_at_the_same_indices():
    """A motion artifact is common-mode; per-channel repair would invent colour."""
    from rppg.preprocess import despike

    x = _pulse_with_spikes(0)
    x[300] += np.array([9.0, 9.0, 9.0])
    y = despike(x)
    changed = np.abs(y - x) > 1e-9
    # Every repaired sample is repaired in all three channels, or none.
    assert set(changed.sum(axis=1)) <= {0, 3}


def test_despike_refuses_when_most_of_the_signal_trips_the_test():
    """If 'spikes' are everywhere they are the signal, not artifacts."""
    from rppg.preprocess import despike

    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, (400, 3))  # broadband noise, no clean baseline
    assert np.allclose(despike(x, max_fraction=0.05), x)


def test_despike_handles_a_one_dimensional_signal():
    from rppg.preprocess import despike

    x = np.sin(np.arange(300) * 0.2) * 2 + 100
    x[150] += 20
    y = despike(x)
    assert y.shape == x.shape
    assert abs(y[150] - x[150]) > 5


def test_pipeline_despike_is_on_by_default_and_can_be_turned_off():
    from rppg.pipeline import PipelineConfig, project_and_filter

    fs = 30.0
    x = _pulse_with_spikes(3)
    on = project_and_filter(x, fs, "green", PipelineConfig(fs=fs, despike=True))
    off = project_and_filter(x, fs, "green", PipelineConfig(fs=fs, despike=False))
    assert PipelineConfig(fs=fs).despike is True
    assert not np.allclose(on, off)
