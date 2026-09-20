"""Respiratory rate from PPG: three modulations, agreement-based fusion.

The synthetic waveform here imposes each modulation separately, so a test can
assert that a given route recovers the rate *it* is responsible for — which is
the only way to tell a working demodulator from one that is picking up a
neighbour's signal.
"""

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.beats import detect_beats
from rppg.respiration import (
    RespEstimate,
    analyse,
    estimate_rate,
    fuse,
    pulse_amplitudes,
    riav,
    rifv,
    riiv,
)

FS = 125.0


def synth(bpm=75.0, resp_bpm=15.0, dur=120.0, fs=FS,
          riiv_amp=0.0, riav_amp=0.0, rifv_amp=0.0, seed=0):
    """PPG with each respiratory modulation individually switchable.

    riiv_amp modulates the baseline, riav_amp the pulse height, rifv_amp the
    interval. Turning on exactly one isolates the route under test.
    """
    rng = np.random.default_rng(seed)
    f_resp = resp_bpm / 60.0
    base = 60.0 / bpm

    beat_times, t_now = [], 0.0
    while t_now < dur:
        beat_times.append(t_now)
        # Frequency modulation: intervals shorten and lengthen with the breath.
        factor = 1.0 + rifv_amp * np.sin(2 * np.pi * f_resp * t_now)
        t_now += base * factor

    t = np.arange(0, dur, 1 / fs)
    x = np.zeros_like(t)
    for bt in beat_times:
        u = (t - bt) / base
        m = (u >= 0) & (u < 1)
        uu = u[m]
        height = 1.0 + riav_amp * np.sin(2 * np.pi * f_resp * bt)
        x[m] += height * np.exp(-((uu - 0.15) ** 2) / (2 * 0.055**2))
        x[m] += 0.3 * height * np.exp(-((uu - 0.40) ** 2) / (2 * 0.06**2))
    if riiv_amp:
        x = x + riiv_amp * np.sin(2 * np.pi * f_resp * t)
    return x + rng.normal(0, 0.005, len(t))


# --------------------------------------------------------------------------
# Individual routes
# --------------------------------------------------------------------------


def test_riiv_recovers_baseline_modulation():
    x = synth(resp_bpm=15.0, riiv_amp=0.30)
    sig, sig_fs = riiv(x, FS)
    est = estimate_rate(sig, sig_fs, method="riiv")
    assert est.rate_bpm == pytest.approx(15.0, abs=1.5)


def test_riav_recovers_amplitude_modulation():
    x = synth(resp_bpm=18.0, riav_amp=0.35)
    peaks = detect_beats(x, FS)
    sig, sig_fs = riav(x, peaks, FS)
    est = estimate_rate(sig, sig_fs, method="riav")
    assert est.rate_bpm == pytest.approx(18.0, abs=2.0)


def test_rifv_recovers_interval_modulation():
    x = synth(resp_bpm=12.0, rifv_amp=0.10)
    peaks = detect_beats(x, FS)
    sig, sig_fs = rifv(peaks, FS)
    est = estimate_rate(sig, sig_fs, method="rifv")
    assert est.rate_bpm == pytest.approx(12.0, abs=2.0)


def test_routes_do_not_invent_a_rate_from_an_unmodulated_signal():
    """With no respiration present, agreement must fail rather than produce a number."""
    x = synth(resp_bpm=15.0)  # all modulation amplitudes zero
    out = analyse(x, FS)
    assert not np.isfinite(out["fused"].rate_bpm) or not out["fused"].high_confidence


def test_pulse_amplitudes_track_the_imposed_modulation():
    x = synth(resp_bpm=15.0, riav_amp=0.4)
    peaks = detect_beats(x, FS)
    times, amps = pulse_amplitudes(x, peaks, FS)
    assert len(amps) > 50
    assert amps.std() / amps.mean() > 0.05, "amplitude modulation should be visible"


def test_amplitude_is_measured_peak_to_foot_not_against_a_global_baseline():
    """Respiration moves the baseline, so a global reference would double-count it."""
    x = synth(resp_bpm=15.0, riiv_amp=1.0)  # huge baseline swing, no real AM
    peaks = detect_beats(x, FS)
    _, amps = pulse_amplitudes(x, peaks, FS)
    assert amps.std() / amps.mean() < 0.25, (
        "baseline drift is leaking into the amplitude measurement"
    )


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def _est(v, method="x"):
    return RespEstimate(v, float("nan"), method)


def test_fusion_averages_two_routes_that_agree():
    out = fuse({"riiv": _est(15.0), "rifv": _est(15.5), "riav": _est(28.0)})
    assert out.rate_bpm == pytest.approx(15.25)
    assert not out.high_confidence


def test_fusion_flags_high_confidence_when_all_three_agree():
    out = fuse({"riiv": _est(15.0), "rifv": _est(16.0), "riav": _est(17.0)})
    assert out.high_confidence
    assert out.rate_bpm == pytest.approx(16.0)


def test_fusion_refuses_when_nothing_agrees():
    """Measured: 65% coverage at 1.99 MAE beats 100% coverage at 2.66."""
    out = fuse({"riiv": _est(10.0), "rifv": _est(20.0), "riav": _est(30.0)})
    assert not np.isfinite(out.rate_bpm)
    assert "no agreement" in out.method


def test_fusion_ignores_routes_that_returned_nothing():
    out = fuse({"riiv": _est(15.0), "rifv": _est(float("nan")), "riav": _est(15.4)})
    assert out.rate_bpm == pytest.approx(15.2)


def test_fusion_needs_at_least_two_routes():
    out = fuse({"riiv": _est(15.0), "rifv": _est(float("nan")),
                "riav": _est(float("nan"))})
    assert not np.isfinite(out.rate_bpm)


def test_all_three_branch_takes_precedence_over_any_pair():
    """At the default tolerances the pair path is only reached when one pair agrees.

    If two pairs are each within 1.0, all three are necessarily within 2.0 and
    so within strict_tolerance — the unanimous branch fires first, by
    construction. Pinning that here so the precedence is not changed by
    accident when the tolerances are tuned.
    """
    out = fuse({"riiv": _est(15.0), "rifv": _est(15.9), "riav": _est(15.1)})
    assert out.high_confidence
    assert out.rate_bpm == pytest.approx(15.333, abs=0.01)


def test_fusion_uses_the_only_pair_that_agrees():
    out = fuse({"riiv": _est(15.0), "rifv": _est(15.8), "riav": _est(26.0)})
    assert not out.high_confidence
    assert out.rate_bpm == pytest.approx(15.4)
    assert "riiv+rifv" in out.method


# --------------------------------------------------------------------------
# Band limits and aliasing
# --------------------------------------------------------------------------


def test_beat_sampled_routes_report_their_nyquist_limit():
    """RIFV/RIAV are sampled once per beat; RIIV is not, and has no such limit."""
    x = synth(bpm=60.0, resp_bpm=15.0, riav_amp=0.3, riiv_amp=0.2)
    out = analyse(x, FS)
    # 60 BPM => 1 Hz beat sampling => 0.5 Hz Nyquist => 30 br/min.
    assert out["rifv"].nyquist_bpm == pytest.approx(30.0, abs=2.0)
    assert not np.isfinite(out["riiv"].nyquist_bpm)


def test_respiration_near_the_beat_nyquist_is_flagged_as_aliased():
    """A slow heart and fast breathing is a genuine limit, not a bug to hide."""
    x = synth(bpm=50.0, resp_bpm=24.0, riav_amp=0.4)
    out = analyse(x, FS)
    # 50 BPM => 25 br/min Nyquist; 24 br/min sits right at it.
    assert out["riav"].nyquist_bpm == pytest.approx(25.0, abs=2.0)


def test_estimate_rate_returns_nan_for_a_dead_signal():
    est = estimate_rate(np.zeros(500), 4.0, method="riiv")
    assert not np.isfinite(est.rate_bpm)


def test_estimate_rate_handles_a_series_too_short_to_analyse():
    est = estimate_rate(np.array([1.0, 2.0, 3.0]), 4.0)
    assert not np.isfinite(est.rate_bpm)


def test_analyse_returns_all_four_entries():
    x = synth(resp_bpm=15.0, riiv_amp=0.3, riav_amp=0.2)
    out = analyse(x, FS)
    assert set(out) == {"riiv", "rifv", "riav", "fused"}


def test_analyse_finds_the_rate_when_all_modulations_are_present():
    x = synth(resp_bpm=15.0, riiv_amp=0.30, riav_amp=0.30, rifv_amp=0.08)
    out = analyse(x, FS)
    assert out["fused"].rate_bpm == pytest.approx(15.0, abs=2.0)
