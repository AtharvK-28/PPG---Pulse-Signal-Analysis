"""BIDMC validation helpers.

The dataset itself is not in the repo, so these test the logic that would
silently corrupt the benchmark if it were wrong -- anti-aliasing on decimation,
window/reference alignment, and the gate table the SNR default is derived from.
"""

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.bidmc import gate_table, operating_curve, to_webcam_rate, windows

FS = 30.0


def _ppg(bpm=75.0, dur=120.0, fs=125.0, hf=0.0, seed=0):
    """Contact-PPG-like waveform: fundamental plus harmonics, as the real one is."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    f = bpm / 60
    x = (np.sin(2 * np.pi * f * t)
         + 0.4 * np.sin(4 * np.pi * f * t)
         + 0.15 * np.sin(6 * np.pi * f * t))
    if hf:
        # Content above the 30 Hz target's Nyquist, to catch naive subsampling.
        x = x + hf * np.sin(2 * np.pi * 27.0 * t)
    return x + rng.normal(0, 0.01, len(t)), t


def test_decimation_rejects_content_above_the_new_nyquist():
    """Plain subsampling folds 26.5 Hz onto 1.5 Hz — a fake 90 BPM 'pulse'.

    Isolated deliberately: a pure high tone with no pulse, so the in-band power
    measured afterwards can only be the alias. Mixing it with a real pulse hides
    the effect under the pulse's own harmonics.
    """
    from rppg.spectral import periodogram

    fs = 125.0
    t = np.arange(0, 120.0, 1 / fs)
    # 125/5 = 25 Hz after subsampling; |26.5 - 25| = 1.5 Hz = 90 BPM.
    x = np.sin(2 * np.pi * 26.5 * t)

    y, _ = to_webcam_rate(x, fs)
    naive = x[::5]

    def power_at_90bpm(sig, sig_fs):
        f, p = periodogram(sig - sig.mean(), sig_fs, zero_pad=2)
        m = np.abs(f * 60 - 90) < 6
        return p[m].sum()

    aliased = power_at_90bpm(naive, 25.0)
    filtered = power_at_90bpm(y, FS)
    assert aliased > 1e-3, "the naive path should show the alias, or the test is moot"
    assert filtered < 0.01 * aliased, (
        f"anti-aliasing should suppress the fold: {filtered:.3e} vs {aliased:.3e}"
    )


def test_resampled_signal_lands_on_the_target_rate():
    x, _ = _ppg(dur=60.0)
    y, t = to_webcam_rate(x, 125.0)
    assert abs(np.median(np.diff(t)) - 1 / FS) < 1e-9
    assert abs(len(y) / t[-1] - FS) < 0.5


def test_windows_recovers_a_known_rate_and_aligns_to_the_reference():
    x, _ = _ppg(bpm=75.0, dur=120.0)
    y, t = to_webcam_rate(x, 125.0)
    hr = np.full(int(t[-1]) + 2, 75.0)
    pairs = list(windows(y, t, hr, hop_sec=5.0))
    assert len(pairs) > 10
    err = np.array([abs(p[1]) for p in pairs])
    assert err.mean() < 1.0, f"clean 75 BPM should be recovered, MAE {err.mean():.2f}"


def test_windows_skips_reference_gaps():
    x, _ = _ppg(dur=60.0)
    y, t = to_webcam_rate(x, 125.0)
    hr = np.full(int(t[-1]) + 2, np.nan)
    hr[:20] = 75.0
    pairs = list(windows(y, t, hr, hop_sec=1.0))
    # Only windows ending inside the first 20 s have a reference.
    assert 0 < len(pairs) <= 20


def test_gate_table_is_monotonic_in_the_right_direction():
    """A stricter gate must keep fewer windows and score better, or it is useless."""
    rng = np.random.default_rng(0)
    snr = rng.uniform(-10, 12, 4000)
    # Error falls as SNR rises, which is the relationship the gate exploits.
    err = np.clip(20 * np.exp(-0.35 * (snr + 8)) + rng.normal(0, 0.3, 4000), 0, None)
    rows = gate_table(snr, err)
    kept = [r["kept"] for r in rows]
    maes = [r["mae"] for r in rows]
    assert kept == sorted(kept, reverse=True)
    assert maes == sorted(maes, reverse=True)


def test_gate_table_skips_thresholds_that_keep_nothing():
    snr = np.full(100, -12.0)
    rows = gate_table(snr, np.ones(100), gates=(-20, 0, 5))
    assert [r["gate"] for r in rows] == [-20]


def test_default_snr_gate_matches_the_calibration():
    """The -3 dB default is a measurement; a silent change should fail here."""
    from rppg.pipeline import PipelineConfig

    assert PipelineConfig(fs=FS).snr_threshold == -3.0


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/bidmc/bidmc01.hea").exists(),
    reason="BIDMC not downloaded (python -m rppg.bidmc --download)",
)
def test_real_recording_is_recovered_accurately():
    from rppg.bidmc import evaluate

    r = evaluate("bidmc01")
    assert r["mae"] < 2.0, f"clean ICU PPG should be easy, got {r['mae']:.2f} BPM"
    assert r["within3"] > 95
