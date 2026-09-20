"""The session post-mortem must tell a real pulse from a common-mode artifact.

This is the check that finally located the live-webcam failure, so it needs a
test that would fail if it stopped discriminating.
"""

import numpy as np
import pytest

from rppg import _compat  # noqa: F401
from rppg.session import band_snr, best_projection, grid_quality, load_session, report

FS = 30.0


def _write(tmp_path, t, rgb, ref=None):
    path = tmp_path / "session_test.csv"
    with open(path, "w") as fh:
        if ref is not None:
            fh.write(f"# reference_bpm={ref}\n")
        fh.write("t,R,G,B\n")
        for ti, row in zip(t, rgb):
            fh.write("%.6f,%.6f,%.6f,%.6f\n" % (ti, *row))
    return path


def _pulse_rgb(bpm=86.0, dur=40.0, fs=FS, amp=0.006, noise=0.0004, seed=0):
    """A pulse with the green/red ratio real haemoglobin absorption produces."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    p = np.sin(2 * np.pi * bpm / 60 * t)
    dc = np.array([120.0, 95.0, 88.0])
    gains = np.array([0.45, 1.0, 0.35])  # green modulates ~2x red
    x = dc * (1 + amp * gains * p[:, None])
    return t, x + rng.normal(0, noise * dc, x.shape)


def _common_mode_rgb(bpm=50.0, dur=40.0, fs=FS, amp=0.01, seed=0):
    """An illumination/motion term: identical relative swing in all channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    i = np.sin(2 * np.pi * bpm / 60 * t)
    dc = np.array([120.0, 95.0, 88.0])
    x = dc * (1 + amp * i[:, None])
    return t, x + rng.normal(0, 0.0002 * dc, x.shape)


def test_best_projection_finds_a_real_pulse():
    _, rgb = _pulse_rgb(bpm=86.0)
    snr, _ = best_projection(rgb, FS, 86.0, steps=11)
    assert snr > 3.0, f"a clean 0.6 % pulse should be well above 0 dB, got {snr:.2f}"


def test_best_projection_rejects_a_frequency_with_no_pulse():
    _, rgb = _pulse_rgb(bpm=86.0)
    snr, _ = best_projection(rgb, FS, 130.0, steps=11)
    assert snr < 0.0, "must not find a pulse at a rate the subject does not have"


def test_common_mode_artifact_is_not_mistaken_for_a_pulse_by_the_ratio_test():
    from rppg.session import channel_character

    _, rgb = _common_mode_rgb()
    rel, corr = channel_character(rgb, FS)
    assert rel["G"] / rel["R"] == pytest.approx(1.0, abs=0.1)
    assert min(corr[0, 1], corr[0, 2], corr[1, 2]) > 0.9


def test_real_pulse_shows_the_green_red_ratio():
    _, rgb = _pulse_rgb()
    from rppg.session import channel_character

    rel, _ = channel_character(rgb, FS)
    assert rel["G"] / rel["R"] > 1.5, "haemoglobin absorbs green ~2x more than red"


def test_report_names_the_upstream_fault_when_the_pulse_is_absent(tmp_path):
    t, rgb = _common_mode_rgb(bpm=50.0)
    path = _write(tmp_path, t, rgb, ref=86.0)
    text = report(path)
    assert "below 0 dB" in text
    assert "upstream" in text
    assert "agreement is no evidence here" in text


def test_report_clears_the_estimator_when_the_pulse_is_there(tmp_path):
    t, rgb = _pulse_rgb(bpm=86.0)
    path = _write(tmp_path, t, rgb, ref=86.0)
    text = report(path)
    assert "the pulse is present" in text


def test_report_reads_the_reference_from_the_header(tmp_path):
    t, rgb = _pulse_rgb(bpm=86.0)
    path = _write(tmp_path, t, rgb, ref=86.0)
    meta, _, _ = load_session(path)
    assert meta["reference_bpm"] == "86.0"
    assert "86 BPM" in report(path)


def test_report_asks_for_a_reference_when_the_file_has_none(tmp_path):
    t, rgb = _pulse_rgb()
    path = _write(tmp_path, t, rgb, ref=None)
    assert "--reference" in report(path)


def test_grid_quality_flags_a_stalling_loop():
    # 30 fps with a 500 ms hole punched in it.
    t = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(5.5, 10, 1 / 30)])
    g = grid_quality(t)
    assert g["worst"] > 400
    assert g["late"] >= 1


def test_band_snr_scores_the_named_frequency_not_the_tallest_peak():
    """The distinction that made the diagnosis possible."""
    fs, t = FS, np.arange(0, 40, 1 / FS)
    # Strong tone at 50 BPM, nothing at 86.
    x = np.sin(2 * np.pi * 50 / 60 * t)
    assert band_snr(x, fs, 50.0) > band_snr(x, fs, 86.0)
    assert band_snr(x, fs, 86.0) < 0
