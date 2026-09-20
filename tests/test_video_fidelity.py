"""Video must be lossless, or the pulse does not survive being written to disk.

The pulsatile modulation is 0.1-1% of intensity — around one 8-bit level. A
lossy codec spends its bits on what the eye notices, and a sub-percent uniform
brightness shift across a cheek is exactly what it throws away. UBFC-rPPG ships
uncompressed 8-bit RGB for this reason.

These tests exist because the synthetic video writer defaulted to `mp4v` for a
while, and every experiment built on it was measuring the codec.
"""

import numpy as np
import pytest

from rppg.pipeline import PipelineConfig, analyse_signal
from rppg.spectral import estimate_bpm
from rppg.synthetic import synth_rgb, write_synthetic_video

cv2 = pytest.importorskip("cv2")
FS = 30.0


def _roundtrip(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
    finally:
        cap.release()
    return frames


def test_default_codec_is_lossless(tmp_path):
    clip = synth_rgb(duration=2.0, fs=FS, bpm=72.0, pulse_amplitude=0.02, rng=0)
    path = tmp_path / "clip.avi"
    write_synthetic_video(path, clip, pixel_noise=0.0, rng=0)

    frames = _roundtrip(path)
    assert len(frames) > 30

    # A uniform disc on a uniform field: a lossless codec returns it exactly.
    centre = frames[0][120, 160].astype(int)
    for f in frames[1:]:
        patch = f[110:130, 150:170].reshape(-1, 3).astype(int)
        assert patch.std(axis=0).max() == 0, "disc interior must be flat after roundtrip"


def _recover(tmp_path, clip, codec, ext, pixel_noise):
    """Correlation between the recovered ROI mean and the pulse that went in."""
    from rppg.capture import VideoFileSource
    from rppg.roi import extract_signal, make_roi

    path = tmp_path / f"clip_{codec}_{pixel_noise}{ext}"
    try:
        write_synthetic_video(path, clip, pixel_noise=pixel_noise, codec=codec, rng=3)
    except RuntimeError as exc:  # codec unavailable on this build
        pytest.skip(str(exc))

    roi = make_roi("fixed", rel_rect=(0.36, 0.34, 0.28, 0.32), use_skin_mask=False)
    with VideoFileSource(path) as src:
        _, rgb, _ = extract_signal(src, roi)

    truth = clip.rgb[:, 1] - clip.rgb[:, 1].mean()
    got = rgb[:, 1] - rgb[:, 1].mean()
    n = min(len(got), len(truth))
    if got[:n].std() < 1e-12:
        return 0.0, 0.0  # nothing survived at all
    return float(np.corrcoef(got[:n], truth[:n])[0, 1]), float(np.ptp(got[:n]))


def test_sub_level_pulse_survives_only_with_dither_and_lossless_video(tmp_path):
    """Two independent ways to lose a pulse that is smaller than one 8-bit level.

    A 0.4% modulation on a mid-grey face is ~0.74 levels peak-to-peak — below
    the quantisation step. It is recoverable only because (a) per-pixel sensor
    noise dithers the quantiser, decorrelating the rounding error so the spatial
    average resolves below one LSB, and (b) nothing downstream re-quantises it.

    Remove either and the pulse is gone, with no error anywhere to tell you.
    """
    # Every stochastic term off: this test is about quantisation, and needs a
    # pulse whose amplitude is known exactly.
    clip = synth_rgb(duration=30.0, fs=FS, bpm=72.0, pulse_amplitude=0.004,
                     noise_std=0.0, pink_std=0.0, respiration_amplitude=0.0,
                     hrv_rsa=0.0, hrv_walk=0.0, waveform="sine", rng=3)
    p2p = float(np.ptp(clip.rgb[:, 1]))
    assert p2p < 1.0, f"this test is only meaningful sub-LSB; got {p2p:.2f} levels"

    # No dither: quantisation alone annihilates it, lossless codec or not.
    corr_nodither, _ = _recover(tmp_path, clip, "FFV1", ".avi", pixel_noise=0.0)
    assert abs(corr_nodither) < 0.2, (
        f"expected no recoverable signal without dither, got r={corr_nodither:.3f}"
    )

    # Dither + lossless: near-perfect recovery of a sub-LSB signal.
    corr_lossless, _ = _recover(tmp_path, clip, "FFV1", ".avi", pixel_noise=2.0)
    assert corr_lossless > 0.95, f"lossless+dither should recover it, got r={corr_lossless:.3f}"

    # Dither + lossy: materially degraded.
    corr_lossy, _ = _recover(tmp_path, clip, "mp4v", ".mp4", pixel_noise=2.0)
    assert corr_lossy < corr_lossless - 0.15, (
        f"lossy r={corr_lossy:.3f} vs lossless r={corr_lossless:.3f}"
    )


def test_lossless_video_round_trips_a_recoverable_bpm(tmp_path):
    """End to end through a file: the BPM must come back."""
    from rppg.capture import VideoFileSource
    from rppg.roi import extract_signal, make_roi

    clip = synth_rgb(duration=40.0, fs=FS, bpm=72.0, pulse_amplitude=0.004,
                     noise_std=0.05, rng=1)
    path = tmp_path / "clip.avi"
    write_synthetic_video(path, clip, pixel_noise=2.0, rng=1)

    roi = make_roi("fixed", rel_rect=(0.36, 0.34, 0.28, 0.32), use_skin_mask=False)
    with VideoFileSource(path) as src:
        t, rgb, _ = extract_signal(src, roi)
    df = analyse_signal(t, rgb, PipelineConfig(fs=FS, window_sec=15.0, hop_sec=3.0))
    green = df[df.method == "green"]
    assert abs(np.nanmedian(green.bpm) - 72.0) < 3.0


def test_per_pixel_noise_is_reduced_by_spatial_averaging():
    """The distinction that makes an ROI-size ablation meaningful at all.

    Per-pixel noise falls as 1/sqrt(N) under averaging; noise common to every
    pixel does not. An experiment about how many pixels the ROI covers measures
    nothing unless the dominant noise is the former.
    """
    rng = np.random.default_rng(0)
    base = np.full((200, 200), 100.0)
    for n_side, expected in [(20, 20), (100, 100)]:
        patch = base[:n_side, :n_side] + rng.normal(0, 5.0, (n_side, n_side))
        got = patch.mean(axis=None)
        # std of the mean should track 5/sqrt(N)
        spread = np.std([
            (base[:n_side, :n_side] + rng.normal(0, 5.0, (n_side, n_side))).mean()
            for _ in range(60)
        ])
        assert spread == pytest.approx(5.0 / n_side, rel=0.45), (
            f"{n_side}x{n_side}: spread {spread:.4f}, expected ~{5.0 / n_side:.4f}"
        )
