"""CHROM — de Haan & Jeanne (2013). Chrominance-based projection.

Build two chrominance signals in which the specular component largely cancels,
then combine them with an adaptive scalar tuned so the *motion*-induced
residues in the two destructively interfere:

    Xs = 3Rn - 2Gn
    Ys = 1.5Rn + Gn - 1.5Bn
    S  = Xf - (sigma(Xf)/sigma(Yf)) Yf

The alpha scaling is the clever part and the reason CHROM survives motion that
ICA does not: it is a per-window correction derived from the reflection model,
not a learned unmixing.
"""

from __future__ import annotations

import numpy as np

from .. import BAND_HZ
from ..preprocess import bandpass


def chrom(
    rgb,
    fs: float,
    window_sec: float = 1.6,
    band=BAND_HZ,
    overlap_add: bool = True,
    bandpass_scope: str = "global",
    **kw,
) -> np.ndarray:
    """CHROM with 1.6 s windows, 50% overlap, Hann-weighted overlap-add.

    ``bandpass_scope`` decides where step 4 happens, and it matters more than it
    looks. ``"window"`` is the literal reading of the algorithm — bandpass Xs
    and Ys *inside* each 1.6 s window. At 30 fps that window is 48 samples,
    while a 0.97 Hz fundamental (58 BPM) has a 31-sample period: barely one
    cycle, against three for its second harmonic. The filter therefore
    attenuates the fundamental far more than the harmonic and tilts the
    spectrum upward, which makes the estimator report double the true rate.

    Measured on a 58 BPM synthetic pulse, power ratio 2f0/f0:

        per-window bandpass   2.57   (harmonic dominates — wrong answer)
        global bandpass       1.49
        GREEN, for reference  0.37   (fundamental dominates, as physics says)

    ``"global"`` keeps the per-window skin-tone normalisation and the per-window
    alpha — the parts that do the actual work — but filters once over the whole
    signal, where 0.7 Hz is properly resolvable. That is the default.

    Set ``overlap_add=False`` for the single-window form written in the ideation
    doc — useful for the walkthrough figure, but it normalises skin tone over
    the whole clip, so slow illumination drift leaks into alpha.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB means, got {rgb.shape}")
    n = rgb.shape[0]

    if not overlap_add:
        return _chrom_window(rgb, fs, band)

    if bandpass_scope == "global":
        return _chrom_global(rgb, fs, band, window_sec)

    length = int(round(window_sec * fs))
    stride = max(1, length // 2)
    if n < length:
        return _chrom_window(rgb, fs, band)

    out = np.zeros(n)
    weight = np.zeros(n)
    for start in range(0, n - length + 1, stride):
        seg = rgb[start : start + length]
        s = _chrom_window(seg, fs, band)
        # Hann weighting makes the 50%-overlap sum a smooth partition of unity,
        # so window boundaries do not inject step discontinuities (which would
        # spread broadband energy across the cardiac band).
        w = np.hanning(length)
        out[start : start + length] += (s - s.mean()) * w
        weight[start : start + length] += w

    tail = n - length
    if tail % stride:  # cover the remainder the strided loop misses
        seg = rgb[tail:]
        s = _chrom_window(seg, fs, band)
        w = np.hanning(len(seg))
        out[tail:] += (s - s.mean()) * w
        weight[tail:] += w

    return out / np.maximum(weight, 1e-9)


def _chrom_global(rgb: np.ndarray, fs: float, band, window_sec: float) -> np.ndarray:
    """Per-window skin-tone normalisation, one global bandpass, global alpha."""
    n = rgb.shape[0]
    length = min(int(round(window_sec * fs)), n)
    stride = max(1, length // 2)

    xs = np.zeros(n)
    ys = np.zeros(n)
    weight = np.zeros(n)
    starts = list(range(0, max(1, n - length + 1), stride))
    if starts[-1] + length < n:
        starts.append(n - length)
    for start in starts:
        seg = rgb[start : start + length]
        mean = seg.mean(axis=0)
        rn, gn, bn = (seg / np.where(mean == 0, 1e-9, mean)).T
        w = np.hanning(len(rn))
        xs[start : start + length] += (3.0 * rn - 2.0 * gn) * w
        ys[start : start + length] += (1.5 * rn + gn - 1.5 * bn) * w
        weight[start : start + length] += w

    xs /= np.maximum(weight, 1e-9)
    ys /= np.maximum(weight, 1e-9)

    xf = bandpass(xs, fs, band=band)
    yf = bandpass(ys, fs, band=band)
    sy = yf.std()
    alpha = xf.std() / sy if sy > 1e-12 else 0.0
    return xf - alpha * yf


def _chrom_window(seg: np.ndarray, fs: float, band) -> np.ndarray:
    """The six steps, on one window."""
    mean = seg.mean(axis=0)
    rn, gn, bn = (seg / np.where(mean == 0, 1e-9, mean)).T  # skin-tone normalise

    xs = 3.0 * rn - 2.0 * gn
    ys = 1.5 * rn + gn - 1.5 * bn

    xf = bandpass(xs, fs, band=band)
    yf = bandpass(ys, fs, band=band)

    sy = yf.std()
    alpha = xf.std() / sy if sy > 1e-12 else 0.0
    return xf - alpha * yf
