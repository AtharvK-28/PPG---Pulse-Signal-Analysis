"""Stages 3-4 — detrending and bandpass filtering."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.signal import butter, filtfilt, sosfiltfilt

from . import BAND_HZ


@lru_cache(maxsize=16)
def _smoothness_operator(n: int, lam: float):
    """(I + lam^2 D2^T D2)^-1 applied as a factorised solve.

    Cached because a sliding window calls this with the same (n, lam) every hop.
    """
    ident = sparse.eye(n, format="csc")
    ones = np.ones(n)
    d2 = sparse.spdiags(np.vstack([ones, -2 * ones, ones]), [0, 1, 2], n - 2, n).tocsc()
    return sparse.linalg.factorized((ident + (lam**2) * (d2.T @ d2)).tocsc())


def detrend_smoothness_priors(x, lam: float = 100.0):
    """Tarvainen et al. (2002) smoothness-priors detrending.

    z_stat = (I - (I + lam^2 D2^T D2)^-1) z — a time-varying high-pass with a
    very smooth response, so it removes the ~0.2-0.4 Hz respiration baseline
    without the ringing a sharp high-pass produces near the cardiac band.

    lam = 100 at fs = 30 Hz puts the -3 dB corner near 0.06 Hz.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return np.column_stack([detrend_smoothness_priors(col, lam) for col in x.T])
    n = x.size
    if n < 3:
        return x - x.mean()
    solve = _smoothness_operator(n, float(lam))
    return x - solve(x)


def detrend_moving_average(x, fs: float, win_sec: float = 1.0):
    """Simpler alternative: subtract a centred moving average of ~1 s."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return np.column_stack([detrend_moving_average(col, fs, win_sec) for col in x.T])
    w = max(3, int(round(win_sec * fs)) | 1)  # odd length keeps it centred
    if x.size <= w:
        return x - x.mean()
    pad = w // 2
    padded = np.pad(x, pad, mode="reflect")
    kernel = np.ones(w) / w
    return x - np.convolve(padded, kernel, mode="valid")


def detrend(x, fs: float, method: str = "smoothness", **kw):
    """Dispatch used by the ablation notebook."""
    if method in ("smoothness", "priors", "tarvainen"):
        return detrend_smoothness_priors(x, kw.get("lam", 100.0))
    if method in ("moving_average", "ma"):
        return detrend_moving_average(x, fs, kw.get("win_sec", 1.0))
    if method == "none":
        return np.asarray(x, dtype=float) - np.mean(x, axis=0)
    raise ValueError(f"unknown detrend method {method!r}")


def bandpass(x, fs: float, band=BAND_HZ, order: int = 4):
    """Zero-phase 4th-order Butterworth over 0.7-4.0 Hz (42-240 BPM).

    filtfilt, not lfilter: forward-backward filtering cancels phase distortion
    exactly, at the cost of doubling the effective order and needing the whole
    segment in memory — affordable because we work on buffered windows.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = band
    nyq = fs / 2.0
    if not 0 < lo < hi < nyq:
        raise ValueError(f"band {band} invalid for fs={fs}")
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    # sosfiltfilt is the numerically stable form of filtfilt for cascaded
    # biquads; identical zero-phase response.
    padlen = 3 * (2 * len(sos) + 1)
    if x.shape[0] <= padlen:
        raise ValueError(
            f"segment of {x.shape[0]} samples too short for zero-phase filtering "
            f"(need > {padlen}); use a longer window"
        )
    return sosfiltfilt(sos, x, axis=0)


def bandpass_ba(x, fs: float, band=BAND_HZ, order: int = 4):
    """Transfer-function form — kept to show filtfilt/lfilter equivalence in the notebook."""
    nyq = fs / 2.0
    b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype="bandpass")
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=0)


def normalize_zscore(x, eps: float = 1e-12):
    """Zero mean, unit variance per column — ICA's required pre-whitening step."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def normalize_mean(x, eps: float = 1e-12):
    """Divide by the temporal mean — CHROM's skin-tone normalisation."""
    x = np.asarray(x, dtype=float)
    return x / (x.mean(axis=0) + eps)


def despike(rgb, k: float = 4.0, max_fraction: float = 0.05):
    """Replace motion transients with interpolation across all three channels.

    A blink, a swallow or a head jerk moves the ROI onto different skin for a
    frame or two. In the trace it is a step or a spike; in the spectrum it is
    energy at *every* frequency, because a discontinuity has no frequency of
    its own. One such event in a 15 s window can outweigh the entire pulse,
    which is what makes the four arms agree on nonsense — they are all seeing
    the same broadband artifact.

    Detection is on the first difference of luminance, scored against the MAD
    rather than the standard deviation: the spikes are exactly what would
    inflate an SD-based threshold and hide themselves. Repair is on all three
    channels at the same indices, because a motion artifact is common-mode by
    nature and repairing channels independently would manufacture chrominance
    that was never there.

    If more than ``max_fraction`` of samples trip the test, the "spikes" are
    the signal — a genuinely noisy recording, not a few transients — and the
    input is returned untouched rather than smoothed into something plausible.
    """
    x = np.asarray(rgb, dtype=float)
    single = x.ndim == 1
    if single:
        x = x[:, None]
    if len(x) < 5:
        return (x[:, 0] if single else x).copy()

    lum = x.mean(axis=1)
    d = np.diff(lum)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad <= 0:
        return (x[:, 0] if single else x).copy()
    # 1.4826 * MAD estimates sigma for Gaussian data.
    bad_diff = np.abs(d - med) > k * 1.4826 * mad
    # A bad step implicates the sample on each side of it.
    bad = np.zeros(len(x), dtype=bool)
    bad[:-1] |= bad_diff
    bad[1:] |= bad_diff
    if not bad.any() or bad.mean() > max_fraction:
        return (x[:, 0] if single else x).copy()

    out = x.copy()
    idx = np.arange(len(x))
    good = ~bad
    if good.sum() < 2:
        return (x[:, 0] if single else x).copy()
    for c in range(x.shape[1]):
        out[bad, c] = np.interp(idx[bad], idx[good], x[good, c])
    return out[:, 0] if single else out
