"""POS — Wang, den Brinker, Stuijk & de Haan (2017). Plane orthogonal to skin.

Temporally normalise RGB over a short sliding window, then project onto a plane
orthogonal to the skin-tone direction:

    P = [[0, 1, -1], [-2, 1, 1]]   =>   S1 = Gn - Bn,  S2 = -2Rn + Gn + Bn

and tune the combination so the intensity-distortion component cancels:

    h = S1 + (sigma(S1)/sigma(S2)) S2

Unlike CHROM, POS needs no bandpass inside the loop — the temporal
normalisation over l = 1.6 fs samples is itself the high-pass.
"""

from __future__ import annotations

import numpy as np

_P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos(rgb, fs: float, window_sec: float = 1.6, **kw) -> np.ndarray:
    """POS with the original single-sample stride and overlap-add."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB means, got {rgb.shape}")
    n = rgb.shape[0]
    length = int(round(window_sec * fs))
    if n < length:
        length = n
    if length < 2:
        return np.zeros(n)

    out = np.zeros(n)
    for end in range(length, n + 1):
        start = end - length
        seg = rgb[start:end]

        mean = seg.mean(axis=0)
        cn = seg / np.where(np.abs(mean) < 1e-9, 1e-9, mean)

        s = _P @ cn.T  # (2, length)
        s2_std = s[1].std()
        alpha = s[0].std() / s2_std if s2_std > 1e-12 else 0.0
        h = s[0] + alpha * s[1]

        # Overlap-add the zero-mean window: each sample accumulates `length`
        # independent estimates, which is where POS's noise averaging comes from.
        out[start:end] += h - h.mean()

    return out
