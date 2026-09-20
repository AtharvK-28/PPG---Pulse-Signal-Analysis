"""GREEN — Verkruysse, Svaasand & Nelson (2008). The baseline you must beat.

Projection w = [0, 1, 0]. Green carries the strongest pulsatile signal both
because haemoglobin absorption peaks near 540 nm and because a Bayer sensor has
twice as many green photosites as red or blue, so the channel starts with a
better SNR before any processing.

If a sophisticated arm only ties GREEN, that is a real and reportable result.
"""

from __future__ import annotations

import numpy as np

from ..preprocess import detrend


def green(rgb, fs: float, detrend_method: str = "smoothness", **kw) -> np.ndarray:
    """Spatial-mean green channel, detrended."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB means, got {rgb.shape}")
    return detrend(rgb[:, 1], fs, method=detrend_method, **kw)
