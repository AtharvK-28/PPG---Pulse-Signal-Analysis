"""Stage 7 — de Haan & Jeanne signal-to-noise ratio.

Needs no ground truth, so it works live: it drives the on-screen confidence
colour and lets us compare arms on unlabelled video.
"""

from __future__ import annotations

import numpy as np


def snr_db(freqs, power, f_peak: float, half_width: float = 0.1, band=(0.5, 4.0)) -> float:
    """Power inside +/-``half_width`` of f_peak and 2*f_peak, over everything else.

    SNR = 10 log10( sum M|S|^2 / sum (1-M)|S|^2 ), M the binary mask that is 1
    on the fundamental and first harmonic.
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band) or not np.isfinite(f_peak):
        return float("nan")

    f, p = freqs[in_band], power[in_band]
    mask = (np.abs(f - f_peak) <= half_width) | (np.abs(f - 2.0 * f_peak) <= half_width)
    signal = float(p[mask].sum())
    noise = float(p[~mask].sum())
    if noise <= 0:
        return float("inf") if signal > 0 else float("nan")
    if signal <= 0:
        return -np.inf
    return float(10.0 * np.log10(signal / noise))
