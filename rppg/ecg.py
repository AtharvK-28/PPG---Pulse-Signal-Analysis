"""Pan-Tompkins R-peak detection — the reference the PPG detector is judged against.

This is here to provide ground truth, not because the project is about ECG. A
PPG beat detector validated against its own output is validated against nothing;
BIDMC records ECG simultaneously, and the R wave is the least ambiguous fiducial
in either signal.

Pan & Tompkins (1985) is five stages, each doing one job:

    bandpass 5-15 Hz     keep the QRS, drop P/T waves, baseline and mains
    derivative           QRS slope is the discriminating feature
    square               rectify, and emphasise large slopes over small ones
    moving-window sum    ~150 ms, the QRS width: one hump per complex
    adaptive threshold   ECG amplitude drifts; a fixed threshold does not survive

It remains the standard because every stage is defensible from the physiology,
which is also why it is worth implementing rather than importing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

#: The QRS occupies roughly 5-15 Hz; P and T waves sit below, muscle noise above.
QRS_BAND_HZ = (5.0, 15.0)
#: 200 ms is the physiological refractory period of cardiac muscle — no two
#: genuine R waves can be closer, so anything that is has to be one of them.
REFRACTORY_SEC = 0.20
#: The QRS complex lasts ~100-150 ms; the integration window matches it so each
#: complex produces exactly one hump.
INTEGRATION_SEC = 0.15


def _bandpass(x, fs, band=QRS_BAND_HZ, order=2):
    nyq = fs / 2.0
    high = min(band[1] / nyq, 0.99)
    b, a = butter(order, [band[0] / nyq, high], btype="bandpass")
    return filtfilt(b, a, x)


def pan_tompkins(x, fs: float, refractory: float = REFRACTORY_SEC):
    """R-peak sample indices."""
    x = np.asarray(x, dtype=float)
    if len(x) < int(2 * fs):
        return np.array([], dtype=int)

    filtered = _bandpass(x, fs)
    derivative = np.gradient(filtered) * fs
    squared = derivative**2
    w = max(1, int(round(INTEGRATION_SEC * fs)))
    integrated = np.convolve(squared, np.ones(w) / w, mode="same")

    # Adaptive threshold: a running quantile rather than a fraction of the
    # global maximum, so one motion spike cannot raise the bar for the whole
    # record and silence every subsequent beat.
    block = max(1, int(2 * fs))
    centres = np.arange(0, len(integrated), block)
    level = np.array(
        [np.percentile(integrated[c : c + block * 2], 95) for c in centres]
    )
    thresh = np.interp(np.arange(len(integrated)), centres, level) * 0.35

    peaks, _ = find_peaks(
        integrated, height=thresh, distance=max(1, int(refractory * fs))
    )
    # The integrator delays and broadens; snap back to the true R peak, taken
    # as the largest absolute deflection of the bandpassed ECG nearby.
    out = []
    half = max(1, int(0.05 * fs))
    for p in peaks:
        lo, hi = max(0, p - half), min(len(filtered), p + half + 1)
        if hi > lo:
            out.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    return np.unique(np.asarray(out, dtype=int)) if out else np.array([], dtype=int)


def match_beats(reference, test, fs: float, tolerance: float = 0.15):
    """Match detected beats to reference beats within a tolerance.

    Returns (true positives, false negatives, false positives, signed offsets
    in ms). Greedy nearest-match, which is adequate because the tolerance is
    far smaller than any plausible interval.

    The offsets matter as much as the counts: PPG lags ECG by the pulse transit
    time, 100-300 ms depending on the site. A *consistent* offset is physiology
    and harmless to HRV; a *variable* one is detector jitter and shows up
    directly in RMSSD.
    """
    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)
    if len(reference) == 0 or len(test) == 0:
        return 0, len(reference), len(test), np.array([])

    tol = tolerance * fs
    used = np.zeros(len(test), dtype=bool)
    tp, offsets = 0, []
    for r in reference:
        d = np.abs(test - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
            offsets.append((test[j] - r) / fs * 1000.0)
    return tp, len(reference) - tp, int((~used).sum()), np.asarray(offsets)
