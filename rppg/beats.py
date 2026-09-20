"""Time-domain beat detection: where each pulse actually happened.

The spectral path in `spectral.py` answers "what rate?" over a whole window.
This answers "when?", beat by beat, which is a different question and the one
HRV needs: a 15 s window that reports 72 BPM tells you nothing about whether
the intervals were 830 ms every time or alternated 700/960.

Why not just `find_peaks` on the PPG:

  * The systolic peak is *rounded*. Its location is poorly defined and shifts
    by tens of milliseconds with noise — which is fatal when RMSSD is computed
    from differences between consecutive intervals of ~800 ms.
  * The dicrotic notch puts a second local maximum inside every beat. Naive
    peak finding takes it as a beat and doubles the rate.
  * PPG amplitude varies several-fold with perfusion, posture and motion, so
    any fixed threshold is wrong somewhere in a long recording.

So the detector keys on the *upstroke*, which is the sharpest and least
ambiguous feature of the waveform, via the Slope Sum Function (Zong et al.,
2003):

    SSF[i] = sum over the preceding window of max(0, x[k] - x[k-1])

Only positive differences contribute, so the steep systolic rise accumulates
into a clean unimodal pulse while the slow diastolic decay contributes nothing.
The dicrotic notch has a much smaller rise and does not survive the threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .preprocess import bandpass

#: Wider than the HR estimation band: the upstroke carries energy well above
#: the fundamental, and low-passing at 4 Hz rounds off the very edge the
#: detector depends on.
BEAT_BAND_HZ = (0.5, 8.0)
#: 300 ms => 200 BPM ceiling. Physiological, and it is what stops the dicrotic
#: notch (~250-350 ms after systole) from ever being counted as a beat.
REFRACTORY_SEC = 0.30


def slope_sum(x, fs: float, win_sec: float = 0.128):
    """Accumulated positive slope over a trailing window.

    0.128 s is about the duration of a systolic upstroke, so the window
    integrates one rise and no more. Much longer and adjacent beats merge;
    much shorter and it degenerates towards the raw derivative, which is noisy.
    """
    x = np.asarray(x, dtype=float)
    d = np.diff(x, prepend=x[0])
    np.clip(d, 0.0, None, out=d)
    w = max(1, int(round(win_sec * fs)))
    kernel = np.ones(w)
    # 'full' then trim: the sum must look backwards only, or every detected
    # beat lands early by half the window.
    return np.convolve(d, kernel, mode="full")[: len(x)]


def detect_beats(x, fs: float, refractory: float = REFRACTORY_SEC,
                 band=BEAT_BAND_HZ, threshold_scale: float = 0.6,
                 adapt_sec: float = 10.0):
    """Sample indices of systolic upstrokes.

    The threshold adapts over a trailing window rather than being global: over
    eight minutes of ICU recording the pulse amplitude changes severalfold, and
    a global threshold either misses the quiet stretches or fires twice per
    beat in the loud ones.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < int(2 * fs):
        return np.array([], dtype=int)

    filtered = bandpass(x, fs, band=band)
    ssf = slope_sum(filtered, fs)

    # Local scale from a running median-of-magnitude, robust to the spikes that
    # motion produces (a mean would be dragged up by them and then miss beats).
    w = max(int(adapt_sec * fs), int(2 * fs))
    if len(ssf) <= w:
        level = np.full(len(ssf), np.median(ssf[ssf > 0]) if np.any(ssf > 0) else 0.0)
    else:
        pad = np.pad(ssf, (w // 2, w // 2), mode="edge")
        # Strided median over non-overlapping blocks, then interpolated back:
        # a true running median at every sample costs O(N*w) and buys nothing.
        centres = np.arange(0, len(ssf), max(1, int(fs)))
        vals = np.array([np.median(pad[c : c + w]) for c in centres])
        level = np.interp(np.arange(len(ssf)), centres, vals)

    height = threshold_scale * np.maximum(level, 1e-12) * 3.0
    peaks, _ = find_peaks(ssf, height=height, distance=max(1, int(refractory * fs)))
    if len(peaks) == 0:
        return peaks

    peaks = _enforce_adaptive_refractory(
        peaks, ssf, fs, refractory, expected=_expected_interval(x, fs)
    )

    # Fiducial = steepest point of the upstroke, NOT the systolic maximum.
    # The maximum is rounded, so its location moves tens of ms with noise, and
    # RMSSD is built from differences between consecutive intervals — measured
    # against ECG on BIDMC, snapping to the maximum inflated RMSSD by 2-3x
    # (50.9 vs 13.5 ms on bidmc02) purely as detector jitter. The steepest
    # upslope is the sharpest feature the waveform has.
    derivative = np.gradient(filtered)
    out, back = [], max(1, int(0.20 * fs))
    for p in peaks:
        lo = max(0, p - back)
        seg = derivative[lo : p + 1]
        out.append(lo + int(np.argmax(seg)) if len(seg) else p)
    return np.unique(np.asarray(out, dtype=int))


def _expected_interval(x, fs, block_sec: float = 30.0):
    """Beat interval in seconds, from the spectrum rather than from the beats.

    This is the seed the refractory needs. Deriving the guard from the median
    of the *detected* peaks is circular: if the detector is already firing
    twice per beat, that median is the halved interval and the guard becomes a
    quarter of the truth — which is precisely why bidmc47 stayed at 51 %
    precision after the first fix.

    The spectral estimate is independent of that failure. The dicrotic notch
    contributes to the second harmonic, not the fundamental, so the peak the
    spectrum reports is the true rate; and that path is separately validated at
    0.80 BPM MAE against ECG (see rppg.bidmc). Taking a median over 30 s blocks
    keeps it usable when the rate drifts across a long record.
    """
    from .spectral import estimate_bpm

    cardiac = bandpass(np.asarray(x, dtype=float), fs)
    n = int(block_sec * fs)
    if len(cardiac) < n:
        blocks = [cardiac]
    else:
        blocks = [cardiac[s : s + n] for s in range(0, len(cardiac) - n + 1, n)]
    bpms = []
    for b in blocks:
        try:
            bpms.append(estimate_bpm(b, fs).bpm)
        except ValueError:
            continue
    if not bpms:
        return None
    return 60.0 / float(np.median(bpms))


def _enforce_adaptive_refractory(peaks, ssf, fs, floor_sec, expected=None):
    """Drop the dicrotic notch, which a fixed refractory cannot reach.

    The notch follows systole by 250-400 ms. At 60 BPM a 300 ms fixed window
    excludes it; at 90 BPM the interval is 667 ms and the notch sits well
    outside the window, so it is detected as a beat and the rate doubles —
    which is exactly what happened on bidmc47 (1306 detections against 671 R
    waves, precision 51%).

    The guard is half the expected cycle, because the notch always falls at a
    roughly fixed *fraction* of the cycle rather than a fixed delay.
    """
    peaks = np.asarray(peaks)
    if len(peaks) < 3:
        return peaks
    if expected is not None and np.isfinite(expected):
        reference = expected * fs
    else:
        reference = np.median(np.diff(peaks))
    guard = max(floor_sec * fs, 0.5 * reference)
    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= guard:
            kept.append(p)
        elif ssf[p] > ssf[kept[-1]]:
            # The later candidate is stronger: it is the beat, the earlier one
            # was the artefact.
            kept[-1] = p
    return np.asarray(kept, dtype=int)


def intervals(peaks, fs: float):
    """Inter-beat intervals in milliseconds."""
    peaks = np.asarray(peaks)
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000.0


def clean_intervals(ibi, low: float = 300.0, high: float = 2000.0,
                    max_change: float = 0.2):
    """Drop physiologically impossible and locally implausible intervals.

    HRV is computed from *differences between consecutive* intervals, so a
    single missed beat — which merges two intervals into one double-length one
    — does not add one bad number, it adds two enormous ones. RMSSD is roughly
    doubled by a single missed beat in a five-minute record, which is larger
    than most effects anyone would want to measure.

    ``max_change`` rejects intervals differing by more than 20 % from the local
    median (Task Force 1996 convention). Returns the surviving intervals and a
    boolean mask, because the fraction rejected is itself a quality measure.
    """
    ibi = np.asarray(ibi, dtype=float)
    if len(ibi) == 0:
        return ibi, np.zeros(0, dtype=bool)
    ok = (ibi >= low) & (ibi <= high)
    if ok.sum() >= 3:
        med = np.median(ibi[ok])
        ok &= np.abs(ibi - med) <= max_change * med
    return ibi[ok], ok


def beat_rate(peaks, fs: float, clean: bool = True):
    """Mean heart rate in BPM from beat timings, not from a spectrum."""
    ibi = intervals(peaks, fs)
    if clean:
        ibi, _ = clean_intervals(ibi)
    if len(ibi) == 0:
        return float("nan")
    return 60000.0 / np.mean(ibi)
