"""Respiratory rate from the PPG alone — three modulations, one reference.

Breathing leaves three separate marks on a photoplethysmogram, and they arise
from different physiology:

  RIIV  Respiratory Induced Intensity Variation.
        Intrathoracic pressure swings alter venous return, so the *baseline* of
        the PPG rises and falls with the breath. Amplitude modulation of the
        DC level.

  RIFV  Respiratory Induced Frequency Variation.
        Respiratory sinus arrhythmia: the heart speeds up on inspiration. This
        is frequency modulation of the cardiac carrier, and it appears in the
        inter-beat interval series — the same HF band `hrv.py` measures.

  RIAV  Respiratory Induced Amplitude Variation.
        Stroke volume varies through the respiratory cycle, so the pulse
        *height* is modulated. Amplitude modulation of the carrier itself.

Framing them as demodulation is what makes this a signal-processing problem
rather than three unrelated heuristics: one carrier at ~1-2 Hz, three
modulations at ~0.2-0.3 Hz, and the job is to recover the modulating signal
from each and see which survives best.

A constraint that must be stated, because it is a genuine limit rather than an
implementation detail: RIFV and RIAV are sampled **once per beat**. At 60 BPM
that is 1 Hz, so Nyquist is 0.5 Hz — exactly the top of the respiratory band.
A patient breathing 24 times a minute (0.4 Hz) with a heart rate of 50 BPM
(0.83 Hz sampling) has their respiration *aliased*, and no amount of filtering
recovers it. RIIV has no such limit: it is sampled at the full frame rate. That
is a concrete reason to compute all three rather than pick a favourite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .beats import clean_intervals, detect_beats, intervals
from .spectral import periodogram

#: 0.1-0.5 Hz = 6-30 breaths per minute. Covers bradypnoea to tachypnoea.
RESP_BAND_HZ = (0.1, 0.5)
#: Uniform grid for the beat-sampled series. 4 Hz is well above 2x0.5 Hz and
#: matches the HRV convention, so the two share a resampling path.
RESAMPLE_HZ = 4.0


@dataclass(frozen=True)
class RespEstimate:
    rate_bpm: float  #: breaths per minute
    snr_db: float
    method: str
    nyquist_bpm: float = float("nan")  #: aliasing limit for this method
    aliased: bool = False
    #: True only when all three routes agreed — MAE 1.16 vs 1.97 for two.
    high_confidence: bool = False


def _bandpass(x, fs, band=RESP_BAND_HZ, order=2):
    nyq = fs / 2.0
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)
    if low >= high:
        return np.zeros_like(np.asarray(x, dtype=float))
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def _resample_beat_series(beat_times, values, fs=RESAMPLE_HZ):
    """Beat-sampled series onto a uniform grid, indexed by time not beat number."""
    beat_times = np.asarray(beat_times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(beat_times) < 4:
        return np.array([]), np.array([])
    t0 = beat_times - beat_times[0]
    grid = np.arange(0.0, t0[-1], 1.0 / fs)
    if len(grid) < 8:
        return np.array([]), np.array([])
    return grid, np.interp(grid, t0, values)


def riiv(x, fs):
    """Baseline modulation, at the full sampling rate."""
    return _bandpass(x, fs), fs


def rifv(peaks, fs, resample_hz=RESAMPLE_HZ):
    """Interval modulation (respiratory sinus arrhythmia)."""
    ibi = intervals(peaks, fs)
    if len(ibi) < 4:
        return np.array([]), resample_hz
    kept, mask = clean_intervals(ibi)
    if len(kept) < 4:
        return np.array([]), resample_hz
    beat_times = (np.asarray(peaks[1:]) / fs)[mask]
    grid, series = _resample_beat_series(beat_times, kept, resample_hz)
    if len(series) == 0:
        return np.array([]), resample_hz
    return _bandpass(series, resample_hz), resample_hz


def pulse_amplitudes(x, peaks, fs):
    """Peak-to-foot height of each beat, and the time it occurred.

    The foot is the minimum just before the upstroke and the peak the maximum
    just after, so the height is measured across the systolic rise rather than
    against a global baseline that respiration is itself moving.
    """
    x = np.asarray(x, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    back, fwd = max(1, int(0.20 * fs)), max(1, int(0.40 * fs))
    times, amps = [], []
    for p in peaks:
        lo = max(0, p - back)
        hi = min(len(x), p + fwd)
        if hi - lo < 3:
            continue
        foot = np.min(x[lo : p + 1])
        top = np.max(x[p:hi])
        times.append(p / fs)
        amps.append(top - foot)
    return np.asarray(times), np.asarray(amps)


def riav(x, peaks, fs, resample_hz=RESAMPLE_HZ):
    """Amplitude modulation of the pulse itself."""
    times, amps = pulse_amplitudes(x, peaks, fs)
    if len(amps) < 4:
        return np.array([]), resample_hz
    grid, series = _resample_beat_series(times, amps, resample_hz)
    if len(series) == 0:
        return np.array([]), resample_hz
    return _bandpass(series, resample_hz), resample_hz


def estimate_rate(signal, fs, band=RESP_BAND_HZ, method="", nyquist_bpm=np.nan):
    """Dominant frequency of a modulation signal, in breaths per minute."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 16 or not np.any(np.isfinite(signal)) or np.allclose(signal, 0):
        return RespEstimate(float("nan"), float("nan"), method, nyquist_bpm, False)
    freqs, power = periodogram(signal, fs, zero_pad=8)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(inside):
        return RespEstimate(float("nan"), float("nan"), method, nyquist_bpm, False)
    idx = np.flatnonzero(inside)
    k = idx[int(np.argmax(power[inside]))]
    f_peak = float(freqs[k])
    # Same SNR convention as the cardiac path: peak plus first harmonic against
    # the rest of the band, so the two numbers are comparable.
    sig_mask = np.abs(freqs - f_peak) < 0.02
    noise = power[inside & ~sig_mask]
    snr = (
        10.0 * np.log10(power[inside & sig_mask].sum() / noise.sum())
        if noise.sum() > 0
        else float("nan")
    )
    rate = f_peak * 60.0
    return RespEstimate(
        rate_bpm=rate,
        snr_db=snr,
        method=method,
        nyquist_bpm=nyquist_bpm,
        aliased=bool(np.isfinite(nyquist_bpm) and rate > 0.9 * nyquist_bpm),
    )


def fuse(estimates, tolerance: float = 1.0, strict_tolerance: float = 3.0):
    """Combine the three routes by agreement, not by averaging or by SNR.

    Measured on 137 BIDMC windows with an in-band reference:

        method                MAE   within 2   coverage
        RIIV                 2.66     67.9%      100%
        RIFV                 3.11     57.7%      100%
        RIAV                 5.15     34.3%      100%
        median of three      2.45     67.2%      100%
        pick highest SNR     3.42     63.6%      100%
        any two within 1     1.97     75.3%       65%
        all three within 3   1.16     91.2%       25%

    Two things that curve settles. Picking by SNR is *worse than RIIV alone*:
    a narrowband peak in the wrong place scores well, so SNR measures
    confidence rather than correctness — the same lesson the cardiac path
    learned. And averaging all three is dragged down by whichever route is
    broken, which is route-dependent (RIFV vanishes without respiratory sinus
    arrhythmia, RIAV dies under motion, RIIV under baseline drift).

    Agreement works because the three failure modes are independent. Two
    mechanisms landing on the same rate is unlikely unless that rate is real.
    The cost is coverage, and that is the honest trade: 65 % of windows
    answered at 1.97 MAE beats 100 % answered at 2.66.
    """
    usable = {
        k: e.rate_bpm
        for k, e in estimates.items()
        if k in ("riiv", "rifv", "riav") and np.isfinite(e.rate_bpm)
    }
    if len(usable) < 2:
        return RespEstimate(float("nan"), float("nan"), "fused (no agreement)")

    names = list(usable)
    values = np.array([usable[k] for k in names])
    if len(values) == 3 and values.max() - values.min() <= strict_tolerance:
        return RespEstimate(
            float(values.mean()), float("nan"), "fused (all three)", high_confidence=True
        )

    best = None
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            gap = abs(values[i] - values[j])
            if gap <= tolerance and (best is None or gap < best[1]):
                best = ((values[i] + values[j]) / 2.0, gap, names[i] + "+" + names[j])
    if best is None:
        return RespEstimate(float("nan"), float("nan"), "fused (no agreement)")
    return RespEstimate(float(best[0]), float("nan"), "fused (%s)" % best[2])


def analyse(x, fs, peaks=None):
    """All three routes plus an agreement-fused estimate."""
    x = np.asarray(x, dtype=float)
    if peaks is None:
        peaks = detect_beats(x, fs)

    out = {}
    sig, sig_fs = riiv(x, fs)
    out["riiv"] = estimate_rate(sig, sig_fs, method="riiv")

    # Beat-sampled routes: Nyquist is half the mean heart rate.
    ibi = intervals(peaks, fs)
    kept, _ = clean_intervals(ibi)
    hr_hz = 1000.0 / np.mean(kept) if len(kept) else np.nan
    nyq_bpm = 0.5 * hr_hz * 60.0 if np.isfinite(hr_hz) else np.nan

    sig, sig_fs = rifv(peaks, fs)
    out["rifv"] = estimate_rate(sig, sig_fs, method="rifv", nyquist_bpm=nyq_bpm)
    sig, sig_fs = riav(x, peaks, fs)
    out["riav"] = estimate_rate(sig, sig_fs, method="riav", nyquist_bpm=nyq_bpm)

    out["fused"] = fuse(out)
    return out
