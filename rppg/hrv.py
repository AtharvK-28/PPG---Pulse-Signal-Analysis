"""Heart rate variability from an inter-beat interval series.

HRV is where PPG stops being a heart-rate monitor and becomes a measurement of
autonomic state. The metrics follow the Task Force of the European Society of
Cardiology / North American Society of Pacing and Electrophysiology (1996)
standard, which is what makes numbers from different studies comparable at all.

Two families, measuring different things:

  Time domain  — SDNN is total variability over the whole record, dominated by
                 slow drift. RMSSD and pNN50 use *successive differences*, so
                 slow drift cancels and what remains is beat-to-beat change,
                 which is essentially parasympathetic (vagal) activity.

  Frequency    — the interval series, viewed as a signal sampled once per beat,
                 has structure. HF (0.15-0.40 Hz) is respiratory sinus
                 arrhythmia: the heart speeds up on inspiration. LF
                 (0.04-0.15 Hz) mixes baroreflex and sympathetic activity.

One trap worth stating plainly, because it invalidates a lot of published work:
the interval series is *not uniformly sampled*. Its samples arrive one per
beat, i.e. at the very rate being measured. Running an FFT on the raw series
treats a variable sampling interval as constant, which smears the spectrum.
Interpolating onto a uniform grid first is the standard fix and is what
`frequency_domain` does.

Minimum record lengths also matter: HF needs ~1 minute, LF ~2 minutes, and the
Task Force recommends 5 minutes for the ratio. Shorter windows produce numbers,
not measurements, so `frequency_domain` reports what it had to work with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

#: Task Force band definitions, in Hz.
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
#: 4 Hz is the convention for the interpolated tachogram: comfortably above
#: twice the 0.4 Hz top of the HF band, without inventing resolution.
RESAMPLE_HZ = 4.0


@dataclass(frozen=True)
class TimeDomain:
    mean_nn: float  #: ms
    sdnn: float  #: ms, total variability
    rmssd: float  #: ms, beat-to-beat (vagal)
    pnn50: float  #: %, successive differences over 50 ms
    cv: float  #: %, sdnn / mean_nn
    mean_hr: float  #: BPM
    n_beats: int


@dataclass(frozen=True)
class FrequencyDomain:
    vlf: float
    lf: float
    hf: float
    total: float
    lf_hf: float  #: sympathovagal balance, as usually reported
    lf_nu: float  #: normalised units, LF / (LF + HF)
    hf_nu: float
    duration_sec: float
    reliable: bool  #: at least 2 minutes, per the Task Force minimum for LF


def time_domain(ibi) -> TimeDomain:
    """Task Force time-domain metrics from NN intervals in milliseconds."""
    ibi = np.asarray(ibi, dtype=float)
    if len(ibi) < 2:
        nan = float("nan")
        return TimeDomain(nan, nan, nan, nan, nan, nan, len(ibi))
    diffs = np.diff(ibi)
    mean_nn = float(np.mean(ibi))
    # ddof=1: these are a sample of the intervals, not the population.
    sdnn = float(np.std(ibi, ddof=1))
    return TimeDomain(
        mean_nn=mean_nn,
        sdnn=sdnn,
        rmssd=float(np.sqrt(np.mean(diffs**2))),
        pnn50=float(100.0 * np.mean(np.abs(diffs) > 50.0)),
        cv=float(100.0 * sdnn / mean_nn) if mean_nn else float("nan"),
        mean_hr=float(60000.0 / mean_nn) if mean_nn else float("nan"),
        n_beats=len(ibi) + 1,
    )


def tachogram(ibi, fs: float = RESAMPLE_HZ):
    """Interval series interpolated onto a uniform time grid.

    The x-axis is the cumulative beat time, not the beat number: an interval
    series indexed by beat number is stretched wherever the heart was slow,
    which puts a rate-dependent distortion straight onto the frequency axis.
    """
    ibi = np.asarray(ibi, dtype=float)
    if len(ibi) < 4:
        return np.array([]), np.array([])
    t = np.cumsum(ibi) / 1000.0
    t = t - t[0]
    grid = np.arange(0.0, t[-1], 1.0 / fs)
    if len(grid) < 4:
        return np.array([]), np.array([])
    return grid, np.interp(grid, t, ibi)


def frequency_domain(ibi, fs: float = RESAMPLE_HZ) -> FrequencyDomain:
    """Spectral HRV via Welch on the interpolated tachogram."""
    grid, series = tachogram(ibi, fs)
    nan = float("nan")
    if len(series) < 8:
        return FrequencyDomain(nan, nan, nan, nan, nan, nan, nan, 0.0, False)

    duration = float(grid[-1])
    # 4 segments where possible: Welch trades resolution for variance, and an
    # HRV spectrum estimated from one periodogram is far too noisy to compare.
    nperseg = min(len(series), max(64, int(len(series) / 4)))
    freqs, power = welch(series - series.mean(), fs, nperseg=nperseg)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(power[m], freqs[m])) if m.sum() > 1 else 0.0

    vlf, lf, hf = band(*VLF_BAND), band(*LF_BAND), band(*HF_BAND)
    denom = lf + hf
    return FrequencyDomain(
        vlf=vlf,
        lf=lf,
        hf=hf,
        total=vlf + lf + hf,
        lf_hf=float(lf / hf) if hf > 0 else nan,
        lf_nu=float(100.0 * lf / denom) if denom > 0 else nan,
        hf_nu=float(100.0 * hf / denom) if denom > 0 else nan,
        duration_sec=duration,
        reliable=duration >= 120.0,
    )


def summary(ibi) -> dict:
    """Both families in one dict, for tables and notebooks."""
    td, fd = time_domain(ibi), frequency_domain(ibi)
    return {
        "n_beats": td.n_beats,
        "mean_hr_bpm": td.mean_hr,
        "mean_nn_ms": td.mean_nn,
        "sdnn_ms": td.sdnn,
        "rmssd_ms": td.rmssd,
        "pnn50_pct": td.pnn50,
        "cv_pct": td.cv,
        "lf_ms2": fd.lf,
        "hf_ms2": fd.hf,
        "lf_hf": fd.lf_hf,
        "lf_nu": fd.lf_nu,
        "hf_nu": fd.hf_nu,
        "duration_sec": fd.duration_sec,
        "hrv_reliable": fd.reliable,
    }
