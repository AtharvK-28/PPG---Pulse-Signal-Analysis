"""Stage 5 — windowing, FFT, and sub-bin peak location.

Frequency resolution is set by window *duration*, not FFT length:

    df = 1/T   =>   dBPM = 60/T

so a 10 s window quantises to 6 BPM, which alone would sink any sub-6-BPM MAE
claim. Three mitigations, all implemented here: a longer window (caller's
choice), zero-padding, and parabolic peak interpolation.

Zero-padding interpolates the DTFT densely; it does NOT add resolution. Two
tones 3 BPM apart in a 10 s window stay unresolved no matter how much padding
is applied — padding only locates a single isolated peak more precisely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, welch

from . import BAND_HZ
from .quality import snr_db


@dataclass(frozen=True, eq=False)
class SpectrumResult:
    bpm: float
    f_peak: float
    snr: float
    freqs: np.ndarray
    power: np.ndarray
    peak_bin: int
    #: 60/T — the honest resolution, unchanged by zero-padding.
    resolution_bpm: float = float("nan")


def _parabolic_peak(mag_db: np.ndarray, k: int) -> float:
    """Sub-bin offset of the vertex of the parabola through bins k-1, k, k+1.

    Fitted in log-magnitude, where a windowed sinusoid's main lobe is close to
    quadratic — that is what makes the 3-point fit accurate rather than merely
    smooth.
    """
    if k <= 0 or k >= mag_db.size - 1:
        return 0.0
    y0, y1, y2 = mag_db[k - 1], mag_db[k], mag_db[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if not np.isfinite(denom) or abs(denom) < 1e-20:
        return 0.0
    delta = 0.5 * (y0 - y2) / denom
    # A vertex further than half a bin away means the 3-point fit is not
    # describing this peak; refuse it rather than emit a wild frequency.
    return float(delta) if abs(delta) <= 0.5 else 0.0


def periodogram(x, fs: float, zero_pad: int = 8, window: str = "hann"):
    """Windowed, zero-padded one-sided power spectrum.

    Hann rather than rectangular: -31 dB first sidelobe instead of -13 dB, so a
    strong residual trend cannot leak a false peak into the cardiac band. The
    cost is a main lobe twice as wide, which is why peak *interpolation* rather
    than peak *bin* is used downstream.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        raise ValueError("window too short for a spectrum")
    x = x - x.mean()
    w = get_window(window, n) if window else np.ones(n)
    xw = x * w
    nfft = int(2 ** np.ceil(np.log2(max(n * max(zero_pad, 1), 8))))
    spec = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    # Normalise by window power so amplitudes stay comparable across windows.
    power = (np.abs(spec) ** 2) / (np.sum(w**2) + 1e-20)
    return freqs, power


#: Fundamental heart rate is searched over a narrower range than the analysis
#: band. The band must reach 4 Hz to *contain* the harmonics of a fast pulse,
#: but a fundamental above ~3 Hz (180 BPM) is not what a seated subject is
#: doing, and allowing it invites the estimator to report a harmonic as if it
#: were the rate.
FUNDAMENTAL_HZ = (0.7, 3.0)


def harmonic_score(
    freqs,
    power,
    fundamental=FUNDAMENTAL_HZ,
    weights=(1.0, 0.60, 0.35),
):
    """Score each candidate fundamental by its own power plus its harmonics'.

    A cardiac pulse is not a sinusoid — a steep systolic upstroke and dicrotic
    notch put real energy at 2f and 3f. A periodic *motion* artefact generally
    does not. So summing power across the harmonic series separates them on a
    property that a single-peak search is blind to.

    This matters because the naive maximum is not merely imprecise, it is
    sometimes on the wrong line entirely: CHROM and POS high-pass the signal
    internally, which tilts the spectrum upward and can leave the second
    harmonic taller than the fundamental. The estimator then reports exactly
    twice the true heart rate, confidently and with an excellent SNR.

    Weights decay with harmonic order because higher harmonics are weaker and
    noisier; the fundamental still carries the most vote.

    **Known failure, measured not assumed.** This has a symmetric weakness to
    the one it fixes: a candidate at f0/2 collects w2 * P(f0) from the real
    pulse line, so if low-frequency motion leaves anything at f0/2 to seed it,
    the subharmonic can win on borrowed evidence. It bites hardest at elevated
    heart rates, where the true harmonics fall outside the passband and the
    correct candidate gets no support of its own.

    A minimum-own-power gate was tried and removed: it made no difference on
    real clips (the spurious candidate genuinely has power there) and it blocks
    the weak-fundamental rescue that is the whole point. Normalising by the
    number of in-band harmonics was also tried — it fixes the elevated-HR case
    and breaks the resting case by more. Temporal continuity is what actually
    resolves it; see `rppg.tracking`.
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    sel = (freqs >= fundamental[0]) & (freqs <= fundamental[1])
    if not np.any(sel):
        raise ValueError(f"no bins inside the fundamental range {fundamental}")
    cand = freqs[sel]

    score = np.zeros_like(cand)
    for k, w in enumerate(weights, start=1):
        # Harmonics beyond the analysed band contribute nothing rather than
        # wrapping around, hence left/right = 0.
        score += w * np.interp(k * cand, freqs, power, left=0.0, right=0.0)

    return cand, score


def estimate_bpm(
    x,
    fs: float,
    band=BAND_HZ,
    zero_pad: int = 8,
    window: str = "hann",
    interpolate: bool = True,
    selector: str = "harmonic",  #: "harmonic" | "peak"
    fundamental=FUNDAMENTAL_HZ,
) -> SpectrumResult:
    """Locate the pulse frequency and convert to BPM.

    ``selector="peak"`` takes the tallest in-band bin — the classical choice,
    kept so the harmonic selector can be measured against it rather than
    assumed better.
    """
    x = np.asarray(x, dtype=float)
    freqs, power = periodogram(x, fs, zero_pad=zero_pad, window=window)
    df = freqs[1] - freqs[0]

    if selector == "harmonic":
        cand, score = harmonic_score(freqs, power, fundamental=fundamental)
        j = int(np.argmax(score))
        # Interpolate on the score curve, which is sampled on the same grid.
        score_db = 10.0 * np.log10(score + 1e-20)
        delta = _parabolic_peak(score_db, j) if interpolate else 0.0
        f_peak = float(cand[j] + delta * df)
        k = int(np.argmin(np.abs(freqs - cand[j])))
        f_peak = float(np.clip(f_peak, fundamental[0], fundamental[1]))
    else:
        in_band = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(in_band):
            raise ValueError(f"no FFT bins inside {band} Hz at fs={fs}")
        band_idx = np.flatnonzero(in_band)
        k = int(band_idx[np.argmax(power[in_band])])
        mag_db = 10.0 * np.log10(power + 1e-20)
        k_frac = k + (_parabolic_peak(mag_db, k) if interpolate else 0.0)
        f_peak = float(np.clip(k_frac * df, band[0], band[1]))

    return SpectrumResult(
        bpm=f_peak * 60.0,
        f_peak=f_peak,
        snr=snr_db(freqs, power, f_peak),
        freqs=freqs,
        power=power,
        peak_bin=k,
        resolution_bpm=60.0 * fs / x.size,
    )


def estimate_bpm_welch(
    x, fs: float, band=BAND_HZ, seg_sec: float = 4.0, overlap: float = 0.5
) -> SpectrumResult:
    """Welch comparison arm: lower-variance estimate, worse resolution.

    Splitting a 15 s window into 4 s segments trades the 4 BPM resolution of the
    full window for averaging over ~7 periodograms. Worth measuring against the
    single-FFT path rather than assuming which wins.
    """
    x = np.asarray(x, dtype=float)
    nperseg = min(x.size, max(16, int(round(seg_sec * fs))))
    noverlap = int(nperseg * overlap)
    freqs, power = welch(
        x - x.mean(), fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        nfft=int(2 ** np.ceil(np.log2(nperseg * 8))), detrend=False,
    )
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    band_idx = np.flatnonzero(in_band)
    k = int(band_idx[np.argmax(power[in_band])])
    mag_db = 10.0 * np.log10(power + 1e-20)
    f_peak = float((k + _parabolic_peak(mag_db, k)) * (freqs[1] - freqs[0]))
    f_peak = float(np.clip(f_peak, band[0], band[1]))
    return SpectrumResult(
        bpm=f_peak * 60.0,
        f_peak=f_peak,
        snr=snr_db(freqs, power, f_peak),
        freqs=freqs,
        power=power,
        peak_bin=k,
        resolution_bpm=60.0 * fs / nperseg,
    )


def bpm_resolution(window_sec: float) -> float:
    """60/T — the number to quote when asked about resolution."""
    return 60.0 / window_sec
