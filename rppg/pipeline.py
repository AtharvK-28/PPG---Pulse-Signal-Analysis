"""Stage 6 — the sliding-window pipeline that ties every stage together.

    video -> ROI -> spatial mean -> resample -> detrend -> project
          -> bandpass -> window -> FFT -> peak -> BPM

One code path serves both the batch benchmark and the live app, so a number
shown on stage is produced by exactly the same DSP as a number in the results
table.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import BAND_HZ, METHODS
from .methods import PROJECTIONS
from .preprocess import bandpass, despike
from .resample import max_gap_in_span, resample_uniform
from .spectral import (
    SpectrumResult,
    estimate_bpm,
    estimate_bpm_welch,
    harmonic_score,
    periodogram,
)
from .spectral import FUNDAMENTAL_HZ
from .tracking import track_bpm, viterbi_track


@dataclass
class PipelineConfig:
    fs: float = 30.0
    window_sec: float = 15.0  #: 60/15 = 4 BPM raw resolution
    hop_sec: float = 1.0  #: one estimate per second
    band: tuple = BAND_HZ
    zero_pad: int = 8
    detrend_method: str = "smoothness"
    spectrum: str = "fft"  #: "fft" | "welch"
    median_len: int = 5  #: temporal median over the last N estimates
    #: Below this, reject rather than emit a number. Calibrated on 18,640 BIDMC
    #: windows with ECG-derived ground truth (see rppg.bidmc), not chosen by
    #: taste. Accuracy against measured SNR:
    #:
    #:     gate   kept    MAE    within 3 BPM
    #:     -7 dB  95.8%   9.12      73.2%
    #:     -5 dB  77.6%   5.12      84.3%
    #:     -3 dB  59.4%   1.80      95.1%
    #:      0 dB  45.5%   0.69      98.5%
    #:
    #: The old -7 dB default admitted almost everything and carried a 9 BPM
    #: mean error — it was displaying confident numbers from a regime where the
    #: expected error exceeds the width of the cardiac band. -3 dB keeps most
    #: of the windows worth keeping and is where accuracy turns over.
    snr_threshold: float = -3.0
    #: Reject a window whose sampling gap exceeds this fraction of its length.
    #: Measured, not guessed: with a 15 s window, a single gap costs no
    #: measurable accuracy up to ~3% of the window (2.22 vs 2.49 BPM baseline),
    #: turns marginal at 5% (4.30), and is clearly harmful at 6.7% (7.77).
    #: Expressing it as a fraction is what makes it correct at other window
    #: lengths — 0.5 s is 3% of a 15 s window but 10% of a 5 s one.
    max_gap_fraction: float = 0.05
    #: Absolute override in seconds; derived from the fraction when left None.
    max_gap_sec: float | None = None
    stall_sec: float = 3.0  #: any interval this long invalidates the buffer
    min_effective_fps: float = 10.0  #: below this, the capture rate itself is the problem
    selector: str = "harmonic"  #: "harmonic" | "peak" — how the pulse line is chosen
    #: Repair motion transients before projecting. Measured on synthetic data
    #: with a known 86 BPM pulse: three spikes in a 30 s window drop SNR from
    #: +25.7 to -1.6 dB, and repairing them restores +25.8. The projection
    #: bound — the best any arm could do — gains +15 to +18 dB. On a real
    #: recording with no transients it changed nothing, which is the point:
    #: it repairs what is there and leaves alone what is not.
    despike: bool = True
    track: bool = True  #: resolve octave errors with a continuity-constrained track
    track_slew_bpm_per_sec: float = 12.0
    track_penalty_db_per_bpm: float = 0.15
    track_history: int = 30  #: windows of context the live tracker reconsiders
    methods: tuple = METHODS

    def __post_init__(self):
        if self.max_gap_sec is None:
            self.max_gap_sec = self.max_gap_fraction * self.window_sec


@dataclass
class WindowEstimate:
    t_start: float
    t_end: float
    t_center: float
    method: str
    bpm: float  #: raw, this window only
    bpm_smoothed: float  #: median-filtered over recent accepted windows
    snr: float
    accepted: bool  #: False => a gate rejected it; display the rejection
    resolution_bpm: float
    max_gap: float = 0.0  #: longest sampling hole inside this window
    reject_reason: str = ""
    spectrum: SpectrumResult | None = field(default=None, repr=False, compare=False)


def project_and_filter(rgb, fs: float, method: str, config: PipelineConfig | None = None):
    """Run one arm on a window of RGB means and return the filtered pulse signal."""
    cfg = config or PipelineConfig(fs=fs)
    if method not in PROJECTIONS:
        raise ValueError(f"unknown method {method!r}; have {sorted(PROJECTIONS)}")
    x = np.asarray(rgb, dtype=float)
    # Before projection, not after: a transient is common-mode across R, G and
    # B, so it is still identifiable as one event here. Once projected it is
    # just a spike in a scalar trace, indistinguishable from a fast heartbeat.
    if cfg.despike:
        x = despike(x)
    raw = PROJECTIONS[method](x, fs, detrend_method=cfg.detrend_method, band=cfg.band)
    # Common final bandpass so every arm is judged over the same passband,
    # whatever filtering it does internally.
    return bandpass(raw, fs, band=cfg.band)


def gate(res: SpectrumResult, max_gap: float, config: PipelineConfig) -> tuple[bool, str]:
    """Should this window be reported? Returns (accepted, reason-if-not).

    Two independent failure modes. A low SNR means no narrowband peak stood out.
    A sampling gap means the window contains interpolated fiction — and that one
    is worth catching separately, because a spline overshoot is *narrowband* and
    scores a perfectly respectable SNR while being entirely an artefact.
    """
    if max_gap > config.max_gap_sec:
        return False, (
            f"{max_gap * 1000:.0f} ms gap "
            f"({max_gap / config.window_sec:.0%} of the window)"
        )
    if not np.isfinite(res.snr):
        return False, "no spectrum"
    if res.snr < config.snr_threshold:
        return False, f"SNR {res.snr:+.1f} dB below gate"
    return True, ""


def agreement_spread(estimates) -> float:
    """Max-min BPM across the arms that were accepted.

    A confidence signal that needs no ground truth and, unlike SNR, is not
    fooled by a confident wrong answer: for all four projections to agree, the
    thing they agree on has to survive four different ways of suppressing the
    specular term. A wide spread means at most one of them can be right.
    """
    vals = [
        e.bpm for e in (estimates.values() if hasattr(estimates, "values") else estimates)
        if getattr(e, "accepted", True) and np.isfinite(e.bpm)
    ]
    return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")


def spectrum_of(signal, fs: float, config: PipelineConfig) -> SpectrumResult:
    if config.spectrum == "welch":
        return estimate_bpm_welch(signal, fs, band=config.band)
    return estimate_bpm(
        signal,
        fs,
        band=config.band,
        zero_pad=config.zero_pad,
        selector=config.selector,
    )


def estimate_window(rgb, fs: float, method: str, config: PipelineConfig | None = None):
    """One window, one arm -> SpectrumResult."""
    cfg = config or PipelineConfig(fs=fs)
    return spectrum_of(project_and_filter(rgb, fs, method, cfg), fs, cfg)


def analyse_signal(
    timestamps,
    rgb,
    config: PipelineConfig | None = None,
    already_uniform: bool = False,
) -> pd.DataFrame:
    """Batch: slide over a whole clip's RGB means, every arm, one row per window.

    ``timestamps`` are the per-frame capture times. They are resampled onto a
    uniform grid first even for dataset video where frames are uniform by
    construction, so the live and batch paths stay identical.
    """
    cfg = config or PipelineConfig()
    t = np.asarray(timestamps, dtype=float)
    rgb = np.asarray(rgb, dtype=float)

    if already_uniform:
        t_u, rgb_u = t, rgb
    else:
        t_u, rgb_u = resample_uniform(t, rgb, fs=cfg.fs)

    win = int(round(cfg.window_sec * cfg.fs))
    hop = max(1, int(round(cfg.hop_sec * cfg.fs)))
    if rgb_u.shape[0] < win:
        raise ValueError(
            f"clip has {rgb_u.shape[0]} samples at {cfg.fs} Hz "
            f"({rgb_u.shape[0] / cfg.fs:.1f} s) — shorter than the "
            f"{cfg.window_sec:.1f} s window"
        )

    history = {m: deque(maxlen=cfg.median_len) for m in cfg.methods}
    rows = []
    for start in range(0, rgb_u.shape[0] - win + 1, hop):
        seg = rgb_u[start : start + win]
        t0, t1 = float(t_u[start]), float(t_u[start + win - 1])
        # Gaps are a property of the original capture, so they have to be
        # measured before resampling smooths the evidence away.
        gap = max_gap_in_span(t, t0, t1)
        for method in cfg.methods:
            res = estimate_window(seg, cfg.fs, method, cfg)
            accepted, reason = gate(res, gap, cfg)
            if accepted:
                history[method].append(res.bpm)
            smoothed = float(np.median(history[method])) if history[method] else np.nan
            rows.append(
                dict(
                    t_start=t0,
                    t_end=t1,
                    t_center=0.5 * (t0 + t1),
                    method=method,
                    bpm=res.bpm,
                    bpm_smoothed=smoothed,
                    snr=res.snr,
                    accepted=accepted,
                    reject_reason=reason,
                    max_gap=gap,
                    resolution_bpm=res.resolution_bpm,
                )
            )
    df = pd.DataFrame(rows)

    if cfg.track and not df.empty:
        df["bpm_tracked"] = np.nan
        for method in cfg.methods:
            # Project the whole clip once. CHROM and POS do their own internal
            # windowing, so a single pass over the full signal is both correct
            # and what gives the tracker a continuous spectrogram to work on.
            try:
                whole = project_and_filter(rgb_u, cfg.fs, method, cfg)
                track = track_bpm(
                    whole,
                    cfg.fs,
                    window_sec=cfg.window_sec,
                    hop_sec=cfg.hop_sec,
                    zero_pad=cfg.zero_pad,
                    max_slew_bpm_per_sec=cfg.track_slew_bpm_per_sec,
                    penalty_db_per_bpm=cfg.track_penalty_db_per_bpm,
                )
            except ValueError:
                continue
            sel = df.method == method
            # Track times are relative to the resampled grid's origin.
            df.loc[sel, "bpm_tracked"] = np.interp(
                df.loc[sel, "t_center"], track.times + float(t_u[0]), track.bpm
            )
    return df


class SlidingWindowEstimator:
    """Live path: push timestamped RGB means, pull BPM per arm once per hop.

    Keeps a rolling buffer of raw (non-uniform) samples, resamples the buffer on
    every hop, and applies the same SNR gate and median filter as the batch run.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        span = self.cfg.window_sec
        self._t: deque[float] = deque()
        self._rgb: deque[tuple] = deque()
        self._span = span
        self._history = {m: deque(maxlen=self.cfg.median_len) for m in self.cfg.methods}
        self._scores = {
            m: deque(maxlen=self.cfg.track_history) for m in self.cfg.methods
        }
        self._grid_bpm = np.arange(
            FUNDAMENTAL_HZ[0] * 60.0, FUNDAMENTAL_HZ[1] * 60.0 + 1e-9, 0.5
        )
        self._last_emit: float | None = None
        self.latest: dict[str, WindowEstimate] = {}
        #: Dropouts seen this session, and the size of the most recent one.
        self.gaps_seen = 0
        self.last_gap = 0.0
        self._lost_from: float | None = None

    def mark_lost(self, t: float) -> None:
        """Report a frame in which the ROI could not be extracted.

        The caller knows whether the face was actually found; the estimator
        cannot tell that from timestamps alone. A long interval between samples
        means either "the face went away" or "we were too slow to process the
        frames in between", and those need opposite responses: the first
        invalidates the buffer, the second does not — the samples on either side
        are real, just sparse. Guessing from the interval alone treats a slow
        machine as a permanent dropout and never recovers.
        """
        if self._lost_from is None:
            self._lost_from = float(t)

    def push(self, t: float, rgb_triple) -> None:
        t = float(t)
        lost_for = (t - self._lost_from) if self._lost_from is not None else 0.0
        self._lost_from = None
        stalled = bool(self._t) and (t - self._t[-1]) > self.cfg.stall_sec

        if self._t and (lost_for > self.cfg.max_gap_sec or stalled):
            # A real hole. Splining across one manufactures a step, the step
            # rings through the bandpass, and amplitude normalisation then
            # squashes the real signal against it — one 2 s dropout poisons the
            # whole 15 s window.
            #
            # Discarding the pre-gap samples costs nothing we could have used:
            # no window spanning the hole was analysable anyway. It just means
            # we refill with contiguous data instead of carrying fiction on.
            self.gaps_seen += 1
            self.last_gap = max(lost_for, t - self._t[-1])
            self.reset(keep_history=False)

        self._t.append(t)
        self._rgb.append(tuple(float(v) for v in rgb_triple))
        # Drop everything older than one window; a small margin absorbs jitter.
        cutoff = self._t[-1] - self._span * 1.05
        while len(self._t) > 4 and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    def _push_score(self, method: str, seg) -> None:
        """Record this window's harmonic evidence for the online tracker."""
        if not self.cfg.track:
            return
        try:
            sig = project_and_filter(seg, self.cfg.fs, method, self.cfg)
            freqs, power = periodogram(sig, self.cfg.fs, zero_pad=self.cfg.zero_pad)
            cand, score = harmonic_score(freqs, power)
        except (ValueError, FloatingPointError):
            return
        grid_hz = self._grid_bpm / 60.0
        row = np.interp(grid_hz, cand, score, left=0.0, right=0.0)
        row_db = 10.0 * np.log10(row + 1e-20)
        self._scores[method].append(row_db - row_db.max())

    def _resolve(self, method: str, hist) -> float:
        """The reported BPM: a tracked path when available, else a median.

        The median filter suppresses isolated outliers but cannot repair an
        octave error, because doubling is not an outlier in the statistical
        sense — it is a confident, repeatable wrong answer. Only continuity
        across windows distinguishes it.
        """
        rows = self._scores.get(method)
        if self.cfg.track and rows and len(rows) >= 3:
            scores = np.asarray(rows)
            times = np.arange(len(rows)) * self.cfg.hop_sec
            path = viterbi_track(
                times,
                self._grid_bpm,
                scores,
                max_slew_bpm_per_sec=self.cfg.track_slew_bpm_per_sec,
                penalty_db_per_bpm=self.cfg.track_penalty_db_per_bpm,
            )
            return float(self._grid_bpm[path[-1]])
        return float(np.median(hist)) if hist else float("nan")

    def reset(self, keep_history: bool = True) -> None:
        """Drop the buffer. Also clears the median history unless told otherwise.

        The history is cleared by default on a dropout so the display cannot
        keep showing a pre-dropout BPM as though it were live — during recovery
        there is no estimate, and the UI should say so.
        """
        self._t.clear()
        self._rgb.clear()
        self._last_emit = None
        self._lost_from = None
        if not keep_history:
            for hist in self._history.values():
                hist.clear()
            for rows in self._scores.values():
                rows.clear()
            self.latest = {}

    @property
    def refill_remaining(self) -> float:
        """Seconds of contiguous data still needed before the next estimate."""
        return max(0.0, self._span - self.buffered_sec)

    @property
    def effective_fps(self) -> float:
        """Rate at which usable samples are actually arriving.

        Not the camera's rate — the rate the whole loop sustains. If this falls
        near the 8 Hz needed to satisfy Nyquist for a 4 Hz band edge, the
        bottleneck is throughput, and no amount of buffer management fixes it.
        """
        if len(self._t) < 2 or self._t[-1] <= self._t[0]:
            return float("nan")
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])

    @property
    def too_slow(self) -> bool:
        fps = self.effective_fps
        return bool(np.isfinite(fps)) and fps < self.cfg.min_effective_fps

    @property
    def buffered_sec(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    @property
    def ready(self) -> bool:
        return self.buffered_sec >= self._span

    def due(self) -> bool:
        """True when the buffer is full and a hop has elapsed since the last emit."""
        if not self.ready:
            return False
        return self._last_emit is None or (self._t[-1] - self._last_emit) >= self.cfg.hop_sec

    def estimate(self) -> dict[str, WindowEstimate]:
        """Run every arm on the current buffer. Call when ``due()``."""
        if not self.ready:
            raise RuntimeError(
                f"buffer holds {self.buffered_sec:.1f} s, need {self._span:.1f} s"
            )
        t = np.fromiter(self._t, dtype=float)
        rgb = np.array(self._rgb, dtype=float)
        t_u, rgb_u = resample_uniform(t, rgb, fs=self.cfg.fs)

        win = int(round(self.cfg.window_sec * self.cfg.fs))
        if rgb_u.shape[0] < win:  # jitter left the buffer a sample or two short
            win = rgb_u.shape[0]
        seg = rgb_u[-win:]

        gap = max_gap_in_span(t, float(t_u[-win]), float(t_u[-1]))

        out: dict[str, WindowEstimate] = {}
        for method in self.cfg.methods:
            res = estimate_window(seg, self.cfg.fs, method, self.cfg)
            accepted, reason = gate(res, gap, self.cfg)
            if accepted:
                self._history[method].append(res.bpm)
                self._push_score(method, seg)
            hist = self._history[method]
            out[method] = WindowEstimate(
                t_start=float(t_u[-win]),
                t_end=float(t_u[-1]),
                t_center=float(0.5 * (t_u[-win] + t_u[-1])),
                method=method,
                bpm=res.bpm,
                bpm_smoothed=self._resolve(method, hist),
                snr=res.snr,
                accepted=accepted,
                reject_reason=reason,
                max_gap=gap,
                resolution_bpm=res.resolution_bpm,
                spectrum=res,
            )
        self._last_emit = self._t[-1]
        self.latest = out
        return out

    def to_records(self) -> list[dict]:  # pragma: no cover - convenience
        return [
            {k: v for k, v in asdict(e).items() if k != "spectrum"}
            for e in self.latest.values()
        ]
