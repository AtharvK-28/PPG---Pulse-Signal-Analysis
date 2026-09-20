"""Stage 2 — non-uniform capture grid -> uniform grid.

Webcams do not deliver frames at a constant interval. Every FFT downstream
assumes uniform sampling, so an assumed-30-Hz axis over a true 28.4-Hz mean
capture rate biases every BPM estimate. Timestamp each frame, then cubic-spline
onto a uniform grid before any DSP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class JitterStats:
    """Frame-interval statistics — report these in the write-up."""

    n_frames: int
    mean_dt: float
    std_dt: float
    min_dt: float
    max_dt: float
    mean_fps: float
    dropped_estimate: int

    @property
    def max_gap(self) -> float:
        """Longest interval between consecutive usable samples, in seconds."""
        return self.max_dt

    def __str__(self) -> str:  # pragma: no cover - display only
        gap = f" | worst gap {self.max_dt * 1e3:.0f} ms" if self.max_dt > 0.2 else ""
        return (
            f"{self.n_frames} frames | mean {self.mean_fps:.2f} fps "
            f"(dt {self.mean_dt * 1e3:.2f} +/- {self.std_dt * 1e3:.2f} ms, "
            f"range {self.min_dt * 1e3:.2f}-{self.max_dt * 1e3:.2f} ms) | "
            f"~{self.dropped_estimate} dropped{gap}"
        )


def frame_interval_stats(timestamps) -> JitterStats:
    """Characterise capture jitter. Feeds the week-1 jitter histogram."""
    t = np.asarray(timestamps, dtype=float)
    if t.size < 2:
        raise ValueError("need at least 2 timestamps")
    dt = np.diff(t)
    mean_dt = float(dt.mean())
    # A gap of >1.5x the median interval is most simply explained by a drop.
    med = float(np.median(dt))
    dropped = int(np.sum(np.round(dt / med) - 1)) if med > 0 else 0
    return JitterStats(
        n_frames=int(t.size),
        mean_dt=mean_dt,
        std_dt=float(dt.std()),
        min_dt=float(dt.min()),
        max_dt=float(dt.max()),
        mean_fps=1.0 / mean_dt if mean_dt > 0 else float("nan"),
        dropped_estimate=max(dropped, 0),
    )


def find_gaps(timestamps, max_gap: float = 0.5):
    """Intervals longer than ``max_gap`` seconds, as (start, end) pairs.

    Frames are dropped whenever the face is lost, and the resampler then splines
    straight across the hole. Over one or two missing frames that is exactly
    what you want. Over a second of missing data it is fiction — a cubic through
    widely separated knots overshoots, and the excursion lands right in the
    cardiac band as a large transient shared by every projection. Windows
    spanning a gap this size should be rejected, not analysed.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.size < 2:
        return []
    dt = np.diff(t)
    return [(float(t[i]), float(t[i + 1])) for i in np.flatnonzero(dt > max_gap)]


def max_gap_in_span(timestamps, t0: float, t1: float) -> float:
    """Longest sampling interval overlapping the window [t0, t1]."""
    t = np.asarray(timestamps, dtype=float)
    if t.size < 2:
        return float("inf")
    dt = np.diff(t)
    # An interval counts if any part of it falls inside the window.
    overlaps = (t[1:] >= t0) & (t[:-1] <= t1)
    return float(dt[overlaps].max()) if np.any(overlaps) else 0.0


def resample_uniform(timestamps, signals, fs: float = 30.0):
    """Cubic-spline ``signals`` sampled at ``timestamps`` onto a uniform grid.

    Parameters
    ----------
    timestamps : (N,) seconds, strictly increasing after de-duplication.
    signals    : (N,) or (N, C) — e.g. the (R, G, B) spatial means.
    fs         : target sampling rate in Hz.

    Returns
    -------
    t_uniform : (M,) grid starting at ``timestamps[0]``, step ``1/fs``.
    y_uniform : (M,) or (M, C) interpolated signal.
    """
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(signals, dtype=float)
    if t.ndim != 1:
        raise ValueError("timestamps must be 1-D")
    if y.shape[0] != t.shape[0]:
        raise ValueError(f"length mismatch: {t.shape[0]} timestamps, {y.shape[0]} samples")
    if t.size < 4:
        raise ValueError("cubic spline needs at least 4 samples")

    # CubicSpline requires strictly increasing x; duplicate timestamps happen
    # when a capture backend reuses a clock tick.
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, y = t[keep], y[keep]

    duration = t[-1] - t[0]
    n_out = int(np.floor(duration * fs)) + 1
    t_uniform = t[0] + np.arange(n_out) / fs
    # Guard against floating-point overshoot past the last knot.
    t_uniform = t_uniform[t_uniform <= t[-1] + 1e-12]

    spline = CubicSpline(t, y, axis=0, extrapolate=False)
    y_uniform = spline(np.clip(t_uniform, t[0], t[-1]))
    return t_uniform, y_uniform
