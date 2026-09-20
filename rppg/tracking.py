"""Heart-rate tracking as a global optimisation, not a sequence of guesses.

Every stage up to here treats each window independently: take a spectrum, pick a
peak, emit a number. That throws away the single strongest piece of prior
knowledge available — **a heart rate cannot jump**. A resting HR moves by a few
BPM per second; it never doubles between one window and the next.

An independent peak-picker has no way to express that, so it must resolve
octave ambiguity (is the tallest line the fundamental or its second harmonic?)
using one window's evidence alone. It gets it wrong whenever the harmonic is
briefly taller, and a median filter cannot repair it because the median is a
*local* fix applied after the damage.

Dynamic programming over the whole spectrogram resolves it instead: choose the
frequency *path* maximising total spectral evidence minus a penalty for moving.
A single window whose harmonic outranks its fundamental is outvoted by its
neighbours, because switching to the harmonic and back costs more than the
evidence gained. This is the Viterbi algorithm on a time-frequency lattice, and
it is the same argument that makes pitch tracking work in speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import BAND_HZ
from .spectral import FUNDAMENTAL_HZ, harmonic_score, periodogram


@dataclass
class Track:
    times: np.ndarray  #: window centre times (s)
    bpm: np.ndarray  #: tracked heart rate
    grid_bpm: np.ndarray  #: candidate grid the path was chosen from
    scores: np.ndarray  #: (n_windows, n_candidates) evidence, dB, row-normalised
    path: np.ndarray  #: index into grid_bpm per window


def score_gram(
    signal,
    fs: float,
    window_sec: float = 15.0,
    hop_sec: float = 1.0,
    grid_step_bpm: float = 0.5,
    fundamental=FUNDAMENTAL_HZ,
    zero_pad: int = 8,
    band=BAND_HZ,
):
    """Build the (window x candidate-BPM) evidence matrix a track runs over.

    Rows are normalised to their own maximum: absolute spectral power varies
    hugely between windows (a subject leaning toward the camera changes it by
    more than a heartbeat does), and without normalisation a few bright windows
    would dictate the whole path.
    """
    signal = np.asarray(signal, dtype=float)
    win = int(round(window_sec * fs))
    hop = max(1, int(round(hop_sec * fs)))
    if signal.size < win:
        raise ValueError(f"signal shorter ({signal.size}) than the window ({win})")

    grid_bpm = np.arange(fundamental[0] * 60.0, fundamental[1] * 60.0 + 1e-9, grid_step_bpm)
    grid_hz = grid_bpm / 60.0

    times, rows = [], []
    for start in range(0, signal.size - win + 1, hop):
        seg = signal[start : start + win]
        freqs, power = periodogram(seg, fs, zero_pad=zero_pad)
        cand, score = harmonic_score(freqs, power, fundamental=fundamental)
        # Resample the harmonic score onto the fixed BPM grid so every window
        # shares one candidate axis.
        row = np.interp(grid_hz, cand, score, left=0.0, right=0.0)
        row_db = 10.0 * np.log10(row + 1e-20)
        rows.append(row_db - row_db.max())
        times.append((start + win / 2.0) / fs)

    return np.asarray(times), grid_bpm, np.asarray(rows)


def viterbi_track(
    times,
    grid_bpm,
    scores,
    max_slew_bpm_per_sec: float = 12.0,
    penalty_db_per_bpm: float = 0.15,
) -> np.ndarray:
    """Best-scoring frequency path under a physiological slew limit.

    ``max_slew_bpm_per_sec`` is a hard gate: transitions faster than this are
    forbidden outright, which is what makes an octave jump impossible rather
    than merely expensive. ``penalty_db_per_bpm`` prices the movement that
    remains allowed, so the path prefers to stay put unless the evidence for
    moving is real.

    Returns indices into ``grid_bpm``, one per window.
    """
    scores = np.asarray(scores, dtype=float)
    grid_bpm = np.asarray(grid_bpm, dtype=float)
    times = np.asarray(times, dtype=float)
    n_win, n_cand = scores.shape
    if n_win == 0:
        return np.empty(0, dtype=int)
    if n_win == 1:
        return np.array([int(np.argmax(scores[0]))])

    # |bpm_i - bpm_j| for every candidate pair, computed once.
    delta = np.abs(grid_bpm[:, None] - grid_bpm[None, :])

    total = scores[0].copy()
    back = np.zeros((n_win, n_cand), dtype=np.int32)

    for i in range(1, n_win):
        dt = max(times[i] - times[i - 1], 1e-6)
        allowed = delta <= max_slew_bpm_per_sec * dt
        # transition[j, k] = cost of arriving at k from j
        transition = np.where(allowed, -penalty_db_per_bpm * delta, -np.inf)
        cand_total = total[:, None] + transition  # (from, to)
        best_from = np.argmax(cand_total, axis=0)
        best_val = cand_total[best_from, np.arange(n_cand)]
        # A candidate unreachable from anywhere restarts from its own evidence.
        stuck = ~np.isfinite(best_val)
        best_val = np.where(stuck, scores[i] - 1e3, best_val + scores[i])
        back[i] = best_from
        total = best_val

    path = np.zeros(n_win, dtype=int)
    path[-1] = int(np.argmax(total))
    for i in range(n_win - 1, 0, -1):
        path[i - 1] = int(back[i, path[i]])
    return path


def track_bpm(
    signal,
    fs: float,
    window_sec: float = 15.0,
    hop_sec: float = 1.0,
    max_slew_bpm_per_sec: float = 12.0,
    penalty_db_per_bpm: float = 0.15,
    **kw,
) -> Track:
    """Spectrogram -> harmonic evidence -> best continuous path -> BPM."""
    times, grid_bpm, scores = score_gram(
        signal, fs, window_sec=window_sec, hop_sec=hop_sec, **kw
    )
    path = viterbi_track(
        times,
        grid_bpm,
        scores,
        max_slew_bpm_per_sec=max_slew_bpm_per_sec,
        penalty_db_per_bpm=penalty_db_per_bpm,
    )
    return Track(
        times=times, bpm=grid_bpm[path], grid_bpm=grid_bpm, scores=scores, path=path
    )
