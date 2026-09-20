"""ICA — Poh, McDuff & Picard (2010). Blind source separation.

Three observations (R, G, B), three assumed sources, no redundancy. FastICA
returns components in arbitrary order with arbitrary scale and sign, so the
spectral selector below is *mandatory*, not a convenience: nothing guarantees
"component 3 is the pulse". What the dashboard shows is "the component the
selector chose".

Known failure mode, worth stating before anyone asks: head motion enters
through the shared multiplicative illumination term I(t), which is a
correlated, nonlinear distortion of all three channels — not the independent
additive source ICA assumes. That is why it degrades under motion where the
model-based arms hold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from .. import BAND_HZ
from ..preprocess import detrend, normalize_zscore
from ..spectral import periodogram


@dataclass
class ICAResult:
    """Everything the dashboard's three-component view needs."""

    signal: np.ndarray  #: the selected component
    components: np.ndarray  #: (N, 3), all sources
    chosen: int  #: index of the selected component
    scores: np.ndarray  #: spectral concentration per component
    converged: bool  #: False if FastICA hit max_iter or raised
    note: str = ""
    stats: dict = field(default_factory=dict)


#: Fraction of FastICA calls that failed to converge, over the process
#: lifetime. Report this number — a failure rate is a result, not an
#: embarrassment.
CONVERGENCE_LOG = {"calls": 0, "failures": 0}


def _spectral_concentration(x: np.ndarray, fs: float, band) -> float:
    """How sharply peaked is this component inside the cardiac band?

    Ratio of power within +/-0.1 Hz of the strongest in-band peak to total
    in-band power. A pulse component is narrowband; a motion component is not.
    """
    if np.allclose(x, x[0]):
        return 0.0
    freqs, power = periodogram(x, fs, zero_pad=4)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        return 0.0
    f, p = freqs[in_band], power[in_band]
    total = p.sum()
    if total <= 0:
        return 0.0
    f_peak = f[np.argmax(p)]
    near = np.abs(f - f_peak) <= 0.1
    return float(p[near].sum() / total)


def ica_full(
    rgb,
    fs: float,
    band=BAND_HZ,
    detrend_method: str = "smoothness",
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 0,
    **kw,
) -> ICAResult:
    """Run FastICA on the 3 x N channel matrix and select the pulse component."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB means, got {rgb.shape}")

    x = normalize_zscore(detrend(rgb, fs, method=detrend_method))
    CONVERGENCE_LOG["calls"] += 1

    converged, note = True, ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            transformer = FastICA(
                n_components=3,
                whiten="unit-variance",
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
            )
            components = transformer.fit_transform(x)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False
            note = f"FastICA did not converge in {max_iter} iterations"
    except Exception as exc:  # singular window, all-constant ROI, etc.
        # Fall back to the detrended green channel rather than emitting nothing;
        # the caller sees converged=False and can prefer the previous estimate.
        converged = False
        note = f"FastICA failed: {type(exc).__name__}: {exc}"
        components = np.column_stack([x[:, 1], x[:, 0], x[:, 2]])

    if not converged:
        CONVERGENCE_LOG["failures"] += 1

    scores = np.array([_spectral_concentration(c, fs, band) for c in components.T])
    chosen = int(np.argmax(scores))
    return ICAResult(
        signal=components[:, chosen],
        components=components,
        chosen=chosen,
        scores=scores,
        converged=converged,
        note=note,
        stats=dict(CONVERGENCE_LOG),
    )


def ica(rgb, fs: float, **kw) -> np.ndarray:
    """Selected pulse component only — the shared projection signature."""
    return ica_full(rgb, fs, **kw).signal


def convergence_failure_rate() -> float:
    """Fraction of FastICA calls that failed so far. Quote it in the report."""
    calls = CONVERGENCE_LOG["calls"]
    return CONVERGENCE_LOG["failures"] / calls if calls else 0.0
