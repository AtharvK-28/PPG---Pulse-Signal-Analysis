"""Validation metrics — the part that turns a demo into a project.

Report per method x per condition. MAE is the headline; RMSE exposes the
instability MAE hides; Pearson r answers "does it track HR or just sit near the
population mean?"; Bland-Altman is the correct plot for a method-comparison
study and belongs in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _clean(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, ref {ref.shape}")
    ok = np.isfinite(pred) & np.isfinite(ref)
    return pred[ok], ref[ok]


def mae(pred, ref) -> float:
    p, r = _clean(pred, ref)
    return float(np.mean(np.abs(p - r))) if p.size else float("nan")


def rmse(pred, ref) -> float:
    p, r = _clean(pred, ref)
    return float(np.sqrt(np.mean((p - r) ** 2))) if p.size else float("nan")


def mape(pred, ref) -> float:
    """Percent error — the fair comparison across resting and elevated HR."""
    p, r = _clean(pred, ref)
    nz = np.abs(r) > 1e-9
    return float(np.mean(np.abs((p[nz] - r[nz]) / r[nz])) * 100.0) if np.any(nz) else float("nan")


def pearson(pred, ref) -> float:
    p, r = _clean(pred, ref)
    if p.size < 3 or np.std(p) < 1e-12 or np.std(r) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(p, r).statistic)


@dataclass(frozen=True)
class BlandAltman:
    mean: np.ndarray
    diff: np.ndarray
    bias: float
    sd: float
    loa_lower: float
    loa_upper: float

    def summary(self) -> str:
        return f"bias {self.bias:+.2f} BPM, 95% LoA [{self.loa_lower:.2f}, {self.loa_upper:.2f}]"


def bland_altman(pred, ref) -> BlandAltman:
    """Bias and 95% limits of agreement (bias +/- 1.96 SD of the differences)."""
    p, r = _clean(pred, ref)
    diff = p - r
    mean = 0.5 * (p + r)
    bias = float(np.mean(diff)) if diff.size else float("nan")
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    return BlandAltman(mean, diff, bias, sd, bias - 1.96 * sd, bias + 1.96 * sd)


def summarize(pred, ref, snr=None) -> dict:
    """All headline metrics for one (method, condition) cell."""
    p, r = _clean(pred, ref)
    ba = bland_altman(p, r)
    out = {
        "n": int(p.size),
        "MAE": mae(p, r),
        "RMSE": rmse(p, r),
        "MAPE": mape(p, r),
        "r": pearson(p, r),
        "bias": ba.bias,
        "LoA_lo": ba.loa_lower,
        "LoA_hi": ba.loa_upper,
    }
    if snr is not None:
        s = np.asarray(snr, dtype=float)
        s = s[np.isfinite(s)]
        out["SNR_dB"] = float(np.mean(s)) if s.size else float("nan")
    return out


def results_table(
    df: pd.DataFrame,
    pred_col: str = "bpm_smoothed",
    ref_col: str = "bpm_ref",
    by=("method", "condition"),
) -> pd.DataFrame:
    """The 4x4 centrepiece: one row per (method, condition)."""
    by = [c for c in by if c in df.columns]
    rows = []
    for key, grp in df.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(summarize(grp[pred_col], grp[ref_col], grp.get("snr")))
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("MAE").reset_index(drop=True) if "MAE" in table else table


def plot_bland_altman(pred, ref, ax=None, title: str = ""):  # pragma: no cover - plotting
    import matplotlib.pyplot as plt

    ba = bland_altman(pred, ref)
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    ax.scatter(ba.mean, ba.diff, s=14, alpha=0.6)
    ax.axhline(ba.bias, color="C3", label=f"bias {ba.bias:+.2f}")
    ax.axhline(ba.loa_upper, color="C3", ls="--", label="95% LoA")
    ax.axhline(ba.loa_lower, color="C3", ls="--")
    ax.set_xlabel("mean of (rPPG, reference)  [BPM]")
    ax.set_ylabel("rPPG - reference  [BPM]")
    ax.set_title(title or "Bland-Altman")
    ax.legend(fontsize=8)
    return ax
