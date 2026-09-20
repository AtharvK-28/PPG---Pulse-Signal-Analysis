"""Post-mortem on a recorded session against a known reference BPM.

    python -m rppg.session data/sessions/session_*.csv

The app can only ever show you *an* answer. This asks the prior question:
is the pulse in the recording at all?

The decisive test is the projection sweep. Every method in this project is one
choice of a projection w applied to (R, G, B); GREEN fixes it, ICA learns it,
CHROM and POS derive it from a reflection model. So sweeping w over the whole
simplex and taking the best SNR at the reference frequency upper-bounds what
*any* of them could achieve. If that bound is below 0 dB, the pulse is not in
the data and no estimator can recover it — which is worth knowing before
spending another evening tuning peak selection.

The second test separates a pulse from an intensity artifact. Haemoglobin
absorbs green far more than red, so a real pulse has a green/red relative
amplitude around 1.5-2.5x. A shared illumination or motion term has a ratio
near 1.0 and near-unity correlation between all three channels. Cross-method
agreement cannot see this: a common-mode artifact survives every projection,
so all four arms agree on it confidently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .preprocess import bandpass, detrend
from .resample import resample_uniform
from .spectral import periodogram

FS = 30.0


def load_session(path):
    """Read a session CSV: `# key=value` headers, then t,R,G,B."""
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition("=")
                meta[key.strip()] = value.strip()
            elif line and not line.lower().startswith("t,"):
                rows.append([float(x) for x in line.split(",")])
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{path}: expected columns t,R,G,B")
    return meta, arr[:, 0], arr[:, 1:4]


def band_snr(sig, fs, f0_bpm, half_bw=3.0, band=(40.0, 200.0)):
    """de Haan SNR at a *given* frequency and its second harmonic.

    Not at the peak — the whole point is to ask how much energy sits where the
    pulse is known to be, rather than where the spectrum happens to be tallest.
    """
    b = bandpass(detrend(np.asarray(sig, float), fs=fs), fs)
    freqs, power = periodogram(b, fs, zero_pad=8)
    bpm = freqs * 60.0
    inside = (bpm >= band[0]) & (bpm <= band[1])
    target = (np.abs(bpm - f0_bpm) < half_bw) | (np.abs(bpm - 2 * f0_bpm) < 2 * half_bw)
    s = power[inside & target].sum()
    n = power[inside & ~target].sum()
    return 10.0 * np.log10(s / n) if n > 0 else float("nan")


def best_projection(rgb, fs, f0_bpm, steps=21):
    """Upper bound over all linear projections of (R, G, B)."""
    best_w, best = None, -np.inf
    for a in np.linspace(-1.0, 1.0, steps):
        for b in np.linspace(-1.0, 1.0, steps):
            w = np.array([a, b, 1.0 - abs(a) - abs(b)])
            norm = np.linalg.norm(w)
            if norm < 1e-6:
                continue
            w = w / norm
            score = band_snr(rgb @ w, fs, f0_bpm)
            if np.isfinite(score) and score > best:
                best, best_w = score, w
    return best, best_w


def grid_quality(t):
    dt = np.diff(np.asarray(t, float)) * 1000.0
    return dict(
        fps=len(t) / (t[-1] - t[0]),
        median=float(np.median(dt)),
        p99=float(np.percentile(dt, 99)),
        worst=float(dt.max()),
        late=int((dt > 2 * np.median(dt)).sum()),
        n=len(dt),
    )


def channel_character(rgb, fs):
    """Green/red relative amplitude and inter-channel correlation, in band."""
    ac, rel = {}, {}
    for i, c in enumerate("RGB"):
        b = bandpass(detrend(rgb[:, i], fs=fs), fs)
        ac[c] = b
        rel[c] = b.std() / rgb[:, i].mean()
    corr = np.corrcoef(np.array([ac[c] for c in "RGB"]))
    return rel, corr


def report(path, reference_bpm=None, fs=FS):
    meta, t, rgb = load_session(path)
    ref = reference_bpm if reference_bpm is not None else float(
        meta.get("reference_bpm", "nan")
    )
    lines = [f"session: {Path(path).name}", "=" * 70]

    g = grid_quality(t)
    lines.append(
        "sampling grid   %.1f fps, median %.1f ms, p99 %.0f ms, worst %.0f ms"
        % (g["fps"], g["median"], g["p99"], g["worst"])
    )
    lines.append(
        "                %d of %d intervals more than 2x the median"
        % (g["late"], g["n"])
    )
    if g["p99"] > 4 * g["median"]:
        lines.append(
            "  -> the loop is stalling. Spline resampling fills each hole with"
        )
        lines.append(
            "     invented smooth data, which removes energy at the pulse rate."
        )

    tu, X = resample_uniform(t, rgb, fs)
    rel, corr = channel_character(X, fs)
    lines += ["", "in-band AC/DC per channel (a real pulse is ~0.3-1 % in green)"]
    for c in "RGB":
        lines.append("  %s  %.3f %%" % (c, 100 * rel[c]))
    gr = rel["G"] / rel["R"] if rel["R"] else float("nan")
    lines.append("  green/red relative amplitude %.2f  (pulse ~1.5-2.5, artifact ~1.0)" % gr)
    lines.append(
        "  inter-channel correlation  R-G %+.3f  R-B %+.3f  G-B %+.3f"
        % (corr[0, 1], corr[0, 2], corr[1, 2])
    )
    if gr < 1.3 and min(corr[0, 1], corr[0, 2], corr[1, 2]) > 0.9:
        lines.append(
            "  -> flat ratio with near-unity correlation: a single common"
        )
        lines.append(
            "     intensity term dominates. Every projection sees it, so all"
        )
        lines.append(
            "     four arms will agree on it: agreement is no evidence here."
        )

    if np.isfinite(ref):
        lines += ["", "is the pulse present at the reference rate?"]
        snr, w = best_projection(X, fs, ref)
        lines.append(
            "  best of all projections at %.0f BPM: %+.2f dB  w = [%+.2f %+.2f %+.2f]"
            % (ref, snr, w[0], w[1], w[2])
        )
        lines.append("  this bounds GREEN, ICA, CHROM and POS alike.")
        if snr < 0:
            lines.append(
                "  -> below 0 dB: less power at the pulse rate than in the rest"
            )
            lines.append(
                "     of the band. No choice of estimator recovers this; the"
            )
            lines.append("     fault is upstream, in capture or ROI.")
        else:
            lines.append("  -> the pulse is present; a wrong reading is the estimator's fault.")
    else:
        lines.append("")
        lines.append("no reference_bpm in the file and none given: pass --reference")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="session CSV written by the app")
    ap.add_argument("--reference", type=float, default=None, help="known BPM")
    ap.add_argument("--fs", type=float, default=FS, help="resampling rate")
    args = ap.parse_args(argv)
    print(report(args.path, args.reference, args.fs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
