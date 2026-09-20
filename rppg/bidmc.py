"""Ground-truth validation against contact PPG with ECG-derived heart rate.

    python -m rppg.bidmc --download
    python -m rppg.bidmc

BIDMC (Pimentel et al., 2017) is 53 ICU recordings: photoplethysmogram at
125 Hz alongside a heart rate derived from simultaneous ECG at 1 Hz. Decimating
the PPG to 30 Hz puts it on the same sampling grid a webcam gives us, so this
validates every stage downstream of the ROI -- detrending, bandpass, windowing,
peak selection, sub-bin interpolation, the SNR gate -- against a reference that
does not come from me.

What it does NOT validate: GREEN, ICA, CHROM and POS. Those are projections
from three colour channels to one, and contact PPG has no colour channels. That
half needs face video with a reference, which is what UBFC-rPPG is for.

Being explicit about the split matters. Every previous number in this project
came from synthetic data whose difficulty I chose, which makes it evidence
about my imagination rather than about the algorithm.

The headline result, on ten records: MAE 0.74 BPM where the pulse is clean, and
an operating curve showing accuracy as a function of measured SNR. That curve
is what sets the gate; before it, the -7 dB default was a guess.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .preprocess import bandpass, detrend
from .spectral import estimate_bpm

#: Ten records spanning clean and difficult pulses. bidmc40 is deliberately
#: included: it is the one that fails, and a benchmark that quietly excludes
#: its failures is not a benchmark.
DEFAULT_RECORDS = tuple(f"bidmc{i:02d}" for i in (1, 2, 3, 5, 8, 13, 21, 34, 40, 47))
DEFAULT_ROOT = Path("data/bidmc")
FS_TARGET = 30.0


def download(records=DEFAULT_RECORDS, root: Path = DEFAULT_ROOT):
    """Fetch records from PhysioNet. Open access, no request form."""
    import wfdb

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    wanted = []
    for rec in records:
        wanted += [rec, rec + "n"]
    wfdb.dl_database("bidmc", str(root), records=wanted)
    return root


def load(record: str, root: Path = DEFAULT_ROOT):
    """(ppg, fs, hr_reference_at_1hz). Signal names carry a trailing comma."""
    import wfdb

    root = Path(root)
    sig = wfdb.rdrecord(str(root / record))
    num = wfdb.rdrecord(str(root / (record + "n")))
    names = [s.strip().rstrip(",") for s in sig.sig_name]
    nnames = [s.strip().rstrip(",") for s in num.sig_name]
    return (
        sig.p_signal[:, names.index("PLETH")],
        float(sig.fs),
        num.p_signal[:, nnames.index("HR")],
    )


def to_webcam_rate(x, fs: float, fs_target: float = FS_TARGET):
    """125 Hz -> 30 Hz, anti-aliased.

    `decimate` rather than slicing: the PPG carries real content above 15 Hz,
    and plain subsampling folds it straight into the cardiac band, which would
    manufacture exactly the kind of spurious peak this project exists to avoid.
    """
    factor = 5
    y = decimate(np.asarray(x, float), factor, ftype="fir", zero_phase=True)
    t_in = np.arange(len(y)) / (fs / factor)
    t_out = np.arange(0.0, t_in[-1], 1.0 / fs_target)
    return np.interp(t_out, t_in, y), t_out


def windows(x, t, hr_ref, fs=FS_TARGET, window_sec=15.0, hop_sec=1.0, selector="harmonic"):
    """Yield (measured_snr, absolute_error) for every window with a reference."""
    n, step = int(window_sec * fs), max(1, int(hop_sec * fs))
    for s in range(0, len(x) - n + 1, step):
        est = estimate_bpm(
            bandpass(detrend(x[s : s + n], fs=fs), fs), fs, selector=selector
        )
        i = int(round(t[s + n - 1]))
        if i < len(hr_ref) and np.isfinite(hr_ref[i]):
            yield est.snr, est.bpm - hr_ref[i]


def evaluate(record, root=DEFAULT_ROOT, noise: float = 0.0, seed: int = 0, **kw):
    """Per-record error summary, optionally after degrading the SNR."""
    ppg, fs, hr = load(record, root)
    x, t = to_webcam_rate(ppg, fs)
    if noise:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, noise * np.std(x), len(x))
    pairs = list(windows(x, t, hr, **kw))
    if not pairs:
        return None
    snr = np.array([p[0] for p in pairs])
    err = np.array([p[1] for p in pairs])
    return dict(
        record=record,
        n=len(err),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        rmse=float(np.sqrt(np.mean(err**2))),
        within3=float(100 * np.mean(np.abs(err) <= 3)),
        snr=float(np.mean(snr)),
    )


def operating_curve(records=DEFAULT_RECORDS, root=DEFAULT_ROOT,
                    noise_levels=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0), seed=0):
    """Accuracy as a function of measured SNR, pooled over records and noise.

    Sweeping added noise is what makes the curve cover the low-SNR regime a
    webcam actually operates in; the clean recordings alone only populate the
    right-hand end, where everything works and nothing is learned.
    """
    rng = np.random.default_rng(seed)
    out = []
    for rec in records:
        ppg, fs, hr = load(rec, root)
        base, t = to_webcam_rate(ppg, fs)
        for nz in noise_levels:
            x = base + rng.normal(0, nz * np.std(base), len(base)) if nz else base
            out.extend(windows(x, t, hr, hop_sec=2.0))
    arr = np.asarray(out)
    return arr[:, 0], np.abs(arr[:, 1])


def gate_table(snr, abs_err, gates=(-7, -5, -4, -3, -2, -1, 0, 2)):
    """What each SNR threshold buys and costs. This is how the default was set."""
    rows = []
    for g in gates:
        keep = snr >= g
        if keep.sum() == 0:
            continue
        rows.append(dict(
            gate=g,
            kept=float(100 * keep.mean()),
            mae=float(abs_err[keep].mean()),
            within3=float(100 * np.mean(abs_err[keep] <= 3)),
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--curve", action="store_true", help="SNR operating curve (slow)")
    args = ap.parse_args(argv)

    if args.download:
        print(f"downloading {len(DEFAULT_RECORDS)} records to {args.root} ...")
        download(root=Path(args.root))
        print("done")
        return 0

    print("BIDMC contact PPG decimated to 30 fps, 15 s windows")
    print(f"{'record':10s}{'n':>6}{'MAE':>8}{'bias':>8}{'RMSE':>8}{'<=3BPM':>9}{'SNR':>8}")
    print("-" * 55)
    rows = []
    for rec in DEFAULT_RECORDS:
        r = evaluate(rec, Path(args.root))
        if r is None:
            continue
        rows.append(r)
        print("%-10s%6d%8.2f%+8.2f%8.2f%8.1f%%%+8.1f"
              % (r["record"], r["n"], r["mae"], r["bias"], r["rmse"],
                 r["within3"], r["snr"]))
    if rows:
        print("-" * 55)
        clean = [r for r in rows if r["snr"] > 8]
        print("%-10s%6d%8.2f%+8.2f%8.2f%8.1f%%"
              % ("mean", sum(r["n"] for r in rows),
                 np.mean([r["mae"] for r in rows]),
                 np.mean([r["bias"] for r in rows]),
                 np.mean([r["rmse"] for r in rows]),
                 np.mean([r["within3"] for r in rows])))
        if clean:
            print("%-10s%6s%8.2f   (records with mean SNR above +8 dB)"
                  % ("clean", "", np.mean([r["mae"] for r in clean])))

    if args.curve:
        print("\nSNR operating curve")
        snr, err = operating_curve(root=Path(args.root))
        print(f"{len(snr)} windows")
        for row in gate_table(snr, err):
            print("  gate %+3d dB -> keeps %5.1f%%, MAE %6.2f BPM, %5.1f%% within 3"
                  % (row["gate"], row["kept"], row["mae"], row["within3"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
