"""Analyse a recording made by `rppg.record`, with no real-time budget.

    python -m rppg.offline data/recordings/subject01
    python -m rppg.offline data/recordings/subject01 --reference 86 --sweep

Two stages, deliberately separated:

**Extract** runs the ROI over every frame and writes the (t, R, G, B) means to
a CSV. This is the expensive stage — MediaPipe is ~18.5 ms/frame — and it is
run once. Nothing here is throttled or skipped; the landmark detector sees
every frame, which the live path cannot afford.

**Analyse** reads that CSV and runs the DSP. It is fast, so parameter sweeps
are cheap: window length, selector, detrend method and projection can all be
compared on *identical* data, which is the only way a comparison between them
means anything.

The separation matters beyond speed. In the live path, changing the window
length changes which frames were captured, so two settings are never compared
on the same data. Here they are.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .preprocess import bandpass, detrend
from .record import load_timestamps
from .resample import find_gaps, frame_interval_stats, resample_uniform
from .spectral import estimate_bpm

DEFAULT_FS = 30.0


def extract(recording, backend: str = "mediapipe", roi_kw=None, progress=None):
    """Every frame's ROI mean, using the recorded timestamps. Cached to CSV."""
    import cv2

    from .capture import Frame
    from .roi import make_roi

    recording = Path(recording)
    video = recording.with_suffix(".avi")
    if not video.exists():
        raise FileNotFoundError("%s not found" % video)
    timestamps = load_timestamps(recording.parent / (recording.name + "_timestamps.csv"))

    # redetect_every=1: the live path amortises the landmark detector across
    # frames to buy back its budget. Offline there is no budget, so every frame
    # gets its own landmarks and the ROI never lags a moving face.
    roi = make_roi(backend, redetect_every=1, **(roi_kw or {}))
    cap = cv2.VideoCapture(str(video))
    rows, missed, i = [], 0, 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if i >= len(timestamps):
                # More frames than timestamps means the writer and the grabber
                # disagree; trusting the container's nominal rate here would
                # silently put a scale error on the whole frequency axis.
                break
            sample = roi(Frame(index=i, timestamp=float(timestamps[i]), image=image))
            if sample.ok and np.all(np.isfinite(sample.rgb)):
                rows.append((timestamps[i], *sample.rgb, sample.n_pixels))
            else:
                missed += 1
            i += 1
            if progress is not None and i % 60 == 0:
                progress(i, len(timestamps))
    finally:
        cap.release()
        roi.close()

    arr = np.asarray(rows, dtype=float)
    out = recording.parent / (recording.name + "_rgb.csv")
    with open(out, "w") as fh:
        fh.write("# frames=%d detected=%d missed=%d\n" % (i, len(rows), missed))
        fh.write("t,R,G,B,n_pixels\n")
        for row in arr:
            fh.write("%.9f,%.9f,%.9f,%.9f,%d\n" % (*row[:4], int(row[4])))
    return arr, dict(frames=i, detected=len(rows), missed=missed, path=out)


def regularise_timestamps(t, nominal_fps: float = DEFAULT_FS, tolerance: float = 0.03):
    """Reconstruct exposure instants when delivery was bursty but lossless.

    The project's founding rule is to timestamp every frame rather than assume
    uniform sampling, because dropped and delayed frames are real and an
    assumed grid puts a scale error on the whole frequency axis. That rule is
    right when the *sensor* misses frames. It is wrong when the sensor is
    clocked perfectly and only the *driver* delivers unevenly.

    Measured here: MSMF holds two frames and releases them together, giving
    16 ms / 50 ms alternating intervals around a 33 ms mean, with 67 % of
    intervals short. The sensor exposed those frames 33 ms apart. Resampling
    onto the delivery times warps time locally by tens of percent and smears
    the spectrum — the correction is worse than the disease.

    The physical argument is what makes this safe. A CMOS sensor exposes on a
    crystal-derived clock; its frame instants are uniform to within
    microseconds and it does not jitter. Every irregularity `perf_counter()`
    sees at `read()` was therefore added *after* exposure, by the driver, the
    USB stack or the OS scheduler. So the only question is whether any frames
    were genuinely lost: if none were, the exposure grid was uniform, and
    reconstructing it removes transport noise rather than inventing data.

    Measured on a 72 BPM clip, SNR before -> after reconstruction:

        uniform delivery          +42.2  ->  (not applied)
        alternating pairs         +42.2  ->  +42.2   harmless either way
        irregular bursts          +21.6  ->  +42.2   recovers 20.5 dB
        white jitter, sd 17 ms     -6.8  ->  +42.2   recovers 49 dB

    Strictly alternating delivery — the MSMF signature measured here — turns
    out to cost nothing, because the error lands at Nyquist and the bandpass
    removes it. The irregular cases are where this earns its place.

    Returns (timestamps, changed, reason).
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 10:
        return t, False, "too few frames to judge"

    span = t[-1] - t[0]
    if span <= 0:
        return t, False, "timestamps do not advance"

    expected = span * nominal_fps + 1.0
    shortfall = abs(len(t) - expected) / expected
    if shortfall > tolerance:
        return t, False, (
            "%d frames present against %.0f a uniform clock would have "
            "produced (%.1f%% off): frames were genuinely lost, so the gaps "
            "are real and the measured timestamps are kept"
            % (len(t), expected, 100 * shortfall)
        )

    # Self-consistent period from the data rather than the nominal rate: the
    # sensor's true clock is rarely exactly 30.000 fps, and using the nominal
    # value would reintroduce the scale error this project exists to avoid.
    period = span / (len(t) - 1)
    dt = np.diff(t) * 1000.0
    jitter = float(np.std(dt))
    if jitter < 0.05 * period * 1000.0:
        return t, False, "delivery is already even (interval sd %.1f ms)" % jitter
    return (
        t[0] + np.arange(len(t)) * period,
        True,
        "all %d of %.0f expected frames present but delivery jittered "
        "(interval sd %.1f ms); exposure reconstructed on a uniform %.2f ms "
        "grid (%.3f fps)"
        % (len(t), expected, jitter, period * 1000.0, 1.0 / period),
    )


def load_rgb(recording):
    """Read the cached extraction."""
    recording = Path(recording)
    path = recording.parent / (recording.name + "_rgb.csv")
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("t,"):
                continue
            rows.append([float(x) for x in line.split(",")])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:4], arr[:, 4]


def analyse(
    t, rgb, fs: float = DEFAULT_FS, window_sec: float = 30.0, hop_sec: float = 1.0,
    methods=("green", "ica", "chrom", "pos"), selector: str = "harmonic",
    despike: bool = True,
):
    """Sliding-window estimates over the whole recording, per method."""
    from .pipeline import PipelineConfig, project_and_filter

    cfg = PipelineConfig(
        fs=fs, window_sec=window_sec, hop_sec=hop_sec,
        selector=selector, despike=despike, methods=tuple(methods),
    )
    tu, xu = resample_uniform(t, rgb, fs)
    n, step = int(window_sec * fs), max(1, int(hop_sec * fs))
    out = {m: [] for m in methods}
    if len(tu) < n:
        return out, tu, xu
    for s in range(0, len(tu) - n + 1, step):
        seg = xu[s : s + n]
        for m in methods:
            sig = project_and_filter(seg, fs, m, cfg)
            est = estimate_bpm(sig, fs, selector=selector)
            out[m].append((float(tu[s + n - 1]), est.bpm, est.snr))
    return {m: np.asarray(v) for m, v in out.items()}, tu, xu


def summarise(results, reference=None):
    """Per-method BPM, SNR and — when a reference exists — error."""
    rows = []
    for method, arr in results.items():
        if len(arr) == 0:
            continue
        bpm, snr = arr[:, 1], arr[:, 2]
        row = dict(
            method=method, n=len(bpm), bpm_median=float(np.median(bpm)),
            bpm_iqr=float(np.percentile(bpm, 75) - np.percentile(bpm, 25)),
            snr_median=float(np.median(snr)),
        )
        if reference is not None and np.isfinite(reference):
            row["mae"] = float(np.mean(np.abs(bpm - reference)))
            row["bias"] = float(np.mean(bpm - reference))
            row["within3"] = float(100 * np.mean(np.abs(bpm - reference) <= 3))
        rows.append(row)
    return rows


def sweep(t, rgb, reference=None, fs=DEFAULT_FS,
          windows=(10.0, 15.0, 20.0, 30.0, 45.0), selectors=("harmonic", "peak")):
    """Compare settings on identical data.

    This is the comparison the live path cannot make: there, changing the
    window changes which frames were captured, so any difference confounds the
    setting with the recording.
    """
    rows = []
    for w in windows:
        for sel in selectors:
            res, _, _ = analyse(t, rgb, fs=fs, window_sec=w, selector=sel)
            for row in summarise(res, reference):
                row["window_sec"] = w
                row["selector"] = sel
                # 60/T, unchanged by zero-padding.
                row["resolution_bpm"] = 60.0 / w
                rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", help="path without extension, as given to rppg.record")
    ap.add_argument("--reference", type=float, default=None, help="known BPM")
    ap.add_argument("--fs", type=float, default=DEFAULT_FS)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--backend", default="mediapipe")
    ap.add_argument("--regions", default="forehead+cheeks",
                    choices=["forehead+cheeks", "forehead", "cheeks"])
    ap.add_argument("--re-extract", action="store_true", help="ignore the cached CSV")
    ap.add_argument("--sweep", action="store_true", help="compare windows and selectors")
    ap.add_argument("--raw-timestamps", action="store_true",
                    help="never reconstruct the exposure grid, even if delivery was bursty")
    args = ap.parse_args(argv)

    rec = Path(args.recording)
    cached = rec.parent / (rec.name + "_rgb.csv")
    if args.re_extract or not cached.exists():
        roi_kw = dict(
            forehead=args.regions != "cheeks",
            cheeks=args.regions != "forehead",
        )
        print("Extracting ROI means (every frame, no throttling)...")
        _, info = extract(
            rec, backend=args.backend, roi_kw=roi_kw,
            progress=lambda i, n: print("\r  %d / %d frames" % (i, n), end="", flush=True),
        )
        print("\n  %d frames, face found in %d, missed %d (%.1f%%)"
              % (info["frames"], info["detected"], info["missed"],
                 100 * info["missed"] / max(info["frames"], 1)))
    t, rgb, npix = load_rgb(rec)
    if len(t) < 10:
        print("Too few usable frames (%d)." % len(t))
        return 1

    if not args.raw_timestamps:
        t, changed, reason = regularise_timestamps(t, nominal_fps=args.fs)
        print("\nTimestamps: %s" % reason)
        if changed:
            print("  (use --raw-timestamps to analyse the delivery times instead)")

    stats = frame_interval_stats(t)
    gaps = find_gaps(t, max_gap=0.2)
    print("\nSampling grid: %s" % stats)
    print("  gaps over 200 ms: %d" % len(gaps))
    print("  ROI: %.0f px mean, %.0f px min" % (npix.mean(), npix.min()))
    print("  duration %.1f s, %.2f effective fps" % (t[-1] - t[0], len(t) / (t[-1] - t[0])))

    if args.sweep:
        rows = sweep(t, rgb, args.reference, fs=args.fs)
        print("\n%-8s %7s %7s %8s %8s %8s %8s"
              % ("method", "window", "sel", "res BPM", "median", "SNR", "MAE"))
        print("-" * 62)
        for r in sorted(rows, key=lambda r: (r["method"], r["window_sec"], r["selector"])):
            print("%-8s %7.0f %7s %8.2f %8.1f %+8.1f %8s"
                  % (r["method"], r["window_sec"], r["selector"][:4],
                     r["resolution_bpm"], r["bpm_median"], r["snr_median"],
                     ("%.2f" % r["mae"]) if "mae" in r else "-"))
        return 0

    results, _, _ = analyse(t, rgb, fs=args.fs, window_sec=args.window)
    print("\n%-8s %6s %9s %8s %8s %8s"
          % ("method", "n", "BPM med", "IQR", "SNR", "MAE"))
    print("-" * 52)
    for r in summarise(results, args.reference):
        print("%-8s %6d %9.1f %8.1f %+8.1f %8s"
              % (r["method"], r["n"], r["bpm_median"], r["bpm_iqr"], r["snr_median"],
                 ("%.2f" % r["mae"]) if "mae" in r else "-"))
    if args.reference is None:
        print("\nNo --reference given: these are estimates without ground truth.")
        print("Cross-method spread is the only check available, and it cannot")
        print("detect an artifact that survives every projection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
