"""Command-line runner: video or webcam in, BPM table out.

    python -m rppg.run --source 0 --seconds 60          # live webcam
    python -m rppg.run --source clip.mp4                # recorded file
    python -m rppg.run --demo                           # synthetic, no camera

Week 1's deliverable is a CSV of timestamped RGB means. `--save-signal` writes
exactly that, so the capture step can be validated before any of the DSP is
trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import METHODS
from .pipeline import PipelineConfig, analyse_signal
from .resample import frame_interval_stats


def collect(source, backend: str, seconds: float | None, show: bool = False, quiet=False):
    """Pull timestamped RGB means from a frame source through an ROI extractor."""
    import cv2

    from .capture import open_source
    from .roi import draw_overlay, make_roi

    src = open_source(source)
    try:
        roi = make_roi(backend)
    except RuntimeError as exc:
        print(f"{exc}\nFalling back to the Haar backend.", file=sys.stderr)
        roi = make_roi("haar")

    ts, rgbs, lost = [], [], 0
    try:
        for frame in src:
            if seconds is not None and frame.timestamp > seconds:
                break
            s = roi(frame)
            if s.ok and np.all(np.isfinite(s.rgb)):
                ts.append(s.timestamp)
                rgbs.append(s.rgb)
            else:
                lost += 1
            if show:
                cv2.imshow("rppg", draw_overlay(frame.image, s))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if not quiet and frame.index % 60 == 0:
                print(
                    f"\r{frame.timestamp:6.1f} s  {len(ts)} usable frames  "
                    f"{lost} lost",
                    end="",
                    flush=True,
                )
    finally:
        src.close()
        roi.close()
        if show:
            cv2.destroyAllWindows()
    if not quiet:
        print()
    if lost:
        print(f"face not found in {lost} frames — dropped, not interpolated")
    return np.array(ts), np.array(rgbs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None, help="camera index or video path")
    ap.add_argument("--demo", action="store_true", help="synthetic clip, no camera needed")
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "haar", "fixed"])
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=15.0, help="window length in seconds")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--spectrum", default="fft", choices=["fft", "welch"])
    ap.add_argument("--detrend", default="smoothness", choices=["smoothness", "ma", "none"])
    ap.add_argument("--show", action="store_true", help="live ROI overlay window")
    ap.add_argument("--save-signal", type=Path, help="write timestamped RGB means to CSV")
    ap.add_argument("--save-results", type=Path, help="write per-window estimates to CSV")
    args = ap.parse_args(argv)

    if args.demo:
        from .synthetic import nonuniform_timestamps, synth_rgb

        print("synthetic clip: 72 BPM, 0.4% modulation, respiration + capture jitter")
        t_ns = nonuniform_timestamps(args.seconds or 45.0, args.fs, jitter_ms=3.0, rng=0)
        clip = synth_rgb(timestamps=t_ns, bpm=72.0, rng=0)
        ts, rgb = clip.t, clip.rgb
    elif args.source is not None:
        ts, rgb = collect(args.source, args.backend, args.seconds, args.show)
    else:
        ap.error("pass --source (camera index or video path) or --demo")

    if len(ts) < 4:
        print("not enough usable frames — is the face visible and lit?", file=sys.stderr)
        return 1

    stats = frame_interval_stats(ts)
    print(f"\ncapture: {stats}")

    if args.save_signal:
        args.save_signal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"t": ts, "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2]}).to_csv(
            args.save_signal, index=False
        )
        print(f"signal -> {args.save_signal}")

    cfg = PipelineConfig(
        fs=args.fs,
        window_sec=args.window,
        hop_sec=args.hop,
        methods=tuple(args.methods),
        spectrum=args.spectrum,
        detrend_method={"ma": "moving_average"}.get(args.detrend, args.detrend),
    )
    df = analyse_signal(ts, rgb, cfg)

    summary = (
        df.groupby("method")
        .agg(
            BPM=("bpm_smoothed", "median"),
            spread=("bpm", "std"),
            SNR_dB=("snr", "mean"),
            accepted=("accepted", "mean"),
            windows=("bpm", "size"),
        )
        .sort_values("SNR_dB", ascending=False)
    )
    print(f"\n{cfg.window_sec:.0f} s window => {60 / cfg.window_sec:.1f} BPM raw resolution")
    print(summary.round(2).to_string())

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_results, index=False)
        print(f"\nresults -> {args.save_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
