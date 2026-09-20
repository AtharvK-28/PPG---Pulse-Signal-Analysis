"""Record raw video to disk, losslessly, with exact per-frame timestamps.

    python -m rppg.record --seconds 60 --out data/recordings/subject01

The live path has to make the whole chain — capture, face landmarks, four
projections, spectra, plots — fit inside 33 ms. Measured on this machine it
does not: `read()` costs 13.8 ms and the landmark stage 18.5 ms, so the loop
runs at 97 % utilisation and stalls in bursts, producing p99 = 438 ms holes in
a sampling grid the camera was delivering at p99 = 49 ms.

Recording removes that constraint rather than engineering around it. Nothing
happens per frame except a memory copy and a queue push; every measurement is
made later, at whatever speed it needs, with every frame present. This is also
what the field does — UBFC, PURE, COHFACE and MAHNOB are all recorded
datasets, and no published rPPG result is computed in a live loop.

Three properties matter for the recording to be worth analysing:

**Lossless.** The pulse is a 0.1-1 % uniform brightness change, which is
exactly what a lossy codec is designed to discard as imperceptible. FFV1 is
used here and the round-trip is verified by test_video_fidelity.

**Honest timestamps.** Taken with `perf_counter()` the instant `read()`
returns, before the frame is queued or encoded, so encoder backpressure can
never reach the sampling grid.

**Encoding off the capture thread.** FFV1 at 640x480x30 is tens of MB/s of
work; doing it inline would reintroduce the stall this module exists to avoid.
The writer runs on its own thread behind a deep queue, and the queue depth is
reported so you can see whether it ever came close to filling.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

#: Lossless. Anything else silently removes the signal.
CODEC = "FFV1"
CONTAINER = ".avi"


def burst_score(dt_ms) -> tuple[float, int]:
    """How much of the delivery arrives in bursts rather than evenly.

    Some drivers hold two frames and release them together. The pair then looks
    like a 16 ms interval followed by a 50 ms one, even though the sensor
    exposed them 33 ms apart — the timestamps describe *transport*, not
    exposure, and resampling onto them warps time locally by tens of percent.

    Measured on this machine over 8 s: MSMF gives 67 % short intervals and 80
    short-then-long pairs; DirectShow gives 0 and 0.

    Returns (fraction of intervals under 60 % of the mean, short-then-long
    pair count). A uniform delivery scores near zero on both.
    """
    dt = np.asarray(dt_ms, dtype=float)
    if len(dt) < 3:
        return 0.0, 0
    short = dt < 0.6 * dt.mean()
    pairs = int(np.sum(short[:-1] & (dt[1:] > 1.4 * dt.mean())))
    return float(short.mean()), pairs


@dataclass
class CaptureReport:
    frames: int
    duration_sec: float
    fps: float
    dt_median_ms: float
    dt_p99_ms: float
    dt_max_ms: float
    late_frames: int
    dropped: int
    max_queue: int
    width: int
    height: int
    fourcc: str
    mean_luma: float
    #: Fraction of intervals far below the mean, and short-then-long pair count.
    burst_fraction: float = 0.0
    burst_pairs: int = 0
    backend: str = ""

    def verdict(self) -> tuple[bool, str]:
        """Is this recording worth analysing?"""
        problems = []
        # Not fatal: rppg.offline can reconstruct a uniform grid when nothing
        # was actually lost. But it must be visible, because silently trusting
        # burst timestamps warps the frequency axis.
        if self.burst_pairs > 0.1 * self.frames:
            problems.append(
                "delivery is bursty (%.0f%% short intervals, %d short-then-long "
                "pairs) - timestamps describe transport, not exposure; try "
                "--backend dshow" % (100 * self.burst_fraction, self.burst_pairs)
            )
        if self.fps < 10:
            problems.append(
                "%.1f fps is below the 10 fps floor; Nyquist for the 4 Hz band "
                "edge needs 8+, so the top of the cardiac band is aliased" % self.fps
            )
        if self.dt_p99_ms > 4 * self.dt_median_ms:
            problems.append(
                "p99 interval %.0f ms against a %.0f ms median: the capture "
                "stalled, and spline interpolation will fill those holes with "
                "invented data" % (self.dt_p99_ms, self.dt_median_ms)
            )
        if self.dropped:
            problems.append("%d frames dropped by the writer queue" % self.dropped)
        if self.mean_luma < 25:
            problems.append("mean luma %.0f - too dark to carry the modulation"
                            % self.mean_luma)
        elif self.mean_luma > 245:
            problems.append("mean luma %.0f - clipping, modulation truncated"
                            % self.mean_luma)
        return (not problems), "; ".join(problems)


def record(
    device: int = 0,
    seconds: float = 60.0,
    out: str | Path = "data/recordings/session",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    backend: int | None = None,
    fix_exposure: bool = False,
    progress=None,
) -> CaptureReport:
    """Capture to `<out>.avi` + `<out>_timestamps.csv` + `<out>_capture.json`.

    `fix_exposure` defaults to False: measured on this hardware, locking
    exposure halved the frame rate (30.1 -> 15.9 fps) and the light (168 -> 80)
    while reporting success. See capture.WebcamSource._lock_exposure.
    """
    from .capture import WebcamSource

    if backend is None and hasattr(cv2, "CAP_DSHOW"):
        # Measured over 8 s on this machine: DirectShow delivered every frame
        # evenly (mean 100.0 ms, median 96.7, zero short-then-long pairs) while
        # MSMF burst-delivered (mean 33.2, median 15.9, 80 pairs) at the same
        # nominal rate. For a recording the even one is worth more than the
        # faster one, because bursty timestamps cannot be undone reliably.
        backend = cv2.CAP_DSHOW
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video_path = out.with_suffix(CONTAINER)

    cam = WebcamSource(
        device, width=width, height=height, fps=fps,
        fix_exposure=fix_exposure, backend=backend,
    )
    actual_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fourcc = int(cam.cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_name = (
        "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        if raw_fourcc else "(none)"
    )

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*CODEC), fps, (actual_w, actual_h)
    )
    if not writer.isOpened():
        cam.close()
        raise RuntimeError(
            "Could not open a %s writer. FFV1 needs an OpenCV build with "
            "FFmpeg support; without a lossless codec the recording is not "
            "worth making." % CODEC
        )

    # Deep enough to absorb any plausible encoder hiccup: 10 s at 30 fps.
    q: queue.Queue = queue.Queue(maxsize=300)
    stop = threading.Event()
    state = {"dropped": 0, "max_queue": 0}

    def _write():
        while True:
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            if frame is None:
                return
            writer.write(frame)

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()

    times, lumas = [], []
    t0 = None
    try:
        for frame in cam:
            t = time.perf_counter()
            if t0 is None:
                t0 = t
            elapsed = t - t0
            if elapsed > seconds:
                break
            times.append(elapsed)
            # Luma on a subsample: a full-frame mean is ~0.5 ms and this runs
            # on the capture thread, where every millisecond is the grid.
            lumas.append(float(frame.image[::8, ::8].mean()))
            try:
                q.put_nowait(frame.image)
            except queue.Full:
                state["dropped"] += 1
            state["max_queue"] = max(state["max_queue"], q.qsize())
            if progress is not None and len(times) % 30 == 0:
                progress(elapsed, seconds)
    finally:
        stop.set()
        q.put(None)
        thread.join(timeout=10.0)
        writer.release()
        cam.close()

    times = np.asarray(times)
    if len(times) < 10:
        raise RuntimeError(
            "Only %d frames captured. Another application probably holds the "
            "camera - a shared webcam drops to about 1 fps." % len(times)
        )
    dt = np.diff(times) * 1000.0
    frac, pairs = burst_score(dt)
    report = CaptureReport(
        burst_fraction=frac,
        burst_pairs=pairs,
        backend={cv2.CAP_DSHOW: "dshow", cv2.CAP_MSMF: "msmf"}.get(backend, "default"),
        frames=len(times),
        duration_sec=float(times[-1]),
        fps=len(times) / float(times[-1]),
        dt_median_ms=float(np.median(dt)),
        dt_p99_ms=float(np.percentile(dt, 99)),
        dt_max_ms=float(dt.max()),
        late_frames=int((dt > 2 * np.median(dt)).sum()),
        dropped=state["dropped"],
        max_queue=state["max_queue"],
        width=actual_w,
        height=actual_h,
        fourcc=fourcc_name,
        mean_luma=float(np.mean(lumas)),
    )

    # Timestamps are the measurement; keep them beside the pixels, not inside
    # a container whose frame rate field is a nominal constant.
    ts_path = out.parent / (out.name + "_timestamps.csv")
    with open(ts_path, "w") as fh:
        fh.write("frame,timestamp\n")
        for i, t in enumerate(times):
            fh.write("%d,%.9f\n" % (i, t))
    with open(out.parent / (out.name + "_capture.json"), "w") as fh:
        json.dump(asdict(report), fh, indent=2)
    return report


def load_timestamps(path) -> np.ndarray:
    """Per-frame capture instants written alongside a recording."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("frame"):
                continue
            rows.append(float(line.split(",")[1]))
    return np.asarray(rows)


def describe(report: CaptureReport) -> str:
    ok, problems = report.verdict()
    lines = [
        "%d frames in %.1f s = %.2f fps" % (report.frames, report.duration_sec, report.fps),
        "interval: median %.1f ms, p99 %.1f ms, max %.1f ms, %d late"
        % (report.dt_median_ms, report.dt_p99_ms, report.dt_max_ms, report.late_frames),
        "%dx%d, camera format %s, backend %s, mean luma %.0f"
        % (report.width, report.height, report.fourcc, report.backend, report.mean_luma),
        "delivery: %.0f%% short intervals, %d short-then-long pairs%s"
        % (100 * report.burst_fraction, report.burst_pairs,
           "  <- bursty" if report.burst_pairs > 0.1 * report.frames else "  (even)"),
        "writer queue peaked at %d of 300, %d frames dropped"
        % (report.max_queue, report.dropped),
    ]
    lines.append("VERDICT: usable" if ok else "VERDICT: not usable - " + problems)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="data/recordings/session")
    # Noise falls as 1/sqrt(N) over the ROI, so 1280x720 triples the pixel
    # count against 640x480 and buys about 4.8 dB - potentially the difference
    # between an unusable and a usable recording. It may cost frame rate; the
    # report says which happened.
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lock-exposure", action="store_true")
    ap.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "default"],
                    help="dshow delivers evenly on this hardware; msmf bursts")
    args = ap.parse_args(argv)
    backend = {"dshow": getattr(cv2, "CAP_DSHOW", None),
               "msmf": getattr(cv2, "CAP_MSMF", None),
               "default": None}[args.backend]

    def _progress(elapsed, total):
        print("\r  recording %5.1f / %.0f s" % (elapsed, total), end="", flush=True)

    print("Recording %s for %.0f s at %dx%d" %
          (args.out, args.seconds, args.width, args.height))
    print("Sit still, face the camera, and put the light in front of you.")
    report = record(
        device=args.device, seconds=args.seconds, out=args.out,
        width=args.width, height=args.height, fps=args.fps,
        fix_exposure=args.lock_exposure, backend=backend, progress=_progress,
    )
    print("\n")
    print(describe(report))
    print("\nNext: python -m rppg.offline %s" % args.out)
    return 0 if report.verdict()[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
