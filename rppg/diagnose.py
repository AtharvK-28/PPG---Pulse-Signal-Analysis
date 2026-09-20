"""Check the capture chain before blaming the DSP.

    python -m rppg.diagnose

Most "the heart rate is wrong" failures are not signal-processing failures.
They are a camera delivering 1 fps because another process has it open, or a
face too dark to carry a sub-percent modulation, or an ROI a tenth of the size
it should be. Those are all measurable in twenty seconds, and none of them are
visible from a BPM readout - which will happily show a confident wrong number.

Every check prints the measurement, the threshold, and *why* the threshold is
what it is.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {PASS: "[ok]  ", WARN: "[warn]", FAIL: "[FAIL]"}

#: Nyquist for the 4 Hz top of the cardiac band. Below this the upper half of
#: the band is not measured, it is aliased.
MIN_FPS = 10.0
#: Spatial averaging cuts noise as 1/sqrt(N); below this the sub-percent
#: modulation does not clear the sensor noise floor.
MIN_ROI_PIXELS = 8000
#: Below this the face is under-exposed and the pulse rides on almost nothing.
MIN_BRIGHTNESS = 25.0
#: Above this the skin is clipping and the modulation is being truncated away.
MAX_BRIGHTNESS = 245.0


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _fourcc(v) -> str:
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip() if v else "(none)"


def check_capture(device=0, seconds=6.0, backend=None, width=640, height=480):
    """Open the camera and measure what it actually delivers."""
    import cv2

    checks: list[Check] = []
    cap = cv2.VideoCapture(device, backend) if backend else cv2.VideoCapture(device)
    if not cap.isOpened():
        from .capture import list_cameras

        return None, [
            Check(
                "camera opens",
                FAIL,
                f"could not open camera {device}",
                f"available indices: {list_cameras() or 'none found'}",
            )
        ]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    t0 = time.perf_counter()
    ok, _ = cap.read()
    first_ms = (time.perf_counter() - t0) * 1000.0
    # Warm-up: exposure and gain settle, and MSMF needs a moment to start.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        cap.read()

    times, brights = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        t = time.perf_counter()
        if ok and frame is not None:
            times.append(t)
            brights.append(float(frame.mean()))

    fourcc = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if len(times) < 4:
        return None, [
            Check(
                "frame delivery",
                FAIL,
                f"only {len(times)} frames in {seconds:.0f} s (first frame {first_ms:.0f} ms)",
                "another application almost certainly has the camera open - "
                "close other video apps, browser tabs with camera permission, "
                "and any leftover copy of this app, then retry",
            )
        ]

    times = np.asarray(times)
    dt = np.diff(times) * 1000.0
    fps = len(times) / seconds
    brightness = float(np.mean(brights))

    if fps < 3.0:
        checks.append(
            Check(
                "frame rate",
                FAIL,
                f"{fps:.1f} fps (interval {dt.mean():.0f} +/- {dt.std():.0f} ms)",
                "a rate this low and this steady means the camera is shared. "
                "Close every other program using it - including any still-running "
                "copy of the Streamlit app - and retry. rPPG is impossible here: "
                f"Nyquist is {fps / 2:.1f} Hz, and the cardiac band starts at 0.7 Hz.",
            )
        )
    elif fps < MIN_FPS:
        checks.append(
            Check(
                "frame rate",
                FAIL,
                f"{fps:.1f} fps, below the {MIN_FPS:.0f} fps floor",
                "Nyquist for the 4 Hz band edge needs 8+ fps. Improve lighting "
                "(a dim scene makes the camera lengthen its exposure), lower the "
                "resolution, or close other camera users.",
            )
        )
    else:
        checks.append(
            Check("frame rate", PASS, f"{fps:.1f} fps, interval sd {dt.std():.1f} ms")
        )

    if brightness < 1.0:
        checks.append(
            Check(
                "exposure",
                FAIL,
                f"mean frame brightness {brightness:.1f} - frames are black",
                "lens cover or privacy shutter closed, or the camera is held by "
                "another process and returning empty frames",
            )
        )
    elif brightness < MIN_BRIGHTNESS:
        checks.append(
            Check(
                "exposure",
                WARN,
                f"mean frame brightness {brightness:.1f}, under {MIN_BRIGHTNESS:.0f}",
                "add light. The pulse is 0.1-1% of intensity, so a dark image "
                "carries proportionally less signal and the camera also lengthens "
                "its exposure, costing frame rate.",
            )
        )
    elif brightness > MAX_BRIGHTNESS:
        checks.append(
            Check("exposure", WARN, f"mean brightness {brightness:.1f} - likely clipping",
                  "reduce light or exposure; clipped skin has no modulation left")
        )
    else:
        checks.append(Check("exposure", PASS, f"mean frame brightness {brightness:.1f}"))

    compressed = fourcc.upper() in {"MJPG", "H264", "X264", "AVC1", "HEVC"}
    checks.append(
        Check(
            "pixel format",
            WARN if compressed else PASS,
            f"{fourcc} at {w}x{h}",
            "the camera is delivering a lossy-compressed stream. Compression "
            "discards exactly the sub-percent uniform brightness changes the "
            "pulse consists of. Prefer an uncompressed mode (YUY2) if the "
            "camera offers one at a usable frame rate."
            if compressed
            else "",
        )
    )
    return dict(fps=fps, brightness=brightness, fourcc=fourcc, size=(w, h)), checks


def check_roi(device=0, seconds=10.0, backend_name="mediapipe"):
    """Measure ROI size, detection reliability and how much the region jitters."""
    import numpy as np

    from .capture import WebcamSource
    from .roi import make_roi

    checks: list[Check] = []
    try:
        roi = make_roi(backend_name)
    except RuntimeError as exc:
        return None, [Check("ROI backend", WARN, str(exc), "falling back to haar")]

    pixels, centroids, ok_count, total = [], [], 0, 0
    try:
        with WebcamSource(device) as cam:
            for frame in cam:
                if frame.timestamp > seconds:
                    break
                total += 1
                s = roi(frame)
                if s.ok:
                    ok_count += 1
                    pixels.append(s.n_pixels)
                    pts = np.concatenate([np.asarray(p) for p in s.polygons])
                    centroids.append(pts.mean(axis=0))
    finally:
        roi.close()

    if total == 0:
        return None, [Check("ROI", FAIL, "no frames captured", "see capture checks above")]

    rate = ok_count / total
    if rate < 0.5:
        checks.append(
            Check("face detection", FAIL, f"face found in {rate:.0%} of frames",
                  "sit square to the camera, remove obstructions, add light")
        )
    elif rate < 0.95:
        checks.append(
            Check("face detection", WARN, f"face found in {rate:.0%} of frames",
                  "dropouts punch holes in the sampling grid and cost whole windows")
        )
    else:
        checks.append(Check("face detection", PASS, f"face found in {rate:.0%} of frames"))

    if pixels:
        px = float(np.mean(pixels))
        checks.append(
            Check(
                "ROI size",
                PASS if px >= MIN_ROI_PIXELS else FAIL,
                f"{px:,.0f} skin pixels averaged per frame",
                "" if px >= MIN_ROI_PIXELS else
                "move closer to the camera. Noise falls as 1/sqrt(N), so a small "
                f"ROI gives away SNR directly; aim for {MIN_ROI_PIXELS:,}+.",
            )
        )
    if len(centroids) > 2:
        c = np.asarray(centroids)
        jitter = float(np.mean(np.std(c, axis=0)))
        checks.append(
            Check(
                "ROI stability",
                PASS if jitter < 3.0 else WARN,
                f"region centroid wanders {jitter:.1f} px (sd)",
                "" if jitter < 3.0 else
                "the ROI is moving across a shaded face, so different pixels enter "
                "the average each frame. That injects noise directly into the "
                "signal - hold still, and keep the lighting even.",
            )
        )
    return dict(rate=rate, pixels=np.mean(pixels) if pixels else 0.0), checks


def report(checks) -> str:
    worst = PASS
    lines = []
    for c in checks:
        lines.append(f"  {_MARK[c.status]} {c.name:18s} {c.detail}")
        if c.remedy:
            for i, chunk in enumerate(_wrap(c.remedy, 66)):
                lines.append(f"{'':<9}{'-> ' if i == 0 else '   '}{chunk}")
        if c.status == FAIL or (c.status == WARN and worst == PASS):
            worst = c.status if c.status == FAIL else WARN
    return "\n".join(lines), worst


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0, help="ROI measurement length")
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "haar"])
    ap.add_argument("--skip-roi", action="store_true", help="capture checks only")
    args = ap.parse_args(argv)

    print("rPPG capture diagnostic")
    print("=" * 72)
    print("\nCapture chain")
    _, cap_checks = check_capture(args.device)
    text, worst = report(cap_checks)
    print(text)

    if worst == FAIL:
        print("\n" + "=" * 72)
        print("Stop here - fix the capture problems above before looking at any BPM.")
        print("A camera below 10 fps aliases every heart rate; the numbers the app")
        print("shows in that state are arithmetic on noise, not measurements.")
        return 1

    if not args.skip_roi:
        print(f"\nROI and face tracking ({args.seconds:.0f} s - please sit still, face the camera)")
        _, roi_checks = check_roi(args.device, args.seconds, args.backend)
        text2, worst2 = report(roi_checks)
        print(text2)
        worst = FAIL if FAIL in (worst, worst2) else (WARN if WARN in (worst, worst2) else PASS)

    print("\n" + "=" * 72)
    if worst == PASS:
        print("Capture chain looks healthy. If the BPM is still wrong, the problem")
        print("is downstream - check cross-method agreement in the app.")
    elif worst == WARN:
        print("Usable, but degraded. Address the warnings above for a better estimate.")
    else:
        print("Not usable yet. Fix the failures above first.")
    return 0 if worst != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
