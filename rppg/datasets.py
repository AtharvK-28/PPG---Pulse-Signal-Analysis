"""Dataset loaders and the window-to-reference alignment protocol.

UBFC-rPPG (Bobbia et al., 2019) is the primary benchmark: Logitech C920, 30 fps,
640x480, uncompressed 8-bit RGB, CMS50E pulse oximeter ground truth, subjects
seated ~1 m from the camera. It matches our hardware assumption closely, which
is why numbers from it transfer to the webcam demo.

Access is by request to the authors — send it in week 1, before writing code.
PURE, LGI-PPGI and COHFACE are the fallbacks if it stalls.

Known limitation to state up front rather than be asked about: UBFC-rPPG is not
skin-tone diverse, and rPPG degrades on darker skin because melanin absorbs
more incident light, so less reaches the capillary bed and less returns to the
sensor. Every number produced from this dataset inherits that bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GroundTruth:
    t: np.ndarray  #: seconds
    ppg: np.ndarray  #: raw oximeter waveform
    hr: np.ndarray  #: device-reported BPM

    def hr_at(self, t_query) -> np.ndarray:
        return np.interp(np.asarray(t_query, dtype=float), self.t, self.hr)


@dataclass
class Recording:
    subject: str
    video: Path
    ground_truth: GroundTruth | None
    condition: str = "A"


def load_ubfc_ground_truth(path, layout: str = "auto") -> GroundTruth:
    """Read UBFC-rPPG ground truth.

    DATASET_2 ships `ground_truth.txt`: three whitespace-separated rows — PPG
    waveform, HR, timestamps. DATASET_1 ships `gtdump.xmp`, a comma-separated
    dump whose columns are (ms, ?, HR, SpO2, PPG).
    """
    path = Path(path)
    if layout == "auto":
        layout = "dataset1" if path.suffix.lower() == ".xmp" else "dataset2"

    if layout == "dataset2":
        rows = [
            np.array([float(v) for v in line.split()])
            for line in path.read_text().strip().splitlines()
            if line.strip()
        ]
        if len(rows) < 3:
            raise ValueError(f"{path}: expected 3 rows (ppg, hr, time), got {len(rows)}")
        ppg, hr, t = rows[0], rows[1], rows[2]
    else:
        arr = np.loadtxt(path, delimiter=",")
        t, hr, ppg = arr[:, 0] / 1000.0, arr[:, 2], arr[:, 4]

    n = min(len(t), len(hr), len(ppg))
    t = np.asarray(t[:n], dtype=float)
    return GroundTruth(t=t - t[0], ppg=np.asarray(ppg[:n]), hr=np.asarray(hr[:n]))


def find_ubfc_recordings(root, condition: str = "A") -> list[Recording]:
    """Discover `subject*/vid.avi` + ground truth under a UBFC-rPPG root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Request UBFC-rPPG access from the authors and "
            "unpack it here, or point --data at another root."
        )
    out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        videos = sorted(list(sub.glob("vid.avi")) + list(sub.glob("*.avi")))
        if not videos:
            continue
        gt_file = next(
            (p for p in (sub / "ground_truth.txt", sub / "gtdump.xmp") if p.exists()), None
        )
        gt = load_ubfc_ground_truth(gt_file) if gt_file else None
        out.append(Recording(sub.name, videos[0], gt, condition))
    if not out:
        raise FileNotFoundError(f"no subject folders with videos under {root}")
    return out


def find_own_recordings(root) -> list[Recording]:
    """Self-collected stress set: `<condition>/<clip>.mp4` + optional `<clip>_gt.csv`.

    Conditions A (still), B (motion), C (illumination), D (post-exercise). D
    matters more than it looks — a method that always outputs ~72 BPM scores
    beautifully at rest and means nothing.
    """
    root = Path(root)
    out = []
    for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video in sorted(
            list(cond_dir.glob("*.mp4")) + list(cond_dir.glob("*.avi"))
        ):
            gt_csv = video.with_name(video.stem + "_gt.csv")
            gt = None
            if gt_csv.exists():
                df = pd.read_csv(gt_csv)
                gt = GroundTruth(
                    t=df["t"].to_numpy(float),
                    ppg=df.get("ppg", pd.Series(np.zeros(len(df)))).to_numpy(float),
                    hr=df["hr"].to_numpy(float),
                )
            out.append(Recording(video.stem, video, gt, cond_dir.name[:1].upper()))
    return out


def attach_reference(df: pd.DataFrame, gt: GroundTruth) -> pd.DataFrame:
    """Add the reference BPM for each analysis window.

    The reference for a window is the *mean* device HR over that window's span,
    not the instantaneous value at its centre: our estimate is itself an average
    over the window, so comparing it to a point sample would charge the method
    for the reference's own within-window variation.
    """
    df = df.copy()
    ref = []
    for t0, t1 in zip(df.t_start, df.t_end):
        span = (gt.t >= t0) & (gt.t <= t1)
        ref.append(float(np.mean(gt.hr[span])) if np.any(span) else float(np.mean(gt.hr_at([0.5 * (t0 + t1)]))))
    df["bpm_ref"] = ref
    return df
