"""Stage 1 — ROI extraction and the per-frame (R, G, B) spatial mean.

The pulsatile modulation is 0.1-1% of pixel intensity, below the noise floor of
a single 8-bit pixel. It only clears the floor because we average thousands of
skin pixels, so which pixels enter the average is a first-order concern: every
non-skin pixel (hair, eyebrow, glasses frame, background) dilutes the pulse
without contributing any.

Backends
--------
mediapipe : primary. Dense 3D landmarks track through moderate motion, so head
            tilt moves the ROI with the face instead of destroying it.
haar      : documented fallback. Detection + CSRT tracking for continuity.
fixed     : a static rectangle. For synthetic clips and plumbing tests, where
            there is no face to find.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

#: YCrCb skin thresholds. Chrominance-only, so the test is largely invariant to
#: how brightly the face is lit — which is the point, since luminance is the
#: thing we are measuring and must not be used to segment.
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)


@dataclass
class RoiSample:
    timestamp: float
    rgb: np.ndarray  #: (3,) spatial mean over masked skin pixels, RGB order
    n_pixels: int
    ok: bool
    polygons: list = field(default_factory=list, repr=False)
    mask: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def empty(cls, timestamp: float):
        return cls(timestamp, np.full(3, np.nan), 0, False)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask by YCrCb chrominance thresholding."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    m = (
        (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])
    ).astype(np.uint8)
    # Close pinholes from specular highlights, then drop isolated speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)


def spatial_mean(bgr: np.ndarray, polygons, use_skin_mask: bool = True):
    """Mean (R, G, B) over the union of ``polygons``, optionally skin-masked.

    Returns ``(rgb, n_pixels, mask)``. Averaging is over uint8 pixels promoted
    to float — the mean of thousands of them carries far more precision than
    any single pixel's quantisation step, which is exactly why sub-1%
    modulation is recoverable at all.
    """
    h, w = bgr.shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(region, [pts], 1)
    if use_skin_mask:
        region &= skin_mask(bgr)

    n = int(region.sum())
    if n == 0:
        return np.full(3, np.nan), 0, region
    mean_bgr = cv2.mean(bgr, mask=region)[:3]
    return np.array(mean_bgr[::-1], dtype=float), n, region


# --------------------------------------------------------------------------
# MediaPipe backend
# --------------------------------------------------------------------------

# Canonical FaceMesh indices used to build the ROI frame. Polygons are derived
# geometrically from these rather than hard-coded as long index lists, so the
# regions rotate and scale with the head instead of drifting off it. Verify
# them once visually with `python -m rppg.roi --preview`.
#
# Everything is anchored to the eye corners and the hairline. Interocular
# distance is the most stable metric on a face — it barely changes with
# expression — so using it as the unit makes the regions scale correctly with
# distance from the camera without any per-subject tuning.
IDX_HAIRLINE = 10  # top-centre of the forehead
IDX_EYE_L = (33, 133)  # outer, inner corner
IDX_EYE_R = (362, 263)
IDX_FACE_L = 234
IDX_FACE_R = 454
IDX_CHIN = 152

# Region geometry, in units of interocular distance (IOD) or of the
# eye-line -> hairline vector. Generous on purpose: the YCrCb skin mask removes
# eyebrows, hair, glasses frames and background that overspill catches, so a
# box slightly too large costs nothing while a box too small costs SNR you
# cannot get back. Spatial averaging beats noise down as 1/sqrt(N), so a region
# 8x smaller than it should be gives away a factor of ~3 in SNR.
FOREHEAD_BOTTOM = 0.32  # fraction of eye-line -> hairline; clears the eyebrows
FOREHEAD_TOP = 0.88  # stops just below the hairline
FOREHEAD_HALF_WIDTH = 0.95  # x IOD
#: Cheek centre, as a fraction of the eye-line -> chin vector. Far enough down
#: to clear the lower rim of a pair of glasses: lens reflection is pure specular
#: (no pulse) and swings violently with head movement, so it is worse than
#: useless in the average.
CHEEK_DROP = 0.42
CHEEK_HALF_WIDTH = 0.42  # x IOD
CHEEK_HALF_HEIGHT = 0.28  # x IOD — kept short to stay clear of nose and mouth


def download_model(dest: Path = MODEL_PATH) -> Path:
    """Fetch the `face_landmarker.task` bundle (~3.7 MB) from Google's CDN."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(MODEL_URL, dest)
    return dest


def _quad(centre, axis_h, axis_v, half_w: float, half_h: float):
    """Corners of a quad in the face's own (h, v) frame — tilt-covariant."""
    c = np.asarray(centre, dtype=float)
    a = np.asarray(axis_h, dtype=float) * half_w
    b = np.asarray(axis_v, dtype=float) * half_h
    return np.array([c - a - b, c + a - b, c + a + b, c - a + b])


def landmarks_to_polygons(pts: np.ndarray, forehead: bool = True, cheeks: bool = True):
    """Forehead and upper-cheek polygons from (N, 2) pixel landmarks.

    The face frame is built from the eye corners (horizontal axis) and the
    eye-line-to-hairline vector (vertical axis), so both regions rotate with a
    head tilt rather than staying axis-aligned while the face turns underneath.
    """
    polys = []
    eye_l = 0.5 * (pts[IDX_EYE_L[0]] + pts[IDX_EYE_L[1]])
    eye_r = 0.5 * (pts[IDX_EYE_R[0]] + pts[IDX_EYE_R[1]])
    eye_mid = 0.5 * (eye_l + eye_r)

    across = eye_r - eye_l
    iod = float(np.linalg.norm(across))
    up = pts[IDX_HAIRLINE] - eye_mid
    up_len = float(np.linalg.norm(up))
    if iod < 1.0 or up_len < 1.0:
        return polys

    h_unit = across / iod
    v_unit = up / up_len

    if forehead:
        centre = eye_mid + 0.5 * (FOREHEAD_BOTTOM + FOREHEAD_TOP) * up
        half_h = 0.5 * (FOREHEAD_TOP - FOREHEAD_BOTTOM) * up_len
        polys.append(_quad(centre, h_unit, v_unit, FOREHEAD_HALF_WIDTH * iod, half_h))

    if cheeks:
        # Built from the eye-line -> chin vector rather than a named cheek
        # landmark: it needs only indices whose position is unambiguous, and it
        # follows head pitch, which a fixed landmark offset does not.
        down = pts[IDX_CHIN] - eye_mid
        for eye_c in (eye_l, eye_r):
            polys.append(
                _quad(
                    eye_c + CHEEK_DROP * down,
                    h_unit,
                    v_unit,
                    CHEEK_HALF_WIDTH * iod,
                    CHEEK_HALF_HEIGHT * iod,
                )
            )

    return polys


class MediaPipeROI:
    """Face Landmarker (Tasks API) in VIDEO mode."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        forehead: bool = True,
        cheeks: bool = True,
        use_skin_mask: bool = True,
        auto_download: bool = False,
        min_detection_confidence: float = 0.3,
        min_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        redetect_every: int = 3,
    ):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            if auto_download:
                download_model(model_path)
            else:
                raise FileNotFoundError(
                    f"{model_path} not found. Run `python -m rppg.roi --download-model` "
                    "(fetches ~3.7 MB from Google), or use backend='haar'."
                )
        self._vision = vision
        # Thresholds well below the 0.5 default. Losing the face is far more
        # costly here than a briefly imprecise ROI: a dropout punches a hole in
        # the sampling grid and costs a whole window, whereas slightly noisy
        # landmarks only jitter the region by a few pixels, which the spatial
        # average largely absorbs.
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self.forehead, self.cheeks, self.use_skin_mask = forehead, cheeks, use_skin_mask
        # Landmark inference costs ~18.5 ms/frame, against a 33 ms budget that
        # cv2.read() has already spent 13.8 ms of — leaving ~1 ms for everything
        # else, so the loop saturates and stalls in bursts. The measurement we
        # actually need, the masked mean, costs 0.28 ms. Detecting every Nth
        # frame and reusing the polygon in between buys back most of the budget;
        # a seated face moves a few pixels in 100 ms, which the spatial average
        # absorbs. Set to 1 to detect on every frame.
        self.redetect_every = max(1, int(redetect_every))
        self._polys = None
        self._since_detect = 1 << 30  # force detection on the first frame

    def _detect(self, frame):
        import mediapipe as mp

        h, w = frame.image.shape[:2]
        rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # VIDEO mode wants monotonically increasing integer milliseconds.
        result = self.landmarker.detect_for_video(mp_image, int(frame.timestamp * 1000))
        if not result.face_landmarks:
            return None
        pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]])
        return landmarks_to_polygons(pts, self.forehead, self.cheeks) or None

    def __call__(self, frame) -> RoiSample:
        if self._since_detect >= self.redetect_every:
            self._polys = self._detect(frame)
            # 1, not 0: this frame is itself the first of the new interval, so
            # resetting to 0 would stretch the period to redetect_every + 1 and
            # make redetect_every=1 detect on every *other* frame.
            self._since_detect = 1
            # A failed detection drops the stale polygon rather than averaging
            # over wherever the face used to be: a hole in the sampling grid is
            # recoverable, a plausible-looking wrong signal is not.
            if self._polys is None:
                return RoiSample.empty(frame.timestamp)
        else:
            self._since_detect += 1
            if self._polys is None:
                return RoiSample.empty(frame.timestamp)

        rgb, n, mask = spatial_mean(frame.image, self._polys, self.use_skin_mask)
        return RoiSample(frame.timestamp, rgb, n, n > 0, self._polys, mask)

    def close(self):
        self.landmarker.close()


# --------------------------------------------------------------------------
# Haar + CSRT fallback
# --------------------------------------------------------------------------


class HaarROI:
    """Cascade detection with CSRT tracking between detections.

    Documented as the fallback, not the plan: the box is axis-aligned, so a head
    tilt rotates the face inside a static rectangle and pulls hair and
    background into the average.
    """

    def __init__(
        self,
        redetect_every: int = 30,
        use_skin_mask: bool = True,
        forehead: bool = True,
        cheeks: bool = True,
        cascade_path: str | None = None,
    ):
        path = cascade_path or (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"could not load cascade {path}")
        self.redetect_every = redetect_every
        self.use_skin_mask, self.forehead, self.cheeks = use_skin_mask, forehead, cheeks
        self.tracker = None
        self.box = None
        self._since_detect = 0

    def _detect(self, gray):
        faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        # Largest face — the subject is nearest the camera.
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

    def _boxes_from_face(self, box):
        x, y, w, h = box
        polys = []
        if self.forehead:
            fx, fy, fw, fh = x + 0.30 * w, y + 0.10 * h, 0.40 * w, 0.14 * h
            polys.append(_rect_poly(fx, fy, fw, fh))
        if self.cheeks:
            polys.append(_rect_poly(x + 0.12 * w, y + 0.55 * h, 0.20 * w, 0.16 * h))
            polys.append(_rect_poly(x + 0.68 * w, y + 0.55 * h, 0.20 * w, 0.16 * h))
        return polys

    def __call__(self, frame) -> RoiSample:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        need_detect = self.tracker is None or self._since_detect >= self.redetect_every
        if not need_detect:
            ok, box = self.tracker.update(frame.image)
            if ok:
                self.box = tuple(int(v) for v in box)
                self._since_detect += 1
            else:
                need_detect = True
        if need_detect:
            box = self._detect(gray)
            if box is None:
                self.tracker = None
                return RoiSample.empty(frame.timestamp)
            self.box = box
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame.image, box)
            self._since_detect = 0

        polys = self._boxes_from_face(self.box)
        rgb, n, mask = spatial_mean(frame.image, polys, self.use_skin_mask)
        return RoiSample(frame.timestamp, rgb, n, n > 0, polys, mask)

    def close(self):
        self.tracker = None


def _rect_poly(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


class FixedROI:
    """A static region. No detection — for synthetic clips and plumbing tests."""

    def __init__(self, rect=None, rel_rect=(0.35, 0.35, 0.30, 0.30), use_skin_mask=False):
        self.rect, self.rel_rect, self.use_skin_mask = rect, rel_rect, use_skin_mask

    def __call__(self, frame) -> RoiSample:
        h, w = frame.image.shape[:2]
        if self.rect is not None:
            x, y, rw, rh = self.rect
        else:
            rx, ry, rw_, rh_ = self.rel_rect
            x, y, rw, rh = rx * w, ry * h, rw_ * w, rh_ * h
        polys = [_rect_poly(x, y, rw, rh)]
        rgb, n, mask = spatial_mean(frame.image, polys, self.use_skin_mask)
        return RoiSample(frame.timestamp, rgb, n, n > 0, polys, mask)

    def close(self):
        pass


def make_roi(backend: str = "mediapipe", **kw):
    """Build an ROI extractor, falling back to Haar if MediaPipe is unavailable."""
    backend = backend.lower()
    if backend == "mediapipe":
        try:
            return MediaPipeROI(**kw)
        except (ImportError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"MediaPipe backend unavailable: {exc}\n"
                "Use make_roi('haar') to fall back."
            ) from exc
    if backend == "haar":
        kw.pop("model_path", None)
        kw.pop("auto_download", None)
        return HaarROI(**kw)
    if backend == "fixed":
        return FixedROI(**{k: v for k, v in kw.items() if k in ("rect", "rel_rect", "use_skin_mask")})
    raise ValueError(f"unknown ROI backend {backend!r}")


def draw_overlay(image: np.ndarray, sample: RoiSample, colour=(0, 255, 0)):
    """ROI polygons + skin mask for the dashboard's left column."""
    out = image.copy()
    if sample.mask is not None:
        tint = np.zeros_like(out)
        tint[:, :, 1] = 255
        out = np.where(sample.mask[:, :, None] > 0, (0.75 * out + 0.25 * tint), out).astype(
            np.uint8
        )
    for poly in sample.polygons:
        cv2.polylines(out, [np.asarray(poly, np.int32)], True, colour, 2)
    return out


def extract_signal(source, roi, max_frames: int | None = None, progress=None):
    """Run a frame source through an ROI extractor -> (timestamps, rgb, samples).

    Frames where the face is lost are dropped rather than filled: a NaN-free,
    honestly non-uniform time series is exactly what the resampler expects.
    """
    ts, rgbs, samples = [], [], []
    for i, frame in enumerate(source):
        if max_frames is not None and i >= max_frames:
            break
        s = roi(frame)
        samples.append(s)
        if s.ok and np.all(np.isfinite(s.rgb)):
            ts.append(s.timestamp)
            rgbs.append(s.rgb)
        if progress is not None:
            progress(i, s)
    return np.array(ts), np.array(rgbs), samples


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="ROI utilities")
    ap.add_argument("--download-model", action="store_true", help="fetch face_landmarker.task")
    ap.add_argument("--preview", action="store_true", help="webcam preview with ROI overlay")
    ap.add_argument("--backend", default="mediapipe")
    args = ap.parse_args()

    if args.download_model:
        print("downloading ->", download_model())
    if args.preview:
        from .capture import WebcamSource

        # The pixel count is the number to watch: the pulse is 0.1-1% of
        # intensity, and only survives because averaging N skin pixels cuts
        # noise by sqrt(N). Under ~8000 px, expect a poor SNR no matter how
        # good the downstream DSP is.
        MIN_PIXELS = 8000
        roi = make_roi(args.backend)
        print("ESC to quit. Watch the pixel count — aim for >8000.")
        with WebcamSource() as cam:
            for frame in cam:
                s = roi(frame)
                out = draw_overlay(frame.image, s)
                ok = s.n_pixels >= MIN_PIXELS
                cv2.putText(
                    out,
                    f"{s.n_pixels} px" if s.ok else "no face",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 0) if (s.ok and ok) else (0, 0, 255),
                    2,
                )
                cv2.imshow("roi", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()
