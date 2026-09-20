"""Live demo — four rPPG arms side by side.

    streamlit run app/streamlit_app.py

Left column is the problem: three noisy, drifting traces that look nothing like
a heartbeat. Right column is the solution: the same data under four different
projections. The bottom strip is the actual demo — shake your head and watch
the GREEN and ICA lines diverge while CHROM and POS hold. That is a
measurement, not a claim.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rppg import METHODS  # noqa: E402
from rppg.capture import (  # noqa: E402
    ThreadedWebcamSource,
    VideoFileSource,
    WebcamSource,
    list_cameras,
)
from rppg.methods import ica_full  # noqa: E402
from rppg.pipeline import (  # noqa: E402
    PipelineConfig,
    SlidingWindowEstimator,
    agreement_spread,
    project_and_filter,
)
from rppg.resample import frame_interval_stats, resample_uniform  # noqa: E402
from rppg.roi import RoiSample, draw_overlay, make_roi  # noqa: E402
from rppg.sampler import ThreadedSampler  # noqa: E402
from rppg.spectral import estimate_bpm, periodogram  # noqa: E402
from rppg.beats import clean_intervals, detect_beats, intervals  # noqa: E402
from rppg.hrv import summary, tachogram  # noqa: E402
from rppg.preprocess import bandpass, detrend  # noqa: E402
from rppg.respiration import analyse, riav, rifv, riiv  # noqa: E402

COLOURS = {"green": "#2e9e4f", "ica": "#c2410c", "chrom": "#1d4ed8", "pos": "#7c3aed"}

#: Preview budget. The camera feed is a convenience, not the measurement — it
#: must never compete with the capture loop for time.
PREVIEW_WIDTH = 320
PREVIEW_PERIOD = 0.10  # seconds between preview frames (10 fps is plenty)
#: Figure redraw cadence, adapted at runtime. Plots are diagnostics, not
#: the measurement, so they yield to the capture loop when it is starved.
MIN_PLOT_PERIOD = 0.8
MAX_PLOT_PERIOD = 5.0

st.set_page_config(page_title="rPPG — contactless pulse", layout="wide")


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

st.sidebar.title("rPPG")
st.sidebar.caption("Contactless pulse estimation — DTSP mini-project")

mode = st.sidebar.radio(
    "Source",
    ["Webcam", "Video file", "Synthetic (no camera)", "PPG record (BIDMC)"],
)

#: Contact-PPG mode exists because it is the only path with a clinical
#: reference attached. The webcam arms are the extension; this is the part of
#: the project whose numbers are validated, so the demo should be able to show
#: it rather than only the half that is still being argued about.
BIDMC_ROOT = Path(__file__).resolve().parent.parent / "data" / "bidmc"
bidmc_record = bidmc_start = bidmc_len = None
if mode == "PPG record (BIDMC)":
    available = sorted(p.stem for p in BIDMC_ROOT.glob("bidmc??.hea"))
    if not available:
        st.sidebar.error(
            "No BIDMC records found. Run `python -m rppg.bidmc --download` "
            "(open access, no request form)."
        )
    else:
        bidmc_record = st.sidebar.selectbox("Record", available)
        bidmc_start = st.sidebar.slider("Start (s)", 0, 400, 0, 10)
        # 120 s by default: LF needs ~2 minutes before the ratio means anything,
        # and the app should not offer a number the standard says is unreliable.
        bidmc_len = st.sidebar.slider("Length (s)", 30, 300, 120, 30)


@st.cache_data(show_spinner="looking for cameras…")
def _cameras():
    """Probed once per session — opening every index takes a second each."""
    return list_cameras()


device = 0
if mode == "Webcam":
    cams = _cameras()
    if cams:
        # A dropdown of indices that actually work, rather than a free number
        # box that happily accepts a camera you do not have.
        device = st.sidebar.selectbox(
            "Camera", cams, format_func=lambda i: f"index {i}"
        )
    else:
        st.sidebar.error(
            "No camera detected. Close any app using the webcam, check "
            "Windows Settings › Privacy & security › Camera, or pick "
            "*Synthetic* to run without one."
        )
        device = None

upload = st.sidebar.file_uploader("Clip", ["mp4", "avi", "mov"]) if mode == "Video file" else None

#: The four benchmark conditions, as synth_rgb keyword arguments. Being able to
#: pick B here is what lets the rolling strip demonstrate its own caption:
#: without motion in the source, all four arms agree and the panel promises a
#: divergence it never shows.
CONDITIONS = {
    "A — still": dict(bpm=72.0),
    "B — motion": dict(bpm=72.0, motion_amplitude=0.05, motion_bpm=42.0),
    "C — illumination": dict(bpm=72.0, motion_amplitude=0.01, illumination_drift=0.25),
    "D — post-exercise": dict(bpm=(100.0, 140.0), motion_amplitude=0.02),
}
condition = (
    st.sidebar.selectbox("Condition", list(CONDITIONS), help="the four benchmark conditions")
    if mode == "Synthetic (no camera)"
    else "A — still"
)

st.sidebar.subheader("ROI")
backend = st.sidebar.selectbox("Backend", ["mediapipe", "haar", "fixed"])
use_skin_mask = st.sidebar.checkbox("YCrCb skin mask", True)
#: Glasses are the common case where cheeks hurt rather than help: a lens is
#: pure specular reflection, so those pixels carry no pulse and swing hard with
#: any head movement. Averaging them in dilutes the signal twice over.
regions = st.sidebar.selectbox(
    "Regions",
    ["forehead + cheeks", "forehead only", "cheeks only"],
    help="Pick 'forehead only' if you wear glasses — the cheek boxes can land on the lenses.",
)
roi_kw = dict(
    use_skin_mask=use_skin_mask,
    forehead=regions != "cheeks only",
    cheeks=regions != "forehead only",
)

st.sidebar.subheader("DSP")
window_sec = st.sidebar.slider("Window (s)", 5.0, 30.0, 15.0, 1.0)
st.sidebar.caption(f"Raw resolution: **{60 / window_sec:.1f} BPM** (60/T)")
hop_sec = st.sidebar.slider("Hop (s)", 0.5, 5.0, 1.0, 0.5)
detrend_method = st.sidebar.selectbox("Detrend", ["smoothness", "moving_average", "none"])
spectrum = st.sidebar.selectbox("Spectrum", ["fft", "welch"])
zero_pad = st.sidebar.select_slider("Zero-pad", [1, 2, 4, 8, 16], 8)
snr_threshold = st.sidebar.slider(
    "SNR gate (dB)", -20.0, 10.0, -3.0, 0.5,
    help="Calibrated on 18,640 BIDMC windows against ECG ground truth: "
         "-3 dB keeps 59% of windows at 1.80 BPM MAE, -7 dB keeps 96% at 9.12.",
)
#: Exposed because it is the fastest way to identify an octave error on real
#: data: if switching to "peak" doubles the answer, the harmonic selector has
#: locked onto a subharmonic.
selector = st.sidebar.selectbox("Peak selector", ["harmonic", "peak"])
methods = st.sidebar.multiselect("Methods", list(METHODS), list(METHODS))

st.sidebar.subheader("Record")
record = st.sidebar.checkbox(
    "Save this session", help="writes timestamped R,G,B means for offline analysis"
)
record_note = st.sidebar.text_input(
    "Reference BPM (optional)", "",
    help="your own wrist count, so the recording carries its own ground truth",
) if record else ""
# A fixed length so the run ends by itself and the file is definitely written;
# a Streamlit rerun can kill the loop before any cleanup would fire.
record_seconds = st.sidebar.slider("Record for (s)", 30, 180, 60, 10) if record else None

fs = 30.0
cfg = PipelineConfig(
    fs=fs,
    window_sec=window_sec,
    hop_sec=hop_sec,
    detrend_method=detrend_method,
    spectrum=spectrum,
    zero_pad=int(zero_pad),
    snr_threshold=snr_threshold,
    selector=selector,
    methods=tuple(methods) if methods else METHODS,
)

col_a, col_b = st.sidebar.columns(2)
start = col_a.button("Start", type="primary", use_container_width=True)
col_b.button("Stop", use_container_width=True)  # a click reruns the script, ending the loop


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    return ax


def fig_raw_rgb(t, rgb):
    fig = Figure(figsize=(5, 2.1), dpi=85)
    ax = fig.subplots()
    if len(t):
        for i, (name, colour) in enumerate(zip("RGB", ["#dc2626", "#16a34a", "#2563eb"])):
            ax.plot(t - t[0], rgb[:, i], lw=0.8, color=colour, label=name)
        ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
    ax.set_title("raw spatial means — drifting, no visible pulse", fontsize=8)
    ax.set_xlabel("s", fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.22)
    return _style(ax).figure


def fig_pulses(signals, fs):
    n = max(len(signals), 1)
    fig = Figure(figsize=(5, 1.0 * n + 0.6), dpi=85)
    axes = fig.subplots(n, 1, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, sig) in zip(axes, signals.items()):
        t = np.arange(len(sig)) / fs
        ax.plot(t, sig / (np.std(sig) + 1e-12), lw=0.9, color=COLOURS.get(name, "k"))
        ax.set_ylabel(name.upper(), fontsize=7, color=COLOURS.get(name, "k"))
        _style(ax)
        ax.set_yticks([])
    axes[-1].set_xlabel("s", fontsize=7)
    axes[0].set_title("extracted pulse (amplitude-normalised)", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.12, hspace=0.18)
    return fig


def fig_spectra(results):
    n = max(len(results), 1)
    fig = Figure(figsize=(5, 1.0 * n + 0.6), dpi=85)
    axes = fig.subplots(n, 1, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, results.items()):
        band = (res.freqs >= 0.5) & (res.freqs <= 4.5)
        ax.plot(res.freqs[band] * 60, res.power[band], lw=0.9, color=COLOURS.get(name, "k"))
        ax.axvline(res.bpm, color="k", ls="--", lw=0.8)
        ax.annotate(
            f"{res.bpm:.1f}", (res.bpm, ax.get_ylim()[1]), fontsize=7, ha="left", va="top"
        )
        ax.set_ylabel(name.upper(), fontsize=7, color=COLOURS.get(name, "k"))
        ax.set_yticks([])
        _style(ax)
    axes[-1].set_xlabel("BPM", fontsize=7)
    axes[0].set_title("spectrum, peak marked", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.12, hspace=0.18)
    return fig


def fig_rolling(history):
    fig = Figure(figsize=(11, 2.4), dpi=85)
    ax = fig.subplots()
    for name, pts in history.items():
        if not pts:
            continue
        t, bpm = zip(*pts)
        ax.plot(t, bpm, lw=1.4, color=COLOURS.get(name, "k"), label=name.upper(), marker="o", ms=2)
    ax.set_ylabel("BPM", fontsize=8)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_title(
        "rolling estimate — under head motion GREEN/ICA diverge, CHROM/POS hold",
        fontsize=9,
    )
    if any(history.values()):
        ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.22)
    return _style(ax).figure


def fig_ica_components(res, fs):
    fig = Figure(figsize=(5, 3.0), dpi=85)
    axes = fig.subplots(3, 1, sharex=True)
    t = np.arange(res.components.shape[0]) / fs
    for i, ax in enumerate(axes):
        picked = i == res.chosen
        ax.plot(t, res.components[:, i], lw=0.9, color="#c2410c" if picked else "#9ca3af")
        ax.set_ylabel(f"IC{i}", fontsize=7)
        ax.set_yticks([])
        ax.text(
            0.99,
            0.85,
            f"concentration {res.scores[i]:.2f}" + ("  <- selected" if picked else ""),
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            color="#c2410c" if picked else "#6b7280",
        )
        _style(ax)
    axes[-1].set_xlabel("s", fontsize=7)
    axes[0].set_title("ICA components — the one the selector chose", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.12, hspace=0.18)
    return fig


def confidence(snr: float) -> tuple:
    """SNR-driven colouring. High SNR means narrowband, which is not the same
    as correct — a method locked onto a motion peak scores well too."""
    if not np.isfinite(snr):
        return "grey", "no estimate"
    if snr >= 3:
        return "#16a34a", "strong"
    if snr >= -3:
        return "#ca8a04", "usable"
    return "#dc2626", "weak"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

st.title("Contactless pulse — four projections, one pipeline")
st.caption(
    "Estimate with reported error bounds, not a clinical measurement. "
    "Every arm is one choice of projection w such that wᵀ[R,G,B]ᵀ suppresses "
    "the specular term and keeps the pulsatile diffuse component."
)

metric_row = st.container()
health_slot = st.empty()
left, right = st.columns([1.05, 1])
with left:
    video_slot = st.empty()
    roi_slot = st.empty()
    raw_slot = st.empty()
    status_slot = st.empty()
with right:
    pulse_slot = st.empty()
    spectra_slot = st.empty()
rolling_slot = st.empty()
with st.expander("ICA components (blind source separation)"):
    ica_slot = st.empty()

# One st.empty() per column, not the column itself: writing into a column
# *appends*, so a once-per-second update would stack 45 readouts down the page
# and push every plot off screen. A placeholder replaces its content.
metric_slots = [c.empty() for c in metric_row.columns(max(len(cfg.methods), 1))]


#: Below this the spatial average stops beating the sensor noise floor down far
#: enough to expose a sub-1% modulation. Chosen from the 1/sqrt(N) argument, not
#: from taste — see roi.py.
MIN_ROI_PIXELS = 8000


def blank_metrics():
    """Wipe the readouts. A stale BPM left on screen reads as a live one."""
    for slot, name in zip(metric_slots, cfg.methods):
        slot.markdown(
            f"<span style='font-size:0.75rem;color:#6b7280'>{name.upper()}</span><br>"
            f"<span style='font-size:2rem;color:#9ca3af'>—</span>",
            unsafe_allow_html=True,
        )
    health_slot.empty()


def render_health(estimates, n_pixels, eff_fps=None, face_rgb=None):
    """The numbers that say whether to believe anything else on screen."""
    spread = agreement_spread(estimates)
    bits = []

    if face_rgb is not None and np.all(np.isfinite(face_rgb)):
        # Luma of the ROI itself, not the frame. A bright background behind the
        # subject makes the camera expose for the wall and leaves the face dark,
        # and a dark face carries proportionally less pulse.
        luma = float(0.299 * face_rgb[0] + 0.587 * face_rgb[1] + 0.114 * face_rgb[2])
        ok = 60.0 <= luma <= 235.0
        note = ""
        if luma < 60.0:
            note = " - face underexposed; add light in front, not behind"
        elif luma > 235.0:
            note = " - face clipping; reduce light"
        bits.append(
            f"<span style='color:{'#16a34a' if ok else '#dc2626'}'>"
            f"face luma {luma:.0f}{note}</span>"
        )

    if eff_fps is not None and np.isfinite(eff_fps):
        # Nyquist for the 4 Hz band edge needs 8 Hz of real samples. Below that
        # the upper half of the band is not measured, it is invented.
        ok = eff_fps >= cfg.min_effective_fps
        bits.append(
            f"<span style='color:{'#16a34a' if ok else '#dc2626'}'>"
            f"{eff_fps:.1f} fps processed"
            f"{'' if ok else ' — too slow; Nyquist needs 8+ fps for a 4 Hz band'}</span>"
        )

    if n_pixels is not None:
        ok = n_pixels >= MIN_ROI_PIXELS
        bits.append(
            f"<span style='color:{'#16a34a' if ok else '#dc2626'}'>"
            f"ROI {n_pixels:,} px{'' if ok else f' — below {MIN_ROI_PIXELS:,}, expect a poor SNR'}"
            "</span>"
        )
    if np.isfinite(spread):
        ok = spread <= 5.0
        bits.append(
            f"<span style='color:{'#16a34a' if ok else '#dc2626'}'>"
            f"arms agree to {spread:.1f} BPM"
            f"{'' if ok else ' — at most one of them can be right'}</span>"
        )
    gaps = [e.max_gap for e in estimates.values() if np.isfinite(e.max_gap)]
    if gaps and max(gaps) > 0.2:
        bits.append(
            f"<span style='color:#ca8a04'>worst sampling gap {max(gaps) * 1000:.0f} ms</span>"
        )
    if bits:
        health_slot.markdown(
            "<div style='font-size:0.78rem'>" + " &nbsp;·&nbsp; ".join(bits) + "</div>",
            unsafe_allow_html=True,
        )


def render_metrics(estimates):
    """Just the four numbers. Markdown only, ~1 ms — safe to call every hop."""
    for slot, name in zip(metric_slots, cfg.methods):
        est = estimates.get(name)
        if est is None:
            slot.markdown(
                f"<span style='font-size:0.75rem;color:#6b7280'>{name.upper()}</span><br>"
                f"<span style='font-size:2rem;color:#9ca3af'>—</span>",
                unsafe_allow_html=True,
            )
            continue
        colour, label = confidence(est.snr)
        if not est.accepted:
            colour = "#dc2626"
        tail = (
            f"SNR {est.snr:+.1f} dB · {label}"
            if est.accepted
            else f"REJECTED — {est.reject_reason}"
        )
        slot.markdown(
            f"<div style='line-height:1.15'>"
            f"<span style='font-size:0.75rem;color:#6b7280'>{name.upper()}</span><br>"
            f"<span style='font-size:2rem;font-weight:600;color:{colour}'>"
            f"{est.bpm_smoothed:.1f}</span>"
            f"<span style='font-size:0.8rem;color:#6b7280'> BPM</span><br>"
            f"<span style='font-size:0.7rem;color:{colour}'>{tail}</span></div>",
            unsafe_allow_html=True,
        )


#: Panels are redrawn one per tick, in rotation. Drawing all five at once blocks
#: the capture loop for ~440 ms locally and considerably longer through
#: Streamlit's encode-and-send path — and that block is not merely slow, it is a
#: real hole in the sampling record. The gap gate then correctly rejects every
#: window containing one, so the app renders beautiful plots and reports nothing.
#: One panel per tick keeps each block near 100 ms, well under max_gap_sec.
PLOT_ROTATION = ("raw", "pulses", "spectra", "rolling", "ica")


def draw_panel(which, *, estimates, history, t, rgb, seg, note=""):
    """Redraw exactly one panel. Data is computed lazily, only for that panel."""
    if which == "raw" and len(t):
        raw_slot.pyplot(fig_raw_rgb(t, rgb), clear_figure=True)
        if note:
            status_slot.caption(note)
    elif which == "pulses" and seg is not None:
        signals = {n: project_and_filter(seg, fs, n, cfg) for n in cfg.methods}
        pulse_slot.pyplot(fig_pulses(signals, fs), clear_figure=True)
    elif which == "spectra":
        # Free: the estimator already computed these spectra this hop.
        results = {n: e.spectrum for n, e in estimates.items() if e.spectrum is not None}
        if results:
            spectra_slot.pyplot(fig_spectra(results), clear_figure=True)
    elif which == "rolling":
        rolling_slot.pyplot(fig_rolling(history), clear_figure=True)
    elif which == "ica" and seg is not None and "ica" in cfg.methods:
        ica_slot.pyplot(
            fig_ica_components(ica_full(seg, fs, detrend_method=cfg.detrend_method), fs),
            clear_figure=True,
        )


def save_session(t, rgb, note=""):
    """Write the timestamped spatial means so a bad run can be diagnosed offline.

    The RGB means are the entire input to the DSP. With them plus a reference
    heart rate, any disagreement is reproducible away from the camera — which
    is the difference between debugging and guessing.
    """
    import pandas as pd

    out = Path("data") / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame({"t": t, "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2]})
    header = "# reference_bpm=%s\n# fps_nominal=%s\n" % (note or "unknown", fs)
    with path.open("w", newline="") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return path


def run_stream(
    frames,
    show_video: bool,
    roi_extractor,
    max_seconds: float | None = None,
    realtime: bool = True,
):
    estimator = SlidingWindowEstimator(cfg)
    history = {m: deque(maxlen=400) for m in cfg.methods}
    raw_t, raw_rgb = deque(maxlen=int(window_sec * fs * 2)), deque(maxlen=int(window_sec * fs * 2))
    last_draw = 0.0
    last_plot = 0.0
    plot_tick = 0
    plot_period = MIN_PLOT_PERIOD
    n_pixels = None
    last_rgb = None
    lost = 0
    session_t, session_rgb = [], []
    seen_gaps = 0

    def _consume(sample):
        """Everything that must happen for *every* sample, drained or not."""
        nonlocal history, seen_gaps, n_pixels, last_rgb, lost
        if sample.ok and np.all(np.isfinite(sample.rgb)):
            estimator.push(sample.timestamp, sample.rgb)
            if estimator.gaps_seen != seen_gaps:
                # The estimator just discarded its buffer; the plots and
                # readouts on screen describe data that no longer exists.
                seen_gaps = estimator.gaps_seen
                raw_t.clear()
                raw_rgb.clear()
                history = {m: deque(maxlen=400) for m in cfg.methods}
                blank_metrics()
            raw_t.append(sample.timestamp)
            raw_rgb.append(sample.rgb)
            session_t.append(sample.timestamp)
            session_rgb.append(sample.rgb)
            # 0 means "no image behind this sample" (the synthetic source), not
            # "found no skin" — that case leaves sample.ok False and lands below.
            n_pixels = sample.n_pixels or None
            last_rgb = sample.rgb
        else:
            # Tell the estimator the face was genuinely absent, so it can tell
            # a real dropout apart from the loop merely running slowly.
            estimator.mark_lost(sample.timestamp)
            lost += 1
            status_slot.warning(
                f"face not detected — ROI lost ({lost} frames). A loss longer "
                f"than {cfg.max_gap_sec:.1f} s resets the buffer.",
                icon="⚠️",
            )

    # Capture and ROI run on their own thread. Draining a batch means a slow
    # redraw costs latency, not samples: the alternative — extracting the ROI
    # inline — put every Streamlit render directly into the sampling grid, and
    # produced 495 ms holes at 21.6 fps while the camera was delivering 30.
    sampler = ThreadedSampler(frames, roi_extractor, drop_when_full=realtime)
    stop = False
    while not stop:
        batch = sampler.drain()
        if not batch:
            if not sampler.running:
                break
            continue
        if sampler.error is not None:
            raise sampler.error

        now = time.perf_counter()
        if show_video and now - last_draw > PREVIEW_PERIOD:
            # last_draw MUST be updated here, not where the estimate is drawn:
            # the estimate only happens once the buffer is full, so throttling
            # on it means no throttling at all until then — which is exactly
            # when the extra load prevents the buffer from ever filling.
            last_draw = now
            newest = sampler.latest()
            if newest is not None:
                frame, overlay_sample = newest
                preview = draw_overlay(frame.image, overlay_sample)
                # Downscale and JPEG: a full-resolution PNG costs ~70 ms to
                # encode and ~190 KB per frame, which alone exceeds the 33 ms
                # frame budget. This is ~0.3 ms and ~5 KB.
                scale = PREVIEW_WIDTH / preview.shape[1]
                if scale < 1.0:
                    preview = cv2.resize(
                        preview, (PREVIEW_WIDTH, int(preview.shape[0] * scale))
                    )
                video_slot.image(
                    preview[:, :, ::-1],
                    channels="RGB",
                    use_container_width=True,
                    output_format="JPEG",
                )

        for sample in batch:
            if max_seconds is not None and sample.timestamp > max_seconds:
                stop = True
                break
            _consume(sample)
        if stop:
            break

        if estimator.due():
            estimates = estimator.estimate()
            t_arr = np.array(raw_t)
            rgb_arr = np.array(raw_rgb)
            for name in cfg.methods:
                if estimates[name].accepted:
                    history[name].append((estimates[name].t_end, estimates[name].bpm_smoothed))

            # The numbers are cheap (markdown, ~1 ms), so they update every hop.
            render_metrics(estimates)
            render_health(estimates, n_pixels, estimator.effective_fps, last_rgb)

            # The figures are not. Five matplotlib redraws cost ~1200 ms against
            # ~113 ms for all of the DSP — 12x the signal processing — and at a
            # 1 s hop that alone drags the capture loop down to ~1 fps, which
            # aliases every heart rate. So they redraw on their own, slower
            # cadence, and back off further if the loop is still behind.
            now = time.perf_counter()
            eff = estimator.effective_fps
            if np.isfinite(eff) and eff < cfg.min_effective_fps * 1.5:
                plot_period = min(plot_period * 1.5, MAX_PLOT_PERIOD)
            else:
                plot_period = max(plot_period * 0.9, MIN_PLOT_PERIOD)

            if now - last_plot < plot_period:
                continue
            last_plot = now

            which = PLOT_ROTATION[plot_tick % len(PLOT_ROTATION)]
            plot_tick += 1

            seg = None
            if which in ("pulses", "ica"):
                _, rgb_u = resample_uniform(t_arr, rgb_arr, fs=fs)
                seg = rgb_u[-int(window_sec * fs) :]
            stats = frame_interval_stats(t_arr) if len(t_arr) > 2 else None

            draw_panel(
                which,
                estimates=estimates,
                history=history,
                t=t_arr,
                rgb=rgb_arr,
                seg=seg,
                note=f"capture {stats}" if stats else "",
            )
            # Deliberately NOT touching last_draw here. The preview throttle owns
            # its own timestamp at the point it draws; coupling the two is what
            # made the throttle dead in the first place.
        elif not estimator.ready:
            if estimator.too_slow and estimator.buffered_sec > 2.0:
                status_slot.error(
                    f"Only {estimator.effective_fps:.1f} usable frames per second. "
                    "Nyquist needs 8+ fps for a 4 Hz band, so every heart rate is "
                    "aliased below that — the numbers would be arithmetic on noise. "
                    "**Most likely another program has the camera** (another browser "
                    "tab, a video call, or a second copy of this app); a shared "
                    "webcam drops to ~1 fps. Otherwise: add light, or try the "
                    "'haar' ROI backend. Run `python -m rppg.diagnose` to find out "
                    "which.",
                    icon="🐢",
                )
            elif estimator.gaps_seen:
                # One clear cause-and-effect line, rather than the same
                # rejection repeated once per arm.
                status_slot.warning(
                    f"Face lost for {estimator.last_gap:.1f} s — buffer reset. "
                    f"Next estimate in {estimator.refill_remaining:.0f} s. "
                    f"({estimator.gaps_seen} dropout"
                    f"{'s' if estimator.gaps_seen > 1 else ''} so far — "
                    "keep your face in frame and unobstructed.)",
                    icon="🔄",
                )
            else:
                status_slot.info(
                    f"filling buffer — {estimator.buffered_sec:.1f} / {window_sec:.0f} s",
                    icon="⏳",
                )

    sampler.close()
    return np.array(session_t), np.array(session_rgb)


def fig_beats(t, x, peaks, seconds=10.0, fs=125.0):
    """Waveform with detected beats — the plot that shows the notch is avoided."""
    fig = Figure(figsize=(9, 2.2), dpi=90)
    ax = fig.subplots()
    n = int(seconds * fs)
    ax.plot(t[:n], x[:n], lw=1.0, color="#1d4ed8")
    pk = peaks[peaks < n]
    ax.plot(t[pk], x[pk], "v", ms=6, color="#dc2626", label="detected beat")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_xlabel("s", fontsize=7)
    ax.set_title("PPG with beat detections (first %.0f s)" % seconds, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22)
    return _style(ax).figure


def fig_hrv(ibi):
    """Tachogram, Poincare and the HRV spectrum with the Task Force bands."""
    fig = Figure(figsize=(9, 2.4), dpi=90)
    axes = fig.subplots(1, 3)
    axes[0].plot(np.cumsum(ibi) / 1000.0, ibi, lw=0.8, color="#1d4ed8")
    axes[0].set_xlabel("s", fontsize=7)
    axes[0].set_ylabel("NN (ms)", fontsize=7)
    axes[0].set_title("tachogram", fontsize=8)

    axes[1].scatter(ibi[:-1], ibi[1:], s=4, alpha=0.4, color="#7c3aed")
    axes[1].set_xlabel("NN$_i$", fontsize=7)
    axes[1].set_ylabel("NN$_{i+1}$", fontsize=7)
    axes[1].set_title("Poincare", fontsize=8)

    grid, series = tachogram(ibi)
    if len(series) > 8:
        f, p = periodogram(series - series.mean(), 4.0, zero_pad=4)
        m = f <= 0.5
        axes[2].semilogy(f[m], p[m], lw=1, color="#16a34a")
        axes[2].axvspan(0.04, 0.15, alpha=0.12, color="#1d4ed8")
        axes[2].axvspan(0.15, 0.40, alpha=0.12, color="#dc2626")
    axes[2].set_xlabel("Hz", fontsize=7)
    axes[2].set_title("HRV spectrum (LF | HF)", fontsize=8)
    for a in axes:
        _style(a)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.24, wspace=0.35)
    return fig


def fig_respiration(routes):
    """The three modulations, each labelled with the rate it recovered."""
    fig = Figure(figsize=(9, 3.0), dpi=90)
    axes = fig.subplots(3, 1, sharex=True)
    colours = {"riiv": "#b91c1c", "rifv": "#1d4ed8", "riav": "#7c3aed"}
    titles = {
        "riiv": "RIIV — baseline (venous return)",
        "rifv": "RIFV — intervals (sinus arrhythmia)",
        "riav": "RIAV — pulse height (stroke volume)",
    }
    for ax, key in zip(axes, ("riiv", "rifv", "riav")):
        sig, sig_fs, est = routes[key]
        if len(sig):
            ax.plot(np.arange(len(sig)) / sig_fs, sig, lw=0.9, color=colours[key])
        label = (
            "%s  ->  %.1f br/min" % (titles[key], est.rate_bpm)
            if np.isfinite(est.rate_bpm)
            else "%s  ->  no estimate" % titles[key]
        )
        ax.set_title(label, fontsize=8)
        _style(ax)
    axes[-1].set_xlabel("s", fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.14, hspace=0.55)
    return fig


def run_ppg_record(record, start_sec, length_sec):
    """Contact PPG analysis with the clinical reference alongside."""
    import wfdb

    from rppg.bidmc import load

    ppg, fs, hr_ref = load(record, BIDMC_ROOT)
    num = wfdb.rdrecord(str(BIDMC_ROOT / (record + "n")))
    nnames = [s.strip().rstrip(",") for s in num.sig_name]
    rr_ref = num.p_signal[:, nnames.index("RESP")]

    a, b = int(start_sec * fs), int((start_sec + length_sec) * fs)
    seg = np.nan_to_num(ppg[a:b], nan=float(np.nanmean(ppg)))
    if len(seg) < int(20 * fs):
        st.error("Segment too short — reduce the start time or increase the length.")
        return
    t = np.arange(len(seg)) / fs

    peaks = detect_beats(seg, fs)
    ibi, _ = clean_intervals(intervals(peaks, fs))
    hrv = summary(ibi)
    routes = {}
    resp = analyse(seg, fs, peaks=peaks)
    routes["riiv"] = riiv(seg, fs) + (resp["riiv"],)
    routes["rifv"] = rifv(peaks, fs) + (resp["rifv"],)
    routes["riav"] = riav(seg, peaks, fs) + (resp["riav"],)

    ref_hr = float(np.nanmean(hr_ref[int(start_sec) : int(start_sec + length_sec)]))
    window = rr_ref[int(start_sec) : int(start_sec + length_sec)]
    ref_rr = float(np.nanmean(window)) if np.any(np.isfinite(window)) else float("nan")

    # Frequency-domain HR on the same segment, at the webcam rate, so the two
    # estimators are compared on identical data.
    from rppg.bidmc import to_webcam_rate

    x30, _ = to_webcam_rate(seg, fs)
    spec = estimate_bpm(bandpass(detrend(x30, fs=30.0), 30.0), 30.0, selector=selector)

    cols = st.columns(4)
    cols[0].metric("HR — beats", "%.1f" % hrv["mean_hr_bpm"],
                   "%+.2f vs ECG" % (hrv["mean_hr_bpm"] - ref_hr))
    cols[1].metric("HR — spectrum", "%.1f" % spec.bpm,
                   "%+.2f vs ECG" % (spec.bpm - ref_hr))
    cols[2].metric("ECG reference", "%.1f BPM" % ref_hr)
    fused = resp["fused"]
    if np.isfinite(fused.rate_bpm):
        delta = ("%+.2f vs ref" % (fused.rate_bpm - ref_rr)) if np.isfinite(ref_rr) else None
        cols[3].metric("Respiratory rate", "%.1f br/min" % fused.rate_bpm, delta)
    else:
        cols[3].metric("Respiratory rate", "no agreement")

    st.pyplot(fig_beats(t, seg, peaks, seconds=min(10.0, length_sec), fs=fs))

    st.subheader("Heart rate variability")
    if not hrv["hrv_reliable"]:
        st.warning(
            "Segment is %.0f s. The Task Force standard wants 2 minutes before LF "
            "means anything and 5 for LF/HF, so the spectral figures below are "
            "shown but should not be quoted." % hrv["duration_sec"],
            icon="⏱️",
        )
    h = st.columns(5)
    h[0].metric("mean NN", "%.0f ms" % hrv["mean_nn_ms"])
    h[1].metric("SDNN", "%.1f ms" % hrv["sdnn_ms"], help="total variability")
    h[2].metric("RMSSD", "%.1f ms" % hrv["rmssd_ms"], help="beat-to-beat, vagal")
    h[3].metric("pNN50", "%.1f %%" % hrv["pnn50_pct"])
    h[4].metric("LF/HF", "%.2f" % hrv["lf_hf"] if np.isfinite(hrv["lf_hf"]) else "—")
    st.pyplot(fig_hrv(ibi))
    st.caption(
        "PPG-derived RMSSD runs systematically above the ECG value. Part of that "
        "is real — pulse transit time varies beat to beat with blood pressure, "
        "which is why the literature separates *pulse rate* variability from "
        "*heart rate* variability — and part is residual detector jitter. The "
        "two have not been separated here."
    )

    st.subheader("Respiratory rate — three modulations of one carrier")
    st.pyplot(fig_respiration(routes))
    st.caption(
        "Fused by **agreement**, not by SNR: picking the highest-SNR route scored "
        "3.42 br/min MAE on BIDMC, worse than RIIV alone at 2.66, because a "
        "narrowband peak in the wrong place still scores well. Two routes "
        "agreeing gives 1.99 at 65 % coverage; all three gives 1.16 at 25 %. "
        "RIFV and RIAV are sampled once per beat, so at %.0f BPM their Nyquist "
        "limit is %.0f br/min." % (hrv["mean_hr_bpm"], hrv["mean_hr_bpm"] / 2)
    )


if start:
    if mode == "PPG record (BIDMC)":
        if bidmc_record is None:
            st.error("No BIDMC record selected. Download the dataset first.")
            st.stop()
        st.info(
            "Contact PPG with a clinical reference — this is the validated core "
            "of the project. Heart rate from beats agrees with ECG to 0.039 BPM "
            "across ten records; see notebook 02.",
            icon="🫀",
        )
        run_ppg_record(bidmc_record, bidmc_start, bidmc_len)

    elif mode == "Synthetic (no camera)":
        from rppg.capture import Frame
        from rppg.synthetic import nonuniform_timestamps, synth_rgb

        kw = CONDITIONS[condition]
        st.info(
            f"Synthetic clip, condition **{condition}** — no camera involved. "
            + (
                "Watch the rolling strip: GREEN and ICA should lock onto the "
                f"{kw['motion_bpm']:.0f} BPM motion while CHROM and POS hold at 72."
                if "motion_bpm" in kw
                else "Ground truth is "
                + (
                    f"{kw['bpm'][0]:.0f} → {kw['bpm'][1]:.0f} BPM."
                    if isinstance(kw["bpm"], tuple)
                    else f"{kw['bpm']:.0f} BPM."
                )
            )
        )
        ts = nonuniform_timestamps(60.0, fs, jitter_ms=3.0, rng=0)
        clip = synth_rgb(timestamps=ts, rng=0, **kw)

        frames = (Frame(i, t, np.zeros((2, 2, 3), np.uint8)) for i, t in enumerate(clip.t))
        lookup = {round(t, 6): rgb for t, rgb in zip(clip.t, clip.rgb)}

        def _synthetic_roi(frame):
            # The real RoiSample rather than a look-alike: a hand-rolled shim
            # silently goes stale the moment RoiSample gains a field, and the
            # synthetic path is exactly the one nobody re-tests by hand.
            # n_pixels = 0 because there is no image to average over.
            return RoiSample(
                timestamp=frame.timestamp,
                rgb=lookup[round(frame.timestamp, 6)],
                n_pixels=0,
                ok=True,
            )

        run_stream(
            frames, show_video=False, roi_extractor=_synthetic_roi, realtime=False
        )
        st.success(
            f"Synthetic run complete — condition {condition}."
            + (
                " Compare the arms against each other, not against 72: under motion "
                "GREEN and ICA are expected to be wrong, and confidently so."
                if "motion_bpm" in kw
                else " Every arm should track the ground truth above."
            )
        )

    else:
        try:
            roi_extractor = make_roi(backend, **roi_kw)
        except RuntimeError as exc:
            st.warning(f"{exc}")
            roi_extractor = make_roi("haar", **roi_kw)

        if mode == "Webcam":
            if device is None:
                st.error("No camera to open. Pick *Synthetic* to run without one.")
                st.stop()
            try:
                cam = ThreadedWebcamSource(int(device))
            except RuntimeError as exc:
                # Surface the actionable message instead of a traceback.
                st.error(str(exc), icon="📷")
                st.stop()
            with cam:
                if cam.exposure_revert_reason:
                    st.info(f"Auto-exposure left on: {cam.exposure_revert_reason}", icon="🎚️")
                elif not cam.exposure_locked:
                    st.warning(
                        "Could not lock auto-exposure on this backend. It is a closed "
                        "loop driven by scene brightness — the same sub-percent changes "
                        "we are measuring — so expect degraded results.",
                        icon="⚠️",
                    )
                st_t, st_rgb = run_stream(
                    cam, True, roi_extractor, max_seconds=record_seconds
                )
                if record and len(st_t) > 30:
                    st.success(f"session saved to {save_session(st_t, st_rgb, record_note)}")
        else:
            if upload is None:
                st.error("Upload a clip first.")
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir) / upload.name
                    tmp.write_bytes(upload.getbuffer())
                    with VideoFileSource(tmp) as src:
                        st_t, st_rgb = run_stream(src, True, roi_extractor, realtime=False)
                        if record and len(st_t) > 30:
                            st.success(
                                f"session saved to {save_session(st_t, st_rgb, record_note)}"
                            )
                st.success("Clip complete.")
else:
    st.info(
        "Pick a source in the sidebar and press **Start**. "
        "No camera? Choose *Synthetic* — it runs the identical DSP on a clip "
        "with a known 72 BPM pulse.",
        icon="👈",
    )
