"""Smoke tests for the Streamlit demo.

Run in a subprocess: importing the app executes Streamlit's module-level UI
calls, which pollute global state and emit a warning per widget. Isolating it
keeps the rest of the suite clean.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"

SMOKE = r"""
import importlib.util, sys
sys.argv = ["streamlit_app.py"]
spec = importlib.util.spec_from_file_location("stapp", r"{app}")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from rppg.synthetic import synth_rgb
from rppg.pipeline import PipelineConfig, SlidingWindowEstimator, project_and_filter
from rppg.methods import ica_full
from rppg.spectral import estimate_bpm

clip = synth_rgb(duration=20.0, fs=30.0, bpm=72.0, rng=0)
cfg = PipelineConfig(fs=30.0, window_sec=15.0)
est = SlidingWindowEstimator(cfg)
for t, rgb in zip(clip.t, clip.rgb):
    est.push(t, rgb)
out = est.estimate()

seg = clip.rgb[:450]
signals = {{k: project_and_filter(seg, 30.0, k, cfg) for k in cfg.methods}}
results = {{k: estimate_bpm(v, 30.0) for k, v in signals.items()}}

m.fig_raw_rgb(clip.t, clip.rgb)
m.fig_pulses(signals, 30.0)
m.fig_spectra(results)
m.fig_rolling({{k: [(1.0, 72.0), (2.0, 72.5)] for k in cfg.methods}})
m.fig_ica_components(ica_full(seg, 30.0), 30.0)

# The health strip: healthy, starved ROI, and no-pixel-count-yet.
m.render_health(out, 25000)
m.render_health(out, 900)
m.render_health(out, None)

# Drive run_stream end to end over the synthetic source. This is the path that
# broke when RoiSample gained a field and the app's look-alike shim did not:
# every figure function passed, and the app still crashed on the first frame.
from rppg.capture import Frame
from rppg.roi import RoiSample

short = synth_rgb(duration=20.0, fs=30.0, bpm=72.0, rng=0)
lookup = {{round(t, 6): rgb for t, rgb in zip(short.t, short.rgb)}}
frames = (Frame(i, t, __import__("numpy").zeros((2, 2, 3), "uint8"))
          for i, t in enumerate(short.t))
m.run_stream(
    frames,
    show_video=False,
    roi_extractor=lambda f: RoiSample(
        timestamp=f.timestamp, rgb=lookup[round(f.timestamp, 6)], n_pixels=0, ok=True
    ),
)

# The contact-PPG figures. Built from a synthesised PPG rather than BIDMC so
# the test runs without the dataset, but through exactly the app's own code.
import numpy as _np
from rppg.beats import detect_beats, intervals, clean_intervals
from rppg.hrv import summary as _hrv_summary
from rppg.respiration import analyse as _resp_analyse, riiv as _riiv, rifv as _rifv, riav as _riav

_fs = 125.0
_t = _np.arange(0, 120.0, 1 / _fs)
_beats = _np.arange(0, 120.0, 0.8)
_ppg = _np.zeros_like(_t)
for _b in _beats:
    _u = (_t - _b) / 0.8
    _mm = (_u >= 0) & (_u < 1)
    _ppg[_mm] += _np.exp(-((_u[_mm] - 0.15) ** 2) / (2 * 0.055 ** 2))
    _ppg[_mm] += 0.3 * _np.exp(-((_u[_mm] - 0.40) ** 2) / (2 * 0.06 ** 2))
_ppg += 0.25 * _np.sin(2 * _np.pi * 0.25 * _t)

_pk = detect_beats(_ppg, _fs)
_ibi, _ = clean_intervals(intervals(_pk, _fs))
assert len(_ibi) > 100, "beat detection produced too few intervals for the app panels"
_s = _hrv_summary(_ibi)
assert 70 < _s["mean_hr_bpm"] < 80

_resp = _resp_analyse(_ppg, _fs, peaks=_pk)
_routes = {{
    "riiv": _riiv(_ppg, _fs) + (_resp["riiv"],),
    "rifv": _rifv(_pk, _fs) + (_resp["rifv"],),
    "riav": _riav(_ppg, _pk, _fs) + (_resp["riav"],),
}}

m.fig_beats(_t, _ppg, _pk, seconds=10.0, fs=_fs)
m.fig_hrv(_ibi)
m.fig_respiration(_routes)

# A route that returned nothing must not break the figure — RIFV vanishes in
# patients without respiratory sinus arrhythmia, and the panel still renders.
from rppg.respiration import RespEstimate as _RE
_routes["rifv"] = (_np.array([]), 4.0, _RE(float("nan"), float("nan"), "rifv"))
m.fig_respiration(_routes)

assert m.confidence(10.0)[1] == "strong"
assert m.confidence(float("nan"))[1] == "no estimate"
assert all(abs(out[m].bpm - 72.0) < 3.0 for m in ("green", "ica", "pos"))

# Every advertised condition must actually build a clip. These are the four
# benchmark conditions the sidebar offers; a bad kwarg here is a crash the
# moment someone picks it on stage.
assert set(m.CONDITIONS) == {{
    "A — still", "B — motion", "C — illumination", "D — post-exercise"
}}
for name, kw in m.CONDITIONS.items():
    c = synth_rgb(duration=20.0, fs=30.0, rng=0, **kw)
    assert c.rgb.shape[1] == 3 and len(c.t) > 500, name
    assert __import__("numpy").isfinite(c.rgb).all(), name

print("APP_SMOKE_OK")
"""


def test_app_imports_and_renders_every_figure():
    proc = subprocess.run(
        [sys.executable, "-c", SMOKE.format(app=str(APP))],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(APP.parent.parent),
    )
    assert proc.returncode == 0, f"app smoke failed:\n{proc.stdout}\n{proc.stderr}"
    assert "APP_SMOKE_OK" in proc.stdout
