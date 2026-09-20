# Pulse Signal Analysis using PPG

*DTSP mini-project #16.*

Classical DSP on the photoplethysmogram: heart rate in both the frequency and
time domains, heart rate variability, and respiratory rate — each **validated
against a clinical reference**, not against itself.

The extension chapter asks whether the same pulse can be recovered from an
ordinary webcam (rPPG) and benchmarks four projection methods for doing so.

## Headline results

Against **BIDMC** (PhysioNet): 53 ICU recordings carrying PPG at 125 Hz, ECG
lead II, impedance respiration, and ECG-derived reference heart and respiratory
rates. The PPG is decimated to 30 Hz — a webcam's frame rate — so these numbers
also bound what the camera path could achieve with perfect capture.

| measurement | result | reference | notebook |
|---|---|---|---|
| Heart rate, frequency domain | **0.80 BPM** MAE | ECG-derived HR | [02](notebooks/02_ground_truth_validation.ipynb) |
| Beat detection | **F1 0.972** | Pan–Tompkins on ECG | 02 |
| Heart rate, time domain | **0.039 BPM** | ECG R–R intervals | 02 |
| HRV (SDNN/RMSSD/pNN50/LF-HF) | RMSSD ±7.7 ms | ECG R–R | 02 |
| Respiratory rate | **1.99 br/min** at 65 % coverage | impedance pneumography | 02 |
| | **1.16 br/min** at 25 % coverage | | 02 |

```bash
python -m rppg.bidmc --download    # open access, no request form
python -m rppg.bidmc               # reproduces the heart-rate table
python -m rppg.session <csv>       # post-mortem on a recorded session
```

Design rationale, physics and the rPPG benchmark protocol live in
[rPPG_Ideation_Doc.md](rPPG_Ideation_Doc.md). This file is how to run it.

## Three findings the ground truth forced

1. **The SNR gate was set by taste and was wrong.** At −7 dB it admitted 96 % of
   windows carrying a **9.12 BPM** mean error — confident numbers from a regime
   where the expected error exceeds the width of the cardiac band. Calibrated on
   18,640 windows, the knee is sharp and sits at −3 dB (59 % kept, 1.80 BPM).
   That single change explains most of the live app's plausible wrong answers.

2. **Fiducial choice dominates HRV.** RMSSD is built from differences between
   consecutive intervals, so snapping beats to the rounded systolic *peak*
   inflated it 2–3× (50.9 ms against ECG's 13.5) as pure detector jitter.
   Keying on the steepest *upslope* roughly halved the error.

3. **Choosing by SNR is worse than not choosing.** For respiratory rate, picking
   the highest-SNR of three routes scored 3.42 — worse than the best single
   route at 2.66. A narrowband peak in the wrong place scores well, so SNR
   measures confidence, not correctness. Fusion by **agreement** instead gives
   1.99 at 65 % coverage.

**Two additions beyond the ideation doc**, both because measurement demanded
them rather than because they sounded good:

- **Harmonic-sum peak selection** ([spectral.py](rppg/spectral.py)). A cardiac
  pulse has real energy at 2f and 3f; a periodic motion artefact usually does
  not. Scoring the harmonic series instead of the tallest single bin separates
  them on a property a single-peak search is blind to — and fixes CHROM
  reporting exactly double the true rate.
- **Viterbi heart-rate tracking** ([tracking.py](rppg/tracking.py)). Picking a
  peak per window throws away the strongest prior available: *a heart rate
  cannot jump*. Choosing the best frequency **path** through the spectrogram,
  under a physiological slew limit, outvotes windows where a harmonic briefly
  looks taller. A median filter cannot do this — doubling is not a statistical
  outlier, it is a confident repeatable wrong answer.

Not a medical device. The output is an estimate with reported error bounds.

---

## Quick start

```bash
pip install -r requirements.txt
python -m rppg.roi --download-model      # face_landmarker.task, ~3.7 MB
pytest -q                                # 163 tests, ~2 min
```

**Before trusting any webcam number, measure the capture chain.** This is not
optional advice — most "the heart rate is wrong" failures are not DSP failures:

```bash


           # live ROI overlay + pixel count
```

### The single most common failure

**Another program holding the camera.** A shared webcam does not fail loudly —
it drops to about **1 frame per second**. At 1 fps Nyquist is 30 BPM, below the
bottom of the cardiac band, so *every* heart rate is aliased and the app shows
confident nonsense. Measured on this machine:

| | camera shared | camera exclusive |
|---|---|---|
| DSHOW | 1.0 fps | 16.0 fps |
| MSMF | 0 frames | 29.8 fps |

Close other browser tabs with camera permission, video-call apps, and any
leftover copy of the Streamlit app before recording. `rppg.diagnose` reports
this in the first ten seconds; the app's health strip shows `fps processed`
live.

No camera and no dataset needed to see the whole pipeline work:

```bash
python -m rppg.run --demo
```

```
capture: 1350 frames | mean 30.07 fps (dt 33.26 +/- 2.95 ms, ...) | ~0 dropped

15 s window => 4.0 BPM raw resolution
          BPM  spread  SNR_dB  accepted  windows
ica     71.97    0.08   13.11       1.0       30
green   71.94    0.09   12.06       1.0       30
pos     71.87    0.27    5.51       1.0       30
chrom   71.86    0.51    0.55       1.0       30
```

Then the notebooks:

```bash
jupyter lab notebooks/01_signal_walkthrough.ipynb        # every stage plotted, known 72 BPM
jupyter lab notebooks/02_ground_truth_validation.ipynb   # BIDMC vs ECG — the real numbers
jupyter lab notebooks/03_ablations.ipynb                 # every design choice, measured (~15 min)
```

Live demo:

```bash
streamlit run app/streamlit_app.py        # webcam, video file, or synthetic
python -m rppg.run --source 0 --seconds 60 --show
python -m rppg.run --source clip.mp4 --save-results data/out.csv
```

---

## What is implemented

| Stage | Module | Notes |
|---|---|---|
| Capture + timestamps | [capture.py](rppg/capture.py) | `perf_counter()` per frame at grab time; threaded grabber; exposure lock verified by measurement and reverted if it costs frame rate |
| Threaded sampling | [sampler.py](rppg/sampler.py) | Capture + ROI off the UI thread, so a slow redraw costs latency, not samples |
| ROI + skin mask | [roi.py](rppg/roi.py) | MediaPipe landmarks (primary), Haar+CSRT (fallback), fixed (tests); YCrCb skin mask |
| Resample | [resample.py](rppg/resample.py) | Cubic spline onto a uniform grid; jitter statistics |
| Detrend | [preprocess.py](rppg/preprocess.py) | Smoothness priors (λ=100) or moving average |
| Bandpass | [preprocess.py](rppg/preprocess.py) | 4th-order Butterworth, 0.7–4.0 Hz, zero-phase `filtfilt` |
| Projections | [methods/](rppg/methods/) | GREEN, ICA, CHROM, POS |
| Spectrum | [spectral.py](rppg/spectral.py) | Hann, 8× zero-pad, parabolic interpolation, Welch, **harmonic-sum selection** |
| HR tracking | [tracking.py](rppg/tracking.py) | **Viterbi path through the spectrogram** under a physiological slew limit |
| **Beat detection** | [beats.py](rppg/beats.py) | Slope Sum Function; upstroke fiducial; adaptive refractory seeded from the spectrum |
| **ECG reference** | [ecg.py](rppg/ecg.py) | Pan–Tompkins, implemented not imported; beat matching |
| **HRV** | [hrv.py](rppg/hrv.py) | Task Force (1996): SDNN, RMSSD, pNN50, LF/HF on an interpolated tachogram |
| **Respiratory rate** | [respiration.py](rppg/respiration.py) | RIIV / RIFV / RIAV demodulation, agreement-based fusion |
| **Ground truth** | [bidmc.py](rppg/bidmc.py) | BIDMC loader, anti-aliased decimation, SNR operating curve |
| **Session post-mortem** | [session.py](rppg/session.py) | Is the pulse in this recording at all? Projection-bound SNR |
| Quality | [quality.py](rppg/quality.py) | de Haan & Jeanne SNR — no ground truth needed |
| Sliding window | [pipeline.py](rppg/pipeline.py) | 15 s / 1 s hop, median-of-5, SNR gate, gap gate, dropout recovery |
| Metrics | [metrics.py](rppg/metrics.py) | MAE, RMSE, MAPE, Pearson r, Bland–Altman |
| Datasets | [datasets.py](rppg/datasets.py) | UBFC-rPPG loader, window↔reference alignment |
| Diagnostics | [diagnose.py](rppg/diagnose.py) | measures the capture chain and says what to fix |
| Synthetic ground truth | [synthetic.py](rppg/synthetic.py) | Known-BPM clips; the week-3 sanity check |

Batch and live share one code path, so a number shown on stage comes from
exactly the same DSP as a number in the results table — asserted by
`test_live_and_batch_paths_agree`.

### Not yet built, and not yet verified

**The colour projections are unvalidated against ground truth.** GREEN, ICA,
CHROM and POS map three colour channels to one, and contact PPG has no colour
channels — BIDMC cannot test them. Every number for those four still comes from
synthetic clips. **Send the UBFC-rPPG access request**; notebook 04 (stress
conditions) is blocked on it.

The contact-PPG chain, by contrast, is validated end to end in
[notebook 02](notebooks/02_ground_truth_validation.ipynb).

Two known failures, both left in every table rather than excluded:

- **`bidmc03`** — beat detection F1 0.851. Its ECG RMSSD of 68.6 ms with SDNN
  49.5 ms strongly suggests an arrhythmia.
- **`bidmc40`** — 13.02 BPM MAE on frequency-domain HR, with a mean SNR of
  +5.0 dB against +10 to +19 for the rest.

And one unresolved question: **PPG-derived RMSSD runs systematically high** even
after the fiducial fix. Part is real — pulse transit time varies beat to beat
with blood pressure, which is why the literature separates *pulse rate*
variability from *heart rate* variability. Part is residual detector jitter.
The two have not been separated here.

Two things are written but unexercised, because this machine has had no camera
and no face video pointed at it:

- **The MediaPipe ROI on a real face.** The model bundle downloads and the
  Tasks API initialises, but the landmark indices in `roi.py` that build the
  forehead and cheek polygons have not been checked against an actual face.
  Run `python -m rppg.roi --preview` first and confirm the polygons sit where
  they should before trusting any webcam number.
- **The Haar + CSRT fallback**, for the same reason.

Everything downstream of the spatial mean is tested end to end, including on a
real `.mp4` decoded through `VideoFileSource`.

---

## Results so far (synthetic, four conditions)

MAE in BPM against known ground truth, 15 s window. **old** = tallest-bin peak
pick + median filter, CHROM filtering per-window. **new** = harmonic-sum
selection + Viterbi tracking, CHROM filtering globally.

| condition | method | old | new |
|---|---|---|---|
| A still | green / ica / pos | 0.24 / 0.26 / 0.39 | 0.23 / 0.20 / 0.36 |
| A still | **chrom** | 8.67 | **2.31** |
| B motion | green | 18.45 | **12.19** |
| B motion | **ica** | 20.06 | **1.71** |
| B motion | **chrom** | 13.65 | **3.18** |
| B motion | pos | 1.09 | 1.09 |
| C illum | green / pos | 0.23 / 0.51 | 0.19 / 0.63 |
| C illum | **chrom** | 10.51 | **4.68** |
| C illum | ica | 20.40 | 27.57 ⚠ |
| D exercise | green / ica / pos | 3.08 / 6.06 / 1.59 | 1.99 / 3.30 / 0.48 |
| D exercise | chrom | 18.25 | 28.46 ⚠ |

**mean 7.72 → 5.54, median 4.57 → 1.85, better in 12 of 16 cells.**

Two regressions (⚠), both in cells that were already unusable at 18–20 BPM
error. Both are the same mechanism: at elevated HR the pulse's harmonics fall
outside the 0.7–4.0 Hz passband, so the correct candidate gets no harmonic
support while a subharmonic can borrow the true line's power. Documented in
`harmonic_score`, pinned by
`test_chrom_is_octave_sensitive_at_elevated_heart_rate`.

### Reading these numbers honestly

These are **synthetic** results and they justify the design, not the accuracy.
Published rPPG on UBFC-rPPG lands around 1–3 BPM MAE; anything far below that
means the test is too easy, not that the estimator is exceptional. The model
here deliberately includes heart-rate variability, a harmonic-rich PPG
waveform, 1/f noise, burst motion and capture jitter for exactly that reason —
an earlier sine-wave model reported 0.06 BPM, which was meaningless.

**SNR measures narrowbandness, not correctness.** Under motion, GREEN and ICA
report their *highest* confidence while being ~29 BPM wrong, because a periodic
artefact is genuinely narrowband. Cross-method agreement is the honest
no-ground-truth signal; ground truth is the only real one.

**The synthetic model is built from POS's skin-tone and specular assumptions,
so it plausibly favours POS.** It cannot settle CHROM vs POS — only UBFC can.

---

## Recording your own clips — read this first

**Record losslessly, or the benchmark measures your codec.** The pulsatile
modulation is under one 8-bit level. It survives quantisation only because
per-pixel sensor noise *dithers* the quantiser, decorrelating the rounding
error so the spatial average resolves below one LSB. Anything that re-quantises
downstream throws that away.

Measured on a 0.74-level (sub-LSB) synthetic pulse, correlation with the truth:

| dither | codec | recovered |
|---|---|---|
| none | lossless | nothing at all |
| 2 levels | **lossless (FFV1)** | **r = +0.995** |
| 2 levels | mp4v | r = +0.678 |

So: record raw frames or FFV1 (`.avi`), never MP4. Better still, use
`--save-signal`, which writes the RGB means straight out of the live capture
and sidesteps the question. `write_synthetic_video` defaults to FFV1 for the
same reason.

A pleasing corollary for the viva: **a noiseless camera would be worse than a
noisy one.** The sensor noise is what makes a sub-LSB measurement possible.

## Known environment issue on Windows

With pandas 3.0.3 + pyarrow 24 + scipy 1.17 + matplotlib 3.10.8 on Python 3.13,
this import order hard-crashes the interpreter:

```python
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
pd.DataFrame([{"a": 1.0}])       # access violation in ArrowStringArray
```

It is order-sensitive (importing pandas first, or scipy.signal before
matplotlib, both survive) and nothing to do with the data — a native-library
conflict between the wheels.

`import rppg` sets `pd.options.mode.string_storage = "python"` to route around
it, which keeps `str` dtype and changes only the storage backend. Details and
the opt-out (`RPPG_KEEP_ARROW_STRINGS=1`) are in [_compat.py](rppg/_compat.py).
If you build DataFrames in a notebook **before** importing `rppg`, you can
still hit it — import `rppg` early.

---

## Repository layout

```
rppg/                     core library — shared by notebooks and app
  capture.py  roi.py  resample.py  preprocess.py
  methods/{green,ica,chrom,pos}.py
  spectral.py  quality.py  metrics.py  datasets.py
  pipeline.py  synthetic.py  run.py  _compat.py
notebooks/01_signal_walkthrough.ipynb
notebooks/03_ablations.ipynb
app/streamlit_app.py
tests/  test_dsp.py  test_roi_and_gates.py  test_video_fidelity.py  test_app.py
data/                     gitignored — datasets and captures
models/                   gitignored — face_landmarker.task
```

## When a live reading looks wrong

The app shows three numbers above the plots. Check them before the BPM.

| Reading | Means |
|---|---|
| **ROI px** | under ~8000 and the spatial average cannot beat the noise floor |
| **arms agree to X BPM** | over ~5 BPM and at most one arm can be right |
| **fps processed** | under 10 and the loop is the bottleneck, not the signal — Nyquist needs 8 fps for a 4 Hz band |

Rejections are shown with a reason. Two independent gates: SNR, and sampling
gaps. The gap gate exists because a spline drawn across dropped frames is
*smooth*, therefore narrowband, therefore scores a fine SNR while being
entirely an artefact — SNR structurally cannot catch it.

**SNR measures narrowbandness, not correctness.** Under motion, GREEN and ICA
report their highest confidence while being ~29 BPM wrong, because a periodic
motion artefact is genuinely narrowband. Cross-method agreement is the honest
confidence signal; ground truth is the only real one.

## Data layout

```
data/ubfc/subject1/vid.avi + ground_truth.txt     # UBFC-rPPG (DATASET_2)
data/own/A_still/clip1.mp4 + clip1_gt.csv         # self-collected, columns t,hr[,ppg]
```

Conditions: **A** still · **B** motion · **C** illumination · **D**
post-exercise. D is the one that catches a method which always outputs ~72 BPM.