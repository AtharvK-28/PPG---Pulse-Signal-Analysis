# Contactless Pulse Signal Analysis via rPPG — Ideation Document

**Course:** Digital Time Signal Processing (DTSP)
**Project:** Pulse Signal Analysis (Heart Rate) — Project 16
**Scope:** Pure software. Python. No custom hardware.
**Status:** v1.0 — ideation, open for revision
**Basis:** Revises the proposal by Atharv Sushen Kharade (*Contactless Vital Sign Extraction via rPPG*)

---

## 0. TL;DR — What changes from the original proposal

| # | Original proposal | This document | Why |
|---|---|---|---|
| 1 | ICA is *the* solution | ICA is **one arm of four** (GREEN, ICA, CHROM, POS) | ICA is the weakest of the four in every published benchmark. Betting the project on it is a defensibility risk. |
| 2 | Framed as "build a robust HR meter" | Framed as **"benchmark four rPPG projections and report which wins, with evidence"** | Turns a demo into a study. Survives "how do you know it works?" |
| 3 | React frontend + Python backend | **Streamlit app + Jupyter benchmark notebooks** | The grade is for DSP, not for WebSocket plumbing. Cuts ~2 weeks of non-DSP work. |
| 4 | Validation unspecified | **UBFC-rPPG public dataset + pulse-oximeter ground truth**, MAE/RMSE/Pearson/Bland–Altman | No ground truth = no result. This is the single biggest gap in the original. |
| 5 | Assumes 30 fps uniform sampling | **Timestamp every frame, resample to a uniform grid** | Webcams do not deliver uniform frame intervals. Skipping this makes the frequency axis wrong. |
| 6 | "clinical-grade vital sign" | "estimate with reported error bounds" | Overclaim. Invites an easy attack in viva and is not true of webcam rPPG. |
| 7 | FFT peak → BPM | FFT with **detrending, Hann window, zero-padding, parabolic peak interpolation** | A 10 s window gives 6 BPM raw resolution. That alone would sink the accuracy numbers. |

Everything else in Atharv's proposal — the physics, the ROI strategy, the three-way component visualization, the live "break it on stage" demo — is good and is retained.

---

## 1. What the project actually is

Estimate a person's heart rate from ordinary RGB video of their face, using nothing but signal processing in Python, and **quantify how well each method works and under what conditions it fails.**

The deliverable is two things:

1. **A working pipeline** — video in, BPM out, live and on recorded files.
2. **A results table** — four extraction methods × four recording conditions, scored against ground truth.

The second is what makes it a DTSP project rather than a CV project.

### Explicit non-goals

- Not blood pressure, SpO₂, or HRV-based stress inference. (HRV is a stretch goal only; it needs beat-level timing accuracy that webcam rPPG barely supports.)
- Not a deep-learning model. Classical DSP is the point of the course.
- Not a medical device. State this in the report.

---

## 2. Physics — why this works at all

Each cardiac cycle pushes a volume of blood through the capillary beds of the face. Oxygenated hemoglobin absorbs light strongly around **540 nm (green)** and weakly in the red. So facial skin darkens fractionally during systole and lightens during diastole.

The modulation depth is roughly **0.1–1% of pixel intensity** — well below visual perception, but above the noise floor of an 8-bit sensor once you spatially average over thousands of skin pixels.

Two useful facts to have ready for viva:

- Green channel carries the strongest pulsatile signal (Verkruysse et al., 2008), both because of the hemoglobin absorption peak **and** because a Bayer sensor has twice as many green photosites as red or blue, so the green channel has a better SNR before anything else happens.
- Red penetrates deeper into tissue but is dominated by non-pulsatile diffuse reflection; blue is mostly surface specular reflection. This asymmetry is exactly what CHROM and POS exploit.

### The signal model

For each channel $c \in \{R,G,B\}$, the spatially-averaged intensity is:

$$C_c(t) = I(t)\big(v_s(t) + v_d(t)\big) + v_n(t)$$

- $I(t)$ — illumination intensity (varies with lighting, distance, auto-exposure)
- $v_s(t)$ — **specular** reflection off the skin surface: carries **no** pulse, varies violently with head motion
- $v_d(t)$ — **diffuse** reflection from subsurface tissue: this is where the pulse lives
- $v_n(t)$ — sensor noise

Every method below is, at heart, **one choice of a projection vector** $\mathbf{w}$ such that $\mathbf{w}^\top [R, G, B]^\top$ suppresses $I(t)v_s(t)$ and keeps the pulsatile part of $I(t)v_d(t)$. That framing unifies the whole project — it is worth stating explicitly in the report, because it makes four seemingly unrelated algorithms into one comparable family.

---

## 3. The four method arms

| Arm | Projection | Type | Origin |
|---|---|---|---|
| **GREEN** | $[0, 1, 0]$ | Fixed, trivial | Verkruysse et al., 2008 |
| **ICA** | Learned per window | Data-driven | Poh et al., 2010 |
| **CHROM** | Fixed model + adaptive scalar | Model-based | de Haan & Jeanne, 2013 |
| **POS** | Fixed model + adaptive scalar | Model-based | Wang et al., 2017 |

### 3.1 GREEN — the baseline you must beat

Take the spatial mean of the green channel, detrend, bandpass, FFT. Three lines of real work.

Include it because **a baseline you can't beat means your method doesn't work.** If ICA ties GREEN, you have learned something real and reportable.

### 3.2 ICA — blind source separation (Atharv's original core)

Normalize each channel to zero mean and unit variance, then run FastICA on the $3 \times N$ matrix to get three components. Select the pulse component by which one has the sharpest spectral peak in $[0.7, 4.0]$ Hz.

**Be honest in the doc about its known weaknesses** — this is what makes the project defensible rather than naive:

- **Permutation ambiguity.** ICA does not order its outputs. Component 1 is not reliably "noise" — the ordering changes window to window. The FFT-based selection heuristic is *mandatory*, not a convenience. (The original proposal's labelled "Component 1 = noise, Component 2 = respiration, Component 3 = pulse" is not something ICA guarantees; it is what the selection step decides after the fact.)
- **Scale and sign ambiguity.** Amplitude is meaningless; only frequency content is usable.
- **Exactly determined system.** Three observations, three assumed sources. No redundancy at all. Any fourth real source breaks the model.
- **The independence assumption is violated by the very thing it's meant to fix.** A head tilt changes all three channels through a shared multiplicative $I(t)$ term. That is not an independent additive source — it's a nonlinear, correlated distortion. This is precisely why ICA underperforms under motion.

**The evidence:** across published benchmarks, ICA is consistently the weakest of the classical family. In one motion benchmark with 117 subjects in vigorous motion, ICA produced the correct pulse rate ~4% of the time while chrominance methods reached ~48%. Model-based methods (CHROM, PBV, POS) beat non-model-based methods (GREEN, PCA, ICA) on motion robustness in essentially every comparison study. A 2026 benchmarking paper across four public datasets found CHROM the most reliable algorithm across all RGB datasets.

This is not a reason to drop ICA. It is a reason to **measure it** — a project that says "we implemented ICA and here is the data showing where it breaks and why" is stronger than one that asserts it's robust.

### 3.3 CHROM — chrominance-based

De Haan & Jeanne's insight: build two chrominance signals in which the specular component largely cancels, then combine them adaptively.

1. Skin-tone normalize: $R_n = R/\bar{R}$, $G_n = G/\bar{G}$, $B_n = B/\bar{B}$ over the window
2. $X_s = 3R_n - 2G_n$
3. $Y_s = 1.5R_n + G_n - 1.5B_n$
4. Bandpass both → $X_f, Y_f$
5. $\alpha = \sigma(X_f)/\sigma(Y_f)$
6. $S = X_f - \alpha Y_f$

The $\alpha$ scaling is the clever bit: it tunes the cancellation so that the motion-induced components in $X_f$ and $Y_f$ destructively interfere.

### 3.4 POS — plane orthogonal to skin

Wang et al.'s refinement: project onto a plane orthogonal to the skin-tone direction in temporally normalized RGB space.

1. Temporally normalize over a sliding sub-window of $l = 1.6 \times f_s$ samples: $C_n = C / \bar{C}_{\text{window}}$
2. Project with $P = \begin{bmatrix} 0 & 1 & -1 \\ -2 & 1 & 1 \end{bmatrix}$ → $S_1 = G_n - B_n$, $S_2 = -2R_n + G_n + B_n$
3. $h = S_1 + \dfrac{\sigma(S_1)}{\sigma(S_2)} S_2$
4. Overlap-add $h - \bar{h}$ into the output signal

POS and CHROM trade places at the top depending on the dataset; both consistently beat GREEN and ICA. Expect one of these two to win.

---

## 4. The DSP pipeline (this is the graded core)

```
video ──► ROI ──► spatial mean ──► RESAMPLE ──► detrend ──► project ──► bandpass ──► window ──► FFT ──► peak ──► BPM
                    (R,G,B)      (uniform fs)              (4 arms)                  (Hann)  (0-pad)  (interp)
```

### Stage 1 — ROI extraction

- **Primary:** MediaPipe Face Landmarker (Tasks API, `pip install mediapipe`, plus the `face_landmarker.task` model bundle). Gives dense 3D landmarks; use them to define forehead and upper-cheek polygons. Landmarks track through moderate motion, which removes the "head tilt destroys the ROI" failure the original proposal describes.
- **Fallback if MediaPipe misbehaves:** Haar cascade for detection + CSRT tracker for continuity. Documented as a fallback, not the plan.
- **Skin masking:** inside the ROI, threshold in YCrCb (roughly $133 \le C_r \le 173$, $77 \le C_b \le 127$) to drop eyebrows, glasses frames, hair, and background pixels that leak in. Cheap and gives a measurable SNR improvement — worth an ablation.
- Output per frame: $(\bar{R}, \bar{G}, \bar{B})$ over masked skin pixels **plus the frame timestamp**.

### Stage 2 — Resampling to a uniform grid ⚠️

**This is the step the original proposal is missing and the one most likely to silently ruin the results.**

`cv2.VideoCapture` does not deliver frames at a constant interval. Real webcam capture jitters — dropped frames, exposure-dependent readout, OS scheduling. Nominal 30 fps might be an actual mean of 28.4 fps with a standard deviation of several milliseconds.

Every FFT you run assumes uniform sampling. If the true sampling is non-uniform and you assume 30 Hz, your frequency axis is scaled wrong and your BPM is systematically biased.

**Fix:**
1. Record `time.perf_counter()` for every frame alongside the RGB triple.
2. Cubic-spline interpolate (`scipy.interpolate.CubicSpline`) each channel onto a uniform grid at a chosen $f_s$ (30 Hz).
3. Do all downstream DSP on the resampled signal.

This costs ~10 lines and is an excellent viva answer. Report the measured frame-interval jitter as a figure in the write-up.

*(On recorded dataset videos, frames are uniform by construction — but keep the step in the pipeline so both paths are identical.)*

### Stage 3 — Detrending

Respiration produces a large, slow (~0.2–0.4 Hz) baseline wander that the cardiac signal rides on top of. A naive high-pass with a sharp cutoff near the respiration band will ring.

- **Preferred:** smoothness-priors detrending (Tarvainen et al., 2002), $\lambda \approx 100$ at $f_s = 30$ Hz. Effectively a time-varying high-pass with a very smooth response.
- **Simpler alternative:** subtract a moving average of ~1 s.

Compare both in an ablation — this is a clean, self-contained DSP experiment.

### Stage 4 — Bandpass filtering

- 4th-order Butterworth, **0.7–4.0 Hz** (42–240 BPM).
- Apply with `scipy.signal.filtfilt`, **not** `lfilter`.

**Why `filtfilt`:** it runs the filter forward then backward, which cancels phase distortion exactly (zero-phase response) at the cost of doubling the effective order and needing the whole segment in memory. Since we work on buffered windows, we can afford it. Phase linearity matters if you ever want beat-to-beat timing.

Justify the band from physiology: 42 BPM covers a resting athlete, 240 BPM covers maximal exertion. Anything outside is not a human heart rate.

### Stage 5 — Windowing and the FFT resolution problem ⚠️

This is the most important numerical issue in the project and the question most likely to be asked.

**Frequency resolution of an FFT is set by window duration, not by FFT length:**

$$\Delta f = \frac{1}{T} \quad\Longrightarrow\quad \Delta\text{BPM} = \frac{60}{T}$$

| Window $T$ | Resolution |
|---|---|
| 5 s | 12 BPM |
| **10 s (300 frames — the original proposal's choice)** | **6 BPM** |
| 20 s | 3 BPM |
| 30 s | 2 BPM |

A 6 BPM quantization step makes any sub-6-BPM MAE claim meaningless. Three mitigations, all worth implementing:

1. **Longer window.** $T = 15$–20 s. Trade-off: slower response to real HR changes, and more chance of motion contaminating the window. Run a window-length sweep and plot MAE vs $T$ — that plot is a genuine result.
2. **Zero-padding.** Pad to 8× the window length before the FFT. This does *not* create new information (the underlying resolution is unchanged), but it densely samples the DTFT, so the peak is located far more precisely. Say this correctly in viva — "zero-padding interpolates the spectrum, it does not improve resolution" is exactly the distinction examiners probe for.
3. **Parabolic peak interpolation.** Fit a parabola through the peak bin and its two neighbours in the log-magnitude spectrum; take the vertex. Sub-bin accuracy for ~5 lines of code.

Apply a **Hann window** before the FFT to suppress spectral leakage from the finite window (rectangular windowing leaks at −13 dB first sidelobe; Hann at −31 dB, at the cost of a wider main lobe). Note the trade-off explicitly.

**Also consider Welch's method** as a comparison: split the window into overlapping segments, average the periodograms. Lower variance estimate, worse resolution. Comparing single-FFT vs Welch is another clean, cheap experiment.

### Stage 6 — Sliding window and temporal smoothing

- Window 10–15 s, **hop 1 s** → one BPM estimate per second.
- **Median filter** the last 5 BPM estimates to suppress single-window outliers.
- **Reject** windows whose SNR falls below threshold rather than emitting a bad number — and *display* the rejection, so the demo is honest about when it can't see a pulse.

### Stage 7 — Quality metric (no ground truth needed)

De Haan & Jeanne's SNR: with $f_{\text{peak}}$ the estimated pulse frequency, define a binary mask $M$ that is 1 within ±0.1 Hz of $f_{\text{peak}}$ and of $2f_{\text{peak}}$, and 0 elsewhere in $[0.5, 4]$ Hz. Then:

$$\text{SNR} = 10\log_{10}\frac{\sum M(f)|S(f)|^2}{\sum (1-M(f))|S(f)|^2}$$

This is genuinely useful: it gives a per-window confidence score that works live, with no reference sensor. Use it to drive the on-screen confidence indicator **and** as a comparison metric between arms on unlabelled video.

---

## 5. Validation — the part that turns a demo into a project

### 5.1 Primary: public dataset

**UBFC-rPPG** (Univ. Bourgogne Franche-Comté). Recorded with a Logitech C920 at 30 fps, 640×480, uncompressed 8-bit RGB, with a CMS50E pulse oximeter providing ground-truth PPG waveform and heart rate. Subjects seated ~1 m from camera under mixed indoor and natural light. It is the standard benchmark for exactly this task, and it matches our hardware assumption (consumer webcam) closely.

Access is by request to the authors — **send the request in week 1**, before writing any code, because turnaround is not instant.

Alternatives if access stalls: **PURE**, **LGI-PPGI**, **COHFACE**.

### 5.2 Secondary: self-collected stress set

Record ~8–10 clips of team members, ~60 s each, with a fingertip pulse oximeter or smartwatch visible in frame (or logged) as reference. Four conditions:

| Condition | Purpose |
|---|---|
| **A — Still** | Best case. Establishes the ceiling. |
| **B — Motion** | Talking, head rotation, natural fidgeting. Where ICA is predicted to fail. |
| **C — Illumination** | Change screen brightness mid-clip; walk a shadow across. Triggers auto-exposure. |
| **D — Post-exercise** | HR elevated to 100–140. Tests the upper band and the dynamic response. |

Condition D matters more than it looks: almost all naive rPPG demos are validated only at resting HR, and a method that always outputs ~72 BPM will look great and mean nothing.

### 5.3 Metrics

| Metric | What it tells you |
|---|---|
| **MAE** (BPM) | Headline number |
| **RMSE** (BPM) | Penalizes large failures — exposes instability that MAE hides |
| **MAPE** (%) | Fair across resting and elevated HR |
| **Pearson $r$** | Does it *track* HR, or just sit near the population mean? |
| **Bland–Altman** | Bias and limits of agreement. The right plot for method-comparison studies. Include it. |
| **SNR** (dB) | Signal quality without ground truth |

**Report per method × per condition.** A 4×4 grid of MAE with a Bland–Altman plot for the winning method is the centrepiece figure of the whole report.

### 5.4 The one honest limitation to state up front

rPPG performance degrades on darker skin tones — melanin absorbs more incident light, so less reaches the capillary bed and less returns to the sensor, reducing the pulsatile modulation depth. UBFC-rPPG is not skin-tone diverse. **State this as a known limitation of both the dataset and the reported numbers.** Volunteering it is far stronger than being asked.

---

## 6. Architecture

### Recommendation: cut the React frontend

The original proposal specifies a React frontend over a Python backend. For this project that is the wrong trade:

- It adds WebSocket/streaming plumbing, CORS handling, a build toolchain, and frame-encoding latency — none of which is DSP.
- It is roughly 2 weeks of work that earns zero marks in a signal processing course.
- Live plotting is the *only* thing it buys, and Streamlit does that in ~15 lines.

**Proposed instead:**

| Component | Tool | Role |
|---|---|---|
| Live demo app | **Streamlit** | Webcam feed, ROI overlay, live plots, 4-arm comparison, BPM + confidence |
| Benchmark | **Jupyter notebooks** | Batch runs over UBFC-rPPG, metrics, figures |
| Core library | Plain Python package | Shared by both — no logic duplicated |

Alternative for the live demo if Streamlit's webcam handling is awkward: a plain OpenCV window plus a Matplotlib animated figure. Uglier, zero framework risk.

**React stays as a stretch goal** — build it only if the benchmark is fully done and there is time left. Do not start with it.

### 6.1 Stack

```
python 3.11
numpy, scipy          # DSP core
opencv-python         # capture, ROI
mediapipe             # face landmarks
scikit-learn          # FastICA
pandas                # results tables
matplotlib            # figures
streamlit             # demo app
```

Deliberately no PyTorch, no TensorFlow. If asked "why no deep learning?" — because the course is DSP, because the classical methods have interpretable, defensible mechanisms, and because a supervised model trained on UBFC-rPPG and tested on UBFC-rPPG would prove nothing.

### 6.2 Repository structure

```
rppg-dtsp/
├── rppg/
│   ├── capture.py         # webcam + video-file readers, timestamps
│   ├── roi.py             # MediaPipe landmarks, skin mask, spatial means
│   ├── resample.py        # non-uniform → uniform grid
│   ├── preprocess.py      # detrend, normalize, bandpass
│   ├── methods/
│   │   ├── green.py
│   │   ├── ica.py
│   │   ├── chrom.py
│   │   └── pos.py
│   ├── spectral.py        # Hann, zero-pad, FFT, parabolic peak, Welch
│   ├── quality.py         # SNR
│   └── metrics.py         # MAE, RMSE, MAPE, Pearson, Bland-Altman
├── notebooks/
│   ├── 01_signal_walkthrough.ipynb   # one clip, every stage plotted
│   ├── 02_benchmark_ubfc.ipynb
│   ├── 03_ablations.ipynb            # window length, detrend, skin mask, Welch
│   └── 04_stress_conditions.ipynb
├── app/
│   └── streamlit_app.py
├── data/                  # gitignored
└── report/
```

**A note on `01_signal_walkthrough.ipynb`:** build this first. One clip, every intermediate signal plotted — raw RGB, resampled, detrended, each of the four projections, filtered, spectrum, peak. It is simultaneously your debugging tool, the best figure set in your report, and the artefact that makes the viva easy.

---

## 7. Dashboard design

Keep Atharv's core idea — it is genuinely good. The panel layout:

**Left column — the problem**
- Webcam feed with ROI polygon and skin mask overlaid
- Raw RGB traces: three noisy, drifting, obviously-not-a-heartbeat lines

**Right column — the solution**
- Four extracted pulse traces, one per method, stacked and aligned in time
- Four spectra with the detected peak marked
- Four BPM readouts with SNR-driven confidence colouring
- Reference BPM (from oximeter) when available, in a contrasting colour

**Bottom strip**
- Rolling BPM-vs-time for all four methods on shared axes. When you shake your head, the ICA line should visibly diverge while POS and CHROM hold — **that is the demo**, and unlike the original narrative it is a measurement rather than a claim.

For the ICA arm specifically, keep the three-component view (noise / respiration / cardiac), since visualizing blind source separation is genuinely instructive — but label it as "the component the selector chose," which is what it actually is.

---

## 8. Timeline

Six weeks. Adjust to your actual deadline, but preserve the ordering — **the benchmark must exist before the polish.**

| Week | Deliverable | Done means |
|---|---|---|
| **1** | Requests + ROI + capture | UBFC access request **sent day 1**. Webcam capture with per-frame timestamps writing CSV. ROI + skin mask working. Frame-jitter histogram plotted. |
| **2** | GREEN end-to-end | Full pipeline for one arm. Notebook 01 complete with every stage plotted. Produces a plausible BPM on a still clip. |
| **3** | All four arms | CHROM, POS, ICA implemented and unit-tested against a synthetic signal of known frequency. All four run on the same input. |
| **4** | Benchmark | Notebook 02 runs over UBFC-rPPG. First full metrics table exists, even if the numbers are bad. |
| **5** | Ablations + stress set | Window-length sweep, detrend comparison, skin-mask on/off, Welch vs FFT. Own clips recorded and scored. |
| **6** | App + report | Streamlit demo working. Report written. Rehearse the live demo on the actual presentation machine. |

**Sanity check for week 3:** feed the pipeline a synthetic signal — a 1.2 Hz sinusoid plus noise plus a 0.3 Hz drift — and confirm it returns 72 BPM. If it can't do that, no amount of real video will help. This test takes 20 minutes to write and will save days.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **UBFC access delayed** | Medium | Request on day 1. Fall back to PURE or LGI-PPGI. Worst case, self-collected set only — degraded but survivable. |
| **FastICA fails to converge** | Medium | Wrap in try/except, cap `max_iter`, fall back to previous window's estimate. **Log the failure rate and report it** — it is a result, not an embarrassment. |
| **MediaPipe install friction** | Low–Medium | Pin versions early. Haar + CSRT fallback path kept working. |
| **Auto-exposure destroys everything** | Medium | Disable it explicitly: `cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)` on V4L2 backends, plus fixed white balance. Document that you did — and run one clip *with* it on to show the damage. |
| **All four methods perform badly** | Low | Still a valid report if the failure is characterized. But de-risk with the synthetic-signal test in week 3. |
| **Scope creep into HRV / BP / SpO₂** | **High** | Explicitly out of scope in §1. Revisit only after the 4×4 table exists. |
| **Documentation substituting for building** | **High** | Freeze this document after v1.1. The only tracked task after that is code. |

That last row is the real one. This document is now written; its job is done. Week 1's deliverable is a CSV of timestamped RGB means, not a revision of section 4.

---

## 10. Viva preparation

Likely questions and short answers:

**Why green?** Hemoglobin absorption peaks near 540 nm, and Bayer sensors have twice as many green photosites, so the green channel starts with better SNR.

**Why 0.7–4.0 Hz?** 42–240 BPM. Covers resting athlete to maximal exertion. Outside that isn't a human pulse.

**What is your frequency resolution?** $60/T$ BPM. At $T=10$ s that's 6 BPM, which is why we use a 15 s window with 8× zero-padding and parabolic peak interpolation. Zero-padding interpolates the spectrum; it does not add resolution.

**Why `filtfilt` and not `lfilter`?** Zero-phase response. Forward-backward filtering cancels phase distortion, at the cost of doubling effective order and requiring the full segment.

**Is 30 fps enough?** Nyquist is 15 Hz, well above the 4 Hz band edge, so the fundamental and first two harmonics are safely captured. Mains flicker at 100 Hz (50 Hz supply) aliases to 10 Hz at 30 fps sampling — above our passband, so the bandpass removes it. Beat frequencies between flicker and auto-exposure hunting are a real concern, which is another reason to fix exposure.

**Why did ICA underperform?** Because its core assumption fails here. Motion enters through a shared multiplicative illumination term, not as an independent additive source; the system is exactly determined with no redundancy; and outputs have permutation and scale ambiguity requiring a heuristic selector that can itself pick wrong. Model-based projections encode the skin-reflection physics directly instead of trying to learn it from three observations.

**How do you know your BPM is right?** Ground-truth pulse oximeter waveforms from UBFC-rPPG, scored with MAE, RMSE, Pearson $r$, and Bland–Altman across four recording conditions.

---

## 11. Open decisions

Flagged for the team, not resolved here:

1. **Project name.** Optional, but useful for the report cover and slides. Candidates: *PULSE* (Photoplethysmographic Un-mixing & Live Spectral Estimation), *NADI*, *CardioCam*. Low priority.
2. **Window length** — 10 s vs 15 s vs 20 s. Resolve empirically with the week-5 sweep rather than by argument.
3. **Whether to add PBV or 2SR as a fifth arm.** Only if weeks 1–4 finish early. Four arms is already a complete story.
4. **HRV as a stretch goal.** Requires beat-level peak detection and much tighter timing than BPM estimation. Decide after the main table exists; likely defer.
5. **Division of labour.** Suggested split: one person on capture/ROI/resampling, one on the four method arms, one on benchmarking and metrics. The `spectral.py` module should be written once and shared, not per-arm.

---

## 12. References

- Verkruysse, Svaasand & Nelson (2008). *Remote plethysmographic imaging using ambient light.* Optics Express. — GREEN.
- Poh, McDuff & Picard (2010). *Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.* Optics Express. — ICA.
- de Haan & Jeanne (2013). *Robust pulse rate from chrominance-based rPPG.* IEEE TBME. — CHROM, and the SNR metric.
- Wang, den Brinker, Stuijk & de Haan (2017). *Algorithmic principles of remote PPG.* IEEE TBME. — POS.
- Tarvainen, Ranta-aho & Karjalainen (2002). *An advanced detrending method with application to HRV analysis.* IEEE TBME. — smoothness-priors detrending.
- Bobbia et al. (2019). *Unsupervised skin tissue segmentation for remote photoplethysmography.* Pattern Recognition Letters. — UBFC-rPPG dataset paper.
- Liu et al. (2023). *rPPG-Toolbox: Deep Remote PPG Toolbox.* NeurIPS Datasets & Benchmarks. — reference implementations and evaluation protocol.
- Boccignone et al. *pyVHR* — Python framework implementing GREEN, ICA, PCA, CHROM, POS, SSR, LGI, PBV, OMIT with dataset APIs.

**On pyVHR and rPPG-Toolbox:** read them, cross-check your outputs against them, cite them. Do **not** import them as your implementation — the point of a DSP course project is that you wrote the DSP. Use them as an oracle: if your CHROM output doesn't match theirs on the same clip, you have a bug.

---

*End of v1.0. Next revision only after the week-2 milestone produces a real signal.*
