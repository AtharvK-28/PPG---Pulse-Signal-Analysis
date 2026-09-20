"""Ground truth you control — the week-3 sanity check, available from day one.

Feed the pipeline a signal of known frequency plus noise plus drift and confirm
it returns the right BPM. If it cannot do that, no amount of real video will
help. This module builds those signals from the same reflection model the
methods assume:

    C_c(t) = I(t) (v_s(t) + v_d(t)) + v_n(t)

so the arms can be separated on the axis that actually distinguishes them:
specular/motion contamination, which enters multiplicatively through I(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Standardised skin-tone direction (Wang et al., 2017).
SKIN_TONE = np.array([0.7682, 0.5121, 0.3841])
#: Blood-volume pulse signature (de Haan & van Leest, 2014), normalised.
PULSE_SIGNATURE = np.array([0.33, 0.77, 0.53]) / np.linalg.norm([0.33, 0.77, 0.53])
#: Specular reflection of a white illuminant is colour-neutral.
SPECULAR = np.ones(3) / np.sqrt(3.0)


@dataclass
class SyntheticClip:
    t: np.ndarray  #: frame timestamps (seconds)
    rgb: np.ndarray  #: (N, 3) spatial means, 0-255 scale
    bpm_true: np.ndarray  #: instantaneous ground-truth HR per frame
    pulse: np.ndarray  #: the clean pulse waveform that was injected
    fs_nominal: float

    @property
    def mean_bpm(self) -> float:
        return float(np.mean(self.bpm_true))

    def mean_bpm_over(self, t0: float, t1: float) -> float:
        """Mean true HR across a window — the correct reference for a spectral estimate.

        A window-based method estimates the *average* rate over its window, not
        the instantaneous rate at any point in it. Once the HR varies within a
        window (which it always does, through respiratory sinus arrhythmia),
        scoring against an instantaneous value charges the estimator for
        variation it was never able to resolve.
        """
        sel = (self.t >= t0) & (self.t <= t1)
        return float(np.mean(self.bpm_true[sel])) if np.any(sel) else float("nan")


def _pink_noise(n: int, rng) -> np.ndarray:
    """Unit-variance 1/f noise.

    Real sensor and illumination noise is not white — it has far more power at
    low frequencies, which is exactly where detrending and the lower edge of the
    cardiac band live. Benchmarking against white noise flatters every method.
    """
    if n < 4:
        return rng.normal(size=n)
    white = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(white / np.sqrt(f), n=n)
    return x / (x.std() + 1e-12)


def nonuniform_timestamps(
    duration: float,
    fs: float = 30.0,
    jitter_ms: float = 3.0,
    drop_prob: float = 0.0,
    rng=None,
) -> np.ndarray:
    """Realistic webcam capture times: nominal rate + jitter + dropped frames.

    A nominal 30 fps capture is really a ~28-29 fps mean with several ms of
    standard deviation. Feeding these into the pipeline is how we prove the
    resampling stage matters.
    """
    rng = np.random.default_rng(rng)
    n = int(round(duration * fs))
    dt = 1.0 / fs + rng.normal(0.0, jitter_ms * 1e-3, size=n)
    dt = np.clip(dt, 1e-4, None)
    t = np.cumsum(dt)
    if drop_prob > 0:
        keep = rng.random(n) >= drop_prob
        keep[0] = keep[-1] = True
        t = t[keep]
    return t - t[0]


def synth_rgb(
    duration: float = 30.0,
    fs: float = 30.0,
    bpm: float | tuple = 72.0,
    pulse_amplitude: float = 0.004,  #: 0.4% modulation — realistic for webcam rPPG
    respiration_bpm: float = 15.0,
    respiration_amplitude: float = 0.02,
    motion_amplitude: float = 0.0,
    motion_bpm: float = 40.0,
    motion_kind: str = "sinusoid",  #: "sinusoid" | "burst"
    illumination_drift: float = 0.0,
    noise_std: float = 0.15,  #: white sensor noise, 8-bit levels
    pink_std: float = 0.25,  #: 1/f noise, 8-bit levels
    hrv_rsa: float = 0.04,  #: respiratory sinus arrhythmia, fraction of HR
    hrv_walk: float = 0.02,  #: slow random-wander of HR, fraction
    waveform: str = "ppg",  #: "ppg" (harmonic-rich) | "sine"
    dc: float = 128.0,
    timestamps=None,
    rng=None,
) -> SyntheticClip:
    """Build spatially-averaged RGB means containing a known pulse.

    ``bpm`` may be a scalar or ``(start, end)`` for a linear ramp — use the ramp
    to check the pipeline tracks a changing HR instead of latching onto one
    value.

    ``motion_amplitude`` drives a specular term through the shared I(t), which
    is the distortion ICA's independence assumption cannot represent. Turning it
    up is how the benchmark separates ICA from CHROM/POS.
    """
    rng = np.random.default_rng(rng)
    t = np.asarray(timestamps, dtype=float) if timestamps is not None else np.arange(
        int(round(duration * fs))
    ) / fs
    n = t.size

    if np.isscalar(bpm):
        f_hr = np.full(n, float(bpm) / 60.0)
    else:
        profile = np.asarray(bpm, dtype=float)
        if profile.size == 2:
            f_hr = np.linspace(profile[0], profile[1], n) / 60.0
        else:
            # An arbitrary profile, resampled onto the frame grid. A linear ramp
            # is a deceptively easy target: the mean HR over a window equals the
            # instantaneous HR at the window's centre, so windowing costs
            # nothing. Anything curved — a step, a recovery curve — is what
            # actually exposes the window-length trade-off.
            f_hr = np.interp(
                np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, profile.size), profile
            ) / 60.0
    # Heart rate is never constant. Respiratory sinus arrhythmia modulates it at
    # the breathing rate (a few BPM peak to peak), on top of a slow wander. This
    # is the single biggest reason a real spectral peak is broader than a test
    # tone's, and omitting it is what makes a synthetic benchmark report
    # implausibly small errors.
    if hrv_rsa:
        f_hr = f_hr * (
            1.0 + hrv_rsa * np.sin(2.0 * np.pi * (respiration_bpm / 60.0) * t + 0.3)
        )
    if hrv_walk:
        wander = np.cumsum(rng.normal(0.0, 1.0, size=n))
        wander = wander / (np.abs(wander).max() + 1e-12)
        f_hr = f_hr * (1.0 + hrv_walk * wander)

    # Integrate instantaneous frequency so a varying HR stays phase-continuous.
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if n > 1 else 1.0 / fs))
    phase = 2.0 * np.pi * np.cumsum(f_hr * dt)

    if waveform == "sine":
        pulse = np.sin(phase)
    else:
        # A real PPG is far from sinusoidal: a steep systolic upstroke, a
        # dicrotic notch, then a diastolic decay. Spectrally that is a strong
        # fundamental plus decaying harmonics. The harmonic structure is not
        # decoration — it is what distinguishes a pulse from a periodic motion
        # artefact, and it is what the harmonic-sum selector exploits.
        pulse = (
            np.sin(phase)
            + 0.42 * np.sin(2.0 * phase + 1.15)
            + 0.16 * np.sin(3.0 * phase + 2.40)
        )
    pulse /= np.abs(pulse).max()

    illum = np.ones(n)
    if respiration_amplitude:
        illum += respiration_amplitude * np.sin(
            2.0 * np.pi * (respiration_bpm / 60.0) * t + 0.7
        )
    if illumination_drift:
        illum += illumination_drift * np.sin(2.0 * np.pi * 0.05 * t + 1.1)

    diffuse = SKIN_TONE[None, :] * 0.9 + PULSE_SIGNATURE[None, :] * (
        pulse_amplitude * pulse
    )[:, None]

    specular = np.zeros((n, 3))
    if motion_amplitude:
        if motion_kind == "burst":
            # Real head motion is transient: a shift, a nod, a swallow. A
            # continuous sinusoid is the easiest possible artefact to reject —
            # it is narrowband and stationary, so a long window averages it in
            # a stable way. Bursts are broadband and non-stationary, which is
            # what actually breaks estimators.
            motion = np.zeros(n)
            span = max(t[-1] - t[0], 1e-6)
            for _ in range(max(1, int(round(span / 8.0)))):
                centre = rng.uniform(t[0], t[-1])
                width = rng.uniform(0.25, 1.0)
                freq = rng.uniform(0.4, 2.5)
                env = np.exp(-0.5 * ((t - centre) / width) ** 2)
                motion += env * np.sin(2.0 * np.pi * freq * (t - centre))
            motion /= np.abs(motion).max() + 1e-12
        else:
            motion = np.sin(2.0 * np.pi * (motion_bpm / 60.0) * t)
            motion = motion + 0.5 * rng.normal(size=n).cumsum() / np.sqrt(n)
        specular = SPECULAR[None, :] * (motion_amplitude * motion)[:, None]

    rgb = dc * illum[:, None] * (diffuse + specular)
    if noise_std:
        rgb = rgb + rng.normal(0.0, noise_std, size=(n, 3))
    if pink_std:
        # Correlated across time, independent across channels.
        rgb = rgb + pink_std * np.column_stack([_pink_noise(n, rng) for _ in range(3)])
    return SyntheticClip(
        t=t, rgb=rgb, bpm_true=f_hr * 60.0, pulse=pulse, fs_nominal=fs
    )


def write_synthetic_video(
    path,
    clip: SyntheticClip,
    size: tuple = (320, 240),
    radius: int = 70,
    bg: int = 40,
    pixel_noise: float = 2.0,
    codec: str = "FFV1",
    rng=None,
):  # pragma: no cover - I/O
    """Render a clip as an actual video file: a skin-coloured disc on a dark field.

    Not a face — a face detector will not find it. Its purpose is to exercise
    capture -> ROI -> spatial mean -> DSP end to end with the "fixed" ROI
    backend, so the plumbing can be tested before any dataset arrives.

    ``pixel_noise`` is *per-pixel* and therefore falls as 1/sqrt(N) under
    spatial averaging. The noise already baked into ``clip.rgb`` is common to
    every pixel of the disc and does not. The distinction matters whenever an
    experiment turns on how many pixels the ROI averages — a skin-mask ablation
    run with only common-mode noise measures nothing, because the term masking
    would reduce is not the term that dominates.

    ``codec`` defaults to **FFV1 (lossless)** and the path should end in
    ``.avi``. This is not a detail: the pulsatile modulation is 0.1-1% of
    intensity, roughly one 8-bit level, while `mp4v` on this material has a
    readback error of ~50 levels. A lossy codec does not degrade the signal, it
    obliterates it. UBFC-rPPG ships uncompressed 8-bit RGB for exactly this
    reason. Pass ``codec="mp4v"`` only to demonstrate the damage.
    """
    import cv2

    rng = np.random.default_rng(rng)
    w, h = size
    fps = clip.fs_nominal
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"could not open VideoWriter for {path} with codec {codec!r}"
            + (" — FFV1 needs an .avi container" if codec == "FFV1" else "")
        )
    # Anti-aliased disc as a float alpha mask, computed once. The whole frame is
    # then composited in float and quantised exactly once, at the end.
    #
    # Drawing straight into a uint8 array instead would round the disc colour to
    # an integer at draw time — before the noise is added — and a 0.1-1% pulse
    # is well under one 8-bit level, so it would be annihilated before it ever
    # reached the sensor model. Dither only linearises quantisation when it
    # precedes it; noise added afterwards cannot restore what rounding removed.
    alpha_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(alpha_u8, (w // 2, h // 2), radius, 255, -1, lineType=cv2.LINE_AA)
    alpha = (alpha_u8.astype(np.float64) / 255.0)[:, :, None]

    try:
        for row in clip.rgb:
            colour = np.asarray(row[::-1], dtype=np.float64)  # RGB -> BGR
            frame = bg + (colour[None, None, :] - bg) * alpha
            # Per-pixel noise, in float and before quantisation, so it acts as
            # dither: it decorrelates the rounding error across pixels, letting
            # the spatial mean recover sub-level detail.
            if pixel_noise:
                frame = frame + rng.normal(0.0, pixel_noise, size=(h, w, 3))
            writer.write(np.clip(np.round(frame), 0, 255).astype(np.uint8))
    finally:
        writer.release()
    return path
