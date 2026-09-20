"""rPPG — contactless pulse estimation from RGB video by classical DSP.

Four projection arms (GREEN, ICA, CHROM, POS) over one shared pipeline:

    video -> ROI -> spatial mean -> resample -> detrend -> project
          -> bandpass -> window -> FFT -> peak -> BPM
"""

__version__ = "0.1.0"

BAND_HZ = (0.7, 4.0)  # 42-240 BPM
METHODS = ("green", "ica", "chrom", "pos")

# Applied at import because the crash it avoids takes down the whole
# interpreter, and every notebook here hits the triggering import order.
# See rppg/_compat.py for the repro and RPPG_KEEP_ARROW_STRINGS=1 to opt out.
from ._compat import apply_pandas_arrow_guard  # noqa: E402

apply_pandas_arrow_guard()

__all__ = ["BAND_HZ", "METHODS", "apply_pandas_arrow_guard", "__version__"]
