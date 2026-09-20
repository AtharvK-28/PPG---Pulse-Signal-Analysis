"""The four projection arms.

Every method here is one choice of a projection w such that w^T [R,G,B]^T
suppresses the specular term I(t)v_s(t) and keeps the pulsatile part of
I(t)v_d(t). GREEN fixes w = [0,1,0]; ICA learns it per window; CHROM and POS
derive it from a skin-reflection model and adapt one scalar. Framing them as
one family is what makes the comparison meaningful.

All arms share a signature::

    project(rgb: (N, 3), fs: float, **kw) -> (N,) pulse signal

Amplitude is not comparable across arms (ICA has scale/sign ambiguity by
construction); only frequency content is.
"""

from .chrom import chrom
from .green import green
from .ica import ICAResult, ica, ica_full
from .pos import pos

PROJECTIONS = {
    "green": green,
    "ica": ica,
    "chrom": chrom,
    "pos": pos,
}

__all__ = ["PROJECTIONS", "green", "ica", "ica_full", "ICAResult", "chrom", "pos"]
