"""Workaround for a native-library conflict in this environment.

On Windows / Python 3.13 with pandas 3.0.3 + pyarrow 24 + scipy 1.17 +
matplotlib 3.10.8, this exact import order hard-crashes the interpreter:

    import matplotlib.pyplot as plt
    import scipy.signal
    import pandas as pd
    pd.DataFrame([{"a": 1.0}])     # access violation in ArrowStringArray

The fault is inside `pandas.core.arrays.string_arrow._from_sequence` while
building the *column* Index, and it is order-sensitive: importing pandas before
scipy.signal, or scipy.signal before matplotlib, both survive. That points at a
shared native dependency being initialised differently depending on which
extension module loads first — nothing to do with the data being passed in.

It matters here because every notebook in this project imports matplotlib and
scipy.signal and then builds a results table.

The fix is to keep pandas off the Arrow-backed string path. String columns
stay `str` dtype; only the storage changes, so nothing downstream is affected.
Set ``RPPG_KEEP_ARROW_STRINGS=1`` to skip it and get stock pandas behaviour.
"""

from __future__ import annotations

import os
import sys

#: True if the guard was applied in this process.
ARROW_STRINGS_DISABLED = False


def apply_pandas_arrow_guard() -> bool:
    """Route pandas string storage away from pyarrow. Returns True if applied."""
    global ARROW_STRINGS_DISABLED
    if os.environ.get("RPPG_KEEP_ARROW_STRINGS") == "1":
        return False
    if sys.platform != "win32":
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False  # no Arrow backend, no crash path

    try:
        pd.options.mode.string_storage = "python"
    except (AttributeError, ValueError):
        return False  # option gone in a future pandas; nothing to guard
    ARROW_STRINGS_DISABLED = True
    return True
