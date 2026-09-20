"""The capture diagnostic's verdict logic.

Written after a full session of debugging "bad heart rates" that turned out to
be a camera pinned at 1 fps by another process holding it open. Nothing in the
DSP could have revealed that; a twenty-second capture measurement would have.
"""

import pytest

from rppg.diagnose import FAIL, PASS, WARN, Check, _wrap, report


def test_report_marks_and_surfaces_remedies():
    text, worst = report([
        Check("frame rate", PASS, "30.2 fps"),
        Check("exposure", FAIL, "brightness 0.0", "lens cover closed"),
    ])
    assert "[ok]" in text and "[FAIL]" in text
    assert "lens cover closed" in text
    assert worst == FAIL


def test_worst_status_wins():
    assert report([Check("a", PASS, "")])[1] == PASS
    assert report([Check("a", PASS, ""), Check("b", WARN, "")])[1] == WARN
    assert report([Check("a", WARN, ""), Check("b", FAIL, "")])[1] == FAIL
    # A later PASS must not mask an earlier FAIL.
    assert report([Check("a", FAIL, ""), Check("b", PASS, "")])[1] == FAIL


def test_output_is_ascii_only():
    """Windows terminals are cp1252; anything else renders as mojibake."""
    import pathlib

    src = pathlib.Path(report.__module__.replace(".", "/") + ".py")
    text = pathlib.Path("rppg/diagnose.py").read_text(encoding="utf-8")
    offenders = sorted({c for c in text if ord(c) > 127})
    assert not offenders, f"non-ascii in diagnostic output: {offenders}"


def test_wrap_does_not_split_words():
    out = _wrap("the quick brown fox jumps over the lazy dog", 12)
    assert all(len(line) <= 12 for line in out)
    assert " ".join(out) == "the quick brown fox jumps over the lazy dog"
