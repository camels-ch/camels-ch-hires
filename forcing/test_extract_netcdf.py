"""Tests for the post-processing steps in extract_netcdf."""

import numpy as np
import pandas as pd

from extract_netcdf import _deaccumulate


def _day(increments):
    """Cumulative accumulation (resetting at 00 UTC) from hourly increments."""
    return list(np.cumsum(increments))


def test_deaccumulate_recovers_hourly_amounts():
    # Two days, 01:00..00:00 each; constant increments per day.
    idx = pd.date_range("2020-06-01 01:00", "2020-06-03 00:00", freq="h")
    df = pd.DataFrame({"a": _day([2.0] * 24) + _day([0.5] * 24)}, index=idx)
    out = _deaccumulate(df)
    # Each hour recovers its increment; the reset does not leak a huge value.
    assert np.allclose(out["a"].iloc[:24], 2.0)
    assert np.allclose(out["a"].iloc[24:], 0.5)


def test_deaccumulate_window_start_uses_value_itself():
    idx = pd.date_range("2020-06-01 01:00", "2020-06-01 04:00", freq="h")
    df = pd.DataFrame({"a": [0.3, 0.5, 0.9, 1.0]}, index=idx)  # accumulation
    out = _deaccumulate(df)
    # 01:00 is the window start -> value itself; the rest are differences.
    np.testing.assert_allclose(out["a"].values, [0.3, 0.2, 0.4, 0.1])


def test_deaccumulate_handles_gaps_without_crossing_them():
    # A missing 03:00 step: 04:00 must not be differenced against 02:00.
    idx = pd.DatetimeIndex(
        ["2020-06-01 01:00", "2020-06-01 02:00", "2020-06-01 04:00"]
    )
    df = pd.DataFrame({"a": [0.2, 0.5, 0.9]}, index=idx)
    out = _deaccumulate(df)
    # 04:00's predecessor is 2 h earlier -> treated as a window start.
    np.testing.assert_allclose(out["a"].values, [0.2, 0.3, 0.9])


def test_deaccumulate_clips_rounding_negatives_and_keeps_nan():
    idx = pd.date_range("2020-06-01 01:00", "2020-06-01 03:00", freq="h")
    # Accumulation dips by a rounding-sized amount, and a NaN cell.
    df = pd.DataFrame(
        {"a": [0.5, 0.5 - 1e-6, 0.7], "b": [np.nan, np.nan, np.nan]}, index=idx
    )
    out = _deaccumulate(df)
    assert (out["a"].values >= 0).all()
    assert out["b"].isna().all()
