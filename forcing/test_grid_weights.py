"""Tests for the area-weighted aggregation in grid_weights."""

import numpy as np
import pytest
from pyproj import Transformer
from scipy import sparse

from grid_weights import axes_from_2d_lonlat, weighted_mean, SENTINEL_GUARD

# A WRF-like spherical Mercator grid, regular in projected metres.
_PROJ = "+proj=merc +lat_ts=45 +lon_0=8 +R=6370000 +units=m"


def _lonlat_grid(x, y, skew=0.0):
    """2D lon/lat of a projected grid, optionally skewed (curvilinear)."""
    xx, yy = np.meshgrid(x, y)
    xx = xx + skew * (yy - yy[0, 0])
    lon, lat = Transformer.from_crs(
        _PROJ, "EPSG:4326", always_xy=True).transform(xx, yy)
    return lon, lat

# Two catchments over a 1x4 grid: catchment 0 covers cells 0-1, catchment 1
# covers cells 2-3, each cell weighted equally within its catchment.
WEIGHTS = sparse.csr_matrix(
    np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
)


def _run(row):
    """Aggregate a single time step given as a length-4 cell vector."""
    data = np.asarray(row, dtype=np.float32).reshape(1, 1, 4)
    return weighted_mean(WEIGHTS, data).ravel()


def test_plain_mean():
    np.testing.assert_allclose(_run([1.0, 3.0, 2.0, 6.0]), [2.0, 4.0])


def test_nan_cells_are_renormalized():
    # Catchment 0 has one NaN cell -> mean of the remaining valid cell only.
    out = _run([np.nan, 3.0, 2.0, 6.0])
    np.testing.assert_allclose(out, [3.0, 4.0])


def test_all_missing_gives_nan():
    out = _run([np.nan, np.nan, 2.0, 6.0])
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1], 4.0)


def test_undecoded_fill_sentinel_is_excluded():
    # An undecoded no-data sentinel (float32 max, as in the CombiPrecip grids
    # when a source file omits the _FillValue attribute) must be treated as
    # missing, never averaged into the catchment mean.
    sentinel = np.float32(3.4028235e38)
    assert sentinel >= SENTINEL_GUARD
    # Catchment 0 fully over no-data -> NaN; catchment 1 half no-data ->
    # mean of the single real cell, not a fraction of the sentinel.
    out = _run([sentinel, sentinel, sentinel, 6.0])
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1], 6.0)


def test_axes_from_2d_lonlat_recovers_projected_axes():
    # A grid regular in the projection, stored only as 2D lon/lat (WRF-style),
    # must round-trip back to its 1D projected axes.
    x = 100_000.0 + 3000.0 * np.arange(10)
    y = 200_000.0 + 3000.0 * np.arange(8)
    lon, lat = _lonlat_grid(x, y)
    rx, ry = axes_from_2d_lonlat(lon, lat, _PROJ)
    np.testing.assert_allclose(rx, x, atol=1e-3)
    np.testing.assert_allclose(ry, y, atol=1e-3)


def test_axes_from_2d_lonlat_rejects_curvilinear():
    # A skewed grid is not axis-aligned in the projection -> must be rejected
    # rather than silently mis-weighted.
    x = 100_000.0 + 3000.0 * np.arange(10)
    y = 200_000.0 + 3000.0 * np.arange(8)
    lon, lat = _lonlat_grid(x, y, skew=0.3)
    with pytest.raises(ValueError, match="curvilinear"):
        axes_from_2d_lonlat(lon, lat, _PROJ)
