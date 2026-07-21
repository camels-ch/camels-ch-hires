"""
Exact area-weighted extraction of catchment values from a regular grid.

The weights of each grid cell contributing to a catchment are computed from the
exact intersection area between the catchment polygon and the cell polygon, so
partly-covered cells contribute proportionally to their covered fraction.
"""

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from scipy import sparse

logger = logging.getLogger(__name__)

# CRS pairs that are exact constant offsets of each other: the Swiss LV95
# frame (EPSG:2056) equals the LV03 frame (EPSG:21781) shifted by exactly
# +2'000'000 / +1'000'000 m for grids defined that way (e.g. the MeteoSwiss
# products). A true datum reprojection would add sub-metre distortions that
# are meaningless at the km grid scale, so the constant shift is used.
CONSTANT_CRS_SHIFTS = {(21781, 2056): (2_000_000.0, 1_000_000.0)}

# Magnitude above which a source cell is treated as a no-data sentinel rather
# than a measurement. Gridded products flag no-data with a large fill value
# (the CombiPrecip grids use the float32 maximum, 3.4028235e+38); this is
# normally decoded to NaN on read, but a source file that omits the
# _FillValue attribute leaves the raw sentinel in place, and area-averaging it
# into a catchment mean produces huge garbage values (a catchment fully over
# no-data yields the sentinel, one partly covered yields a fraction of it).
SENTINEL_GUARD = 1e20


def _constant_crs_shift(
    from_epsg: int, to_epsg: int
) -> tuple[float, float] | None:
    """The exact (dx, dy) between two CRS, or None if they are not an
    exact-offset pair."""
    if (from_epsg, to_epsg) in CONSTANT_CRS_SHIFTS:
        return CONSTANT_CRS_SHIFTS[(from_epsg, to_epsg)]
    if (to_epsg, from_epsg) in CONSTANT_CRS_SHIFTS:
        dx, dy = CONSTANT_CRS_SHIFTS[(to_epsg, from_epsg)]
        return -dx, -dy
    return None


def compute_weights(
    geometries,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cell_size: float | None = None,
) -> sparse.csr_matrix:
    """
    Compute the exact area weights of the grid cells for each polygon.

    Parameters
    ----------
    geometries
        Sequence of shapely polygons (same CRS as the grid coordinates).
    x_centers
        X coordinates of the grid cell centers (regularly spaced).
    y_centers
        Y coordinates of the grid cell centers (regularly spaced, ascending or
        descending).
    cell_size
        Size of the (square) grid cells in coordinate units, used as a
        validation of the coordinate spacing. If None (default), the cell
        width and height are inferred from the spacing of the x and y
        coordinates (cells are assumed contiguous).

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix of shape (n_polygons, ny * nx) over the row-major flattened grid
        (y first, then x, in the order of the provided coordinate arrays).
        Each row sums to 1.
    """
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)
    nx = len(x_centers)
    ny = len(y_centers)

    dx = _get_regular_step(x_centers, "x")
    dy = _get_regular_step(y_centers, "y")
    if cell_size is not None:
        for step, name in ((dx, "x"), (dy, "y")):
            if not np.isclose(abs(step), cell_size):
                raise ValueError(
                    f"The {name} coordinate spacing ({abs(step)}) does not "
                    f"match the cell size ({cell_size})."
                )
    half_x = abs(dx) / 2.0
    half_y = abs(dy) / 2.0

    rows = []
    cols = []
    vals = []
    for i_poly, poly in enumerate(geometries):
        x_min, y_min, x_max, y_max = poly.bounds

        ix0, ix1 = _index_range(x_min, x_max, x_centers[0], dx, nx)
        iy0, iy1 = _index_range(y_min, y_max, y_centers[0], dy, ny)
        if ix0 > ix1 or iy0 > iy1:
            raise ValueError(
                f"Polygon {i_poly} does not overlap the grid extent "
                f"(bounds: {poly.bounds})."
            )

        xs, ys = np.meshgrid(x_centers[ix0:ix1 + 1], y_centers[iy0:iy1 + 1])
        cells = shapely.box(xs - half_x, ys - half_y, xs + half_x, ys + half_y)
        areas = shapely.area(shapely.intersection(cells, poly))

        total = areas.sum()
        if total <= 0:
            raise ValueError(
                f"Polygon {i_poly} does not intersect any grid cell "
                f"(bounds: {poly.bounds})."
            )

        iy_idx, ix_idx = np.nonzero(areas)
        weights = areas[iy_idx, ix_idx] / total
        flat_idx = (iy_idx + iy0) * nx + (ix_idx + ix0)

        rows.append(np.full(len(flat_idx), i_poly))
        cols.append(flat_idx)
        vals.append(weights)

    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(geometries), ny * nx),
    )

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0), "Weight rows do not sum to 1."

    return matrix


def load_or_compute_weights(
    shapefile: str | Path,
    id_field: str,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cell_size: float | None = None,
    cache_dir: str | Path | None = '.cache',
    data_crs: int | None = None,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """
    Load the weight matrix from cache or compute (and cache) it.

    Parameters
    ----------
    shapefile
        Path to the shapefile with the catchment polygons.
    id_field
        Name of the attribute holding the unique catchment id.
    x_centers, y_centers, cell_size
        Grid definition (see compute_weights).
    cache_dir
        Directory for the cached weights. If None, no caching is done.
    data_crs
        EPSG code of the grid coordinates. If provided and different from
        the shapefile CRS, the polygons are brought to the grid CRS before
        the weight computation: for CRS pairs that are exact constant
        offsets of each other (LV03/LV95, EPSG 21781/2056) the offset is
        applied, otherwise the polygons are reprojected. If None, the grid
        is assumed to be in the CRS of the shapefile.

    Returns
    -------
    (ids, weights)
        The catchment ids (in row order of the matrix) and the sparse weight
        matrix of shape (n_catchments, ny * nx).
    """
    shapefile = Path(shapefile)
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(
            shapefile, id_field, x_centers, y_centers, cell_size, data_crs)
        cache_path = Path(cache_dir) / f"weights_{shapefile.stem}_{key}.npz"
        if cache_path.exists():
            logger.info(f"Loading cached weights from {cache_path}")
            cached = np.load(cache_path)
            weights = sparse.csr_matrix(
                (cached["data"], cached["indices"], cached["indptr"]),
                shape=tuple(cached["shape"]),
            )
            return cached["ids"], weights

    logger.info(f"Reading catchments from {shapefile}")
    gdf = gpd.read_file(shapefile)
    if not gdf[id_field].is_unique:
        raise ValueError(f"The field '{id_field}' is not unique in {shapefile}.")
    ids = gdf[id_field].to_numpy()

    shp_crs = gdf.crs.to_epsg()
    if data_crs is not None and shp_crs != data_crs:
        shift = _constant_crs_shift(shp_crs, data_crs)
        if shift is not None:
            logger.info(
                f"Translating catchments by {shift} "
                f"(EPSG:{shp_crs} -> EPSG:{data_crs}).")
            gdf.geometry = gdf.geometry.translate(*shift)
        else:
            logger.info(f"Reprojecting catchments to EPSG:{data_crs}")
            gdf = gdf.to_crs(epsg=data_crs)

    logger.info(f"Computing exact weights for {len(gdf)} catchments...")
    weights = compute_weights(gdf.geometry.values, x_centers, y_centers, cell_size)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            data=weights.data,
            indices=weights.indices,
            indptr=weights.indptr,
            shape=np.array(weights.shape),
            ids=ids,
        )
        logger.info(f"Cached weights to {cache_path}")

    return ids, weights


def weighted_mean(weights: sparse.csr_matrix, data: np.ndarray) -> np.ndarray:
    """
    Apply the weights to gridded data, renormalizing over valid (non-NaN) cells.

    Parameters
    ----------
    weights
        Sparse weight matrix of shape (n_catchments, ny * nx).
    data
        Gridded data of shape (n_time, ny, nx). Missing cells are marked by
        NaN or by an undecoded no-data sentinel (magnitude >= SENTINEL_GUARD);
        both are excluded and the weights renormalized over the valid cells.

    Returns
    -------
    np.ndarray
        Array of shape (n_time, n_catchments). A value is NaN only when all
        cells contributing to the catchment are missing at that time step.
    """
    n_time = data.shape[0]
    flat = data.reshape(n_time, -1)
    # Exclude NaN/inf and undecoded no-data sentinels (see SENTINEL_GUARD), so
    # a source file missing its _FillValue attribute cannot leak the raw fill
    # value into the area-weighted mean.
    valid = np.isfinite(flat) & (np.abs(flat) < SENTINEL_GUARD)

    num = weights @ np.where(valid, flat, 0.0).T  # (n_catchments, n_time)
    den = weights @ valid.T.astype(np.float64)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)

    return out.T


def _get_regular_step(coords: np.ndarray, name: str) -> float:
    """Return the (signed) step of a regular coordinate axis, validating it."""
    steps = np.diff(coords)
    step = np.median(steps)
    # Coordinates stored as float32 (e.g. ERA5's 0.1 deg grid) make the step
    # wobble by a few 1e-6; allow deviations up to 0.1 % of the step so such
    # grids are accepted while genuinely irregular axes are still rejected.
    if not np.allclose(steps, step, rtol=1e-3, atol=0.0):
        raise ValueError(f"The {name} coordinates are not regularly spaced.")
    return float(step)


def _index_range(
    v_min: float, v_max: float, origin: float, step: float, n: int
) -> tuple[int, int]:
    """
    Indices of the cells whose extent overlaps [v_min, v_max] on one axis.

    Cell i covers [origin + (i - 0.5) * step, origin + (i + 0.5) * step]
    (bounds swapped when step is negative).
    """
    t0 = (v_min - origin) / step
    t1 = (v_max - origin) / step
    t_min, t_max = min(t0, t1), max(t0, t1)
    i0 = max(int(np.ceil(t_min - 0.5)), 0)
    i1 = min(int(np.floor(t_max + 0.5)), n - 1)
    return i0, i1


def _cache_key(
    shapefile: Path,
    id_field: str,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cell_size: float | None,
    data_crs: int | None,
) -> str:
    """Hash the shapefile identity and grid definition for cache naming."""
    stat = shapefile.stat()
    parts = (
        f"{shapefile.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{id_field}|"
        f"{len(x_centers)}|{len(y_centers)}|{x_centers[0]}|{y_centers[0]}|"
        f"{x_centers[-1]}|{y_centers[-1]}|{cell_size}|{data_crs}"
    )
    return hashlib.sha1(parts.encode()).hexdigest()[:12]
