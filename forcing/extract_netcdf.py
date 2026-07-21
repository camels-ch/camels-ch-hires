"""
Generic extraction of catchment-average time series from gridded netCDF
datasets (regular grid, one file per month or per year).

The gridded values are aggregated to exact area-weighted catchment means and
written as one CSV and/or netCDF file per year with one column/coordinate per
catchment. Variable and dimension names, the file naming pattern and the CRS
handling are all parameters, so any regular-grid netCDF dataset can be
processed; see main.py for the hourly CombiPrecip defaults.

The timestamps are kept as in the source files unless a time shift is given.
For accumulation products (e.g. CombiPrecip) they typically label the END of
the accumulation interval: the row labeled T holds the precipitation summed
over the interval ending at T (verified against the 5-min CombiPrecip
product). Start-labeled datasets (e.g. the TabsH hourly means) can be aligned
on that convention with time_shift=1.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from grid_weights import (
    axes_from_2d_lonlat,
    load_or_compute_weights,
    weighted_mean,
)
from output import write_csv, write_netcdf

logger = logging.getLogger(__name__)


def _deaccumulate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recover per-step amounts from a daily-accumulated field (e.g. ERA5-Land
    total precipitation, which accumulates from 00 UTC and resets each day).

    The value at each hourly step is the accumulation since the last 00:00, so
    the per-hour amount is the difference to the previous step, taking the
    value itself at each accumulation-window start (baseline zero). A window
    starts at the step ending at 01:00 UTC, and also wherever the previous step
    is not exactly one hour earlier (series start or a gap between files), so
    no spurious value is carried across a discontinuity.

    De-accumulation and the area-weighted mean are both linear, so applying it
    to the catchment-average series is equivalent to differencing every grid
    cell first, provided the valid-cell mask is constant in time (it is for the
    static ERA5-Land land/sea mask).
    """
    df = df.sort_index()
    out = df - df.shift(1)
    dt = df.index.to_series().diff()
    window_start = (df.index.hour == 1) | (dt != pd.Timedelta(hours=1))
    out.loc[window_start] = df.loc[window_start]
    # An accumulation is non-decreasing within its window, so the only way to
    # get a negative here is float rounding in the source (a known ERA5
    # artifact of the order of 1e-5 mm); clip it away (NaNs are preserved).
    return out.clip(lower=0.0)


def extract_from_netcdf(
    shapefile: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    year_start: int,
    year_end: int,
    var_name: str,
    dim_time: str = "time",
    dim_x: str = "x",
    dim_y: str = "y",
    file_pattern: str = "{year}{month:02d}.nc",
    output_prefix: str | None = None,
    output_var: str | None = None,
    units: str = "",
    threshold: float = 0.0,
    time_shift: float = 0.0,
    deaccumulate: bool = False,
    scale: float = 1.0,
    data_crs: int | None = None,
    lon2d: str | None = None,
    lat2d: str | None = None,
    grid_proj: str | None = None,
    id_field: str = "EZGNR",
    formats: tuple[str, ...] = ("csv",),
    time_chunk: int = 96,
    cache_dir: str | Path | None = None,
) -> None:
    """
    Extract catchment-average time series from netCDF files and write yearly
    files.

    Parameters
    ----------
    shapefile
        Path to the shapefile with the catchment polygons.
    data_dir
        Directory containing the netCDF files.
    out_dir
        Output directory for the yearly files (<output_prefix>_<year>.csv/.nc).
    year_start, year_end
        Year range to process (inclusive). Years whose outputs all exist are
        skipped.
    var_name
        Name of the variable to extract from the netCDF files.
    dim_time, dim_x, dim_y
        Names of the time and spatial dimensions in the netCDF files.
    file_pattern
        Name of the data files, with '{year}' and '{month}' placeholders
        (e.g. '{year}{month:02d}.nc'). If '{month}' is absent, one file per
        year is expected. The formatted name may contain glob wildcards
        (e.g. 'TabsH_*_{year}{month:02d}010000_*.nc' for files whose names
        embed a varying end date); all matching files are processed in
        sorted order.
    output_prefix
        Prefix of the output file names (default: var_name).
    output_var
        Name of the variable in the output netCDF files (default: var_name).
    units
        Units of the variable, written to the output netCDF files.
    threshold
        Catchment values below this threshold are set to 0, to remove
        small precipitation amounts. Use 0 (default) to disable; keep it
        disabled for non-precipitation variables. NaNs are preserved.
    time_shift
        Hours added to the source timestamps, to align datasets on the
        end-of-interval labeling convention (e.g. 1 for start-labeled
        hourly means such as TabsH). Default: 0 (timestamps kept as in
        the source files).
    deaccumulate
        Treat the variable as a daily-resetting accumulation (as ERA5-Land
        total precipitation, accumulated from 00 UTC) and difference it to
        per-step amounts. Applied to the catchment-average series before the
        time shift and threshold. Default: False (values used as-is).
    scale
        Multiplicative factor applied to the extracted values, e.g. 1000 to
        convert ERA5-Land precipitation from m to mm. Default: 1.0 (no
        scaling). NaNs are preserved.
    data_crs
        EPSG code of the data grid. If provided and different from the
        shapefile CRS, the polygons are brought to the grid CRS before the
        weight computation: for CRS pairs that are exact constant offsets of
        each other (LV03/LV95, e.g. the CombiPrecip LV03-style grid vs. an
        LV95 shapefile) the offset is applied, otherwise the polygons are
        reprojected.
    lon2d, lat2d, grid_proj
        For grids georeferenced only by 2D longitude/latitude arrays (no 1D
        coordinate axes, e.g. the WRF-based CHAPTER output): the names of the
        2D coordinate variables and a PROJ string for the projection the grid
        is regular in. When given, the 1D axes are recovered by projecting
        lon2d/lat2d into grid_proj (which also becomes the CRS the catchment
        polygons are reprojected to), and dim_x/dim_y/data_crs are ignored.
    id_field
        Shapefile attribute used as column names / catchment coordinate.
    formats
        Output formats, any of 'csv' and 'netcdf'.
    time_chunk
        Number of time steps read from the netCDF files at once.
    cache_dir
        Directory for the cached weight matrix (default: out_dir).
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = out_dir
    if output_prefix is None:
        output_prefix = var_name
    if output_var is None:
        output_var = var_name
    if bool(grid_proj) != bool(lon2d and lat2d):
        raise ValueError(
            "lon2d, lat2d and grid_proj must be given together to use "
            "2D-coordinate (WRF-style) grids."
        )

    year_files = _list_year_files(
        data_dir, file_pattern, year_start, year_end)

    x_centers, y_centers = _get_grid_coords(
        year_files, dim_x, dim_y, lon2d, lat2d, grid_proj)
    ids, weights = load_or_compute_weights(
        shapefile,
        id_field,
        x_centers,
        y_centers,
        cache_dir=cache_dir,
        data_crs=data_crs,
        grid_proj=grid_proj,
    )

    for year in range(year_start, year_end + 1):
        out_paths = {}
        if "csv" in formats:
            out_paths["csv"] = out_dir / f"{output_prefix}_{year}.csv"
        if "netcdf" in formats:
            out_paths["netcdf"] = out_dir / f"{output_prefix}_{year}.nc"
        if all(path.exists() for path in out_paths.values()):
            logger.info(f"Skipping {year} (outputs already exist).")
            continue

        frames = []
        for nc_path in year_files[year]:
            if not nc_path.exists():
                logger.warning(f"Missing file: {nc_path}")
                continue

            logger.info(f"Processing {nc_path.name}...")
            frames.append(
                _extract_file(nc_path, weights, ids, x_centers, y_centers,
                              var_name, dim_time, dim_x, dim_y, time_chunk,
                              lon2d, lat2d, grid_proj)
            )

        if not frames:
            logger.warning(f"No data found for {year}.")
            continue

        df = pd.concat(frames)
        if deaccumulate:
            df = _deaccumulate(df)
        if scale != 1.0:
            df = df * scale
        if time_shift != 0:
            df.index = df.index + pd.Timedelta(hours=time_shift)
        if threshold > 0:
            # Remove small precipitation amounts (NaNs are preserved).
            df = df.mask(df < threshold, 0.0)
        if "csv" in out_paths:
            write_csv(df, out_paths["csv"])
        if "netcdf" in out_paths:
            write_netcdf(
                {output_var: df},
                out_paths["netcdf"],
                units=units,
                var_attrs={
                    "long_name": f"catchment-average {var_name}",
                },
            )


def _read_axes(
    ds: xr.Dataset,
    dim_x: str,
    dim_y: str,
    lon2d: str | None,
    lat2d: str | None,
    grid_proj: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 1D grid axes (x_centers, y_centers) of an open dataset.

    Regular datasets carry them as 1D coordinate variables (`dim_x`/`dim_y`).
    Grids georeferenced only by 2D lon/lat arrays (`lon2d`/`lat2d`, e.g. WRF
    output) supply `grid_proj`, and the axes are recovered by projecting those
    coordinates into that CRS.
    """
    if grid_proj is not None:
        return axes_from_2d_lonlat(
            ds[lon2d].values, ds[lat2d].values, grid_proj)
    return ds[dim_x].values, ds[dim_y].values


def _extract_file(
    nc_path: Path,
    weights,
    ids: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    var_name: str,
    dim_time: str,
    dim_x: str,
    dim_y: str,
    time_chunk: int,
    lon2d: str | None = None,
    lat2d: str | None = None,
    grid_proj: str | None = None,
) -> pd.DataFrame:
    """Extract the catchment means for one netCDF file."""
    with xr.open_dataset(nc_path) as ds:
        x_axis, y_axis = _read_axes(ds, dim_x, dim_y, lon2d, lat2d, grid_proj)
        if not (
            np.allclose(x_axis, x_centers) and np.allclose(y_axis, y_centers)
        ):
            raise RuntimeError(
                f"The grid of {nc_path} does not match the reference grid."
            )

        time = pd.DatetimeIndex(ds[dim_time].values)
        var = ds[var_name]

        blocks = []
        for i in range(0, len(time), time_chunk):
            data = var.isel({dim_time: slice(i, i + time_chunk)}).to_numpy()
            blocks.append(weighted_mean(weights, data))

    return pd.DataFrame(np.vstack(blocks), index=time, columns=ids)


def _list_year_files(
    data_dir: Path, file_pattern: str, year_start: int, year_end: int
) -> dict[int, list[Path]]:
    """
    Build the expected data file paths per year from the file pattern.

    Formatted names containing glob wildcards ('*', '?' or '[') are expanded
    against data_dir (sorted); names without wildcards are kept as paths even
    if the file is missing, so that the caller can report it.

    Returns
    -------
    dict
        Mapping of year to the list of file paths (one per month if the
        pattern contains a '{month}' placeholder, one per year otherwise).
    """
    monthly = "{month" in file_pattern
    year_files = {}
    for year in range(year_start, year_end + 1):
        if monthly:
            names = [
                file_pattern.format(year=year, month=month)
                for month in range(1, 13)
            ]
        else:
            names = [file_pattern.format(year=year)]

        paths = []
        for name in names:
            if any(char in name for char in "*?["):
                matches = sorted(data_dir.glob(name))
                if not matches:
                    logger.warning(
                        f"No file matching {name} in {data_dir}.")
                paths.extend(matches)
            else:
                paths.append(data_dir / name)
        year_files[year] = paths
    return year_files


def _get_grid_coords(
    year_files: dict[int, list[Path]],
    dim_x: str,
    dim_y: str,
    lon2d: str | None = None,
    lat2d: str | None = None,
    grid_proj: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the grid coordinates from the first available data file."""
    for paths in year_files.values():
        for nc_path in paths:
            if nc_path.exists():
                with xr.open_dataset(nc_path) as ds:
                    x_axis, y_axis = _read_axes(
                        ds, dim_x, dim_y, lon2d, lat2d, grid_proj)
                    return np.asarray(x_axis).copy(), np.asarray(y_axis).copy()
    raise FileNotFoundError("No data file found for the requested years.")
