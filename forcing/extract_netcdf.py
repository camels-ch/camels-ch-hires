"""
Generic extraction of catchment-average time series from gridded netCDF
datasets (regular grid, one file per month or per year).

The gridded values are aggregated to exact area-weighted catchment means and
written as one CSV and/or netCDF file per year with one column/coordinate per
catchment. Variable and dimension names, the file naming pattern and the CRS
handling are all parameters, so any regular-grid netCDF dataset can be
processed; see main.py for the hourly CombiPrecip defaults.

The timestamps are kept as in the source files. For accumulation products
(e.g. CombiPrecip) they typically label the END of the accumulation interval:
the row labeled T holds the precipitation summed over the interval ending
at T (verified against the 5-min CombiPrecip product).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from grid_weights import load_or_compute_weights, weighted_mean
from output import write_csv, write_netcdf

logger = logging.getLogger(__name__)


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
    data_crs: int | None = None,
    coord_shift: tuple[float, float] = (0.0, 0.0),
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
        year is expected.
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
    data_crs
        EPSG code of the data grid. If provided and different from the
        shapefile CRS, the polygons are reprojected before the weight
        computation. Mutually exclusive with coord_shift.
    coord_shift
        (x, y) offset added to the grid coordinates to express them in the
        shapefile CRS. Use when the grid is defined as an exact shift of the
        shapefile CRS (e.g. CombiPrecip LV03-style coordinates = LV95 -
        2'000'000/1'000'000). Mutually exclusive with data_crs.
    id_field
        Shapefile attribute used as column names / catchment coordinate.
    formats
        Output formats, any of 'csv' and 'netcdf'.
    time_chunk
        Number of time steps read from the netCDF files at once.
    cache_dir
        Directory for the cached weight matrix (default: out_dir).
    """
    if data_crs is not None and coord_shift != (0.0, 0.0):
        raise ValueError("Provide either data_crs or coord_shift, not both.")

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = out_dir
    if output_prefix is None:
        output_prefix = var_name
    if output_var is None:
        output_var = var_name

    year_files = _list_year_files(
        data_dir, file_pattern, year_start, year_end)

    x_centers, y_centers = _get_grid_coords(year_files, dim_x, dim_y)
    ids, weights = load_or_compute_weights(
        shapefile,
        id_field,
        x_centers + coord_shift[0],
        y_centers + coord_shift[1],
        cache_dir=cache_dir,
        to_crs=data_crs,
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
                              var_name, dim_time, dim_x, dim_y, time_chunk)
            )

        if not frames:
            logger.warning(f"No data found for {year}.")
            continue

        df = pd.concat(frames)
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
) -> pd.DataFrame:
    """Extract the catchment means for one netCDF file."""
    with xr.open_dataset(nc_path) as ds:
        if not (
            np.allclose(ds[dim_x].values, x_centers)
            and np.allclose(ds[dim_y].values, y_centers)
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
            year_files[year] = [
                data_dir / file_pattern.format(year=year, month=month)
                for month in range(1, 13)
            ]
        else:
            year_files[year] = [data_dir / file_pattern.format(year=year)]
    return year_files


def _get_grid_coords(
    year_files: dict[int, list[Path]], dim_x: str, dim_y: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read the grid coordinates from the first available data file."""
    for paths in year_files.values():
        for nc_path in paths:
            if nc_path.exists():
                with xr.open_dataset(nc_path) as ds:
                    return ds[dim_x].values.copy(), ds[dim_y].values.copy()
    raise FileNotFoundError("No data file found for the requested years.")
