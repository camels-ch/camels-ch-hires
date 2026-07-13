"""
Writers for the extracted catchment time series (CSV and netCDF).
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

FORMATS = ("csv", "netcdf")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write one (time x catchments) data frame to a CSV file."""
    df.index.name = "time"
    df.to_csv(path, float_format="%.3f")
    logger.info(f"Wrote {path}")


def write_netcdf(
    data: dict[str, pd.DataFrame],
    path: Path,
    units: str,
    var_attrs: dict[str, str] | None = None,
) -> None:
    """
    Write (time x catchments) data frames to a compressed netCDF file.

    Parameters
    ----------
    data
        Mapping of variable name to data frame. All frames must share the same
        index (time) and columns (catchment ids).
    path
        Path of the netCDF file to create.
    units
        Units of the variables (e.g. 'mm' or 'mm h-1').
    var_attrs
        Optional additional attributes set on every variable.
    """
    first = next(iter(data.values()))
    ds = xr.Dataset(
        {
            name: (("time", "catchment"), df.to_numpy(dtype="float32"))
            for name, df in data.items()
        },
        coords={"time": first.index, "catchment": first.columns.to_numpy()},
    )
    for name in data:
        ds[name].attrs["units"] = units
        ds[name].attrs.update(var_attrs or {})
    ds["time"].attrs["comment"] = (
        "Timestamps label the end of the aggregation interval."
    )

    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in data
    }
    ds.to_netcdf(path, encoding=encoding)
    logger.info(f"Wrote {path}")
