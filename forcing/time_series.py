import concurrent.futures
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rioxarray as rxr
import pyproj


logger = logging.getLogger(__name__)

class TimeSeries2D:
    """
    Class for generic 2D time series data

    Notes
    -----
    This file original content was extracted from hydrobricks
    (https://github.com/hydrobricks/hydrobricks)
    """

    def __init__(self) -> None:
        """Initialize 2D TimeSeries."""
        super().__init__()
        self.time: list[Any] | pd.Series | pd.DatetimeIndex = []
        self.data: list[np.ndarray] = []
        self.data_name: list[str] = []

    def regrid_from_netcdf(
        self,
        path: str | Path,
        file_pattern: str | None = None,
        data_crs: int | None = None,
        var_name: str | None = None,
        dim_time: str = "time",
        dim_x: str = "x",
        dim_y: str = "y",
        hydro_units: Any | None = None,
        raster_hydro_units: str | Path | None = None,
        weights_block_size: int = 100,
    ) -> None:
        """
        Regrid time series data from netCDF files. The spatialization is done using a
        raster of hydro unit IDs. The meteorological data is resampled to the DEM
        resolution.

        Parameters
        ----------
        path
            Path to a netCDF file containing the data or to a folder containing
            multiple files.
        file_pattern
            Glob pattern of the files to read (e.g., '*.nc'). If None, the path is
            considered to be a single file.
        data_crs
            CRS of the netCDF file (as EPSG code).
            If None, the CRS is read from the file.
        var_name
            Name of the variable to read from the netCDF file.
        dim_time
            Name of the time dimension. Default: 'time'
        dim_x
            Name of the x/longitude dimension. Default: 'x'
        dim_y
            Name of the y/latitude dimension. Default: 'y'
        hydro_units
            HydroUnits object containing the hydro units to use for the spatialization.
            Needed if apply_data_gradient is True.
        raster_hydro_units
            Path to a raster file containing the hydro unit IDs to use for the
            spatialization.
        weights_block_size
            Size of the block of time steps to use for weight computation.
            Default: 100

        Raises
        ------
        RuntimeError
            If raster_hydro_units is not provided, or if time/spatial dimensions
            don't match.
        """
        if raster_hydro_units is None:
            raise RuntimeError(
                "You must provide a raster of the hydro units.",
            )

        # Get unit ids
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
            unit_ids = rxr.open_rasterio(raster_hydro_units)
            unit_ids = unit_ids.squeeze().drop_vars("band")

        logger.debug(
            "Starting regridding from netCDF"
        )

        # Get netCDF dataset
        logger.debug(f"Reading netcdf file(s) from {path}...")
        if file_pattern is None:
            nc_data = xr.open_dataset(path, chunks={})
        else:
            files = sorted(Path(path).glob(file_pattern))
            logger.debug(f"Found {len(files)} files matching pattern '{file_pattern}'")
            nc_data = xr.open_mfdataset(files, chunks={})

        # Get CRS of the netcdf file
        data_crs = self._parse_crs(nc_data, data_crs)
        logger.debug(f"NetCDF CRS: {data_crs}")

        # Get CRS of the unit ids raster
        unit_ids_crs = self._parse_crs(unit_ids, None)
        logger.debug(f"Raster CRS: {unit_ids_crs}")

        if data_crs != unit_ids_crs:
            logger.warning(
                "The CRS of the netcdf file does not match the CRS of the "
                "hydro unit ids raster. Reprojection will be done from "
                f"{unit_ids_crs} to {data_crs}."
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                unit_ids = unit_ids.rio.reproject(f"epsg:{data_crs}")

        # Get list of hydro unit ids
        unit_ids_list = hydro_units["id"].values.squeeze()
        unit_id_count = len(unit_ids_list)
        logger.debug(f"Processing {unit_id_count} hydro units")

        time_method = "full"
        time_nc = nc_data.variables[dim_time][:]
        logger.debug(f"Using full time series with {len(time_nc)} time steps")

        if len(self.time) == 0:
            self.time = pd.Series(time_nc)

        # Check if the time steps are the same
        if len(self.time) != len(time_nc):
            raise RuntimeError(
                f"The length of the netcdf time series ({len(time_nc)}) "
                f"does not match the hydro units data ({len(self.time)})."
            )
        if self.time[0] != time_nc[0]:
            raise RuntimeError(
                f"The first time step of the netcdf time series "
                f"({time_nc[0].data}) does not match the one from the "
                f"hydro units data ({self.time[0]})."
            )
        if self.time[len(self.time) - 1] != time_nc[len(time_nc) - 1]:
            raise RuntimeError(
                f"The last time step of the netcdf time series "
                f"({time_nc[len(time_nc) - 1].data}) does not match "
                f"the one from the hydro units data "
                f"({self.time[len(self.time) - 1]})."
            )

        # Extract the unit id masks
        unit_id_masks = []
        for unit_id in unit_ids_list:
            unit_id_mask = xr.where(unit_ids == unit_id, 1, 0)
            unit_id_masks.append(unit_id_mask)

        # Initialize data array
        data = np.zeros((len(self.time), unit_id_count))
        self.data.append(data)

        # Drop other variables
        other_coords = [
            v
            for v in nc_data.coords
            if v not in [dim_time, dim_x, dim_y, "day_of_year"]
        ]
        nc_data = nc_data.drop_vars(other_coords)

        # Extract variable
        data_var = nc_data[var_name]

        # Specify the CRS if not specified
        if data_var.rio.crs is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                data_var.rio.write_crs(f"epsg:{data_crs}", inplace=True)

        # Get the spatial extent of interest
        ref_data = unit_ids
        data_var = self._select_relevant_extent(
            data_var, data_crs, dim_x, dim_y, ref_data
        )

        # Rename spatial dimensions
        if dim_x != "x":
            data_var = data_var.rename({dim_x: "x"})
        if dim_y != "y":
            data_var = data_var.rename({dim_y: "y"})

        # Time the computation
        start_time = time.time()

        num_threads = os.cpu_count()
        time_len = len(self.time)

        # Create a xarray variable containing the data cell indices
        data_idx = data_var[0].copy()
        data_idx.values = np.arange(data_idx.size).reshape(data_idx.shape)
        data_idx = data_idx.astype(float)

        # Reproject the data cell indices to the hydro unit raster
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
            data_idx.rio.write_crs(f"epsg:{data_crs}", inplace=True)
            data_idx_reproj = data_idx.rio.reproject_match(
                unit_ids, Resampling=rasterio.enums.Resampling.nearest
            )

        # Create the masks (with the original data shape) for each unit with the
        # weights to apply to the gridded data contributing to the unit
        unit_weights = []
        for u in range(unit_id_count):
            # Get the data indices contributing to the unit
            mask_unit_id = xr.where(unit_id_masks[u], data_idx_reproj, -1)
            mask_unit_id = mask_unit_id.to_numpy().astype(int)
            # Get unique values and their counts
            data_idx_values, counts = np.unique(
                mask_unit_id[mask_unit_id >= 0], return_counts=True
            )
            # Create a mask of the weights
            weights_mask = np.zeros(data_idx.shape)
            data_idx_values = np.unravel_index(data_idx_values, data_idx.shape)
            weights_mask[data_idx_values] = counts / np.sum(counts)

            assert np.isclose(np.sum(weights_mask), 1)

            # Add the mask to the list
            unit_weights.append(weights_mask)

        n_steps = 1 + np.ceil(time_len / weights_block_size).astype(int)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit the tasks for each time step to the executor
            futures = [
                executor.submit(
                    self._extract_time_step_data_weights,
                    data_var,
                    unit_weights,
                    t_block,
                    weights_block_size,
                )
                for t_block in range(n_steps)
            ]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

        # Print elapsed time
        elapsed_time = time.time() - start_time
        logger.info(
            f"Elapsed time: {elapsed_time:.2f} seconds "
            f"(using {num_threads} threads)"
        )

    def _extract_time_step_data_weights(
        self, data_var: xr.DataArray, unit_weights: list, i_block: int, block_size: int
    ) -> None:
        """
        Extract time step data and apply spatial weights.

        Extracts meteorological data for a block of time steps and applies weighted
        averaging based on the spatial distribution of data cells within each
        hydro unit.

        Parameters
        ----------
        data_var
            3D xarray DataArray with dimensions (time, y, x) containing gridded data.
        unit_weights
            List of 2D weight arrays for each hydro unit, summing to 1.
        i_block
            Block index for this processing step.
        block_size
            Number of time steps to process in this block.
        """
        i_start = i_block * block_size
        i_end = min((i_block + 1) * block_size, len(self.time))
        i_end = min(i_end, data_var.shape[0])
        if i_start >= len(self.time):
            return

        logger.debug(f"Extracting {self.time[i_start]}")

        # Extract data for each unit
        for u, unit_weight in enumerate(unit_weights):
            # Mask the meteorological data with the unit weights.
            self.data[-1][i_start:i_end, u] = np.nansum(
                data_var[i_start:i_end].to_numpy() * unit_weight, axis=(1, 2)
            )

    def _select_relevant_extent(
        self,
        data_var: xr.DataArray,
        data_crs: int,
        dim_x: str,
        dim_y: str,
        ref_data: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray:
        """
        Select the spatial extent of gridded data relevant to the reference data.

        Clips the input data to the bounding box of the reference data
        (DEM or hydro units), handling CRS transformations if necessary.

        Parameters
        ----------
        data_var
            The gridded data variable to clip.
        data_crs
            CRS of the gridded data (as EPSG code).
        dim_x
            Name of the x/longitude dimension in data_var.
        dim_y
            Name of the y/latitude dimension in data_var.
        ref_data
            Reference dataset (DEM or hydro units raster) to determine extent.

        Returns
        -------
        xr.DataArray
            Clipped data variable containing only the relevant spatial extent.
        """
        x_ref_min, x_ref_max, y_ref_min, y_ref_max = self._get_spatial_bounds(ref_data)

        # Convert the spatial extent to the data CRS
        src_crs = self._parse_crs(ref_data)
        if src_crs != data_crs:
            transformer = pyproj.Transformer.from_crs(src_crs, data_crs, always_xy=True)
            x_min_dat, y_min_dat = transformer.transform(x_ref_min, y_ref_min)
            x_max_dat, y_max_dat = transformer.transform(x_ref_max, y_ref_max)
        else:
            x_min_dat, y_min_dat = x_ref_min, y_ref_min
            x_max_dat, y_max_dat = x_ref_max, y_ref_max

        # Find the coordinates that cover the extent
        x_coords = data_var[dim_x].values
        y_coords = data_var[dim_y].values

        x_start_idx = np.searchsorted(x_coords, x_min_dat, side="right") - 1
        x_end_idx = np.searchsorted(x_coords, x_max_dat, side="left")
        y_start_idx = np.searchsorted(y_coords, y_min_dat, side="right") - 1
        y_end_idx = np.searchsorted(y_coords, y_max_dat, side="left")

        x_start = x_coords[max(x_start_idx, 0)]
        x_end = x_coords[min(x_end_idx, len(x_coords) - 1)]
        y_start = y_coords[max(y_start_idx, 0)]
        y_end = y_coords[min(y_end_idx, len(y_coords) - 1)]

        x_sel = slice(min(x_start, x_end), max(x_start, x_end))
        y_sel = slice(min(y_start, y_end), max(y_start, y_end))

        data_var = data_var.sel({dim_x: x_sel, dim_y: y_sel})

        return data_var

    @staticmethod
    def _parse_crs(data: xr.DataArray | xr.Dataset, file_crs: int | None = None) -> int:
        """
        Extract CRS information from xarray data.

        Attempts to retrieve CRS from multiple sources: explicit parameter,
        data attributes, or rioxarray crs property.
        Raises error if CRS cannot be determined.

        Parameters
        ----------
        data
            xarray DataArray or Dataset to extract CRS from.
        file_crs
            Explicit CRS as EPSG code. If provided, this value is returned directly.

        Returns
        -------
        int
            CRS as EPSG code.

        Raises
        ------
        RuntimeError
            If no CRS is found and file_crs is not provided.
        """
        if file_crs is None:
            if "crs" in data.attrs:
                # Try to get it from the global attributes
                return data.attrs["crs"]
            elif data.rio.crs:
                # Try to get it from the rio crs
                return data.rio.crs.to_epsg()
            else:
                raise RuntimeError(
                    "Could not determine the CRS from the data."
                    "Please provide a CRS (option 'file_crs')."
                )
        return file_crs

    @staticmethod
    def _get_spatial_bounds(ref_data: xr.DataArray | xr.Dataset) -> tuple:
        """
        Extract spatial bounds from xarray data.

        Determines the minimum and maximum coordinates in x and y dimensions,
        automatically detecting dimension names (x/lon/longitude, y/lat/latitude).

        Parameters
        ----------
        ref_data
            xarray DataArray or Dataset containing spatial data.

        Returns
        -------
        tuple
            Tuple of (x_min, x_max, y_min, y_max) spatial bounds.

        Raises
        ------
        RuntimeError
            If spatial dimensions cannot be found in the data.
        """
        # Possible names for spatial dimensions
        x_names = ["x", "lon", "longitude"]
        y_names = ["y", "lat", "latitude"]

        # Find the actual dimension names
        x_dim = next((name for name in x_names if name in ref_data.dims), None)
        y_dim = next((name for name in y_names if name in ref_data.dims), None)

        if x_dim is None or y_dim is None:
            raise RuntimeError(
                f"Could not find spatial dimensions in the reference data. "
                f"Available dimensions: {list(ref_data.dims)}"
            )

        x_ref_min = ref_data[x_dim].min().item()
        x_ref_max = ref_data[x_dim].max().item()
        y_ref_min = ref_data[y_dim].min().item()
        y_ref_max = ref_data[y_dim].max().item()

        return x_ref_min, x_ref_max, y_ref_min, y_ref_max
