import logging
from typing import Any
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from cftime import num2date
from netcdf4 import Dataset

from time_series import TimeSeries2D

logger = logging.getLogger(__name__)


class Forcing:
    """
    Class for managing forcing (meteorological) data.

    Notes
    -----
    This file original content was extracted from hydrobricks
    (https://github.com/hydrobricks/hydrobricks)
    """

    def __init__(self, entities_file_path: str | Path) -> None:
        """
        Initialize Forcing object for a spatial entity.

        Parameters
        ----------
        entities_file_path
            Path to a shapefile with the hydrological entities (e.g. catchments)
        """
        super().__init__()
        self.crs: str | None = None
        self.entities: list | None = None
        self.data2D: TimeSeries2D = TimeSeries2D()
        self._operations: list[dict[str, Any]] = []

        self._extract_entities(entities_file_path)

    def extract_from_gridded_data(self, **kwargs: Any) -> None:
        """
        Define the spatialization operations from gridded data to all hydro units.

        Parameters
        ----------
        variable : str
            Name of the variable to spatialize.
        method : str
            Name of the method to use. Can be:
            * regrid_from_netcdf: regrid data from a single or multiple netCDF files.
        path : str|Path
            Path to the file containing the data or to a folder containing multiple
            files.
        file_pattern : str, optional
            Pattern of the files to read. If None, the path is considered to be
            a single file.
        data_crs : int, optional
            CRS (as EPSG id) of the data file. If None, the CRS is read from the file.
        var_name : str
            Name of the variable to read.
        dim_time : str
            Name of the time dimension.
        dim_x : str
            Name of the x dimension.
        dim_y : str
            Name of the y dimension.
        raster_hydro_units : str|Path
            Path to a raster containing the hydro unit ids to use for the
            spatialization.
        apply_data_gradient : bool, optional
            If True, elevation-based gradients will be retrieved from the data and
            applied to the hydro units (e.g., for temperature and precipitation).
            If False, the data will be regridded without applying any gradient.
            Default is True for temperature and precipitation variables, and False
            for other variables.
        """
        self._operations.append(kwargs)

        logger.debug(f"  Applying spatialize_from_grid operations")

        for operation_ref in self._operations:
            operation = operation_ref.copy()
            self._extract_from_gridded_data(**operation)

    def save_as(self, path: str | Path, max_compression: bool = False) -> None:
        """
        Create a netCDF file with the forcing data.

        Saves the 2D spatialized forcing data to a netCDF4 file with the structure
        suitable for later loading with load_from().

        Parameters
        ----------
        path
            Path of the file to create.
        max_compression
            Option to allow maximum compression for data in file. When True, uses
            compression with least_significant_digit=3 for better storage efficiency.
            Default: False

        Notes
        -----
        If apply_operations() has not been called, it will be called automatically
        before saving to ensure data is properly spatialized.
        """
        if not self.is_initialized():
            logger.info("Applying operations before saving...")
            self.apply_operations()
            self._is_initialized = True

        time = self.data2D.get_dates_as_mjd()

        # Create netCDF file using context manager for proper resource handling
        logger.debug(f"Creating netCDF file: {path}")
        with Dataset(path, "w", "NETCDF4") as nc:
            # Dimensions
            nc.createDimension("hydro_units", len(self.hydro_units))
            nc.createDimension("time", len(time))

            # Variables
            var_id = nc.createVariable("id", "int", ("hydro_units",))
            var_id[:] = self.hydro_units["id"]

            var_time = nc.createVariable("time", "float32", ("time",))
            var_time[:] = time
            var_time.units = "days since 1858-11-17 00:00:00"
            var_time.comment = "Modified Julian Day Number"

            for idx, variable in enumerate(self.data2D.data_name):
                if max_compression:
                    var_data = nc.createVariable(
                        variable,
                        "float32",
                        ("time", "hydro_units"),
                        zlib=True,
                        least_significant_digit=3,
                    )
                else:
                    var_data = nc.createVariable(
                        variable, "float32", ("time", "hydro_units"), zlib=True
                    )
                var_data[:, :] = self.data2D.data[idx]

        logger.debug("NetCDF file created successfully")

    def _extract_entities(self, entities: str | Path) -> None:
        """
        Extract catchment outline geometry from shapefile.

        Reads the shapefile, validates CRS, and stores the polygon geometries.

        Parameters
        ----------
        entities
            Path to shapefile containing the entities
        """
        shapefile = gpd.read_file(entities)
        self._check_crs(shapefile)
        geoms = shapefile.geometry.values
        self.outline = geoms

    def _check_crs(self, data: rasterio.DatasetReader | gpd.GeoDataFrame) -> None:
        """
        Check and validate CRS consistency with the catchment.

        Sets the catchment CRS if not already set, or verifies that the data CRS
        matches the catchment CRS.

        Parameters
        ----------
        data
            Rasterio dataset or GeoDataFrame to check CRS against.

        Raises
        ------
        RuntimeError
            If data CRS doesn't match the catchment CRS.
        """
        data_crs = self._get_crs_from_file(data)
        if self.crs is None:
            self.crs = data_crs
        else:
            if self.crs != data_crs:
                raise RuntimeError(
                    "The CRS of the data does not match the CRS of the catchment."
                )

    @staticmethod
    def _get_crs_from_file(
            data: rasterio.DatasetReader | gpd.GeoDataFrame) -> str:
        """
        Extract CRS from a rasterio dataset or GeoDataFrame.

        Parameters
        ----------
        data
            Rasterio dataset or GeoDataFrame to extract CRS from.

        Returns
        -------
        str
            Coordinate Reference System identifier.

        Raises
        ------
        RuntimeError
            If data format is not recognized.
        """
        if isinstance(data, rasterio.DatasetReader):
            return data.crs
        elif isinstance(data, gpd.GeoDataFrame):
            return data.crs
        else:
            raise RuntimeError("Unknown data format.")

    def _extract_from_gridded_data(
        self, variable: str, **kwargs: Any
    ) -> None:
        """
        Extract forcing data for the provided entities.

        Parameters
        ----------
        variable
            Name or alias of the variable to spatialize.
        **kwargs
            Parameters for the regridding operation including:
            - path: Path to netCDF file(s)
            - file_pattern: Glob pattern for multiple files
            - data_crs: CRS of the data (as EPSG code)
            - var_name: Name of variable in the netCDF file
            - raster_hydro_units: Path to hydro unit IDs raster
            - apply_data_gradient: Whether to apply elevation gradients
            - gradient_type: 'additive' or 'multiplicative'
        """
        variable = self.get_variable_enum(variable)

        path = kwargs.get("path", "")
        file_pattern = kwargs.get("file_pattern", None)
        data_crs = kwargs.get("data_crs", None)
        var_name = kwargs.get("var_name", "")
        dim_time = kwargs.get("dim_time", "time")
        dim_x = kwargs.get("dim_x", "x")
        dim_y = kwargs.get("dim_y", "y")
        raster_hydro_units = kwargs.get("raster_hydro_units", "")

        self.data2D.regrid_from_netcdf(
            path,
            file_pattern=file_pattern,
            data_crs=data_crs,
            var_name=var_name,
            dim_time=dim_time,
            dim_x=dim_x,
            dim_y=dim_y,
            hydro_units=self.hydro_units,
            raster_hydro_units=raster_hydro_units,
        )
        self.data2D.data_name.append(variable)
