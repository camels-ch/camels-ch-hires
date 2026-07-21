"""
Command-line entry point for the catchment forcing extraction.

Examples
--------
Hourly CombiPrecip (netCDF grids) to hourly catchment CSVs/netCDFs
(the defaults of the 'netcdf' command target this dataset; its grid uses
LV03-style coordinates, hence --data-crs 21781):
    python main.py netcdf --data-crs 21781 --years 2005 2023
        --out-dir ./output --format csv netcdf

Any other regular-grid netCDF dataset, e.g. daily RhiresD-like files
(LV95 grid like the shapefile, so no --data-crs needed):
    python main.py netcdf --data-dir /path/to/data --file-pattern "{year}.nc"
        --var-name RhiresD --dim-time time --dim-x E --dim-y N
        --prefix RhiresD --units mm --years 1961 2023

Hourly temperature (TabsH, LV95 grid), monthly files whose names embed the
varying last day of the month (glob wildcards):
    python main.py netcdf --data-dir <...>/TabsH_swiss.lv95
        --file-pattern "TabsH_ch01h.swiss.lv95_{year}{month:02d}010000_*.nc"
        --var-name TabsH --dim-time time --dim-x E --dim-y N
        --prefix TabsH --output-var temp --units degC
        --time-shift 1 --years 2018 2023

Hourly statistics (mean, max, q10, q25, q50, q75, q90) of 5-min CombiPrecip:
    python main.py cpc5min --years 2005 2024 --out-dir ./output
"""

import argparse
import logging

from extract_cpc5min import extract_5min_stats
from extract_netcdf import extract_from_netcdf


def main():
    parser = argparse.ArgumentParser(
        description="Extract catchment forcing from grids."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_nc = subparsers.add_parser(
        "netcdf",
        help="Extract time series from netCDF grids (defaults: hourly "
             "CombiPrecip).")
    parser_nc.add_argument(
        "--data-dir",
        help="Directory with the netCDF files.")
    parser_nc.add_argument(
        "--file-pattern", default="{year}{month:02d}.nc",
        help="Data file names, with {year} and optionally {month} "
             "placeholders; glob wildcards (*, ?) are allowed.")
    parser_nc.add_argument(
        "--var-name", default="CPC",
        help="Name of the variable in the netCDF files.")
    parser_nc.add_argument(
        "--dim-time", default="REFERENCE_TS",
        help="Name of the time dimension.")
    parser_nc.add_argument(
        "--dim-x", default="X", help="Name of the x dimension.")
    parser_nc.add_argument(
        "--dim-y", default="Y", help="Name of the y dimension.")
    parser_nc.add_argument(
        "--prefix", default=None,
        help="Prefix of the output file names (default: the variable name).")
    parser_nc.add_argument(
        "--output-var", default="precip",
        help="Name of the variable in the output netCDF files.")
    parser_nc.add_argument(
        "--units", default="mm",
        help="Units of the variable (netCDF attribute).")
    parser_nc.add_argument(
        "--time-shift", type=float, default=0.0,
        help="Hours added to the source timestamps, to align datasets on "
             "the end-of-interval labeling convention (e.g. 1 for "
             "start-labeled hourly means such as TabsH). Default: 0.")
    parser_nc.add_argument(
        "--deaccumulate", action="store_true",
        help="Treat the variable as a daily-resetting accumulation "
             "(e.g. ERA5-Land total precipitation, accumulated from 00 UTC) "
             "and difference it to per-step amounts. Default: off.")
    parser_nc.add_argument(
        "--scale", type=float, default=1.0,
        help="Multiplicative factor applied to the extracted values "
             "(e.g. 1000 to convert ERA5-Land precipitation from m to mm). "
             "Default: 1.0 (no scaling).")
    parser_nc.add_argument(
        "--data-crs", type=int, default=None,
        help="EPSG code of the data grid; if omitted, the grid is assumed "
             "to be in the shapefile CRS. When it differs from the "
             "shapefile CRS, exact constant-offset pairs (LV03/LV95, EPSG "
             "21781/2056) get the offset applied, otherwise the polygons "
             "are reprojected. The hourly CombiPrecip LV03-style grid "
             "needs 21781.")

    parser_cpc5min = subparsers.add_parser(
        "cpc5min",
        help="Extract hourly statistics from the 5-min CPCH zip files.")
    parser_cpc5min.add_argument(
        "--data-dir",
        help="Root of the <YYYY>/<YYDOY>/CPCH*.zip tree.")
    parser_cpc5min.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for the daily zip files.")

    for sub in (parser_nc, parser_cpc5min):
        sub.add_argument(
            "--shapefile",
            help="Shapefile with the catchment polygons (EPSG:2056).")
        sub.add_argument(
            "--out-dir", default="./output",
            help="Output directory.")
        sub.add_argument(
            "--years", nargs=2, type=int, metavar=("START", "END"),
            required=True, help="Year range to process (inclusive).")
        sub.add_argument(
            "--id-field", default="EZGNR",
            help="Shapefile attribute used as CSV column names.")
        sub.add_argument(
            "--format", nargs="+", choices=["csv", "netcdf"],
            default=["csv"], dest="formats",
            help="Output format(s): csv, netcdf or both (default: csv).")
        sub.add_argument(
            "--threshold", type=float, default=0,
            help="Precipitation threshold in mm/h (default: disable): values "
                 "below it are set to 0 (for cpc5min, all statistics of an "
                 "hour whose mean is below it). Set to 0 to disable, e.g. "
                 "for non-precipitation variables.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "netcdf":
        extract_from_netcdf(
            shapefile=args.shapefile,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            year_start=args.years[0],
            year_end=args.years[1],
            var_name=args.var_name,
            dim_time=args.dim_time,
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            file_pattern=args.file_pattern,
            output_prefix=args.prefix,
            output_var=args.output_var,
            units=args.units,
            threshold=args.threshold,
            time_shift=args.time_shift,
            deaccumulate=args.deaccumulate,
            scale=args.scale,
            data_crs=args.data_crs,
            id_field=args.id_field,
            formats=tuple(args.formats),
        )
    elif args.command == "cpc5min":
        extract_5min_stats(
            shapefile=args.shapefile,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            year_start=args.years[0],
            year_end=args.years[1],
            id_field=args.id_field,
            formats=tuple(args.formats),
            threshold=args.threshold,
            n_workers=args.workers,
        )


if __name__ == "__main__":
    main()
