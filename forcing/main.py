"""
Command-line entry point for the catchment forcing extraction.

Examples
--------
Hourly CombiPrecip (netCDF grids) to hourly catchment CSVs/netCDFs
(the defaults of the 'netcdf' command target this dataset):
    python main.py netcdf --years 2005 2023 --out-dir ./output --format csv netcdf

Any other regular-grid netCDF dataset, e.g. daily RhiresD-like files:
    python main.py netcdf --data-dir /path/to/data --file-pattern "{year}.nc"
        --var-name RhiresD --dim-time time --dim-x E --dim-y N
        --prefix RhiresD --units mm --coord-shift 0 0 --years 1961 2023

Hourly statistics (mean, max, q10, q25, q50, q75, q90) of 5-min CombiPrecip:
    python main.py cpc5min --years 2005 2024 --out-dir ./output
"""

import argparse
import logging

from extract_cpc5min import extract_5min_stats
from extract_netcdf import extract_from_netcdf

# The hourly CombiPrecip files use LV03-style coordinates; the CombiPrecip
# LV95 grid is the same grid shifted by exactly +2'000'000 / +1'000'000 m.
CPCH_HOURLY_COORD_SHIFT = (2_000_000.0, 1_000_000.0)


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
             "placeholders.")
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
        "--prefix", default="CPC_hourly",
        help="Prefix of the output file names.")
    parser_nc.add_argument(
        "--output-var", default="precip",
        help="Name of the variable in the output netCDF files.")
    parser_nc.add_argument(
        "--units", default="mm",
        help="Units of the variable (netCDF attribute).")
    parser_nc.add_argument(
        "--data-crs", type=int, default=None,
        help="EPSG code of the data grid, if it differs from the shapefile "
             "CRS (polygons are reprojected). Exclusive with --coord-shift.")
    parser_nc.add_argument(
        "--coord-shift", nargs=2, type=float,
        default=CPCH_HOURLY_COORD_SHIFT, metavar=("DX", "DY"),
        help="Offset added to the grid coordinates to express them in the "
             "shapefile CRS. Exclusive with --data-crs.")

    parser_cpc5min = subparsers.add_parser(
        "cpc5min",
        help="Extract hourly statistics from the 5-min CPCH zip files.")
    parser_cpc5min.add_argument(
        "--data-dir", default=DEFAULT_CPC5MIN_DIR,
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
        coord_shift = tuple(args.coord_shift)
        if args.data_crs is not None and coord_shift == CPCH_HOURLY_COORD_SHIFT:
            # --data-crs given without an explicit --coord-shift: drop the
            # CombiPrecip default shift, the reprojection takes over.
            coord_shift = (0.0, 0.0)
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
            data_crs=args.data_crs,
            coord_shift=coord_shift,
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
