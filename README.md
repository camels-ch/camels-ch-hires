# camels-ch-hires

Processing scripts for **CAMELS-CH high temporal resolution** (hires): a companion
to the [CAMELS-CH](https://doi.org/10.5194/essd-15-5755-2023) dataset providing
catchment-aggregated hydro-meteorological data for Switzerland at sub-daily
resolution.

The repository contains two independent tool sets:

| Directory  | Purpose                                                                                                       |
|------------|---------------------------------------------------------------------------------------------------------------|
| `forcing/` | Extraction of catchment-average forcing time series (e.g. CombiPrecip precipitation) from gridded products.    |
| `dem/`     | DEM reconditioning (stream burning) with [hydro-snap](https://pypi.org/project/hydro-snap/) prior to catchment delineation. |

## Forcing extraction (`forcing/`)

Aggregates gridded meteorological products to catchment-average time series
using **exact area weighting**: the weight of each grid cell is the exact
intersection area between the catchment polygon and the cell, so
partly-covered cells contribute proportionally. The weight matrix is cached
(`.npz`, keyed by a hash of the shapefile/grid setup) and reused across runs.

Outputs are written per year as CSV (one column per catchment) and/or
compressed netCDF (dimensions `time` × `catchment`). **Timestamps label the
end of the aggregation interval** (matching the CombiPrecip convention).

### Installation

```bash
cd forcing
pip install -r requirements.txt
```

### Usage

The entry point is `main.py` with two subcommands. Both require a catchment
shapefile (EPSG:2056) and an inclusive year range; years whose output files
already exist are skipped, so interrupted runs can simply be restarted.

#### `netcdf` — regular-grid netCDF datasets

Defaults target the hourly CombiPrecip (CPCH) product (its grid uses
LV03-style coordinates, hence `--data-crs 21781`):

```bash
python main.py netcdf --data-crs 21781 --years 2005 2023 --out-dir ./output --format csv netcdf
```

Any other regular-grid netCDF dataset can be processed by overriding the
variable/dimension names and the file naming pattern, e.g. daily RhiresD-like
files (one file per year, LV95 grid like the shapefile so no `--data-crs`
needed):

```bash
python main.py netcdf --data-dir /path/to/data --file-pattern "{year}.nc" \
    --var-name RhiresD --dim-time time --dim-x E --dim-y N \
    --prefix RhiresD --units mm --years 1961 2023
```

The formatted file pattern may contain glob wildcards, e.g. for the hourly
temperature (TabsH) files whose names embed the varying last day of the month
(`TabsH_ch01h.swiss.lv95_201802010000_201802282300.nc`):

```bash
python main.py netcdf --data-dir /path/to/TabsH_swiss.lv95 \
    --file-pattern "TabsH_ch01h.swiss.lv95_{year}{month:02d}010000_*.nc" \
    --var-name TabsH --dim-time time --dim-x E --dim-y N \
    --prefix TabsH --output-var temp --units degC \
    --time-shift 1 --years 2018 2023
```

`--time-shift HOURS` adds an offset to the source timestamps, to align
start-labeled datasets (such as the TabsH hourly means) on the end-of-interval
labeling convention used by the precipitation products.

CRS handling: `--data-crs EPSG` declares the CRS of the data grid; if omitted,
the grid is assumed to be in the shapefile CRS. When it differs from the
shapefile CRS, the catchment polygons are brought to the grid CRS: for CRS
pairs that are exact constant offsets of each other (LV03/LV95, EPSG
21781/2056) the exact offset is applied, any other pair is reprojected. The
hourly CombiPrecip LV03-style grid needs `--data-crs 21781`.

Output: `<prefix>_<year>.csv` / `<prefix>_<year>.nc`.

#### `cpc5min` — hourly statistics of the 5-min CombiPrecip

Reads the zipped ODIM-HDF5 5-min CombiPrecip files (one zip per day, organised
as `<YYYY>/<YYDOY>/CPCH*.zip`), computes the exact area-weighted catchment
mean of every 5-min field, and reduces the 12 values of each hour to
statistics: `mean`, `max`, `q10`, `q25`, `q50`, `q75`, `q90`.

```bash
python main.py cpc5min --years 2005 2024 --out-dir ./output --workers 4
```

All statistics are intensities in mm/h. The
row labeled `T` holds the statistics of the 12 five-minute intervals ending at
`T−55'`, …, `T−5'`, `T`. The 5-min timestamps are parsed from the file names,
because the HDF5 date/time attributes are rounded to the hour in some product
versions.

Output: `CPC_5min_<stat>_<year>.csv` (one file per statistic) and/or
`CPC_5min_stats_<year>.nc` (one variable per statistic).

#### Common options

| Option              | Default           | Description                                               |
|---------------------|-------------------|-----------------------------------------------------------|
| `--shapefile`       | project shapefile | Catchment polygons (EPSG:2056).                           |
| `--id-field`        | `EZGNR`           | Shapefile attribute used as column names / catchment ids. |
| `--years START END` | required          | Year range to process (inclusive).                        |
| `--out-dir`         | `./output`        | Output directory.                                         |
| `--format`          | `csv`             | Output format(s): `csv`, `netcdf` or both.                |
| `--threshold`       | `0` (disabled)    | Precipitation threshold in mm/h: values below it are set to 0 (for `cpc5min`, all statistics of an hour whose mean is below it). Set to 0 to disable, e.g. for non-precipitation variables. |

Run `python main.py <command> --help` for the full list.

## DEM reconditioning (`dem/`)

Reconditions a DEM by enforcing the mapped stream network
([hydro-snap](https://github.com/pascalhorton/hydro-snap)) so that subsequent
flow-accumulation and catchment delineation follow the actual streams.

```bash
cd dem
pip install -r requirements.txt
```

Edit the paths in `dem-reconditioning.py` (DEM raster, streams shapefile,
optional catchment outline, output directory), then:

```bash
python dem-reconditioning.py
```

## License

See [LICENSE](LICENSE).
