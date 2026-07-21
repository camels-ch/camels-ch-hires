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

Handles any regular-grid netCDF dataset: the variable name, dimension names,
file-naming pattern, CRS and unit handling are all options, so the same tool
covers CombiPrecip, ERA5-Land, RhiresD, TabsH, etc. The formatted file pattern
may contain glob wildcards (`*`, `?`), so a single-file-per-year product whose
name embeds a varying end date is matched with `..._{year}*.nc`. Output files
are `<prefix>_<year>.csv` / `.nc`, where `--prefix` defaults to the variable
name.

**Hourly CombiPrecip (CPCH, LV95 grid)** — precipitation, mm, end-labeled hours:

```bash
python main.py netcdf --years 2005 2023 --out-dir ./output --format netcdf \
    --data-dir /path/to/CPCH_hourly_lv95 \
    --file-pattern "CPC_00060_H_{year}*.nc" \
    --var-name CPC --dim-time REFERENCE_TS --dim-x x --dim-y y \
    --prefix CPC_hourly --units mm
```

The older LV03-style `CPCH_hourly` share instead uses uppercase `X`/`Y` and
needs `--data-crs 21781` (its grid is LV03, offset from the LV95 shapefile).

**ERA5-Land precipitation** — the `deaccumulated` share holds per-hour values
already in mm and is used directly:

```bash
python main.py netcdf --years 1950 2022 --out-dir ./output --format netcdf \
    --data-dir /path/to/ERA5-Land/deaccumulated/PRC \
    --file-pattern "PRC-{year}{month:02d}.nc" \
    --var-name tp --dim-time time --dim-x longitude --dim-y latitude \
    --data-crs 4326 --prefix ERA5_PRC --units mm
```

The raw ERA5-Land `PRC` share stores `tp` as a daily-resetting accumulation in
metres; add `--deaccumulate --scale 1000` to obtain per-hour mm. The grid is
geographic (`--data-crs 4326`), so the polygons are reprojected and the area
weighting is computed in degrees (a small latitudinal bias).

**Daily RhiresD** — daily precipitation sums, LV95 grid (no `--data-crs`), one
file per year:

```bash
python main.py netcdf --years 1961 2024 --out-dir ./output --format netcdf \
    --data-dir /path/to/RhiresD_v2.0_swiss.lv95 \
    --file-pattern "RhiresD_ch01h.swiss.lv95_{year}*.nc" \
    --var-name RhiresD --dim-time time --dim-x E --dim-y N \
    --prefix RhiresD --units mm
```

**Hourly TabsH temperature** — start-labeled hourly means, LV95 grid, monthly
files whose names embed the varying last day (hence the glob pattern):

```bash
python main.py netcdf --years 2018 2023 --out-dir ./output --format netcdf \
    --data-dir /path/to/TabsH_swiss.lv95 \
    --file-pattern "TabsH_ch01h.swiss.lv95_{year}{month:02d}010000_*.nc" \
    --var-name TabsH --dim-time time --dim-x E --dim-y N \
    --prefix TabsH --output-var temp --units degC --time-shift 1
```

**Hourly CHAPTER precipitation** — WRF model output (3 km, Mercator grid) with
no 1D coordinate axes, only 2D `XLONG`/`XLAT`; the 1D axes are recovered by
projecting them into the model's Mercator CRS (`--lon2d/--lat2d/--grid-proj`):

```bash
python main.py netcdf --years 1981 2022 --out-dir ./output --format netcdf \
    --data-dir /path/to/CHAPTER/PREC_ACC_NC \
    --file-pattern "PREC_ACC_NC_{year}_CH.nc" \
    --var-name PREC_ACC_NC --dim-time XTIME \
    --lon2d XLONG --lat2d XLAT \
    --grid-proj "+proj=merc +lat_ts=44.671 +lon_0=10.914 +R=6370000 +units=m" \
    --prefix CHAPTER_PREC --units mm
```

Processing options:

- `--deaccumulate` — treat the variable as a daily-resetting accumulation
  (e.g. ERA5-Land `tp`, accumulated from 00 UTC) and difference it to per-step
  amounts (rounding-noise negatives are clipped).
- `--scale FACTOR` — multiply the extracted values, e.g. `1000` for m → mm.
- `--time-shift HOURS` — offset the source timestamps to align start-labeled
  datasets (e.g. TabsH, `1`) on the end-of-interval convention used by the
  precipitation products.
- `--data-crs EPSG` — CRS of the data grid; if omitted it is assumed to match
  the shapefile. Constant-offset pairs (LV03/LV95, EPSG 21781/2056) get the
  exact offset applied, any other pair is reprojected.
- `--lon2d NAME --lat2d NAME --grid-proj PROJ` — for grids georeferenced only
  by 2D lon/lat arrays (e.g. WRF/CHAPTER): recover the regular 1D axes by
  projecting the 2D coordinates into `--grid-proj` (a PROJ string), which also
  becomes the CRS the catchments are reprojected to. Used together; they take
  precedence over `--data-crs`. The grid must be axis-aligned in that
  projection (extraction aborts if it is genuinely curvilinear).

Output: `<prefix>_<year>.csv` / `<prefix>_<year>.nc` (prefix defaults to the
variable name).

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
