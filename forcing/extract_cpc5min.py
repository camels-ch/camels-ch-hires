"""
Extraction of hourly statistics of the 5-minute CombiPrecip (CPCH_5min) data
for catchments.

The zipped ODIM-HDF5 files (one zip per day, organised as <YYYY>/<YYDOY>/
CPCH*.zip) hold 5-min precipitation rates in mm/h. For each 5-min step the
exact area-weighted catchment mean is computed first; the 12 values of each
hour are then reduced to statistics (mean, max, q10, q25, q50, q75, q90),
written per year as one CSV per statistic (one column per catchment) and/or
one netCDF file with one variable per statistic.

Conventions
-----------
- All statistics are intensities in mm/h; the hourly 'mean' is therefore
  numerically equal to the hourly precipitation sum in mm.
- Timestamps label the END of the interval, matching the hourly CPCH netCDF
  product: the row labeled T holds the statistics of the 12 five-minute
  intervals ending at T-55', ..., T-5', T.
- The 5-min timestamps are parsed from the file names (CPC<YYDOY><HHMM>...),
  because the 'what' date/time HDF5 attributes are rounded to the hour in
  some product versions.

The HDF5 decoding (undetect=+inf -> 0, nodata=NaN, quantity RATE in mm/h) was
adapted from swafi (precip_combiprecip_5min.py).
"""

import io
import logging
import warnings
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from grid_weights import load_or_compute_weights, weighted_mean
from output import write_csv, write_netcdf

logger = logging.getLogger(__name__)

# Canonical CombiPrecip grid (EPSG:2056 / CH1903+ / LV95), 1 km cells.
# Pixel centres are offset by +500 m from the cell corner; row 0 is the
# northmost row.
GRID_X_SIZE = 710
GRID_Y_SIZE = 640
GRID_X0 = 2255500.0  # centre of the first (westmost) column
GRID_Y0 = 1479500.0  # centre of the first (northmost) row
CELL_SIZE = 1000.0

STEPS_PER_DAY = 288
STEPS_PER_HOUR = 12

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
STATS = ("mean", "max") + tuple(f"q{int(q * 100)}" for q in QUANTILES)


def extract_5min_stats(
    shapefile: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    year_start: int,
    year_end: int,
    id_field: str = "EZGNR",
    formats: tuple[str, ...] = ("csv",),
    threshold: float = 0.0,
    n_workers: int = 4,
    cache_dir: str | Path | None = None,
) -> None:
    """
    Extract hourly statistics of 5-min catchment precipitation, written per
    year as CSV files (one per statistic) and/or a netCDF file.

    Parameters
    ----------
    shapefile
        Path to the shapefile with the catchment polygons (EPSG:2056).
    data_dir
        Root of the <YYYY>/<YYDOY>/CPCH*.zip tree.
    out_dir
        Output directory for the yearly files (CPCH_5min_<stat>_<year>.csv,
        CPCH_5min_stats_<year>.nc).
    year_start, year_end
        Year range to process (inclusive). Years whose outputs all exist are
        skipped.
    id_field
        Shapefile attribute used as column names / catchment coordinate.
    formats
        Output formats, any of 'csv' and 'netcdf'.
    threshold
        Hours whose mean intensity is below this threshold are considered
        dry: all their statistics are set to 0, to remove small
        precipitation amounts. Use 0 (default) to disable. NaNs are
        preserved.
    n_workers
        Number of parallel workers reading the daily zip files.
    cache_dir
        Directory for the cached weight matrix (default: out_dir).
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = out_dir

    x_centers = GRID_X0 + np.arange(GRID_X_SIZE) * CELL_SIZE
    y_centers = GRID_Y0 - np.arange(GRID_Y_SIZE) * CELL_SIZE
    ids, weights = load_or_compute_weights(
        shapefile, id_field, x_centers, y_centers, CELL_SIZE,
        cache_dir=cache_dir,
    )
    n_catch = len(ids)

    for year in range(year_start, year_end + 1):
        csv_paths = {}
        nc_path = None
        if "csv" in formats:
            csv_paths = {
                stat: out_dir / f"CPC_5min_{stat}_{year}.csv"
                for stat in STATS
            }
        if "netcdf" in formats:
            nc_path = out_dir / f"CPC_5min_stats_{year}.nc"
        expected = list(csv_paths.values()) + ([nc_path] if nc_path else [])
        if all(path.exists() for path in expected):
            logger.info(f"Skipping {year} (outputs already exist).")
            continue

        day_list = _list_daily_zips(data_dir, year)
        if not day_list:
            logger.warning(f"No CPCH zip files found in {data_dir} for {year}.")
            continue

        # The hour labeled 'Jan 1 00:00' needs the last day of the previous
        # year; the hour labeled 'Dec 31 23:00' needs the first 5-min step of
        # the next year (via the day lookahead below).
        day_list = (
            _list_daily_zips(data_dir, year - 1)[-1:]
            + day_list
            + _list_daily_zips(data_dir, year + 1)[:1]
        )

        logger.info(f"Processing {len(day_list)} days for {year}...")
        hours_all = []
        stats_all = []
        nan_row = np.full((1, n_catch), np.nan)
        prev: tuple[pd.Timestamp, np.ndarray] | None = None
        for date, means in _iter_day_means(day_list, weights, n_workers):
            if prev is not None:
                prev_date, prev_means = prev
                if date == prev_date + pd.Timedelta(days=1):
                    next_first = means[0:1]
                else:
                    next_first = nan_row
                hours, stats = _hourly_stats(prev_means, next_first, prev_date)
                hours_all.append(hours)
                stats_all.append(stats)
            prev = (date, means)
        # Last day of the list (no lookahead available)
        hours, stats = _hourly_stats(prev[1], nan_row, prev[0])
        hours_all.append(hours)
        stats_all.append(stats)

        index = pd.DatetimeIndex(np.concatenate(
            [h.to_numpy() for h in hours_all]))
        full_index = pd.date_range(
            f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")

        frames = {}
        for i_stat, stat in enumerate(STATS):
            values = np.vstack([s[i_stat] for s in stats_all])
            df = pd.DataFrame(values, index=index, columns=ids)
            frames[stat] = df[~df.index.duplicated()].reindex(full_index)

        if threshold > 0:
            # Hours with a mean below the threshold are considered dry: all
            # their statistics are zeroed (NaNs are preserved).
            dry = frames["mean"] < threshold
            for stat in STATS:
                frames[stat] = frames[stat].mask(dry, 0.0)

        for stat, path in csv_paths.items():
            write_csv(frames[stat], path)
        if nc_path is not None:
            write_netcdf(
                frames,
                nc_path,
                units="mm h-1",
                var_attrs={
                    "long_name": "hourly statistic of 5-min catchment-average "
                                 "precipitation intensity (CombiPrecip)",
                },
            )


def read_day_zip(zip_path: str | Path, date: pd.Timestamp) -> np.ndarray:
    """
    Read all 5-min HDF5 files contained in a daily CPCH zip into a single
    array of precipitation rates.

    Parameters
    ----------
    zip_path
        The path to the daily zip file (e.g. CPCHhdf524001.zip).
    date
        The date (00:00) of the day covered by the zip.

    Returns
    -------
    np.ndarray
        A (STEPS_PER_DAY, GRID_Y_SIZE, GRID_X_SIZE) float32 array of rates in
        mm/h, ordered by the interval end times 00:00, 00:05, ..., 23:55 (the
        step labeled 00:00 covers 23:55 of the previous day to 00:00). Time
        steps with no corresponding file are filled with NaN.
    """
    out = np.full(
        (STEPS_PER_DAY, GRID_Y_SIZE, GRID_X_SIZE), np.nan, dtype="float32"
    )

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Keep only the 5-min files, ignore 60-min, daily, etc. The file
            # name suffix varies with the product version (.000, .801, .001).
            name = member.rsplit("/", 1)[-1]
            if "_00005." not in name or not name.endswith(".h5"):
                continue

            # Parse the interval end time from the name: CPC<YYDOY><HHMM>...
            # (the HDF5 'what' attributes are rounded to the hour in some
            # product versions and cannot be used).
            hhmm = name[8:12]
            if not name.startswith("CPC") or not hhmm.isdigit():
                logger.warning(f"Unexpected file name: {name} (skipped).")
                continue
            idx = int(hhmm[:2]) * STEPS_PER_HOUR + int(hhmm[2:]) // 5
            if not 0 <= idx < STEPS_PER_DAY:
                logger.warning(f"Unexpected time in file name: {name} (skipped).")
                continue

            with zf.open(member) as fh:
                buf = io.BytesIO(fh.read())

            with h5py.File(buf, "r") as h5:
                raw = h5["dataset1/data1/data"][:]

            # undetect is encoded as +inf (no rain) -> 0; nodata stays NaN.
            out[idx] = np.where(np.isposinf(raw), 0.0, raw).astype("float32")

    return out


def _day_means(zip_path: Path, date: pd.Timestamp, weights) -> np.ndarray:
    """Catchment means of the 5-min rates of one day, (288, n_catchments)."""
    logger.info(f"Processing {date.date()}...")
    rates = read_day_zip(zip_path, date)
    return weighted_mean(weights, rates)


def _iter_day_means(day_list, weights, n_workers):
    """
    Yield (date, catchment means) per day, reading the zips with a bounded
    pool of threads so that memory use stays limited.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending = deque()
        it = iter(day_list)
        for date, zip_path in day_list[: 2 * n_workers]:
            next(it)
            pending.append((date, executor.submit(
                _day_means, zip_path, date, weights)))
        while pending:
            date, future = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt[0], executor.submit(
                    _day_means, nxt[1], nxt[0], weights)))
            yield date, future.result()


def _hourly_stats(
    day_means: np.ndarray, next_first: np.ndarray, date: pd.Timestamp
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Compute the hourly statistics for one day of 5-min catchment means.

    Parameters
    ----------
    day_means
        The (288, n_catchments) catchment means of the day, end-labeled
        00:00 ... 23:55.
    next_first
        The (1, n_catchments) catchment means of the next day's first step
        (end-labeled 00:00 of the next day), or NaN if not available.
    date
        The date (00:00) of the day.

    Returns
    -------
    (hours, stats)
        The 24 hourly end labels (date 01:00 ... next date 00:00) and an
        array of shape (n_stats, 24, n_catchments) ordered as STATS.
    """
    stack = np.vstack([day_means[1:], next_first])
    hourly = stack.reshape(24, STEPS_PER_HOUR, -1)

    with warnings.catch_warnings():
        # All-NaN hours (missing 5-min files) legitimately yield NaN.
        warnings.filterwarnings("ignore", message="All-NaN slice")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        quantiles = np.nanquantile(hourly, QUANTILES, axis=1)
        stats = np.concatenate([
            np.stack([
                np.nanmean(hourly, axis=1),
                np.nanmax(hourly, axis=1),
            ]),
            quantiles,
        ])

    hours = pd.date_range(
        date + pd.Timedelta(hours=1), periods=24, freq="h")

    return hours, stats


def _list_daily_zips(data_dir: Path, year: int) -> list[tuple[pd.Timestamp, Path]]:
    """
    List the daily CPCH zip files of one year, sorted by date.

    Returns
    -------
    list of (pd.Timestamp, Path)
        The (date, zip_path) pairs.
    """
    year_dir = data_dir / str(year)
    day_list = []
    if not year_dir.is_dir():
        return day_list

    for doy_dir in sorted(year_dir.iterdir()):
        if not doy_dir.is_dir():
            continue
        zips = sorted(doy_dir.glob("CPCH*.zip"))
        if not zips:
            continue
        # The DOY folder is named <YY><DOY>, e.g. '24001'.
        doy = int(doy_dir.name[2:])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        day_list.append((date, zips[0]))

    day_list.sort(key=lambda pair: pair[0])

    return day_list
