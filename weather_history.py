"""Fetch NCEI GHCND daily weather history for the OLS training period.

Endpoint: https://www.ncei.noaa.gov/access/services/data/v1
Station:  config.yaml's location.ncei_station -- pick a real nearby surface
          station, not a mountain CRN site; those can carry a large,
          consistent cool bias relative to the service area.
Returns:  daily TMAX, TMIN, PRCP in standard units (°F, inches).

Results are cached as Parquet. The cache is refreshed weekly (TTL 168 h)
so that recent days are gradually added to the training set without hammering
the NCEI API on every run.
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

import data_quality as dq

log = logging.getLogger(__name__)

_NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
_CACHE_FILE = "weather_history.parquet"


def fetch_ghcnd(
    station_id: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    ncei_token: str = "",
    timeout: int = 60,
    cache_ttl_hours: int = 168,
) -> pd.DataFrame:
    """Return daily weather history indexed by date.

    Columns: tmax_f, tmin_f, precip_in

    start_date / end_date: "YYYY-MM-DD" strings.
    ncei_token: optional free token from ncei.noaa.gov/cdo-web/token — lifts
                rate limits but is not required for public historical data.
    """
    cache_file = cache_dir / _CACHE_FILE

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_ttl_hours:
            df = pd.read_parquet(cache_file)
            log.info("Weather history loaded from cache: %d days", len(df))
            return df

    params = {
        "dataset": "daily-summaries",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "TMAX,TMIN,PRCP",
        "units": "standard",    # °F and inches
        "format": "csv",
        "includeAttributes": "false",
    }
    headers = {"token": ncei_token} if ncei_token else {}

    log.info("Fetching GHCND from NCEI: %s %s–%s", station_id, start_date, end_date)
    resp = requests.get(_NCEI_URL, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "TMAX": "tmax_f", "TMIN": "tmin_f", "PRCP": "precip_in"})
    df["date"] = df["date"].dt.date
    df = df.set_index("date")[["tmax_f", "tmin_f", "precip_in"]]
    df = df.apply(pd.to_numeric, errors="coerce")

    # Physical plausibility bounds (added 2026-07-17, data_quality.py).
    # Deliberately generous -- inside Colorado's statewide extremes
    # (-61F/118F, from high-mountain valley cold pools and eastern-plains
    # heat, neither representative of this Front Range station) with
    # margin. This catches corruption/unit errors (Celsius/Fahrenheit
    # swaps, sentinel leaks, decimal-place errors), NOT unusual-but-real
    # weather -- it would NOT have caught M4 (wrong station, ~15-20F cool
    # bias: 73F in July is real, plausible weather, just from the wrong
    # place 30km away -- a climatological/identity problem, not a
    # physical-impossibility one). Closes a different, real gap.
    df = dq.apply_bounds(df, "tmax_f", -40.0, 115.0, action="nan", label="GHCND tmax_f")
    df = dq.apply_bounds(df, "tmin_f", -45.0, 105.0, action="nan", label="GHCND tmin_f")
    df = dq.apply_bounds(df, "precip_in", 0.0, 15.0, action="nan", label="GHCND precip_in")

    n_missing = df["tmax_f"].isna().sum()
    if n_missing > 0:
        log.warning("GHCND: %d days missing TMAX (will be excluded from training)", n_missing)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    log.info("GHCND fetched and cached: %d days", len(df))
    return df
