"""Weather covariate assembly for the CC-DIDS ensemble.

Historical covariates come from the GHCND cache (WHEAT RIDGE 2 — the
proposal's "nearby NOAA station"); reference evapotranspiration is estimated
with Hargreaves-Samani from tmax/tmin because GHCND stations do not report
ET directly. Future covariates use the NWS point forecast for days it covers
and day-of-year climatology beyond.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COVARIATE_COLS = ["tmax_f", "precip_in", "et0_in"]
CALENDAR_COLS = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
                 "is_holiday"]


def parse_holidays(cfg: dict) -> set[date]:
    """Holiday dates from config.yaml's holidays block (repeating MM-DD
    entries expanded over 2019-2032 + fixed one-off dates)."""
    hol = cfg.get("holidays", {}) or {}
    out: set[date] = set()
    for mmdd in hol.get("repeating", []) or []:
        m, d = (int(x) for x in str(mmdd).split("-"))
        for year in range(2019, 2033):
            out.add(date(year, m, d))
    for iso in hol.get("fixed_dates", []) or []:
        out.add(pd.to_datetime(iso).date())
    return out


def calendar_features(dates: list[date], holidays: set[date] | None = None) -> pd.DataFrame:
    """Shared calendar encoding for all ensemble members (single source —
    previously triplicated across svm/gbr/model.py). Includes is_holiday,
    which smooth doy trig cannot represent (holidays shift demand like
    weekends per config.yaml)."""
    idx = pd.to_datetime(pd.Series(list(dates)))
    holidays = holidays or set()
    df = pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * idx.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * idx.dt.dayofweek / 7),
        "doy_sin": np.sin(2 * np.pi * idx.dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * idx.dt.dayofyear / 365.25),
        "is_weekend": (idx.dt.dayofweek >= 5).astype(float),
        "is_holiday": [1.0 if d in holidays else 0.0 for d in dates],
    })
    df.index = list(dates)
    return df


def hargreaves_et0_in(tmax_f: pd.Series, tmin_f: pd.Series, doy: pd.Series,
                      lat_deg: float) -> pd.Series:
    """Hargreaves-Samani daily reference ET in inches.

    ET0 (mm/day) = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin), with Ra
    the extraterrestrial radiation expressed as evaporation equivalent.
    """
    tmax_c = (tmax_f - 32.0) * 5.0 / 9.0
    tmin_c = (tmin_f - 32.0) * 5.0 / 9.0
    tmean_c = (tmax_c + tmin_c) / 2.0
    trange = (tmax_c - tmin_c).clip(lower=0.0)

    lat_rad = np.deg2rad(lat_deg)
    d = doy.astype(float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * d)            # inverse rel. distance
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * d - 1.39)          # solar declination
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1))  # sunset hour angle
    ra_mj = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )
    ra_mm = ra_mj * 0.408  # MJ/m2/day -> mm/day evaporation equivalent
    et0_mm = 0.0023 * ra_mm * (tmean_c + 17.8) * np.sqrt(trange)
    return (et0_mm / 25.4).clip(lower=0.0)


def historical_covariates(weather_hist: pd.DataFrame, lat_deg: float) -> pd.DataFrame:
    """tmax_f / precip_in / et0_in indexed by date, NaN weather rows dropped."""
    df = weather_hist.copy()
    df.index = pd.to_datetime(df.index)
    doy = pd.Series(df.index.dayofyear, index=df.index)
    df["et0_in"] = hargreaves_et0_in(df["tmax_f"], df["tmin_f"], doy, lat_deg)
    # et0_in depends on BOTH tmax_f and tmin_f (Hargreaves-Samani); a day with
    # tmax_f present but tmin_f missing (common GHCND sensor/QC gap) must not
    # be treated as complete, or et0_in silently carries a NaN downstream.
    complete = df["tmax_f"].notna() & df["tmin_f"].notna()
    out = df.loc[complete, COVARIATE_COLS]
    out["precip_in"] = out["precip_in"].fillna(0.0)
    out.index = out.index.date
    return out


def climatology(weather_hist: pd.DataFrame, lat_deg: float) -> pd.DataFrame:
    """Day-of-year mean covariates (smoothed ±7 days) from the full history."""
    hist = historical_covariates(weather_hist, lat_deg)
    hist = hist.copy()
    hist["doy"] = pd.to_datetime(pd.Series(hist.index, index=hist.index)).dt.dayofyear
    rows = []
    for d in range(1, 367):
        window = [(d + off - 1) % 366 + 1 for off in range(-7, 8)]
        sub = hist[hist["doy"].isin(window)]
        rows.append({"doy": d, **{c: sub[c].mean() for c in COVARIATE_COLS}})
    return pd.DataFrame(rows).set_index("doy")


def covariate_frame(dates: list[date], weather_hist: pd.DataFrame,
                    nws_df: pd.DataFrame | None, lat_deg: float,
                    clim: pd.DataFrame | None = None) -> pd.DataFrame:
    """NaN-free covariates for an arbitrary list of days.

    Per-day source priority: observed GHCND history, then NWS point forecast,
    then day-of-year climatology. `source` records which was used, so any day
    — past gap, ingestion-lag day, or future horizon — gets usable values.
    """
    hist = historical_covariates(weather_hist, lat_deg)
    if clim is None:
        clim = climatology(weather_hist, lat_deg)
    nws = None
    if nws_df is not None and len(nws_df):
        nws = nws_df.copy()
        idx = pd.to_datetime(pd.Series(list(nws.index)))
        doy = pd.Series(idx.dt.dayofyear.values, index=nws.index)
        nws["et0_in"] = hargreaves_et0_in(nws["tmax_f"], nws["tmin_f"], doy, lat_deg)

    rows = []
    for day in dates:
        # hist is already filtered to days with BOTH tmax_f and tmin_f
        # present (historical_covariates), so membership alone implies
        # completeness — no separate NaN check needed here.
        if day in hist.index:
            r = {c: float(hist.loc[day, c]) for c in COVARIATE_COLS}
            r["source"] = "observed"
        elif nws is not None and day in nws.index and not pd.isna(nws.loc[day, "tmax_f"]):
            r = {c: float(nws.loc[day, c]) for c in COVARIATE_COLS}
            r["precip_in"] = 0.0 if pd.isna(r["precip_in"]) else r["precip_in"]
            r["source"] = "nws"
        else:
            doy = min(day.timetuple().tm_yday, 366)
            r = {c: float(clim.loc[doy, c]) for c in COVARIATE_COLS}
            r["source"] = "climatology"
        r["date"] = day
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def future_covariates(issue_date: date, n_days: int, weather_hist: pd.DataFrame,
                      nws_df: pd.DataFrame | None, lat_deg: float,
                      clim: pd.DataFrame | None = None) -> pd.DataFrame:
    """Covariates for issue_date+1 .. issue_date+n_days with horizon_days."""
    days = [issue_date + timedelta(days=h) for h in range(1, n_days + 1)]
    out = covariate_frame(days, weather_hist, nws_df, lat_deg, clim=clim)
    out["horizon_days"] = range(1, n_days + 1)
    return out
