"""
Ecospace Zooplankton Evaluation
G Oldford, Mar 12, 2026
==============================

Evaluate Ecospace zooplankton biomass against seasonal survey observations.

This script matches tow-level observational zooplankton biomass to nearest-in-time
Ecospace outputs, builds paired observation-model datasets, computes skill metrics,
generates seasonal and annual anomalies, and produces publication-style plots and
summary tables for major zooplankton groups and their totals.

The workflow is intended to support quantitative assessment of how well Ecospace
reproduces observed seasonal patterns, interannual variability, and group-specific
zooplankton biomass structure.

Major sections
--------------
1. Config adaptor
2. Shared utilities
3. Match
4. Stats + paired summaries
5. Anomaly data build + computation
6. Model-only box anomalies
7. Plotting
8. Table export helpers
9. Workflow drivers / CLI


"""

from __future__ import annotations

import argparse
import os
import re
import string
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field

# Local helpers & config
from helpers import find_closest_date, find_nearest_point_from1D
import ecospace_eval_config as cfg


# =============================================================================
# CONFIG ADAPTOR
# =============================================================================


@dataclass
class Eval5Config:
    # INPUTS
    ecospace_code: str = cfg.ECOSPACE_SC
    ecospace_nc_name: str = f"{cfg.ECOSPACE_SC}_{cfg.ECOSPACE_RN_STR_YR}-{cfg.ECOSPACE_RN_END_YR}.nc"

    # PATHS
    NC_PATH_OUT: str = cfg.NC_PATH_OUT
    EVALOUT_P: str = cfg.EVALOUT_P
    FIGSOUT_P: str = cfg.FIGS_P
    BASEMAP_P: str = cfg.ECOSPACE_MAP_P

    # Zoop & mapping input files (set these in cfg or keep defaults here)
    ZOOP_P: str = getattr(cfg, "Z_P_PREPPED", "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/4. Zooplankton/Zoop_Perryetal_2021/MODIFIED")
    ZOOP_CSV: str = getattr(cfg, "Z_F_TOWLEV", "Zooplankton_B_C_gm2_EWEMODELGRP_Wide.csv")
    GRID_MAP_CSV: str = getattr(cfg, "ECOSPACE_GRID_RC_CSV", "Ecospace_grid_20210208_rowscols.csv")
    ZOOP_CSV_MATCH: str = getattr(cfg, "Z_F_MATCH", f"Zooplankton_matched_to_model_out_{ecospace_code}.csv")

    # Optional external series
    NPGO_CSV: str = getattr(cfg, "NPGO_CSV", "npgo.csv")

    # YEARS
    START_YEAR: int = getattr(cfg, "ZP_YEAR_START", 1980)
    END_YEAR: int = getattr(cfg, "ZP_YEAR_END", 2018)

    # GROUPS (consistent w/ 5b/5c/5d)
    ZOOP_GROUPS: List[str] = field(default_factory=lambda: ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM",
            "ZS2-CTH", "ZS3-CHA", "ZS4-LAR", "ZF1-ICT"])


    # RUN SWITCHES
    RECOMPUTE_MATCH: bool = getattr(cfg, "ZP_RECOMPUTE_MATCH", True)
    MAKE_VIZ: bool = getattr(cfg, "ZP_MAKE_VIZ", True)
    MAKE_ANOM: bool = getattr(cfg, "ZP_MAKE_ANOM", True)

   # climatology weighting for anomalies
    # - ------------------------
    # Options:
    #   "tow"        : current behavior (clim from all tows pooled)
    #   "year_equal" : clim from annual means, each year weight=1
    #   "year_weighted" : clim from annual means, year weight ~ (n_tows**power), optional cap
    ANOM_CLIM_MODE: str = getattr(cfg, "ZP_ANOM_CLIM_MODE", "tow")
    # For year_weighted:
    ANOM_CLIM_WEIGHT_POWER: float = float(getattr(cfg, "ZP_ANOM_CLIM_WEIGHT_POWER", 0.5))  # 0.5=sqrt(n)
    ANOM_CLIM_WEIGHT_CAP: int | None = getattr(cfg, "ZP_ANOM_CLIM_WEIGHT_CAP", None)  # e.g., 20 or 30
    ANOM_CLIM_MIN_TOWS_PER_YEAR: int = int(getattr(cfg, "ZP_ANOM_CLIM_MIN_TOWS_PER_YEAR", 3))
    ANOM_CLIM_EPS_STD: float = float(getattr(cfg, "ZP_ANOM_CLIM_EPS_STD", 1e-12))  # Numerical safety

    # ANOM/FIG OPTIONS
    SEASON_ORDER: List[str] = field(default_factory=lambda: ["Winter", "Spring", "Summer", "Fall"])
    LOG_OFFSET: float = 1e-6
    LAG_YEARS: int = getattr(cfg, "NPGO_LAG_YEARS", 0)

    BOXPLOT_LOG10: bool = getattr(cfg, "ZP_BOXPLOT_LOG10", True)
    SKILL_LOG10: bool = getattr(cfg, "ZP_SKILL_LOG10", False)
    TOTAL_SCATTER_LOG10: bool = getattr(cfg, "ZP_TOTAL_SCATTER_LOG10", True)

    # New anomaly-summary controls for obs-vs-model evaluation
    # These affect only the anomaly workflow, not the tow-level scatter/boxplot views.
    ANOM_USE_SEASONAL_SUMMARY: bool = getattr(cfg, "ZP_ANOM_USE_SEASONAL_SUMMARY", True)
    ANOM_OBS_SUMMARY: str = getattr(cfg, "ZP_ANOM_OBS_SUMMARY", "mean_of_logs")
    ANOM_MODEL_SUMMARY: str = getattr(cfg, "ZP_ANOM_MODEL_SUMMARY", "mean_raw")
    ANOM_LOG_BASE: str = getattr(cfg, "ZP_ANOM_LOG_BASE", "log10")
    ANOM_ALL_MIN_SEASONS: int = int(getattr(cfg, "ZP_ANOM_ALL_MIN_SEASONS", 1))
    COMPARE_ANOM_SUMMARIES: bool = getattr(cfg, "ZP_COMPARE_ANOM_SUMMARIES", False)
    COMPARE_ANOM_OBS_SUMMARIES: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip() for x in (
                getattr(cfg, "ZP_COMPARE_ANOM_OBS_SUMMARIES", ("log_of_mean", "mean_of_logs"))
                if not isinstance(getattr(cfg, "ZP_COMPARE_ANOM_OBS_SUMMARIES", ("log_of_mean", "mean_of_logs")), str)
                else str(getattr(cfg, "ZP_COMPARE_ANOM_OBS_SUMMARIES", "log_of_mean,mean_of_logs")).split(",")
            )
            if str(x).strip()
        )
    )

    ADD_FIG_SUPTITLE: bool = getattr(cfg, "ZP_ADD_FIG_SUPTITLE", True)
    PREFIX_AX_TITLES_WITH_ECOSPACE: bool = getattr(cfg, "ZP_PREFIX_AX_TITLES_WITH_ECOSPACE", True)

    # Match ecosim-style anomaly/scatter knobs where possible
    ANOM_SEASON_CHOICE: str = getattr(cfg, "ZP_SEASON_CHOICE", "Spring")
    ANOM_PLOT_TYPE: str = getattr(cfg, "ZP_PLOT_TYPE", "bar")  # "bar" or "line"
    SHOW_COUNTS: bool = getattr(cfg, "ZP_SHOW_CNTS", True)
    ANOM_ALL_SEASONS: bool = getattr(cfg, "ZP_ANOM_ALL_SEASONS", False)
    MAKE_SCATTER: bool = getattr(cfg, "ZP_MAKE_SCATTER", True)
    SCATTER_LOG10: bool = getattr(cfg, "ZP_SCATTER_LOG10", True)

    # Tow filtering (Ecosim parity)
    TOW_FILTER_MODE: str = getattr(cfg, "ZP_TOW_FILTER_MODE", "none")
    TOW_MIN_PROP: float = float(getattr(cfg, "ZP_TOW_MIN_PROP", 0.7))
    TOW_MIN_START_DEPTH_M: float = float(getattr(cfg, "ZP_TOW_MIN_START_DEPTH_M", 150.0))
    TOW_MAX_BOTTOM_DEPTH_M: Optional[float] = getattr(cfg, "ZP_TOW_MAX_BOTTOM_DEPTH_M", None)

    # options for the two panel plot of seasonal anoms
    PUB_TOTAL_PANEL: bool = getattr(cfg, "ZP_MAKE_PUB_TOTAL_PANEL", False)
    PUB_PANEL_A_MODE: str = getattr(cfg, "ZP_PUB_PANEL_A_MODE", "anomaly_scatter")

    # Model-only box anomaly workflow
    MODELONLY_DO: bool = getattr(cfg, "ZP_MODELONLY_DO", False)
    MODELONLY_YEAR_START: int = int(getattr(cfg, "ZP_MODELONLY_YEAR_START", getattr(cfg, "ZP_FULLRN_START", 1980)))
    MODELONLY_YEAR_END: int = int(getattr(cfg, "ZP_MODELONLY_YEAR_END", getattr(cfg, "ZP_FULLRN_END", 2018)))

    MODELONLY_LAT_MIN: float = float(getattr(cfg, "ZP_MODELONLY_LAT_MIN", 49.00))
    MODELONLY_LAT_MAX: float = float(getattr(cfg, "ZP_MODELONLY_LAT_MAX", 49.90))
    MODELONLY_LON_MIN: float = float(getattr(cfg, "ZP_MODELONLY_LON_MIN", -124.20))
    MODELONLY_LON_MAX: float = float(getattr(cfg, "ZP_MODELONLY_LON_MAX", -123.10))

    MODELONLY_SPATIAL_REDUCER: str = getattr(cfg, "ZP_MODELONLY_SPATIAL_REDUCER", "mean")
    MODELONLY_ANOM_SUMMARY: str = getattr(cfg, "ZP_MODELONLY_ANOM_SUMMARY", "mean_raw")
    MODELONLY_SEASON: Optional[str] = getattr(cfg, "ZP_MODELONLY_SEASON", None)

    MODELONLY_VERBOSE: bool = bool(getattr(cfg, "ZP_MODELONLY_VERBOSE", True))
    MODELONLY_PROGRESS_EVERY: int = int(getattr(cfg, "ZP_MODELONLY_PROGRESS_EVERY", 25))
    MODELONLY_BOX_STRIDE: int = int(getattr(cfg, "ZP_MODELONLY_BOX_STRIDE", 1))

    MODELONLY_SERIES_DEF: dict = field(
        default_factory=lambda: getattr(
            cfg,
            "ZP_MODELONLY_SERIES_DEF",
            {
                "ZC2-AMP": ["ZC2-AMP"],
                "Total": ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"],
            },
        )
    )

    MODELONLY_PLOT_GROUP: str = getattr(cfg, "ZP_MODELONLY_PLOT_GROUP", "Total")

    MODELONLY_LABELS: dict = field(
        default_factory=lambda: getattr(
            cfg,
            "ZP_MODELONLY_LABELS",
            {
                "ZC2-AMP": "Amphipods",
                "Total": "Total crustaceans",
            },
        )
    )

    MODELONLY_SERIES_CSV: str = getattr(
        cfg,
        "ZP_MODELONLY_SERIES_CSV",
        f"ecospace_zoop_model_box_series_{getattr(cfg, 'ECOSPACE_SC', 'run')}.csv"
    )
    MODELONLY_ANOM_CSV: str = getattr(
        cfg,
        "ZP_MODELONLY_ANOM_CSV",
        f"ecospace_zoop_model_box_anom_{getattr(cfg, 'ECOSPACE_SC', 'run')}.csv"
    )

    ANOM_EXCLUDE_STATIONS: List[str] = field(
        default_factory=lambda: list(getattr(cfg, "ZP_ANOM_EXCLUDE_STATIONS", []))
    )

    BOXPLOT_LEVEL: str = getattr(cfg, "ZP_BOXPLOT_LEVEL", "station")

# show-plot toggles
    SHOW_PLTS: bool = getattr(cfg, "ZP_SHOW_PLTS", False)
    SHOW_MODOBS_TOTAL_SCATTER: bool = getattr(cfg, "ZP_SHOW_MODOBS_TOTAL_SCATTER", False)
    SHOW_SEASONAL_BOXPANELS: bool = getattr(cfg, "ZP_SHOW_SEASONAL_BOXPANELS", False)
    SHOW_TOTAL_BY_SEASON_SCATTER: bool = getattr(cfg, "ZP_SHOW_TOTAL_BY_SEASON_SCATTER", False)
    SHOW_PUB_TOTAL_PANEL: bool = getattr(cfg, "ZP_SHOW_PUB_TOTAL_PANEL", False)
    SHOW_MODELONLY_ANOM_BARS: bool = getattr(cfg, "ZP_SHOW_MODELONLY_ANOM_BARS", False)
    SHOW_ANOM_PAIRED_PANEL: bool = getattr(cfg, "ZP_SHOW_ANOM_PAIRED_PANEL", False)
    SHOW_SCATTER_TOWS: bool = getattr(cfg, "ZP_SHOW_SCATTER_TOWS", False)
    SHOW_SCATTER_ANOMS: bool = getattr(cfg, "ZP_SHOW_SCATTER_ANOMS", False)
    SHOW_SCATTER_ANOMS_TOTAL4: bool = getattr(cfg, "ZP_SHOW_SCATTER_ANOMS_TOTAL4", False)
    SHOW_ANOM_BARS_TOTAL4: bool = getattr(cfg, "ZP_SHOW_ANOM_BARS_TOTAL4", False)
    SHOW_ANOM_TOTAL_SINGLE: bool = getattr(cfg, "ZP_SHOW_ANOM_TOTAL_SINGLE", False)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _apply_year_window(df: pd.DataFrame, *, year_col: str, cfg5: Eval5Config) -> pd.DataFrame:
    return df.query("@cfg5.START_YEAR <= " + year_col + " <= @cfg5.END_YEAR").copy()

def _maybe_show(fig, show_plot: bool) -> None:
    if show_plot:
        plt.show()
    plt.close(fig)

def _add_best_fit_line(
    ax,
    x,
    y,
    *,
    color="k",
    linestyle="-",
    linewidth=1.5,
    alpha=0.9,
    show_eq=False,
    eq_loc=(0.98, 0.10),
):
    """
    Add an ordinary least-squares best-fit line to a scatter plot.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : array-like
        Data already on the plotting scale (raw, log10, anomaly, etc.)
    show_eq : bool
        If True, annotate slope/intercept/R^2 in axes coordinates.
    eq_loc : tuple
        Axes-fraction location for the annotation text.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return None

    # fit y = a*x + b
    a, b = np.polyfit(x, y, 1)

    # use visible plotting range
    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax, 200)
    yy = a * xx + b
    ax.plot(xx, yy, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    # stats
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan

    if show_eq:
        txt = f"fit: y = {a:.2f}x + {b:.2f}\n$R^2$ = {r2:.2f}"
        ax.text(
            eq_loc[0], eq_loc[1], txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=10,
        )

    return {"slope": a, "intercept": b, "r2": r2}


def _exclude_stations(df: pd.DataFrame, stations_to_exclude) -> pd.DataFrame:
    """
    Exclude named stations from a matched table before anomaly building.
    Matching is case-insensitive and trims whitespace.
    """
    if not stations_to_exclude:
        return df

    if "Station" not in df.columns:
        print("[5d][WARN] No 'Station' column found; station exclusion skipped.")
        return df

    bad = {str(s).strip().lower() for s in stations_to_exclude if str(s).strip()}
    if not bad:
        return df

    before = len(df)
    keep = ~df["Station"].astype(str).str.strip().str.lower().isin(bad)
    out = df.loc[keep].copy()
    after = len(out)

    removed_stations = sorted(df.loc[~keep, "Station"].astype(str).unique())
    print(
        f"[5d] Station exclusion: removed {before - after} matched rows "
        f"from stations: {removed_stations}"
    )
    return out

def _load_ecospace_times(nc_path: str) -> Tuple[Dict, np.ndarray]:
    with nc.Dataset(nc_path, 'r') as ds:
        time_var = ds.variables['time']
        units = time_var.units
        calendar = getattr(time_var, 'calendar', 'standard')

        # Get times (may be Python datetimes or cftime objects)
        times = nc.num2date(time_var[:], units=units, calendar=calendar)

        # Coerce cftime -> Python datetime when possible
        def to_py_datetime(t):
            # cftime objects have year/month/day/hour/minute/second attributes
            if hasattr(t, 'year') and not isinstance(t, datetime):
                # Handle fractional seconds if present
                sec = float(getattr(t, 'second', 0))
                micro = int(round((sec - int(sec)) * 1_000_000))
                return datetime(t.year, t.month, t.day, getattr(t, 'hour', 0),
                                getattr(t, 'minute', 0), int(sec), micro)
            return t  # already a datetime

        times = np.array([to_py_datetime(t) for t in np.asarray(times).ravel()])

        return {
            "Dimensions": {dim: len(ds.dimensions[dim]) for dim in ds.dimensions},
            "Variables": list(ds.variables.keys())
        }, times

def apply_tow_filters(
    df: pd.DataFrame,
    *,
    filter_mode: str,
    min_prop: float,
    min_start_depth_m: float,
    max_bottom_depth_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    Mirror Ecosim's optional tow-eligibility filter.

    Keeps tows if:
      (tow_prop >= min_prop) OR (Tow_start_depth.m. >= min_start_depth_m)
    where tow_prop = (|start|-|end|) / Bottom_depth.m.

    If required columns are missing, returns df unchanged.
    """
    mode = str(filter_mode).lower().strip()
    if mode in ("none", "all", "", "no_filter", "unfiltered"):
        return df

    req = {"Tow_start_depth.m.", "Tow_end_depth.m.", "Bottom_depth.m."}
    missing = [c for c in req if c not in df.columns]
    if missing:
        print(f"[5b][WARN] Tow filter requested ({filter_mode}) but missing columns: {missing}. Skipping filter.")
        return df

    d = df.copy()
    d["tow_depth_range"] = d["Tow_start_depth.m."].abs() - d["Tow_end_depth.m."].abs()
    d["tow_prop"] = d["tow_depth_range"] / d["Bottom_depth.m."]

    keep = (d["tow_prop"] >= float(min_prop)) | (d["Tow_start_depth.m."] >= float(min_start_depth_m))
    if max_bottom_depth_m is not None:
        keep = keep & (d["Bottom_depth.m."] <= float(max_bottom_depth_m))

    before = len(d)
    d = d.loc[keep].copy()
    after = len(d)
    print(f"[5b] Tow filter '{filter_mode}': kept {after}/{before} rows ({after/before:.1%}).")
    return d

def _perry_zero_replacement(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        nonzero = df[col][df[col] > 0]
        if not nonzero.empty:
            mn = nonzero.min()
            df[col] = df[col].apply(lambda x: np.random.uniform(0, 0.5 * mn) if x == 0 else x)

def _compute_skill_statistics(obs: np.ndarray, mod: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    msk = ~np.isnan(obs) & ~np.isnan(mod)
    obs, mod = obs[msk], mod[msk]
    if len(obs) == 0:
        return {k: np.nan for k in ["obs_std","mod_std","R","RMSE","Bias","WSS"]}

    obs_std = np.std(obs)
    mod_std = np.std(mod)
    R = np.corrcoef(obs, mod)[0, 1] if len(obs) > 1 else np.nan
    RMSE = float(np.sqrt(np.mean((obs - mod) ** 2)))
    Bias = float(np.mean(mod - obs))
    WSS = 1 - (np.sum((obs - mod)**2) / np.sum((np.abs(obs - np.mean(obs)) + np.abs(mod - np.mean(obs)))**2))
    return {"obs_std": obs_std, "mod_std": mod_std, "R": R, "RMSE": RMSE, "Bias": Bias, "WSS": WSS}

def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"

def _spatial_reduce_2d(arr2d: np.ndarray, reducer: str) -> np.ndarray:
    """
    Reduce array shaped (n_cells, n_times) across cells.
    Returns shape (n_times,).
    """
    reducer = str(reducer).strip().lower()
    if reducer == "mean":
        return np.nanmean(arr2d, axis=0)
    if reducer == "median":
        return np.nanmedian(arr2d, axis=0)
    raise ValueError(f"Unknown spatial reducer: {reducer}. Use 'mean' or 'median'.")

def _get_modelonly_series_def(cfg5: Eval5Config) -> dict[str, list[str]]:
    """
    Returns mapping:
        output_series_name -> list of Ecospace groups to sum
    """
    series_def = getattr(cfg5, "ZP_MODELONLY_SERIES_DEF", None)
    if not series_def:
        # safe fallback
        series_def = {
            "ZC2-AMP": ["ZC2-AMP"],
            "Total": ["ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM"],
        }

    clean = {}
    for k, v in series_def.items():
        if isinstance(v, str):
            clean[str(k)] = [v]
        else:
            clean[str(k)] = [str(x) for x in v]
    return clean

def _fmt_elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"

def _finalize_and_save(fig, outpath: str, *, cfg5, st=None, show_plot: bool | None = None):
    """
    Standard figure finalization to avoid suptitle clipping.
    Save first, then optionally show, then close.
    """
    if st is not None:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )

    do_show = cfg5.SHOW_PLTS if show_plot is None else bool(show_plot)
    _maybe_show(fig, do_show)

def _fig_title(cfg5, *, kind: str, years: tuple[int, int] | None = None,
               season: str | None = None, suffix: str | None = None,
               label: str | None = None, plot_type: str | None = None,
               log_state: str | None = None) -> str:
    bits = [f"Ecospace {cfg5.ecospace_code}", kind]
    if years:
        bits.append(f"{years[0]}–{years[1]}")
    if season:
        bits.append(season)
    if plot_type:
        bits.append(plot_type)
    if log_state:
        bits.append(log_state)
    if suffix:
        bits.append(suffix)
    if label:
        bits.append(label)
    return " | ".join(bits)

def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    wsum = float(np.sum(w))
    return float(np.sum(w * x) / wsum) if wsum > 0 else np.nan

def _wstd_unbiased(x: np.ndarray, w: np.ndarray) -> float:
    """
    Unbiased-ish weighted std using the effective-dof correction:
      Var = sum(w*(x-mu)^2) / (sum(w) - sum(w^2)/sum(w))
    Works well for "reliability"/effort weights too (and avoids the cap dominating).
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size < 2:
        return np.nan

    mu = _wmean(x, w)
    sw = float(np.sum(w))
    sw2 = float(np.sum(w * w))
    denom = sw - (sw2 / sw) if sw > 0 else 0.0
    if denom <= 0:
        return np.nan

    var = float(np.sum(w * (x - mu) ** 2) / denom)
    return float(np.sqrt(var))

def _log_array_base(x: pd.Series | np.ndarray, *, base: str, offset: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    z = arr + float(offset)
    if np.any(z <= 0):
        raise ValueError("Encountered non-positive values after applying anomaly log offset.")
    base = str(base).strip().lower()
    if base == "log10":
        return np.log10(z)
    if base in ("ln", "log", "natural"):
        return np.log(z)
    raise ValueError(f"Unknown anomaly log base: {base}")

def _summarize_series_for_anom(values: pd.Series | np.ndarray, *, method: str, log_base: str, log_offset: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    method = str(method).strip().lower()
    if method == "mean_raw":
        return float(np.mean(arr))
    if method == "log_of_mean":
        m = float(np.mean(arr))
        return float(_log_array_base(np.array([m]), base=log_base, offset=log_offset)[0])
    if method == "mean_of_logs":
        return float(np.mean(_log_array_base(arr, base=log_base, offset=log_offset)))

    raise ValueError(
        f"Unknown anomaly summary method '{method}'. "
        "Use 'mean_raw', 'log_of_mean', or 'mean_of_logs'."
    )

# =============================================================================
# MATCH
# =============================================================================

def match_zoop_to_ecospace(cfg5: Eval5Config) -> str:
    """Produce matched CSV if needed; return path to file."""
    ecospace_nc = os.path.join(cfg5.NC_PATH_OUT, cfg5.ecospace_nc_name)
    zoop_csv = os.path.join(cfg5.ZOOP_P, cfg5.ZOOP_CSV)
    grid_csv = os.path.join(cfg5.BASEMAP_P, cfg5.GRID_MAP_CSV)
    zoop_ecospace_match_csv = os.path

    out_csv = os.path.join(cfg5.EVALOUT_P, cfg5.ZOOP_CSV_MATCH)
    if (not cfg5.RECOMPUTE_MATCH) and os.path.exists(out_csv):
        print(f"[5b] Using existing matched CSV: {out_csv}")
        return out_csv

    print("[5b] Matching zooplankton obs to Ecospace…")
    zoop_df = pd.read_csv(zoop_csv)

    n_raw = len(zoop_df)
    zoop_df = apply_tow_filters(
        zoop_df,
        filter_mode=cfg5.TOW_FILTER_MODE,
        min_prop=cfg5.TOW_MIN_PROP,
        min_start_depth_m=cfg5.TOW_MIN_START_DEPTH_M,
        max_bottom_depth_m=cfg5.TOW_MAX_BOTTOM_DEPTH_M,
    )
    print(f"[5b] Obs rows: raw={n_raw}, after_tow_filter={len(zoop_df)}")
    
    grid_df = pd.read_csv(grid_csv)

    model_lats = grid_df['lat']
    model_lons = grid_df['lon']
    model_rows = grid_df['EWE_row']
    model_cols = grid_df['EWE_col']
    model_deps = grid_df['depth']
    mask_ecospace = model_deps != 0

    _, ecospace_times = _load_ecospace_times(ecospace_nc)

    # Prepare columns
    for var in cfg5.ZOOP_GROUPS:
        zoop_df[f"EWE-{var}"] = np.nan

    # force a Date col (mid‑month) if not present
    if 'Date' not in zoop_df.columns:
        zoop_df['Date'] = pd.to_datetime(dict(year=zoop_df['Year'], month=zoop_df['Month'], day=15))

    zoop_df['ecospace_closest_lat'] = np.nan
    zoop_df['ecospace_closest_lon'] = np.nan
    zoop_df['closest_ecospace_time'] = pd.NaT

    with nc.Dataset(ecospace_nc, 'r') as eco_ds:
        for idx, row in zoop_df.iterrows():
            lat, lon, obs_time = row['Latitude.W.'], row['Longitude.N.'], row['Date']
            eco_idx = find_nearest_point_from1D(lon, lat, model_lons, model_lats, mask_ecospace, dx=0.01)
            if eco_idx[0] is np.nan:
                continue

            zoop_df.at[idx, 'ecospace_closest_lat'] = model_lats[eco_idx[0]]
            zoop_df.at[idx, 'ecospace_closest_lon'] = model_lons[eco_idx[0]]
            r, c = model_rows[eco_idx[0]], model_cols[eco_idx[0]]

            closest_time = find_closest_date(ecospace_times, obs_time)
            t_idx = int(np.where(ecospace_times == closest_time)[0][0])
            zoop_df.at[idx, 'closest_ecospace_time'] = closest_time

            for var in cfg5.ZOOP_GROUPS:
                if var in eco_ds.variables:
                    try:
                        val = eco_ds.variables[var][t_idx, r, c]
                        zoop_df.at[idx, f"EWE-{var}"] = np.round(val, 3)
                    except Exception:
                        pass

    zoop_df.to_csv(out_csv, index=False)
    print(f"[5b] Saved: {out_csv}")
    return out_csv

# =============================================================================
# STATS + PAIRED SUMMARIES
# =============================================================================

def compute_stats(df: pd.DataFrame, obs_col: str, mod_col: str, *, log_or_anom: bool = False) -> Dict[str, float]:
    """Return classic skill metrics for two columns in *df* (NaN-robust)."""
    o_ser, m_ser = df[obs_col], df[mod_col]
    mask = o_ser.notna() & m_ser.notna()
    o, m = o_ser[mask].values, m_ser[mask].values
    N = len(o)
    if N == 0:
        return dict(N=0, MB=np.nan, MAE=np.nan, RMSE=np.nan, NRMSE=np.nan,
                    r=np.nan, R2=np.nan, MAPE=np.nan, NSE=np.nan, WSS=np.nan)

    mb = float(np.nanmean(m - o))
    mae = float(np.nanmean(np.abs(m - o)))
    rmse = float(np.sqrt(np.nanmean((m - o) ** 2)))

    # For anomalies/logs, NRMSE isn't meaningful as RMSE / mean(obs)
    if log_or_anom:
        nrmse = rmse
    else:
        denom = float(np.nanmean(o))
        nrmse = rmse / denom if denom != 0 else np.nan

    r = float(np.corrcoef(m, o)[0, 1]) if N > 1 else np.nan
    r2 = float(r ** 2) if not np.isnan(r) else np.nan
    mape = float(np.nanmean(np.abs((m - o) / o)) * 100) if np.all(o != 0) else np.nan

    denom_nse = float(np.nansum((o - np.nanmean(o)) ** 2))
    nse = float(1 - np.nansum((m - o) ** 2) / denom_nse) if denom_nse != 0 else np.nan

    denom_wss = float(np.nansum((np.abs(m - np.nanmean(o)) + np.abs(o - np.nanmean(o))) ** 2))
    wss = float(1 - np.nansum((m - o) ** 2) / denom_wss) if denom_wss != 0 else np.nan

    return dict(N=N, MB=mb, MAE=mae, RMSE=rmse, NRMSE=nrmse,
                r=r, R2=r2, MAPE=mape, NSE=nse, WSS=wss)

def skill_metrics_by_group(annual: pd.DataFrame, *, groups: List[str], log_or_anom: bool) -> pd.DataFrame:
    """Compute skill metrics per group using annual columns obs_anom/model_anom."""
    rows: list[dict] = []
    for g in groups:
        d = annual[annual["group"] == g]
        if len(d) > 1:
            s = compute_stats(d, "obs_anom", "model_anom", log_or_anom=log_or_anom)
            s["Group"] = g
            rows.append(s)
    return pd.DataFrame(rows)

def visualize_and_stats(matched_csv: str, cfg5: Eval5Config) -> str:
    print("[5c] Visualizing & computing station-level skill stats…")
    df = pd.read_csv(matched_csv)

    df = df.query("@cfg5.START_YEAR <= Year <= @cfg5.END_YEAR").copy()

    # summary table
    summarize_tow_replication_for_supplement(
        matched_csv,
        cfg5=cfg5,
        include_all_year=True,
        include_grand_total=True,
    )

    obs_cols = [c for c in cfg5.ZOOP_GROUPS if c in df.columns]
    if not obs_cols:
        raise ValueError("No configured zooplankton observation columns found in matched_csv.")

    model_cols = [f"EWE-{c}" for c in obs_cols]

    # Perry-style zero replacement is applied at the matched-record level
    # before station-year-season-region aggregation.
    _perry_zero_replacement(df, obs_cols)

    # Station-level aggregation used for boxplots and station skill assessment
    df_station = (
        df.groupby(["Station", "Year", "Season", "Region"], as_index=False)[obs_cols + model_cols]
        .mean()
    )

    # Overall total across all included zooplankton groups
    df_station["TOT"] = df_station[obs_cols].sum(axis=1)
    df_station["EWE-TOT"] = df_station[model_cols].sum(axis=1)

    # Group splits
    zc_cols = [g for g in obs_cols if g.startswith("ZC")]
    zs_cols = [g for g in obs_cols if g.startswith("ZS")]
    zf_cols = [g for g in obs_cols if g.startswith("ZF")]

    # Totals for ZC and ZS only
    if zc_cols:
        df_station["TOT_ZC"] = df_station[zc_cols].sum(axis=1)
        df_station["EWE-TOT_ZC"] = df_station[[f"EWE-{g}" for g in zc_cols]].sum(axis=1)

    if zs_cols:
        df_station["TOT_ZS"] = df_station[zs_cols].sum(axis=1)
        df_station["EWE-TOT_ZS"] = df_station[[f"EWE-{g}" for g in zs_cols]].sum(axis=1)


    df_box = df.copy()

    df_box["TOT"] = df_box[obs_cols].sum(axis=1)
    df_box["EWE-TOT"] = df_box[model_cols].sum(axis=1)

    if zc_cols:
        df_box["TOT_ZC"] = df_box[zc_cols].sum(axis=1)
        df_box["EWE-TOT_ZC"] = df_box[[f"EWE-{g}" for g in zc_cols]].sum(axis=1)

    if zs_cols:
        df_box["TOT_ZS"] = df_box[zs_cols].sum(axis=1)
        df_box["EWE-TOT_ZS"] = df_box[[f"EWE-{g}" for g in zs_cols]].sum(axis=1)

    # Export one long-format station skill table using the same station-level
    # data that feed the seasonal boxplots.
    out_station_skill = export_station_skill_long(
        df_station,
        cfg5=cfg5,
        obs_groups=obs_cols,
        seasons=cfg5.SEASON_ORDER,
    )
    print(f"[5c] Saved station skill stats: {out_station_skill}")

    # Tow-level skill tables, exported separately for ZC and ZS
    if zc_cols:
        paired_zc = build_paired_long_ecospace(df, groups=zc_cols)
        paired_zc = paired_zc.query("@cfg5.START_YEAR <= year <= @cfg5.END_YEAR").copy()
        out_tow_zc = export_tow_skill_long(
            paired_zc,
            cfg5=cfg5,
            seasons=cfg5.SEASON_ORDER,
            out_csv=os.path.join(
                cfg5.EVALOUT_P,
                f"ecospace_zoop_tow_skill_forpub_{cfg5.ecospace_code}_ZC.csv",
            ),
        )
        print(f"[5c] Saved tow-level ZC skill stats: {out_tow_zc}")

    if zs_cols:
        paired_zs = build_paired_long_ecospace(df, groups=zs_cols)
        paired_zs = paired_zs.query("@cfg5.START_YEAR <= year <= @cfg5.END_YEAR").copy()
        out_tow_zs = export_tow_skill_long(
            paired_zs,
            cfg5=cfg5,
            seasons=cfg5.SEASON_ORDER,
            out_csv=os.path.join(
                cfg5.EVALOUT_P,
                f"ecospace_zoop_tow_skill_forpub_{cfg5.ecospace_code}_ZS.csv",
            ),
        )
        print(f"[5c] Saved tow-level ZS skill stats: {out_tow_zs}")

    # Figures
    if cfg5.MAKE_VIZ:
        # Total scatter by Region
        fig, ax = plt.subplots(figsize=(6, 6))

        if cfg5.TOTAL_SCATTER_LOG10:
            x = np.log10(df_station["TOT"] + cfg5.LOG_OFFSET)
            y = np.log10(df_station["EWE-TOT"] + cfg5.LOG_OFFSET)
            xlab = "log10(Observed + offset)"
            ylab = "log10(Modelled + offset)"
        else:
            x = df_station["TOT"]
            y = df_station["EWE-TOT"]
            xlab = "Observed (g C m$^{-2}$)"
            ylab = "Modelled (g C m$^{-2}$)"

        sns.scatterplot(
            x=x,
            y=y,
            hue=df_station["Region"],
            alpha=0.7,
            ax=ax,
        )

        # 1:1 line + sensible limits
        lo = np.nanmin(np.r_[x.values, y.values])
        hi = np.nanmax(np.r_[x.values, y.values])
        pad = 0.05 * (hi - lo if hi > lo else 1.0)

        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=1)

        _add_best_fit_line(
            ax,
            x.values,
            y.values,
            color="k",
            linestyle="--",
            linewidth=1,
            show_eq=False,
            eq_loc=(0.98, 0.08),
        )

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(
            f"Ecospace {cfg5.ecospace_code} - Model vs. Obs Total Biomass by Station"
            + (" (log10)" if cfg5.TOTAL_SCATTER_LOG10 else "")
        )

        plt.tight_layout()
        out_plt = os.path.join(
            cfg5.FIGSOUT_P,
            f"ecospace_{cfg5.ecospace_code}_scatter_modobs_total.png",
        )
        plt.savefig(out_plt)
        _maybe_show(fig, cfg5.SHOW_MODOBS_TOTAL_SCATTER or cfg5.SHOW_PLTS)

        box_df = df_station if cfg5.BOXPLOT_LEVEL == "station" else df_box

        # Seasonal boxplots
        if zc_cols:
            panel_seasonal_boxplots(
                box_df,
                groups=zc_cols,
                cfg5=cfg5,
                extra_total="TOT_ZC",
                suffix=f"ZC_{cfg5.BOXPLOT_LEVEL}",
            )

        if zs_cols:
            panel_seasonal_boxplots(
                box_df,
                groups=zs_cols,
                cfg5=cfg5,
                extra_total="TOT_ZS",
                suffix=f"ZS_{cfg5.BOXPLOT_LEVEL}",
            )

        if zf_cols:
            panel_seasonal_boxplots(
                box_df,
                groups=zf_cols,
                cfg5=cfg5,
                suffix=f"ZF_{cfg5.BOXPLOT_LEVEL}",
            )

    return out_station_skill


def export_tow_skill_long(
    paired: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    seasons: list[str] | None = None,
    out_csv: str | None = None,
) -> str:
    """
    Export tow-level skill statistics in a compact, Word-friendly format.

    Output columns:
        group, season, n, bias, obs std, mod std, MAE, RMSE, r, WSS

    Notes
    -----
    - Built from the same long paired tow table used by tow-level scatter plots.
    - Applies the configured year window.
    - If cfg5.SCATTER_LOG10 is True, stats are computed on log10-transformed values
      (matching the tow-level scatter workflow).
    - Values are rounded to 2 decimals for easy cut/paste into Word.
    """
    rows = []
    seasons = seasons or cfg5.SEASON_ORDER

    work_all = paired.copy()
    work_all = work_all.query("@cfg5.START_YEAR <= year <= @cfg5.END_YEAR").copy()

    for group in sorted(work_all["group"].dropna().unique()):
        g = work_all[work_all["group"] == group].copy()
        if g.empty:
            continue

        if cfg5.SKILL_LOG10:
            g = g[(g["obs_biomass"] > 0) & (g["model_biomass"] > 0)].copy()
            g["obs"] = np.log10(g["obs_biomass"] + cfg5.LOG_OFFSET)
            g["mod"] = np.log10(g["model_biomass"] + cfg5.LOG_OFFSET)
            log_or_anom = True
        else:
            g["obs"] = g["obs_biomass"]
            g["mod"] = g["model_biomass"]
            log_or_anom = False



        if g.empty:
            continue

        # All seasons pooled
        s = compute_stats(g, "obs", "mod", log_or_anom=log_or_anom)
        rows.append({
            "group": group,
            "season": "All",
            "n": int(s["N"]) if pd.notna(s["N"]) else 0,
            "bias": s["MB"],
            "obs std": float(np.nanstd(g["obs"])),
            "mod std": float(np.nanstd(g["mod"])),
            "MAE": s["MAE"],
            "RMSE": s["RMSE"],
            "r": s["r"],
            "WSS": s["WSS"],
        })

        # Individual seasons
        for season in seasons:
            sub = g[g["season"] == season].copy()
            if sub.empty:
                continue

            s = compute_stats(sub, "obs", "mod", log_or_anom=log_or_anom)
            rows.append({
                "group": group,
                "season": season,
                "n": int(s["N"]) if pd.notna(s["N"]) else 0,
                "bias": s["MB"],
                "obs std": float(np.nanstd(sub["obs"])),
                "mod std": float(np.nanstd(sub["mod"])),
                "MAE": s["MAE"],
                "RMSE": s["RMSE"],
                "r": s["r"],
                "WSS": s["WSS"],
            })

    out = pd.DataFrame(rows)

    # enforce exact column order
    col_order = ["group", "season", "n", "bias", "obs std", "mod std", "MAE", "RMSE", "r", "WSS"]
    out = out[col_order]

    # round numeric columns to 2 decimals, but keep n as integer
    round_cols = ["bias", "obs std", "mod std", "MAE", "RMSE", "r", "WSS"]
    out[round_cols] = out[round_cols].round(2)

    out.to_csv(out_csv, index=False)
    print(f"[5c] Saved tow-level skill stats: {out_csv}")
    return out_csv


def export_station_skill_long(
    df_station: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    obs_groups: list[str],
    seasons: list[str] | None = None,
) -> str:
    rows = []
    seasons = seasons or cfg5.SEASON_ORDER

    export_groups = list(obs_groups)
    for extra in ["TOT", "TOT_ZC", "TOT_ZS"]:
        if extra in df_station.columns:
            export_groups.append(extra)

    for group in export_groups:
        mod_group = f"EWE-{group}"

        if group not in df_station.columns or mod_group not in df_station.columns:
            continue

        work = df_station[["Season", group, mod_group]].copy()

        if cfg5.SKILL_LOG10:
            work["obs"] = np.log10(work[group] + cfg5.LOG_OFFSET)
            work["mod"] = np.log10(work[mod_group] + cfg5.LOG_OFFSET)
            scale = "log10"
            log_or_anom = True
        else:
            work["obs"] = work[group]
            work["mod"] = work[mod_group]
            scale = "raw"
            log_or_anom = False

        # all seasons pooled
        s = compute_stats(work, "obs", "mod", log_or_anom=log_or_anom)
        s.update({
            "assessment": "station",
            "season": "All",
            "group": group,
            "obs_mean": float(np.nanmean(work["obs"])),
            "mod_mean": float(np.nanmean(work["mod"])),
            "obs_std": float(np.nanstd(work["obs"])),
            "mod_std": float(np.nanstd(work["mod"])),
            "scale": scale,
            "aggregation": "station_year_season_region",
        })
        rows.append(s)

        # season-specific
        for season in seasons:
            sub = work[work["Season"] == season].copy()
            if sub.empty:
                continue
            s = compute_stats(sub, "obs", "mod", log_or_anom=log_or_anom)
            s.update({
                "assessment": "station",
                "group": group,
                "season": season,
                "obs_mean": float(np.nanmean(sub["obs"])),
                "mod_mean": float(np.nanmean(sub["mod"])),
                "obs_std": float(np.nanstd(sub["obs"])),
                "mod_std": float(np.nanstd(sub["mod"])),
                "scale": scale,
                "aggregation": "station_year_season_region",
            })
            rows.append(s)

    out = pd.DataFrame(rows)

    round_cols = [
        "r", "RMSE", "MAE", "WSS", "Bias", "MB",
        "obs_mean", "mod_mean", "obs_std", "mod_std"
    ]
    round_cols = [c for c in round_cols if c in out.columns]
    out[round_cols] = out[round_cols].round(2)

    # I don't need all these stats so keep only some
    first_cols = [
        "assessment", "scale", "aggregation",
        "group", "season",
        "N",
        "obs_mean", "mod_mean",
        "obs_std", "mod_std",
        #"MB",
        "MAE", "RMSE", #"NRMSE",
         "r",
        # #"R2", "MAPE", "NSE",
        "WSS"
    ]
    out = out[[c for c in first_cols if c in out.columns]]

    out_csv = os.path.join(
        cfg5.EVALOUT_P,
        f"ecospace_zoop_station_skill_{scale}_{cfg5.ecospace_code}.csv",
    )
    out.to_csv(out_csv, index=False)
    print(f"[5c] Saved station skill table: {out_csv}")
    return out_csv




def summarize_tow_replication_for_supplement(
    matched_csv: str,
    cfg5=None,
    *,
    out_csv: str | None = None,
    include_all_year: bool = True,
    include_grand_total: bool = True,
    max_station_list: int = 12,
) -> str:
    """
    Summarize whether multiple tows occur within a given Station-Year-Season.

    Output:
        One CSV with one row per Year x Season, plus optional:
          - one 'All' row per Year
          - one grand-total row

    Key columns:
        Year
        Season
        n_rows
        n_unique_tows
        n_stations
        mean_tows_per_station
        median_tows_per_station
        max_tows_per_station
        n_station_groups_with_multi_tows
        pct_station_groups_with_multi_tows
        n_extra_tows_beyond_first
        stations_with_multi_tows

    Notes:
        - Uses unique 'Index' as tow ID, matching your anomaly workflow.
        - Uses 'Station' as the spatial sampling unit, matching your boxplot workflow.
        - If 'Year' is absent but 'Date'/'date' exists, Year is derived from that.
    """

    df = pd.read_csv(matched_csv).copy()

    # -----------------------------
    # Normalize core column names
    # -----------------------------
    if "Station" not in df.columns:
        raise ValueError("Expected a 'Station' column in matched_csv.")

    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(str).str.capitalize()
    elif "season" in df.columns:
        df["Season"] = df["season"].astype(str).str.capitalize()
    else:
        raise ValueError("Expected a 'Season' or 'season' column in matched_csv.")

    if "Year" not in df.columns:
        if "year" in df.columns:
            df["Year"] = df["year"]
        elif "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"]).dt.year
        elif "date" in df.columns:
            df["Year"] = pd.to_datetime(df["date"]).dt.year
        else:
            raise ValueError("Expected 'Year', 'year', 'Date', or 'date' in matched_csv.")

    if "Index" not in df.columns:
        raise ValueError("Expected an 'Index' column in matched_csv (tow ID).")

    # keep only needed columns and drop obvious missing keys
    keep_cols = ["Station", "Year", "Season", "Index"]
    if "Region" in df.columns:
        keep_cols.append("Region")

    work = df[keep_cols].dropna(subset=["Station", "Year", "Season", "Index"]).copy()

    # -----------------------------
    # Station-Year-Season counts
    # -----------------------------
    station_counts = (
        work.groupby(["Year", "Season", "Station"], as_index=False)
            .agg(
                n_rows=("Index", "size"),
                n_unique_tows=("Index", "nunique"),
            )
    )

    station_counts["has_multi_tows"] = station_counts["n_unique_tows"] > 1
    station_counts["extra_tows_beyond_first"] = np.maximum(
        station_counts["n_unique_tows"] - 1, 0
    )

    # -----------------------------
    # Helper to collapse one subset
    # -----------------------------
    def _collapse(sub: pd.DataFrame, year_val, season_val) -> dict:
        if sub.empty:
            return {
                "Year": year_val,
                "Season": season_val,
                "n_station_groups": 0,
                "n_stations": 0,
                "n_rows": 0,
                "n_unique_tows": 0,
                "mean_tows_per_station": np.nan,
                "median_tows_per_station": np.nan,
                "max_tows_per_station": np.nan,
                "min_tows_per_station": np.nan,
                "n_station_groups_with_multi_tows": 0,
                "pct_station_groups_with_multi_tows": np.nan,
                "n_extra_tows_beyond_first": 0,
                "stations_with_multi_tows": "",
            }

        multi = sub[sub["has_multi_tows"]].copy()

        if len(multi) == 0:
            multi_txt = ""
        else:
            parts = [
                f"{row.Station} ({int(row.n_unique_tows)})"
                for row in multi.sort_values(
                    ["n_unique_tows", "Station"], ascending=[False, True]
                ).itertuples(index=False)
            ]
            if len(parts) > max_station_list:
                shown = parts[:max_station_list]
                multi_txt = "; ".join(shown) + f"; ... +{len(parts) - max_station_list} more"
            else:
                multi_txt = "; ".join(parts)

        return {
            "Year": year_val,
            "Season": season_val,
            "n_station_groups": int(len(sub)),
            "n_stations": int(sub["Station"].nunique()),
            "n_rows": int(sub["n_rows"].sum()),
            "n_unique_tows": int(sub["n_unique_tows"].sum()),
            "mean_tows_per_station": float(sub["n_unique_tows"].mean()),
            "median_tows_per_station": float(sub["n_unique_tows"].median()),
            "max_tows_per_station": int(sub["n_unique_tows"].max()),
            "min_tows_per_station": int(sub["n_unique_tows"].min()),
            "n_station_groups_with_multi_tows": int(multi["has_multi_tows"].sum()),
            "pct_station_groups_with_multi_tows": float(100.0 * multi["has_multi_tows"].sum() / len(sub)),
            "n_extra_tows_beyond_first": int(sub["extra_tows_beyond_first"].sum()),
            "stations_with_multi_tows": multi_txt,
        }

    # -----------------------------
    # Main rows: one row per Year x Season
    # -----------------------------
    rows = []

    for (yr, seas), sub in station_counts.groupby(["Year", "Season"], sort=True):
        rows.append(_collapse(sub, yr, seas))

    # -----------------------------
    # Optional 'All' row per year
    # -----------------------------
    if include_all_year:
        for yr, sub in station_counts.groupby("Year", sort=True):
            rows.append(_collapse(sub, yr, "All"))

    # -----------------------------
    # Optional grand-total row
    # -----------------------------
    if include_grand_total:
        rows.append(_collapse(station_counts, "All", "All"))

    out = pd.DataFrame(rows)

    # season ordering
    season_order = {
        "Winter": 0,
        "Spring": 1,
        "Summer": 2,
        "Fall": 3,
        "All": 4,
    }
    out["_year_sort"] = pd.to_numeric(out["Year"], errors="coerce").fillna(999999)
    out["_season_sort"] = out["Season"].map(season_order).fillna(999)
    out = (
        out.sort_values(["_year_sort", "_season_sort"])
           .drop(columns=["_year_sort", "_season_sort"])
           .reset_index(drop=True)
    )

    # rounding
    round_cols = [
        "mean_tows_per_station",
        "median_tows_per_station",
        "pct_station_groups_with_multi_tows",
    ]
    for c in round_cols:
        if c in out.columns:
            out[c] = out[c].round(2)

    if out_csv is None:
        if cfg5 is not None and hasattr(cfg5, "EVALOUT_P") and hasattr(cfg5, "ecospace_code"):
            out_csv = os.path.join(
                cfg5.EVALOUT_P,
                f"ecospace_tow_replication_summary_{cfg5.ecospace_code}.csv",
            )
        else:
            out_csv = str(Path(matched_csv).with_name("tow_replication_summary.csv"))

    out.to_csv(out_csv, index=False)
    print(f"Saved tow replication summary: {out_csv}")
    return out_csv


# =============================================================================
# ANOMALY DATA BUILD + COMPUTATION
# =============================================================================

def build_paired_long_ecospace(df_match: pd.DataFrame,
                               *,
                               groups: List[str]) -> pd.DataFrame:
    """
    Turn the matched Ecospace CSV into a long table with
    obs + model biomass for each group and for Total.

    Expected columns in df_match:
      - 'Index'
      - 'Season' (string; Winter/Spring/Summer/Fall)
      - 'Date' (or 'date')
      - one column per obs group, e.g. 'ZC1-EUP'
      - one column per model group, e.g. 'EWE-ZC1-EUP'
    """
    df = df_match.copy()

    # Normalise date/season column names
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    else:
        raise ValueError("Matched CSV needs a 'Date' or 'date' column.")

    if "season" in df.columns:
        df["season"] = df["season"].astype(str).str.capitalize()
    elif "Season" in df.columns:
        df["season"] = df["Season"].astype(str).str.capitalize()
    else:
        raise ValueError("Matched CSV needs a 'Season' or 'season' column.")

    if "Index" not in df.columns:
        raise ValueError("Matched CSV must have an 'Index' column (tow ID).")

    recs: list[pd.DataFrame] = []

    # Groups + Total
    for g in groups + ["Total"]:
        if g == "Total":
            # Sum obs + model across all zoop groups
            df["Total"] = df[groups].sum(axis=1)
            df["EWE-Total"] = df[[f"EWE-{x}" for x in groups]].sum(axis=1)
            obs_col, mod_col = "Total", "EWE-Total"
        else:
            obs_col, mod_col = g, f"EWE-{g}"
            if obs_col not in df.columns or mod_col not in df.columns:
                # Skip groups that didn't make it into the matched file
                continue

        o = (
            df[["date", "season", "Index", obs_col]]
            .rename(columns={obs_col: "obs_biomass"})
        )
        m = (
            df[["date", "season", "Index", mod_col]]
            .rename(columns={mod_col: "model_biomass"})
        )

        recs.append(
            o.merge(m, on=["date", "season", "Index"])
             .assign(group=g)
        )

    paired = pd.concat(recs, ignore_index=True)
    paired["year"] = paired["date"].dt.year
    return paired

def _seasonal_summary_table(
    paired: pd.DataFrame,
    *,
    season: Optional[str],
    year_range: tuple[int, int],
    obs_summary: str,
    model_summary: str,
    log_base: str,
    log_offset: float,
) -> pd.DataFrame:
    """Build season-year summaries from the paired tow-level table."""
    y0, y1 = year_range
    df = paired.copy()
    df = df.query("@y0 <= year <= @y1").copy()
    if season is not None:
        df = df.query("season == @season").copy()

    if df.empty:
        season_label = season if season is not None else "All"
        raise ValueError(f"No paired data for season={season_label} and years {y0}-{y1}.")

    df = df[np.isfinite(df["obs_biomass"]) & np.isfinite(df["model_biomass"])].copy()
    if df.empty:
        season_label = season if season is not None else "All"
        raise ValueError(f"No finite paired biomass values for season={season_label} and years {y0}-{y1}.")

    keys = ["year", "group"] if season is not None else ["year", "season", "group"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "obs_value": _summarize_series_for_anom(
                g["obs_biomass"],
                method=obs_summary,
                log_base=log_base,
                log_offset=log_offset,
            ),
            "model_value": _summarize_series_for_anom(
                g["model_biomass"],
                method=model_summary,
                log_base=log_base,
                log_offset=log_offset,
            ),
            "n_tows": int(g["Index"].nunique()) if "Index" in g.columns else int(len(g)),
        })

    out = df.groupby(keys, dropna=False).apply(_agg).reset_index()
    return out

def _group_climatology_from_summary(
    summary: pd.DataFrame,
    *,
    value_obs_col: str = "obs_value",
    value_mod_col: str = "model_value",
    clim_mode: str = "year_equal",
    year_weight_power: float = 0.5,
    year_weight_cap: Optional[int] = None,
    min_tows_per_year: int = 3,
    eps_std: float = 1e-12,
) -> pd.DataFrame:
    """Compute climatological mean/std from season-year summaries."""
    clim_mode = str(clim_mode).strip().lower()
    rows: list[dict] = []

    for grp, gdf in summary.groupby("group"):
        good = gdf[gdf["n_tows"] >= int(min_tows_per_year)].copy()
        xo = good[value_obs_col].to_numpy(dtype=float)
        xm = good[value_mod_col].to_numpy(dtype=float)

        if good.empty or np.isfinite(xo).sum() == 0 or np.isfinite(xm).sum() == 0:
            rows.append(dict(group=grp, mean_obs=np.nan, std_obs=np.nan, mean_mod=np.nan, std_mod=np.nan))
            continue

        if clim_mode == "year_equal":
            mean_obs = float(np.nanmean(xo))
            std_obs = float(np.nanstd(xo, ddof=1)) if np.isfinite(xo).sum() > 1 else np.nan
            mean_mod = float(np.nanmean(xm))
            std_mod = float(np.nanstd(xm, ddof=1)) if np.isfinite(xm).sum() > 1 else np.nan
        else:
            ny = good["n_tows"].to_numpy(dtype=float)
            if clim_mode == "tow":
                w = ny
            elif clim_mode == "year_weighted":
                if year_weight_cap is not None:
                    ny = np.minimum(ny, float(year_weight_cap))
                w = np.power(ny, float(year_weight_power))
            else:
                raise ValueError(
                    f"Unknown clim_mode='{clim_mode}'. Use 'tow', 'year_equal', or 'year_weighted'."
                )
            mean_obs = _wmean(xo, w)
            std_obs = _wstd_unbiased(xo, w)
            mean_mod = _wmean(xm, w)
            std_mod = _wstd_unbiased(xm, w)

        if np.isfinite(std_obs) and abs(std_obs) < eps_std:
            std_obs = np.nan
        if np.isfinite(std_mod) and abs(std_mod) < eps_std:
            std_mod = np.nan

        rows.append(dict(group=grp, mean_obs=mean_obs, std_obs=std_obs, mean_mod=mean_mod, std_mod=std_mod))

    return pd.DataFrame(rows)

def compute_anomalies_summary_paired(
    paired: pd.DataFrame,
    *,
    season: Optional[str],
    year_range: tuple[int, int],
    obs_summary: str,
    model_summary: str,
    clim_mode: str = "year_equal",
    log_base: str = "log10",
    log_offset: float = 1e-6,
    year_weight_power: float = 0.5,
    year_weight_cap: Optional[int] = None,
    min_tows_per_year: int = 3,
    eps_std: float = 1e-12,
    min_seasons_all: int = 1,
) -> pd.DataFrame:
    """Compute z-score anomalies from season-year summary values."""
    if season is not None:
        summ = _seasonal_summary_table(
            paired,
            season=season,
            year_range=year_range,
            obs_summary=obs_summary,
            model_summary=model_summary,
            log_base=log_base,
            log_offset=log_offset,
        )
        clim = _group_climatology_from_summary(
            summ,
            clim_mode=clim_mode,
            year_weight_power=year_weight_power,
            year_weight_cap=year_weight_cap,
            min_tows_per_year=min_tows_per_year,
            eps_std=eps_std,
        )
        out = summ.merge(clim, on="group", how="left")
        out["obs_anom"] = (out["obs_value"] - out["mean_obs"]) / out["std_obs"]
        out["model_anom"] = (out["model_value"] - out["mean_mod"]) / out["std_mod"]
        return out[["year", "group", "obs_anom", "model_anom"]].copy()

    summ = _seasonal_summary_table(
        paired,
        season=None,
        year_range=year_range,
        obs_summary=obs_summary,
        model_summary=model_summary,
        log_base=log_base,
        log_offset=log_offset,
    )

    seasonal_out = []
    for seas, sdf in summ.groupby("season"):
        clim = _group_climatology_from_summary(
            sdf,
            clim_mode=clim_mode,
            year_weight_power=year_weight_power,
            year_weight_cap=year_weight_cap,
            min_tows_per_year=min_tows_per_year,
            eps_std=eps_std,
        )
        tmp = sdf.merge(clim, on="group", how="left")
        tmp["obs_anom"] = (tmp["obs_value"] - tmp["mean_obs"]) / tmp["std_obs"]
        tmp["model_anom"] = (tmp["model_value"] - tmp["mean_mod"]) / tmp["std_mod"]
        tmp["season"] = seas
        seasonal_out.append(tmp[["year", "season", "group", "obs_anom", "model_anom"]])

    seasonal_anom = pd.concat(seasonal_out, ignore_index=True)
    annual = (
        seasonal_anom.groupby(["year", "group"], as_index=False)
        .agg(
            obs_anom=("obs_anom", "mean"),
            model_anom=("model_anom", "mean"),
            n_seasons=("season", "nunique"),
        )
    )
    annual = annual[annual["n_seasons"] >= int(min_seasons_all)].copy()
    return annual[["year", "group", "obs_anom", "model_anom"]]

def compute_anomalies_for_eval(
    paired: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    season: Optional[str],
) -> pd.DataFrame:
    """Route anomaly calculation through the selected evaluation method."""
    if getattr(cfg5, "ANOM_USE_SEASONAL_SUMMARY", False):
        return compute_anomalies_summary_paired(
            paired,
            season=season,
            year_range=(cfg5.START_YEAR, cfg5.END_YEAR),
            obs_summary=cfg5.ANOM_OBS_SUMMARY,
            model_summary=cfg5.ANOM_MODEL_SUMMARY,
            clim_mode=cfg5.ANOM_CLIM_MODE,
            log_base=cfg5.ANOM_LOG_BASE,
            log_offset=cfg5.LOG_OFFSET,
            year_weight_power=cfg5.ANOM_CLIM_WEIGHT_POWER,
            year_weight_cap=cfg5.ANOM_CLIM_WEIGHT_CAP,
            min_tows_per_year=cfg5.ANOM_CLIM_MIN_TOWS_PER_YEAR,
            eps_std=cfg5.ANOM_CLIM_EPS_STD,
            min_seasons_all=cfg5.ANOM_ALL_MIN_SEASONS,
        )

    return compute_anomalies_paired(
        paired,
        season=season,
        year_range=(cfg5.START_YEAR, cfg5.END_YEAR),
        log_transform=cfg5.LOG_TRANSFORM,
        clim_mode=cfg5.ANOM_CLIM_MODE,
        log_offset=cfg5.LOG_OFFSET,
        year_weight_power=cfg5.ANOM_CLIM_WEIGHT_POWER,
        year_weight_cap=cfg5.ANOM_CLIM_WEIGHT_CAP,
        min_tows_per_year=cfg5.ANOM_CLIM_MIN_TOWS_PER_YEAR,
        eps_std=cfg5.ANOM_CLIM_EPS_STD,
    )

def compute_anomalies_paired(
    paired: pd.DataFrame,
    *,
    season: Optional[str],
    year_range: tuple[int, int],
    log_transform: bool,
    # --- NEW ---
    clim_mode: str = "tow",                 # "tow" | "year_equal" | "year_weighted"
    log_offset: float = 1e-6,
    year_weight_power: float = 0.5,         # used only for year_weighted
    year_weight_cap: Optional[int] = None,  # used only for year_weighted
    min_tows_per_year: int = 3,
    eps_std: float = 1e-12,
) -> pd.DataFrame:
    """
    Returns: columns ['year','group','obs_anom','model_anom'].

    clim_mode:
      - "tow":       climatology from all tows pooled (your current behavior)
      - "year_equal": climatology from annual means; each year gets weight=1
      - "year_weighted": climatology from annual means; year weight ~ (n_tows**power) with optional cap
    """
    y0, y1 = year_range

    df = paired.copy()
    df = df.query("@y0 <= year <= @y1").copy()
    if season is not None:
        df = df.query("season == @season").copy()

    season_label = season if season is not None else "All"
    if df.empty:
        raise ValueError(f"No data in paired table for season={season_label} and years {y0}-{y1}")

    # Transform on the same scale you want anomalies on
    if log_transform:
        df["obs_biomass"] = np.log(df["obs_biomass"] + log_offset)
        df["model_biomass"] = np.log(df["model_biomass"] + log_offset)

    clim_mode = str(clim_mode).strip().lower()

    # ------------------------------------------------------------------
    # Mode A: tow-weighted climatology (CURRENT behavior)
    # ------------------------------------------------------------------
    print("computing anomalies")
    if clim_mode == "tow":
        clim = (
            df.groupby("group", as_index=False)
              .agg(
                  mean_obs=("obs_biomass", "mean"),
                  std_obs=("obs_biomass", "std"),
                  mean_mod=("model_biomass", "mean"),
                  std_mod=("model_biomass", "std"),
              )
        )

        # Guard against zero/degenerate std
        clim.loc[clim["std_obs"].abs() < eps_std, "std_obs"] = np.nan
        clim.loc[clim["std_mod"].abs() < eps_std, "std_mod"] = np.nan

        df = df.merge(clim, on="group", how="left")
        df["obs_anom"] = (df["obs_biomass"] - df["mean_obs"]) / df["std_obs"]
        df["model_anom"] = (df["model_biomass"] - df["mean_mod"]) / df["std_mod"]

        annual = (
            df.groupby(["year", "group"], as_index=False)
              .agg(
                  obs_anom=("obs_anom", "mean"),
                  model_anom=("model_anom", "mean"),
              )
        )
        return annual

    # ------------------------------------------------------------------
    # Mode B/C: year-based climatology (climatology computed from annual means)
    # ------------------------------------------------------------------
    # Annual means first (this is what you actually plot/analyze)
    if "Index" in df.columns:
        n_tows = df.groupby(["year", "group"])["Index"].nunique()
    else:
        n_tows = df.groupby(["year", "group"]).size()

    annual_means = (
        df.groupby(["year", "group"], as_index=False)
          .agg(
              obs_mean=("obs_biomass", "mean"),
              mod_mean=("model_biomass", "mean"),
          )
    )
    annual_means["n_tows"] = n_tows.reset_index(drop=True).values

    # Use only "reasonably estimated" years to define the climatology baseline
    clim_base = annual_means[annual_means["n_tows"] >= int(min_tows_per_year)].copy()
    if clim_base.empty:
        raise ValueError(
            f"No years have >= {min_tows_per_year} tows for season={season_label}. "
            "Lower min_tows_per_year or use clim_mode='tow'."
        )

    rows = []
    for grp, gdf in clim_base.groupby("group"):
        xo = gdf["obs_mean"].to_numpy(dtype=float)
        xm = gdf["mod_mean"].to_numpy(dtype=float)

        if clim_mode == "year_equal":
            mean_obs = float(np.nanmean(xo))
            std_obs = float(np.nanstd(xo, ddof=1)) if np.isfinite(xo).sum() > 1 else np.nan
            mean_mod = float(np.nanmean(xm))
            std_mod = float(np.nanstd(xm, ddof=1)) if np.isfinite(xm).sum() > 1 else np.nan

        elif clim_mode == "year_weighted":
            ny = gdf["n_tows"].to_numpy(dtype=float)
            if year_weight_cap is not None:
                ny = np.minimum(ny, float(year_weight_cap))
            w = np.power(ny, float(year_weight_power))

            mean_obs = _wmean(xo, w)
            std_obs = _wstd_unbiased(xo, w)
            mean_mod = _wmean(xm, w)
            std_mod = _wstd_unbiased(xm, w)

        else:
            raise ValueError(
                f"Unknown clim_mode='{clim_mode}'. Use one of: "
                "'tow', 'year_equal', 'year_weighted'."
            )

        if np.isfinite(std_obs) and abs(std_obs) < eps_std:
            std_obs = np.nan
        if np.isfinite(std_mod) and abs(std_mod) < eps_std:
            std_mod = np.nan

        rows.append(
            dict(
                group=grp,
                mean_obs=mean_obs,
                std_obs=std_obs,
                mean_mod=mean_mod,
                std_mod=std_mod,
            )
        )

    clim = pd.DataFrame(rows)

    out = annual_means.merge(clim, on="group", how="left")
    out["obs_anom"] = (out["obs_mean"] - out["mean_obs"]) / out["std_obs"]
    out["model_anom"] = (out["mod_mean"] - out["mean_mod"]) / out["std_mod"]

    return out[["year", "group", "obs_anom", "model_anom"]]

def compare_anomaly_obs_summaries(
    cfg5: Eval5Config,
    groups: Optional[List[str]] = None,
    *,
    include_total: bool = True,
    seasons: Optional[List[str]] = None,
) -> tuple[str, str]:
    """Compare alternative observation-side seasonal summary methods."""
    matched_csv = os.path.join(cfg5.EVALOUT_P, cfg5.ZOOP_CSV_MATCH)
    if not os.path.exists(matched_csv):
        raise FileNotFoundError(f"Matched CSV not found: {matched_csv}")

    df_match = pd.read_csv(matched_csv)
    df_match = _exclude_stations(
        df_match,
        getattr(cfg5, "ANOM_EXCLUDE_STATIONS", []),
    )
    candidate = groups or cfg5.ZOOP_GROUPS
    groups_present = [g for g in candidate if g in df_match.columns and f"EWE-{g}" in df_match.columns]
    if not groups_present:
        raise ValueError("No zooplankton group columns found in matched CSV for anomaly comparison.")

    crust_groups = [g for g in groups_present if str(g).startswith("ZC")]
    zs_groups = [g for g in groups_present if str(g).startswith(("ZS2", "ZS3", "ZS4"))]

    passes: list[tuple[str, list[str]]] = []
    if crust_groups:
        passes.append(("ZC", crust_groups))
    if zs_groups:
        passes.append(("ZS", zs_groups))
    if not passes:
        passes.append(("ALL", groups_present))

    seasons_to_run = seasons if seasons is not None else (cfg5.SEASON_ORDER if cfg5.ANOM_ALL_SEASONS else [cfg5.ANOM_SEASON_CHOICE])
    seasons_plus_all = ["All"] + list(seasons_to_run)

    methods = [m for m in cfg5.COMPARE_ANOM_OBS_SUMMARIES if str(m).strip()]
    if not methods:
        methods = [cfg5.ANOM_OBS_SUMMARY]

    annual_rows: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []

    for pass_label, pass_groups in passes:
        paired = build_paired_long_ecospace(df_match, groups=pass_groups)
        paired = paired.query("@cfg5.START_YEAR <= year <= @cfg5.END_YEAR").copy()
        plot_groups = list(pass_groups) + (["Total"] if include_total else [])

        for season_name in seasons_plus_all:
            season_filter = None if str(season_name).lower() == "all" else season_name
            for obs_method in methods:
                annual = compute_anomalies_summary_paired(
                    paired,
                    season=season_filter,
                    year_range=(cfg5.START_YEAR, cfg5.END_YEAR),
                    obs_summary=obs_method,
                    model_summary=cfg5.ANOM_MODEL_SUMMARY,
                    clim_mode=cfg5.ANOM_CLIM_MODE,
                    log_base=cfg5.ANOM_LOG_BASE,
                    log_offset=cfg5.LOG_OFFSET,
                    year_weight_power=cfg5.ANOM_CLIM_WEIGHT_POWER,
                    year_weight_cap=cfg5.ANOM_CLIM_WEIGHT_CAP,
                    min_tows_per_year=cfg5.ANOM_CLIM_MIN_TOWS_PER_YEAR,
                    eps_std=cfg5.ANOM_CLIM_EPS_STD,
                    min_seasons_all=cfg5.ANOM_ALL_MIN_SEASONS,
                ).copy()
                annual["Pass"] = pass_label
                annual["Season"] = season_name
                annual["ObsSummary"] = obs_method
                annual_rows.append(annual)

                stats_df = skill_metrics_by_group(annual, groups=plot_groups, log_or_anom=True)
                if not stats_df.empty:
                    stats_df["Pass"] = pass_label
                    stats_df["Season"] = season_name
                    stats_df["ObsSummary"] = obs_method
                    metric_rows.append(stats_df)

    annual_out = pd.concat(annual_rows, ignore_index=True) if annual_rows else pd.DataFrame()
    metrics_out = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()

    annual_path = os.path.join(cfg5.EVALOUT_P, f"ecospace_zoop_anom_compare_annual_{cfg5.ecospace_code}.csv")
    metrics_path = os.path.join(cfg5.EVALOUT_P, f"ecospace_zoop_anom_compare_metrics_{cfg5.ecospace_code}.csv")

    annual_out.to_csv(annual_path, index=False)
    metrics_out.to_csv(metrics_path, index=False)

    print(f"[5d] Saved anomaly-summary annual comparison: {annual_path}")
    print(f"[5d] Saved anomaly-summary metric comparison: {metrics_path}")
    return annual_path, metrics_path

def counts_by_season_year_from_paired(
    paired: pd.DataFrame,
    *,
    season: Optional[str],
    group_for_counts: str = "Total",
    unique_index: bool = True,
    valid_pairs_only: bool = True,
) -> pd.DataFrame:
    """
    Compute a *single* tow/sample count per (year, season) suitable for annotating
    the anomaly panels.

    By default, counts are based on the 'Total' group and unique Index values.
    This tends to match the "how many tows went into that year/season mean"
    interpretation, and avoids double-counting if the matched table has duplicates.
    """
    df = paired.copy()
    if season is not None:
        df = df.query("season == @season").copy()

    season_label = season if season is not None else "All"

    if group_for_counts is not None and "group" in df.columns:
        df = df[df["group"] == group_for_counts]

    if valid_pairs_only:
        df = df.dropna(subset=["obs_biomass", "model_biomass"])

    if df.empty:
        return pd.DataFrame({"year": [], "season": [], "n_tows": []})

    if unique_index and "Index" in df.columns:
        c = df.groupby("year")["Index"].nunique()
    else:
        c = df.groupby("year").size()

    out = c.rename("n_tows").reset_index()
    out["season"] = season_label
    return out

# =============================================================================
# MODEL-ONLY BOX ANOMALIES
# =============================================================================

def extract_model_box_base_groups(
    cfg5: Eval5Config,
    *,
    base_groups: list[str],
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    spatial_reducer: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Faster version:
    - select wet cells inside lat/lon box
    - compute bounding row/col rectangle
    - read one contiguous NC block per variable
    - apply boolean mask to keep only selected cells
    """
    t0 = time.perf_counter()
    verbose = bool(getattr(cfg, "ZP_MODELONLY_VERBOSE", True))
    progress_every = int(getattr(cfg, "ZP_MODELONLY_PROGRESS_EVERY", 25))

    lat_min = float(getattr(cfg, "ZP_MODELONLY_LAT_MIN", lat_min))
    lat_max = float(getattr(cfg, "ZP_MODELONLY_LAT_MAX", lat_max))
    lon_min = float(getattr(cfg, "ZP_MODELONLY_LON_MIN", lon_min))
    lon_max = float(getattr(cfg, "ZP_MODELONLY_LON_MAX", lon_max))
    spatial_reducer = spatial_reducer or getattr(cfg, "ZP_MODELONLY_SPATIAL_REDUCER", "mean")

    lat_lo, lat_hi = sorted([lat_min, lat_max])
    lon_lo, lon_hi = sorted([lon_min, lon_max])

    ecospace_nc = os.path.join(cfg5.NC_PATH_OUT, cfg5.ecospace_nc_name)
    grid_csv = os.path.join(cfg5.BASEMAP_P, cfg5.GRID_MAP_CSV)

    if verbose:
        print("[5m] --------------------------------------------------")
        print("[5m] Model-only box extraction (fast block-read version)")
        print(f"[5m] NC file:   {ecospace_nc}")
        print(f"[5m] Grid file: {grid_csv}")
        print(f"[5m] Lat box:   {lat_lo:.4f} .. {lat_hi:.4f}")
        print(f"[5m] Lon box:   {lon_lo:.4f} .. {lon_hi:.4f}")
        print(f"[5m] Reducer:   {spatial_reducer}")
        print(f"[5m] Base groups requested ({len(base_groups)}): {base_groups}")

    grid_df = pd.read_csv(grid_csv)
    n_grid = len(grid_df)
    n_wet = int((grid_df["depth"] != 0).sum())

    sel = (
        (grid_df["depth"] != 0) &
        (grid_df["lat"] >= lat_lo) & (grid_df["lat"] <= lat_hi) &
        (grid_df["lon"] >= lon_lo) & (grid_df["lon"] <= lon_hi)
    )

    cells = (
        grid_df.loc[sel, ["EWE_row", "EWE_col", "lat", "lon", "depth"]]
        .dropna()
        .drop_duplicates(subset=["EWE_row", "EWE_col"])
        .copy()
    )

    if verbose:
        print(f"[5m] Grid cells in CSV: {n_grid}")
        print(f"[5m] Wet cells in grid: {n_wet}")
        print(f"[5m] Wet cells in box:  {len(cells)}")

    if cells.empty:
        raise ValueError(
            "No wet Ecospace cells found inside the requested lat/lon box. "
            "Check the bounds against the grid map."
        )

    rows = cells["EWE_row"].astype(int).to_numpy()
    cols = cells["EWE_col"].astype(int).to_numpy()

    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    nr = rmax - rmin + 1
    nc_ = cmax - cmin + 1

    if verbose:
        print(
            f"[5m] Selected row range: {rmin} .. {rmax} ({nr} rows) | "
            f"col range: {cmin} .. {cmax} ({nc_} cols)"
        )
        print(
            f"[5m] Selected cell lat range: {cells['lat'].min():.4f} .. {cells['lat'].max():.4f} | "
            f"lon range: {cells['lon'].min():.4f} .. {cells['lon'].max():.4f}"
        )
        print(
            f"[5m] Depth range in box: {cells['depth'].min():.2f} .. {cells['depth'].max():.2f}"
        )
        print("[5m] First few selected cells:")
        print(cells.head(8).to_string(index=False))

    # Build a boolean mask over the rectangular row/col block
    # True where a selected wet cell exists
    cell_mask_2d = np.zeros((nr, nc_), dtype=bool)
    rr = rows - rmin
    cc = cols - cmin
    cell_mask_2d[rr, cc] = True

    n_rect = nr * nc_
    fill_frac = len(cells) / n_rect if n_rect > 0 else np.nan

    if verbose:
        print(f"[5m] Bounding rectangle size: {nr} x {nc_} = {n_rect} cells")
        print(f"[5m] Selected-cell fill fraction inside rectangle: {fill_frac:.1%}")

    _, ecospace_times = _load_ecospace_times(ecospace_nc)

    model_ts = pd.DataFrame({
        "date": pd.to_datetime(ecospace_times)
    })
    model_ts["year"] = model_ts["date"].dt.year
    model_ts["month"] = model_ts["date"].dt.month
    model_ts["season"] = model_ts["month"].apply(_month_to_season)

    if verbose:
        print(
            f"[5m] Time slices in NC: {len(model_ts)} | "
            f"{model_ts['date'].min().date()} .. {model_ts['date'].max().date()}"
        )

    present_groups = []

    with nc.Dataset(ecospace_nc, "r") as eco_ds:
        nc_vars = set(eco_ds.variables.keys())

        if verbose:
            missing_now = [g for g in base_groups if g not in nc_vars]
            present_now = [g for g in base_groups if g in nc_vars]
            print(f"[5m] Requested groups present in NC: {present_now}")
            if missing_now:
                print(f"[5m][WARN] Requested groups missing in NC: {missing_now}")

        for gi, var in enumerate(base_groups, start=1):
            if var not in eco_ds.variables:
                print(f"[5m][WARN] Variable '{var}' not found in NC. Skipping.")
                continue

            g0 = time.perf_counter()
            if verbose:
                print(
                    f"[5m] [{gi}/{len(base_groups)}] Extracting '{var}' "
                    f"using one block read: [:, {rmin}:{rmax+1}, {cmin}:{cmax+1}] ..."
                )

            # Read one contiguous block: shape = (time, nr, nc_)
            block = np.asarray(
                np.ma.filled(
                    eco_ds.variables[var][:, rmin:rmax+1, cmin:cmax+1],
                    np.nan
                ),
                dtype=float
            )

            if verbose:
                n_mb = block.nbytes / (1024 ** 2)
                print(f"[5m]     block shape={block.shape}, approx memory={n_mb:.1f} MB")

            # Reshape to (time, nr*nc_) and keep only selected cells
            block2 = block.reshape(block.shape[0], -1)
            mask1d = cell_mask_2d.reshape(-1)
            selected = block2[:, mask1d]   # shape = (time, n_selected_cells)

            if verbose:
                print(f"[5m]     selected matrix shape after mask: {selected.shape}")

            # Reduce across selected cells for each time
            if spatial_reducer == "mean":
                series = np.nanmean(selected, axis=1)
            elif spatial_reducer == "median":
                series = np.nanmedian(selected, axis=1)
            else:
                raise ValueError(f"Unknown spatial reducer: {spatial_reducer}")

            model_ts[var] = series
            present_groups.append(var)

            if verbose:
                s = model_ts[var]
                n_nan = int(s.isna().sum())
                print(
                    f"[5m]     finished '{var}' in {_fmt_elapsed(g0)} | "
                    f"min={np.nanmin(s.values):.6g}, mean={np.nanmean(s.values):.6g}, "
                    f"max={np.nanmax(s.values):.6g}, n_nan={n_nan}"
                )

            # free memory before next variable
            del block, block2, selected

    if not present_groups:
        raise ValueError("None of the requested base groups were found in the NetCDF.")

    print(
        f"[5m] Box extraction complete: {len(cells)} wet cells, "
        f"{len(present_groups)} groups extracted, elapsed={_fmt_elapsed(t0)}."
    )

    return model_ts, cells

def build_modelonly_named_series(
    model_ts_base: pd.DataFrame,
    *,
    series_def: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Build named output series from base-group columns.

    Example:
        "ZC2-AMP" -> ["ZC2-AMP"]
        "Total"   -> ["ZC1-EUP","ZC2-AMP",...]
    """
    verbose = bool(getattr(cfg, "ZP_MODELONLY_VERBOSE", True))

    out = model_ts_base[["date", "year", "month", "season"]].copy()

    if verbose:
        print(f"[5m] Building named output series ({len(series_def)} definitions) ...")

    for out_name, members in series_def.items():
        missing = [g for g in members if g not in model_ts_base.columns]
        present = [g for g in members if g in model_ts_base.columns]

        if not present:
            print(f"[5m][WARN] No members present for series '{out_name}'. Skipping.")
            continue

        if missing:
            print(f"[5m][WARN] Series '{out_name}' missing members: {missing}")

        out[out_name] = model_ts_base[present].sum(axis=1)

        if verbose:
            s = out[out_name]
            print(
                f"[5m]   built '{out_name}' from {present} | "
                f"min={np.nanmin(s.values):.6g}, mean={np.nanmean(s.values):.6g}, "
                f"max={np.nanmax(s.values):.6g}"
            )

    series_cols = [c for c in out.columns if c not in ("date", "year", "month", "season")]
    if not series_cols:
        raise ValueError("No named output series could be built from the extracted base groups.")

    if verbose:
        print(f"[5m] Named series available: {series_cols}")

    return out

def build_model_only_long_from_named_series(
    model_ts_named: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a paired-like long table so the existing anomaly engine can be reused.

    obs_biomass and model_biomass are intentionally duplicated from the same
    model-only series; final output uses model_anom only.
    """
    series_cols = [c for c in model_ts_named.columns if c not in ("date", "year", "month", "season")]

    d = model_ts_named.copy()
    d["Index"] = np.arange(len(d), dtype=int)

    recs = []
    for g in series_cols:
        tmp = pd.DataFrame({
            "date": d["date"],
            "season": d["season"],
            "Index": d["Index"],
            "obs_biomass": d[g].astype(float),
            "model_biomass": d[g].astype(float),
            "group": g,
        })
        recs.append(tmp)

    out = pd.concat(recs, ignore_index=True)
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return out

def compute_model_only_box_anomalies(
    cfg5: Eval5Config,
    *,
    season: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end:
      1) extract requested base groups from lat/lon box
      2) build named series (individuals and/or totals)
      3) compute annual model-only anomalies

    Returns
    -------
    annual_anom : DataFrame with columns [year, group, anom]
    model_ts_named : wide time-series DataFrame of named series
    cells : selected cell table
    """
    t0 = time.perf_counter()
    verbose = bool(getattr(cfg, "ZP_MODELONLY_VERBOSE", True))

    series_def = _get_modelonly_series_def(cfg5)
    base_groups = sorted(set(g for members in series_def.values() for g in members))
    season = getattr(cfg, "ZP_MODELONLY_SEASON", season)

    y0 = int(getattr(cfg, "ZP_MODELONLY_YEAR_START", getattr(cfg, "ZP_FULLRN_START", cfg5.START_YEAR)))
    y1 = int(getattr(cfg, "ZP_MODELONLY_YEAR_END",   getattr(cfg, "ZP_FULLRN_END",   cfg5.END_YEAR)))
    summary_method = getattr(cfg, "ZP_MODELONLY_ANOM_SUMMARY", cfg5.ANOM_MODEL_SUMMARY)

    if verbose:
        print("[5m] --------------------------------------------------")
        print("[5m] Computing model-only box anomalies")
        print(f"[5m] Series definition keys: {list(series_def.keys())}")
        print(f"[5m] Unique base groups needed: {base_groups}")
        print(f"[5m] Year window: {y0} .. {y1}")
        print(f"[5m] Season filter: {season if season is not None else 'All seasons'}")
        print(f"[5m] Summary method: {summary_method}")
        print(f"[5m] Climatology mode: {cfg5.ANOM_CLIM_MODE}")

    model_ts_base, cells = extract_model_box_base_groups(
        cfg5,
        base_groups=base_groups,
        lat_min=getattr(cfg, "ZP_MODELONLY_LAT_MIN", None),
        lat_max=getattr(cfg, "ZP_MODELONLY_LAT_MAX", None),
        lon_min=getattr(cfg, "ZP_MODELONLY_LON_MIN", None),
        lon_max=getattr(cfg, "ZP_MODELONLY_LON_MAX", None),
        spatial_reducer=getattr(cfg, "ZP_MODELONLY_SPATIAL_REDUCER", "mean"),
    )

    model_ts_named = build_modelonly_named_series(
        model_ts_base,
        series_def=series_def,
    )

    paired_like = build_model_only_long_from_named_series(model_ts_named)

    if verbose:
        print(
            f"[5m] Long table for anomaly engine: {len(paired_like)} rows | "
            f"groups={sorted(paired_like['group'].dropna().unique().tolist())}"
        )
        if "year" in paired_like.columns and paired_like["year"].notna().any():
            print(
                f"[5m] Long table year coverage: "
                f"{int(paired_like['year'].min())} .. {int(paired_like['year'].max())}"
            )

    annual = compute_anomalies_summary_paired(
        paired_like,
        season=season,
        year_range=(y0, y1),
        obs_summary=summary_method,
        model_summary=summary_method,
        clim_mode=cfg5.ANOM_CLIM_MODE,
        log_base=cfg5.ANOM_LOG_BASE,
        log_offset=cfg5.LOG_OFFSET,
        year_weight_power=cfg5.ANOM_CLIM_WEIGHT_POWER,
        year_weight_cap=cfg5.ANOM_CLIM_WEIGHT_CAP,
        min_tows_per_year=cfg5.ANOM_CLIM_MIN_TOWS_PER_YEAR,
        eps_std=cfg5.ANOM_CLIM_EPS_STD,
        min_seasons_all=cfg5.ANOM_ALL_MIN_SEASONS,
    ).copy()

    annual["anom"] = annual["model_anom"]
    annual = annual[["year", "group", "anom"]].copy()

    if verbose:
        print(f"[5m] Annual anomaly rows: {len(annual)}")
        for g in sorted(annual["group"].dropna().unique()):
            dg = annual[annual["group"] == g].sort_values("year")
            if dg.empty:
                continue
            print(
                f"[5m]   {g}: years={int(dg['year'].min())}..{int(dg['year'].max())}, "
                f"n={len(dg)}, anom_min={np.nanmin(dg['anom'].values):.4f}, "
                f"anom_max={np.nanmax(dg['anom'].values):.4f}"
            )

    print(f"[5m] Model-only anomaly computation complete in {_fmt_elapsed(t0)}.")
    return annual, model_ts_named, cells

# =============================================================================
# PLOTTING
# =============================================================================

def panel_seasonal_boxplots(
    df_station: pd.DataFrame,
    *,
    groups: list[str],
    cfg5,
    extra_total: str | None = None,
    suffix: str = "",
) -> None:
    """
    Make a multi-panel seasonal Obs vs Model boxplot figure.

    - df_station: output of the station-level aggregation in visualize_and_stats
    - groups: list of *observation* group column names (e.g. ["ZC1-EUP", "ZC2-AMP"])
    - extra_total: optional extra column to append (e.g. "TOT_ZC", "TOT_SOFT")
    - suffix: string used in the output filename (e.g. "ZC", "soft")
    """

    if not groups:
        return

    # groups to actually plot (optionally add total column)
    plot_groups = list(groups)
    if extra_total and extra_total in df_station.columns:
        plot_groups.append(extra_total)

    records = []
    for g in plot_groups:
        obs_col = g

        # figure out the matching model column
        if g == extra_total:
            # built these in visualize_and_stats:
            #   "TOT_ZC"   -> "EWE-TOT_ZC"
            #   "TOT_SOFT" -> "EWE-TOT_SOFT"
            mod_col = f"EWE-{extra_total}"
        else:
            mod_col = f"EWE-{g}"

        if obs_col not in df_station.columns or mod_col not in df_station.columns:
            continue

        sub = df_station[["Season", obs_col, mod_col]].copy()

        if cfg5.BOXPLOT_LOG10:
            sub["Observations"] = np.log10(sub[obs_col] + cfg5.LOG_OFFSET)
            sub["SOGEM-LTL"] = np.log10(sub[mod_col] + cfg5.LOG_OFFSET)
        else:
            sub["Observations"] = sub[obs_col]
            sub["SOGEM-LTL"] = sub[mod_col]

        long = sub.melt(
            id_vars="Season",
            value_vars=["Observations", "SOGEM-LTL"],
            var_name="Source",
            value_name="plot_value",
        )
        long["group"] = g
        records.append(long)


    if not records:
        return

    plot_df = pd.concat(records, ignore_index=True)

    # order seasons consistently
    plot_df["Season"] = pd.Categorical(
        plot_df["Season"],
        categories=cfg5.SEASON_ORDER,
        ordered=True,
    )

    unique_groups = list(dict.fromkeys(plot_df["group"]))  # preserve order
    n_groups = len(unique_groups)
    ncols = 2
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, g in enumerate(unique_groups):
        ax = axes[i]
        dd = plot_df[plot_df["group"] == g]

        sns.boxplot(
            data=dd,
            x="Season",
            y="plot_value",
            showfliers=False,  # match Ecosim behaviour
            hue="Source",
            ax=ax,
            width=0.7,
            palette={"Observations": "darkorange", "SOGEM-LTL": "blue"},
            medianprops={"color": "white", "linewidth": 0.8}
        )

        # nicer labels
        if g == "TOT_ZC":
            title = "Total ZC"
        elif g == "TOT_SOFT":
            title = "Total ZS"
        else:
            title = g

        ax.set_title(title)

        panel_lab = f"({string.ascii_lowercase[i]})"
        ax.text(
            0.02, 0.98, panel_lab,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("Season")
        ylabel = (
            "log10(g C m$^{-2}$ + offset)"
            if cfg5.BOXPLOT_LOG10
            else "Biomass (g C m$^{-2}$)"
        )
        ax.set_ylabel(ylabel)

        # if i == 0:
        #     ax.legend(title="")
        # else:
            # remove duplicate legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.grid(True, alpha=0.3)

    # turn off any unused axes
    for j in range(n_groups, len(axes)):
        axes[j].axis("off")

    if cfg5.ADD_FIG_SUPTITLE:

        # pick a sensible year range
        if "Year" in df_station.columns and df_station["Year"].notna().any():
            yr0 = int(df_station["Year"].min())
            yr1 = int(df_station["Year"].max())
        else:
            yr0, yr1 = cfg5.START_YEAR, cfg5.END_YEAR

        log_state = "log10" if cfg5.BOXPLOT_LOG10 else "raw"
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Seasonal biomass (Obs vs Model)",
                years=(yr0, yr1),
                suffix=suffix,
                log_state=log_state,
            ),
            y=0.995,  # <-- was 1.02
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # <-- a touch more room than 0.96
    else:
        st = None
        fig.tight_layout()

    out_plt = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_seasB_modobs_panel_{suffix}.png",
    )

    fig.savefig(
        out_plt,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )

    _maybe_show(fig, cfg5.SHOW_SEASONAL_BOXPANELS or cfg5.SHOW_PLTS)

    print(f"[5c] Saved seasonal panel boxplots: {out_plt}")

def plot_seasonal_boxpanels(
    df_station: pd.DataFrame,
    groups: list[str],
    *,
    suffix: str,
    cfg5,
    title_prefix: str = ""
) -> None:
    """
    Panel of Obs vs Model seasonal boxplots for a set of groups.

    df_station must contain, for each group G in `groups`,
    columns:
      - G         (observed biomass)
      - 'EWE-G'   (model biomass)
      - 'Season'  (categorical)
    """
    import math

    ncols = 3
    nrows = int(math.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(groups):
        ax = axes[i]

        model_col = f"EWE-{group}"
        if group not in df_station.columns or model_col not in df_station.columns:
            ax.axis("off")
            continue

        tmp = df_station[["Season", group, model_col]].copy()
        tmp = tmp.melt(
            id_vars="Season",
            value_vars=[group, model_col],
            var_name="Source",
            value_name="Biomass",
        )

        # Consistent season ordering
        tmp["Season"] = pd.Categorical(
            tmp["Season"],
            categories=cfg5.SEASON_ORDER,
            ordered=True,
        )

        tmp["Source"] = tmp["Source"].map({group: "Observations", model_col: "SOGEM-LTL"})

        sns.boxplot(
            data=tmp,
            x="Season",
            y="Biomass",
            hue="Source",
            ax=ax,
            showfliers=False,
            palette={"Observations": "darkorange", "SOGEM-LTL": "blue"},
            medianprops={"color": "white", "linewidth": 0.8}
        )
        # ax.set_yscale("log")
        ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)

        # nicer titles for the totals
        display_name = {
            "TOT_ZC": "Total (ZC groups)",
            "TOT_SOFT": "Total (soft groups)",
        }.get(group, group)
        ax.set_title(display_name)

        ax.set_xlabel("")
        if i % ncols == 0:
            ax.set_ylabel("Biomass (g C m$^{-2}$)")
        else:
            ax.set_ylabel("")

        # shared legend only on first panel
        if i == 0 and ax.get_legend() is not None:
            ax.legend()
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # hide any unused axes
    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    if cfg5.ADD_FIG_SUPTITLE:
        # pick a sensible year range
        if "Year" in df_station.columns and df_station["Year"].notna().any():
            yr0 = int(df_station["Year"].min())
            yr1 = int(df_station["Year"].max())
        else:
            yr0, yr1 = cfg5.START_YEAR, cfg5.END_YEAR

        log_state = "log10" if cfg5.LOG_TRANSFORM else "raw"
        fig.suptitle(
            _fig_title(
                cfg5,
                kind="Seasonal biomass (Obs vs Model)",
                years=(yr0, yr1),
                suffix=suffix,
                log_state=log_state,
            ),
            y=1.02,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    out_plt = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_seasB_modobs_panel_{suffix}.png",
    )

    fig.savefig(out_plt, dpi=300, bbox_inches="tight")
    _maybe_show(fig, cfg5.SHOW_SEASONAL_BOXPANELS or cfg5.SHOW_PLTS)

    print(f"[5d] Saved seasonal boxplot panels → {out_plt}")

def plot_scatter_total_by_season(
    paired_long: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    total_group: str = "Total",
    seasons: Optional[List[str]] = None,
    label: str = "zoop_crust_total",
    outname: str = "scatter_tows_total4panel",
    log10: bool = True,
    title_text: str = "Total crustaceans",
) -> str:
    """
    Publication-style tow-level scatter:
    one subplot per season, for a single aggregate group
    (e.g. crustacean Total from the crustacean pass).
    """
    seasons = seasons or cfg5.SEASON_ORDER

    d = paired_long.copy()
    d = d.dropna(subset=["group", "obs_biomass", "model_biomass"])
    d = d[d["group"] == total_group].copy()

    if d.empty:
        raise ValueError(f"No paired tow-level values found for group '{total_group}'.")

    if log10:
        eps = 1e-12
        d = d[(d["obs_biomass"] > 0) & (d["model_biomass"] > 0)].copy()
        d["obs_x"] = np.log10(d["obs_biomass"] + eps)
        d["mod_y"] = np.log10(d["model_biomass"] + eps)
        xlab = "Obs (log10 g C m$^{-2}$)"
        ylab = "Model (log10 g C m$^{-2}$)"
    else:
        d["obs_x"] = d["obs_biomass"]
        d["mod_y"] = d["model_biomass"]
        xlab = "Obs (g C m$^{-2}$)"
        ylab = "Model (g C m$^{-2}$)"

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, season in enumerate(seasons[:4]):
        ax = axes[i]
        g = d[d["season"] == season].copy()

        ax.scatter(g["obs_x"], g["mod_y"], s=18, alpha=0.6)

        if len(g) > 0:
            lo = min(g["obs_x"].min(), g["mod_y"].min())
            hi = max(g["obs_x"].max(), g["mod_y"].max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)

            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="#808080", lw=1)

            _add_best_fit_line(
                ax,
                g["obs_x"].values,
                g["mod_y"].values,
                color="k",
                linestyle="--",
                linewidth=1,
                show_eq=False,
            )

            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

            if len(g) > 1:
                r = np.corrcoef(g["obs_x"], g["mod_y"])[0, 1]
                ax.set_title(f"{season}  r={r:.2f}  n={len(g)}")
            else:
                ax.set_title(f"{season}  n={len(g)}")
        else:
            ax.set_title(f"{season}  n=0")

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        log_state = "log10" if log10 else "raw"
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Tow-level scatter (Obs vs Model)",
                suffix=title_text,
                plot_type="scatter",
                log_state=log_state,
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}.png",
    )
    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )

    _maybe_show(fig, cfg5.SHOW_TOTAL_BY_SEASON_SCATTER or cfg5.SHOW_PLTS)

    print(f"[INFO] Saved total-by-season scatter plot: {outpath}")
    return outpath

def _add_statbox(ax, text: str, *, x=0.98, y=0.02):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
        zorder=10,
    )

def _add_colored_statbox(ax, lines, *, x=0.98, y=0.02, lineheight=0.055):
    """
    lines: list of (text, color) tuples
    Places a multi-line stats box in axes coordinates.
    """
    if not lines:
        return

    # crude box height estimate in axes coordinates
    n = len(lines)
    box_h = lineheight * n + 0.02
    box_w = 0.34  # adjust if needed

    # draw background box
    rect = plt.Rectangle(
        (x - box_w, y),
        box_w,
        box_h,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
        zorder=9,
    )
    ax.add_patch(rect)

    # draw each line, top to bottom
    for j, (txt, color) in enumerate(lines):
        yy = y + box_h - 0.015 - j * lineheight
        ax.text(
            x - 0.01,
            yy,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            zorder=10,
        )

def plot_pub_panel_total_single_season(
    paired: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    season: str,
    total_group: str = "Total",
    label: str = "zoop_ZC",
    outname: str = "pub_total_single_season",
    title_text: str = "Total crustaceans",
    log10_scatter: bool = True,
) -> str:
    """
    Publication-style 2-panel figure for a single season:
      (a) tow-level OR seas-lev anom Obs vs Model scatter
      (b) annual anomaly bar plot (Obs vs Model)
    Intended for the crustacean pass, where Total == sum(ZC*).
    """

    panel_a_mode = str(getattr(cfg5, "PUB_PANEL_A_MODE", "anomaly_scatter")).strip().lower()

    annual = compute_anomalies_for_eval(
        paired,
        cfg5=cfg5,
        season=season,
    )
    annual = annual[annual["group"] == total_group].copy()

    counts = counts_by_season_year_from_paired(
        paired,
        season=season,
        group_for_counts=total_group,
        unique_index=True,
        valid_pairs_only=True,
    )

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(5.3, 8),
        gridspec_kw={"height_ratios": [1.3, 1.0]},
        constrained_layout=False,
    )
    ax_scatter, ax_bar = axes

    # =========================================================
    # TOP PANEL: choose between tow-level scatter or anomaly scatter
    # =========================================================
    if panel_a_mode == "tow_scatter":
        d = paired.copy()
        d = d.dropna(subset=["group", "obs_biomass", "model_biomass"])
        d = d[(d["group"] == total_group) & (d["season"] == season)].copy()

        if log10_scatter:
            eps = 1e-12
            d = d[(d["obs_biomass"] > 0) & (d["model_biomass"] > 0)].copy()
            d["x"] = np.log10(d["obs_biomass"] + eps)
            d["y"] = np.log10(d["model_biomass"] + eps)
            xlab = "Observed biomass (log10 g C m$^{-2}$)"
            ylab = "Model biomass (log10 g C m$^{-2}$)"
        else:
            d["x"] = d["obs_biomass"]
            d["y"] = d["model_biomass"]
            xlab = "Observed biomass (g C m$^{-2}$)"
            ylab = "Model biomass (g C m$^{-2}$)"

        ax_scatter.scatter(d["x"], d["y"], s=22, alpha=0.7)

        if len(d) > 0:
            lo = min(d["x"].min(), d["y"].min())
            hi = max(d["x"].max(), d["y"].max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)

            ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="#808080", lw=1)

            _add_best_fit_line(
                ax_scatter,
                d["x"].values,
                d["y"].values,
                color="k",
                linestyle="--",
                linewidth=1,
                show_eq=False,
            )

            ax_scatter.set_xlim(lo - pad, hi + pad)
            ax_scatter.set_ylim(lo - pad, hi + pad)

            if len(d) > 1:
                stats_a = compute_stats(
                    d.rename(columns={"x": "obs_tmp", "y": "mod_tmp"}),
                    "obs_tmp", "mod_tmp",
                    log_or_anom=log10_scatter,
                )
                stat_text_a = f"r = {stats_a['r']:.2f}\nN = {int(stats_a['N'])}"
            elif len(d) == 1:
                stat_text_a = "n = 1"
        else:
            stat_text_a = "n = 0"

            # do not title for publication
            # ax_scatter.set_title(...)
        _add_statbox(ax_scatter, stat_text_a)

        ax_scatter.set_xlabel(xlab)
        ax_scatter.set_ylabel(ylab)

    elif panel_a_mode == "anomaly_scatter":
        d = annual.dropna(subset=["obs_anom", "model_anom"]).copy()

        ax_scatter.scatter(d["obs_anom"], d["model_anom"], s=28, alpha=0.8)

        if len(d) > 0:
            lo = min(d["obs_anom"].min(), d["model_anom"].min())
            hi = max(d["obs_anom"].max(), d["model_anom"].max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)

            ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="#808080", lw=1)

            _add_best_fit_line(
                ax_scatter,
                d["obs_anom"].values,
                d["model_anom"].values,
                color="k",
                linestyle="--",
                linewidth=1,
                show_eq=False,
            )

            ax_scatter.set_xlim(lo - pad, hi + pad)
            ax_scatter.set_ylim(lo - pad, hi + pad)

            if len(d) > 1:
                stats_a = compute_stats(d, "obs_anom", "model_anom", log_or_anom=True)
                stat_text_a = f"r = {stats_a['r']:.2f}\nN = {int(stats_a['N'])}"
            elif len(d) == 1:
                stat_text_a = "n = 1"
        else:
            stat_text_a = "n = 0"

        _add_statbox(ax_scatter, stat_text_a)

        ax_scatter.set_xlabel("Observed anomaly (z-score)")
        ax_scatter.set_ylabel("Model anomaly (z-score)")

        # Optional: label points by year for publication clarity
        for _, row in d.iterrows():
            ax_scatter.text(
                row["obs_anom"],
                row["model_anom"],
                str(int(row["year"])),
                fontsize=7,
                alpha=0.75,
            )

    else:
        raise ValueError(
            f"Unknown PUB_PANEL_A_MODE='{panel_a_mode}'. "
            "Use 'tow_scatter' or 'anomaly_scatter'."
        )

    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.text(
        0.01, 0.98, "(a)",
        transform=ax_scatter.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )


    # =========================================================
    # BOTTOM PANEL: annual anomaly bars
    # =========================================================
    years = sorted(annual["year"].unique().tolist())
    pair_step = 1.18  # spacing between years
    w = 0.28  # bar width
    x = np.arange(len(years)) * pair_step

    if len(years) > 0:
        d_idx = annual.set_index("year")
        x = np.arange(len(years))
        y_obs = [d_idx.loc[yr, "obs_anom"] if yr in d_idx.index else np.nan for yr in years]
        y_mod = [d_idx.loc[yr, "model_anom"] if yr in d_idx.index else np.nan for yr in years]

        ax_bar.axhline(0, lw=1, linestyle="--", alpha=0.7, color="k", zorder=1)
        ax_bar.bar(x - w / 2, y_obs, w, label="Observations", color="darkorange", zorder=2)
        ax_bar.bar(x + w / 2, y_mod, w, label="SOGEM-LTL", color="blue", zorder=2)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax_bar.set_ylabel("Anomaly (z-score)")
        ax_bar.set_xlabel("Year")

        stats_b = compute_stats(annual, "obs_anom", "model_anom", log_or_anom=True)
        if int(stats_b["N"]) > 0 and pd.notna(stats_b["WSS"]):
            stat_text_b = f"WSS = {stats_b['WSS']:.2f}\nN = {int(stats_b['N'])}"
            stat_text_b = f"WSS = {stats_b['WSS']:.2f}"
        else:
            stat_text_b = f"n = {int(stats_b['N'])}"

        _add_statbox(ax_bar, stat_text_b)
        # ax_bar.set_title(f"{season}: annual anomalies")

        ax_bar.grid(True, alpha=0.3, zorder=0)
        ax_bar.legend()

        if cfg5.SHOW_COUNTS and counts is not None and not counts.empty:
            cmap = dict(zip(counts["year"].astype(int), counts["n_tows"]))
            ymin, ymax = ax_bar.get_ylim()
            yrange = ymax - ymin if ymax > ymin else 1.0
            for j, yr in enumerate(years):
                n = cmap.get(int(yr))
                if n is None:
                    continue
                vals = [v for v in [y_obs[j], y_mod[j]] if pd.notna(v)]
                if not vals:
                    continue
                y_top = np.nanmax(vals)
                ax_bar.text(
                    j, y_top + 0.04 * yrange, f"{int(n)}",
                    ha="center", va="bottom", fontsize=8
                )
    else:
        ax_bar.set_title(f"{season}: annual anomalies  (no data)")
        ax_bar.axhline(0, lw=1, linestyle="--", alpha=0.7, color="k")

    ax_bar.text(
        0.01, 0.98, "(b)",
        transform=ax_bar.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )

    # -----------------------------
    # Figure title and save
    # -----------------------------
    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Publication panel",
                years=(cfg5.START_YEAR, cfg5.END_YEAR),
                season=season,
                suffix=title_text,
                label=label,
                plot_type="scatter + anomaly bars",
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{season}_{label}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )

    _maybe_show(fig, cfg5.SHOW_PUB_TOTAL_PANEL or cfg5.SHOW_PLTS)

    print(f"[INFO] Saved publication panel: {outpath}")
    return outpath

def plot_model_only_anomaly_bars(
    annual_anom: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    group: str = "Total",
    season: Optional[str] = None,
    label: str = "zoop_model_box",
    outname: str = "model_box_anomaly_bars",
    title_text: Optional[str] = None,
) -> str:
    """
    Single-series annual anomaly bar plot for one named model-only series.
    """
    d = annual_anom.copy()
    d = d[d["group"] == group].copy()

    if d.empty:
        raise ValueError(f"No anomaly values found for group '{group}'.")

    pretty = getattr(cfg, "ZP_MODELONLY_LABELS", {}).get(group, group)
    title_text = title_text or pretty

    d = d.sort_values("year")
    years = d["year"].astype(int).tolist()
    vals = d["anom"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(12, 4.8))

    x = np.arange(len(years))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]

    ax.axhline(0, lw=1, linestyle="--", alpha=0.7, color="k", zorder=1)
    ax.bar(x, vals, width=0.82, color=colors, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_ylabel("Anomaly (z-score)")
    ax.set_xlabel("Year")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        y0 = int(getattr(cfg, "ZP_MODELONLY_YEAR_START", getattr(cfg, "ZP_FULLRN_START", cfg5.START_YEAR)))
        y1 = int(getattr(cfg, "ZP_MODELONLY_YEAR_END",   getattr(cfg, "ZP_FULLRN_END",   cfg5.END_YEAR)))
        season_text = season if season is not None else "All seasons"
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Model-only annual anomalies",
                years=(y0, y1),
                season=season_text,
                suffix=title_text,
                plot_type="bar",
                log_state="anom z-score",
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{season}_{outname}_{label}_{group}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )

    _maybe_show(fig, cfg5.SHOW_MODELONLY_ANOM_BARS or cfg5.SHOW_PLTS)

    print(f"[5m] Saved model-only anomaly bar plot: {outpath}")
    return outpath


def plot_model_only_anomaly_panel(
    annual_anom: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    top_group: str = "Total",
    bottom_group: str = "ZC2-AMP",
    season: Optional[str] = None,
    label: str = "zoop_model_box",
    outname: str = "model_box_anomaly_panel",
) -> str:
    """
    Two-row publication panel for model-only annual anomalies:
      (a) top_group
      (b) bottom_group

    No subplot titles. Keeps standalone plots separate.
    """
    def _draw_group(ax, group: str, tag: str):
        d = annual_anom.copy()
        d = d[d["group"] == group].copy()
        if d.empty:
            raise ValueError(f"No anomaly values found for group '{group}'.")

        d = d.sort_values("year")
        years = d["year"].astype(int).tolist()
        vals = d["anom"].astype(float).tolist()

        x = np.arange(len(years))
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]

        ax.axhline(0, lw=1, linestyle="--", alpha=0.7, color="k", zorder=1)
        ax.bar(x, vals, width=0.82, color=colors, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax.set_ylabel("Anomaly (z-score)")
        ax.grid(True, axis="y", alpha=0.3, zorder=0)

        # panel tag
        ax.text(
            0.01, 0.97, tag,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold"
        )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7.2),
        sharex=False,
        constrained_layout=False,
    )

    _draw_group(axes[0], top_group, "(a)")
    _draw_group(axes[1], bottom_group, "(b)")

    axes[0].set_xlabel("")      # no x-label on top panel
    axes[1].set_xlabel("Year")  # only bottom panel gets x-label

    # No titles in the panel
    fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}_{top_group}_{bottom_group}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    _maybe_show(fig, cfg5.SHOW_MODELONLY_ANOM_BARS or cfg5.SHOW_PLTS)

    print(f"[5m] Saved model-only anomaly panel: {outpath}")
    return outpath

def plot_anomaly_panel_paired(
    annual: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    plot_groups: List[str],
    label: str,
    season: str,
    plot_type: str = "bar",
    counts: Optional[pd.DataFrame] = None,
    show_counts: bool = True,
    npgo_ann: Optional[pd.Series] = None,
    count_yoffset: float = 0.05,
    count_fontsize: int = 9,
) -> str:
    """
    Ecosim-style anomaly panel: one subplot per group, with Obs vs Model shown
    together (bars or lines) and optional per-year tow counts.
    """
    years = sorted(annual["year"].unique())
    if not years:
        raise ValueError("No years found in annual anomalies table.")

    # year -> n_tows map (for labels)
    counts_map: Dict[int, float] = {}
    if show_counts and counts is not None and not counts.empty:
        c = counts.copy()
        # normalize columns
        if "Year" in c.columns and "year" not in c.columns:
            c = c.rename(columns={"Year": "year"})
        if "Season" in c.columns and "season" not in c.columns:
            c = c.rename(columns={"Season": "season"})
        if "n_tows" not in c.columns:
            for alt in ("n", "count", "n_rows", "N"):
                if alt in c.columns:
                    c = c.rename(columns={alt: "n_tows"})
                    break

        if "season" in c.columns:
            c["season"] = c["season"].astype(str).str.capitalize()
            c = c[c["season"] == season.capitalize()]

        if {"year", "n_tows"}.issubset(c.columns):
            counts_map = dict(zip(c["year"].astype(int), c["n_tows"]))
        else:
            print("[WARN] Counts table missing required columns; skipping annotations.")

    ncols = 2
    nrows = int(np.ceil(len(plot_groups) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharey=True,
        figsize=(4 * ncols, 3 * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    annual_idx = annual.set_index(["group", "year"])
    x = np.arange(len(years))
    pair_step = 1.18  # spacing between years
    w = 0.3  # bar width
    x = np.arange(len(years)) * pair_step

    for i, grp in enumerate(plot_groups):
        ax = axes[i]
        y_obs = [
            annual_idx.loc[(grp, yr), "obs_anom"] if (grp, yr) in annual_idx.index else np.nan
            for yr in years
        ]
        y_mod = [
            annual_idx.loc[(grp, yr), "model_anom"] if (grp, yr) in annual_idx.index else np.nan
            for yr in years
        ]

        ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.5, zorder=1)

        if plot_type.lower() == "bar":
            ax.bar(x - w / 2, y_obs, w, label="Observations", color="darkorange",  zorder=2)
            ax.bar(x + w / 2, y_mod, w, label="SOGEM-LTL", color="blue", zorder=2)
        else:
            ax.plot(years, y_obs, "-o", label="Observations", color="darkorange", zorder=2)
            ax.plot(years, y_mod, "-o", label="SOGEM-LTL", olor="blue", zorder=2)

        # Optional NPGO overlay (same axis, like the current ecospace plot)
        if npgo_ann is not None:
            x_npgo = [yr for yr in years if yr in npgo_ann.index]
            y_npgo = [npgo_ann.loc[yr] for yr in x_npgo]
            if len(x_npgo) > 0:
                ax.plot(
                    [years.index(yr) for yr in x_npgo],
                    y_npgo,
                    "k-",
                    label=f"NPGO (lag {cfg5.LAG_YEARS}y)",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.set_title(grp)

        if i % ncols == 0:
            ax.set_ylabel("Anomaly (z-score)")

        # if i == 0:
        #     ax.legend()

        # Count annotations
        if counts_map:
            ax.margins(y=0.15)
            ymin, ymax = ax.get_ylim()
            yrange = ymax - ymin if ymax > ymin else 1.0
            for j, yr in enumerate(years):
                n = counts_map.get(int(yr))
                if n is None:
                    continue
                vals = []
                if j < len(y_obs):
                    vals.append(y_obs[j])
                if j < len(y_mod):
                    vals.append(y_mod[j])
                if len(vals) == 0 or all(pd.isna(v) for v in vals):
                    continue
                y_top = np.nanmax(vals)
                y_text = y_top + yrange * count_yoffset
                ax.text(
                    j,
                    y_text,
                    f"{int(n)}",
                    ha="center",
                    va="bottom",
                    fontsize=count_fontsize,
                )

    # turn off unused axes
    for j in range(len(plot_groups), len(axes)):
        axes[j].axis("off")

    # ---- Figure title (ONE suptitle only; informative) ----
    log_state = "anom z-score"
    yr0, yr1 = int(years[0]), int(years[-1])

    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Obs vs Model annual anomalies",
                years=(yr0, yr1),
                season=season,
                label=label,
                plot_type=plot_type,
                log_state=log_state,
            ),
            y=0.995,  # keep inside canvas
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # reserve space for title
    else:
        fig.tight_layout()

    # ---- Save (save before show; tight bbox includes suptitle) ----
    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    out_panel = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_anomalies_panel_{season}_{label}.png",
    )

    fig.savefig(
        out_panel,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )
    _maybe_show(fig, cfg5.SHOW_ANOM_PAIRED_PANEL or cfg5.SHOW_PLTS)

    print(f"[INFO] Saved anomaly panel: {out_panel}")
    return out_panel


def plot_scatter_matched_tows(
    paired_long: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    groups: List[str],
    label: str,
    season: str,  # <-- ADD THIS (so it can appear in the suptitle)
    outname: str = "scatter_tows",
    log10: bool = True,
) -> str:
    """
    Scatter of tow-level matched values (obs vs model), one subplot per group.
    Mirrors ecosim_eval_5_zoop.py behaviour.
    """
    d = paired_long.copy()
    d = d.dropna(subset=["group", "obs_biomass", "model_biomass"])
    d = d[d["group"].isin(groups)]

    if d.empty:
        raise ValueError("No paired tow-level values available for scatter plot.")

    if log10:
        eps = 1e-12
        d = d[(d["obs_biomass"] > 0) & (d["model_biomass"] > 0)]
        d["obs_x"] = np.log10(d["obs_biomass"] + eps)
        d["mod_y"] = np.log10(d["model_biomass"] + eps)
        xlab = "Obs (log10 g C m$^{-2}$)"
        ylab = "Model (log10 g C m$^{-2}$)"
    else:
        d["obs_x"] = d["obs_biomass"]
        d["mod_y"] = d["model_biomass"]
        xlab = "Obs (g C m$^{-2}$)"
        ylab = "Model (g C m$^{-2}$)"

    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_colors = {
        "Winter": "tab:blue",
        "Spring": "tab:green",
        "Summer": "tab:orange",
        "Fall": "tab:red",
    }
    season_short = {
        "Winter": "Wi",
        "Spring": "Sp",
        "Summer": "Su",
        "Fall": "Fa",
    }

    for i, grp in enumerate(groups):
        ax = axes[i]
        g = d[d["group"] == grp].copy()

        if g.empty:
            ax.set_title(grp)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.5)
            continue

        # axis limits from all seasons pooled within this group
        lo = min(g["obs_x"].min(), g["mod_y"].min())
        hi = max(g["obs_x"].max(), g["mod_y"].max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)

        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="#808080", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        stat_lines = []

        for season in season_order:
            gs = g[g["season"] == season].copy()
            color = season_colors[season]

            if gs.empty:
                stat_lines.append((f"{season_short[season]}: n=0", color))
                continue

            ax.scatter(
                gs["obs_x"], gs["mod_y"],
                s=6, alpha=0.2,
                color=color,
                label=season if i == 0 else None,
            )

            if len(gs) >= 2:
                _add_best_fit_line(
                    ax,
                    gs["obs_x"].values,
                    gs["mod_y"].values,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    show_eq=False,
                )

            stats = compute_stats(gs, "obs_x", "mod_y", log_or_anom=log10)
            if stats["N"] >= 2 and np.isfinite(stats["r"]):
                stat_lines.append((f"{season_short[season]}: n={int(stats['N'])}, r={stats['r']:.2f}", color))
            else:
                stat_lines.append((f"{season_short[season]}: n={int(stats['N'])}", color))


        ax.set_title(grp)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

        # overlay stats box
        _add_colored_statbox(ax, stat_lines)

    #
    # for i, grp in enumerate(groups):
    #     ax = axes[i]
    #     g = d[d["group"] == grp]
    #     ax.scatter(g["obs_x"], g["mod_y"], s=12, alpha=0.6)
    #
    #     if len(g) > 0:
    #         lo = min(g["obs_x"].min(), g["mod_y"].min())
    #         hi = max(g["obs_x"].max(), g["mod_y"].max())
    #         ax.plot([lo, hi], [lo, hi], "-")
    #
    #         _add_best_fit_line(
    #             ax,
    #             g["obs_x"].values,
    #             g["mod_y"].values,
    #             color="k",
    #             linestyle="--",
    #             linewidth=1,
    #             show_eq=False,
    #         )
    #
    #         if len(g) > 1:
    #             r = np.corrcoef(g["obs_x"], g["mod_y"])[0, 1]
    #             ax.set_title(f"{grp}  r={r:.2f}")
    #         else:
    #             ax.set_title(grp)
    #
    #     ax.set_xlabel(xlab)
    #     ax.set_ylabel(ylab)
    #     ax.grid(True, alpha=0.3)

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    # ---- Figure suptitle (informative + consistent) ----
    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        log_state = "log10" if log10 else "raw"

        if len(season) > 1:
            season_title = "All"
        else:
            season_title = season

        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Tow-level scatter (Obs vs Model)",
                season=season_title,           # <-- ADD THIS
                plot_type="scatter",
                log_state=log_state,
                suffix=label,            # keep label as suffix for consistency
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    # ---- Save ----
    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}.png",
    )
    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )
    _maybe_show(fig, cfg5.SHOW_SCATTER_TOWS or cfg5.SHOW_PLTS)
    print(f"[INFO] Saved scatter plot: {outpath}")
    return outpath



def plot_scatter_anomalies(
    annual: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    groups: List[str],
    label: str,
    season: str,  # <-- ADD THIS
    outname: str = "scatter_anoms",
) -> str:
    """
    Scatter of annual anomalies (obs vs model), one subplot per group.
    Mirrors ecosim_eval_5_zoop.py behaviour.
    """
    d = annual.dropna(subset=["group", "obs_anom", "model_anom"]).copy()
    d = d[d["group"].isin(groups)]

    if d.empty:
        raise ValueError("No annual anomalies available for anomaly scatter plot.")

    ncols = 3
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, grp in enumerate(groups):
        ax = axes[i]
        g = d[d["group"] == grp]
        ax.scatter(g["obs_anom"], g["model_anom"], s=25, alpha=0.7)

        if len(g) > 0:
            lo = min(g["obs_anom"].min(), g["model_anom"].min())
            hi = max(g["obs_anom"].max(), g["model_anom"].max())
            ax.plot([lo, hi], [lo, hi], "-")
            _add_best_fit_line(
                ax,
                g["obs_anom"].values,
                g["model_anom"].values,
                color="k",
                linestyle="--",
                linewidth=1,
                show_eq=False,
            )

            if len(g) > 1:
                r = np.corrcoef(g["obs_anom"], g["model_anom"])[0, 1]
                ax.set_title(f"{grp}  r={r:.2f}")
            else:
                ax.set_title(grp)

        ax.set_xlabel("Obs anomaly (z-score)")
        ax.set_ylabel("Model anomaly (z-score)")
        ax.grid(True, alpha=0.3)

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    # ---- Figure suptitle (informative + consistent) ----
    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        # prefer actual range in the data if present
        if "year" in d.columns and d["year"].notna().any():
            yr0 = int(d["year"].min())
            yr1 = int(d["year"].max())
        elif "Year" in d.columns and d["Year"].notna().any():
            yr0 = int(d["Year"].min())
            yr1 = int(d["Year"].max())
        else:
            # fall back to config
            yr0 = getattr(cfg5, "START_YEAR", None)
            yr1 = getattr(cfg5, "END_YEAR", None)

        years = (yr0, yr1) if (yr0 is not None and yr1 is not None) else None

        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Anomaly scatter (Obs vs Model)",
                years=years,
                season=season,          # <-- ADD THIS
                plot_type="scatter",
                log_state="anom z-score",
                suffix=label,
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    # ---- Save (save before show; include suptitle) ----
    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )
    _maybe_show(fig, cfg5.SHOW_SCATTER_ANOMS or cfg5.SHOW_PLTS)
    print(f"[INFO] Saved anomaly scatter plot: {outpath}")
    return outpath

def plot_scatter_anomalies_total_by_season(
    paired: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    seasons: Optional[list[str]] = None,
    label: str = "zoop_ZC_total",
    outname: str = "scatter_anoms_total4panel",
    title_text: str = "Total crustaceans",
) -> str:
    """
    2x2 panel of annual anomaly scatter plots (Obs vs Model),
    one subplot per season, for the aggregate 'Total' group.
    """

    seasons = seasons or cfg5.SEASON_ORDER
    season_annual = {}

    # build annual anomaly table for each season
    for season in seasons:
        ann = compute_anomalies_for_eval(
            paired,
            cfg5=cfg5,
            season=season,
        )
        ann = ann[ann["group"] == "Total"].copy()
        season_annual[season] = ann

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for i, season in enumerate(seasons[:4]):
        ax = axes[i]
        d = season_annual[season].dropna(subset=["obs_anom", "model_anom"]).copy()

        ax.scatter(d["obs_anom"], d["model_anom"], s=22, alpha=0.7)

        if len(d) > 0:
            lo = min(d["obs_anom"].min(), d["model_anom"].min())
            hi = max(d["obs_anom"].max(), d["model_anom"].max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)

            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="k", lw=1)
            _add_best_fit_line(
                ax,
                d["obs_anom"].values,
                d["model_anom"].values,
                color="k",
                linestyle="--",
                linewidth=1,
                show_eq=False,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

            if len(d) > 1:
                r = np.corrcoef(d["obs_anom"], d["model_anom"])[0, 1]
                ax.set_title(f"{season}  r={r:.2f}  n={len(d)}")
            else:
                ax.set_title(f"{season}  n={len(d)}")
        else:
            ax.set_title(f"{season}  n=0")

        ax.set_xlabel("Obs anomaly (z-score)")
        ax.set_ylabel("Model anomaly (z-score)")
        ax.grid(True, alpha=0.3)

    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Anomaly scatter (Obs vs Model)",
                years=(cfg5.START_YEAR, cfg5.END_YEAR),
                plot_type="scatter",
                log_state="anom z-score",
                suffix=title_text,
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )
    _maybe_show(fig, cfg5.SHOW_SCATTER_ANOMS_TOTAL4 or cfg5.SHOW_PLTS)

    print(f"[INFO] Saved total anomaly scatter panel: {outpath}")
    return outpath

def plot_anomaly_bars_total_by_season(
    paired: pd.DataFrame,
    *,
    cfg5: Eval5Config,
    seasons: Optional[list[str]] = None,
    label: str = "zoop_ZC_total",
    outname: str = "anomaly_bars_total4panel",
    title_text: str = "Total crustaceans",
) -> str:
    """
    2x2 panel of annual anomaly bar plots (Obs vs Model),
    one subplot per season, for the aggregate 'Total' group.
    """

    seasons = seasons or cfg5.SEASON_ORDER
    season_data = {}
    season_counts = {}

    for season in seasons:
        annual = compute_anomalies_for_eval(
            paired,
            cfg5=cfg5,
            season=season,
        )
        annual = annual[annual["group"] == "Total"].copy()
        season_data[season] = annual

        counts = counts_by_season_year_from_paired(
            paired,
            season=season,
            group_for_counts="Total",
            unique_index=True,
            valid_pairs_only=True,
        )
        season_counts[season] = counts

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    pair_step = 1.18  # spacing between years
    w = 0.28  # bar width


    # choose a common y-range across seasons
    vals = []
    for season in seasons:
        d = season_data[season]
        vals.extend(d["obs_anom"].dropna().tolist())
        vals.extend(d["model_anom"].dropna().tolist())
    if vals:
        ymin = min(vals)
        ymax = max(vals)
        pad = 0.12 * (ymax - ymin if ymax > ymin else 1.0)
        ylims = (ymin - pad, ymax + pad)
    else:
        ylims = (-2, 2)

    for i, season in enumerate(seasons[:4]):
        ax = axes[i]
        d = season_data[season]
        years = sorted(d["year"].unique().tolist())

        x = np.arange(len(years)) * pair_step

        if not years:
            ax.set_title(f"{season}  n=0")
            ax.axhline(0, lw=1, linestyle="--", alpha=0.6)
            ax.set_ylim(*ylims)
            continue

        d_idx = d.set_index("year")
        x = np.arange(len(years))
        y_obs = [d_idx.loc[yr, "obs_anom"] if yr in d_idx.index else np.nan for yr in years]
        y_mod = [d_idx.loc[yr, "model_anom"] if yr in d_idx.index else np.nan for yr in years]

        ax.axhline(0, lw=1, linestyle="--", alpha=0.6, zorder=1)
        ax.bar(x - w/2, y_obs, width=w, label="Observations", zorder=2)
        ax.bar(x + w/2, y_mod, width=w, label="SOGEM-LTL", zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax.set_title(season)
        ax.set_ylim(*ylims)
        ax.grid(True, axis="both", alpha=0.3, zorder=0)

        if i % 2 == 0:
            ax.set_ylabel("Anomaly (z-score)")

        if i == 0:
            ax.legend()

        # annotate tow counts
        counts = season_counts[season]
        if cfg5.SHOW_COUNTS and counts is not None and not counts.empty:
            cmap = dict(zip(counts["year"].astype(int), counts["n_tows"]))
            yrange = ylims[1] - ylims[0]
            for j, yr in enumerate(years):
                n = cmap.get(int(yr))
                if n is None:
                    continue
                y_top = np.nanmax([y_obs[j], y_mod[j]])
                ax.text(
                    j,
                    y_top + 0.04 * yrange,
                    f"{int(n)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    st = None
    if getattr(cfg5, "ADD_FIG_SUPTITLE", True):
        st = fig.suptitle(
            _fig_title(
                cfg5,
                kind="Annual anomalies (Obs vs Model)",
                years=(cfg5.START_YEAR, cfg5.END_YEAR),
                plot_type="bar",
                log_state="anom z-score",
                suffix=title_text,
            ),
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
    outpath = os.path.join(
        cfg5.FIGSOUT_P,
        f"ecospace_{cfg5.ecospace_code}_zoop_{outname}_{label}.png",
    )

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        bbox_extra_artists=([st] if st is not None else None),
    )
    _maybe_show(fig, cfg5.SHOW_ANOM_BARS_TOTAL4 or cfg5.SHOW_PLTS)

    print(f"[INFO] Saved total anomaly 4-panel bar plot: {outpath}")
    return outpath

# =============================================================================
# TABLE EXPORT HELPERS
# =============================================================================

SCENARIO = getattr(cfg, "ECOSPACE_SC", "run")
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall", "All"]
TAB_P = getattr(cfg, "EVALOUT_P", ".")

def _parse_filename(path: Path) -> tuple[str, str, str]:
    """
    Expect names like:
        ecospace_zoop_performance_SC215_ZC_Spring.csv
        ecospace_zoop_performance_SC215_ZS_Winter.csv

    Returns
    -------
    scenario, pass_name, season
    """
    m = re.match(
        r"ecospace_zoop_performance_(?P<scenario>.+?)_(?P<pass>ZC|ZS)_(?P<season>All|Winter|Spring|Summer|Fall)\.csv$",
        path.name,
    )
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    return m.group("scenario"), m.group("pass"), m.group("season")

def _season_sort(df: pd.DataFrame, season_col: str = "season") -> pd.DataFrame:
    order_map = {s: i for i, s in enumerate(SEASON_ORDER)}
    return (
        df.assign(_season_order=df[season_col].map(order_map).fillna(999))
          .sort_values(["_season_order"] + [c for c in df.columns if c not in {season_col, "_season_order"}])
          .drop(columns="_season_order")
          .reset_index(drop=True)
    )

def _round_numeric(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    out = df.copy()
    if decimals is None:
        return out
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(decimals)
    return out

def _save_csv(df: pd.DataFrame, out_csv: str | Path, decimals: int | None) -> None:
    out_csv = Path(out_csv)
    if decimals is None:
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False, float_format=f"%.{decimals}f")

def build_manuscript_skill_table(
    scenario: str,
    in_dir: str | Path,
    *,
    out_csv: str | Path | None = None,
    include_all: bool = True,
    decimals: int | None = 2,
    total_name_in_source: str = "Total",
) -> pd.DataFrame:
    """
    Build manuscript table:
        season, N, r, RMSE, MAE, WSS

    Uses the TOTAL row from each seasonal ZC file only.
    """
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob(f"ecospace_zoop_performance_{scenario}_ZC_*.csv"))

    rows = []
    for f in files:
        _, _, season = _parse_filename(f)
        if (not include_all) and season == "All":
            continue

        df = pd.read_csv(f)
        sub = df[df["Group"].astype(str) == total_name_in_source].copy()
        if sub.empty:
            continue

        sub.insert(0, "season", season)
        sub = sub[["season", "N", "r", "RMSE", "MAE", "WSS"]]
        rows.append(sub)

    if not rows:
        raise ValueError(
            f"No matching ZC seasonal files with Group == '{total_name_in_source}' found for scenario {scenario}."
        )

    out = pd.concat(rows, ignore_index=True)
    out = _season_sort(out, "season")
    out = _round_numeric(out, decimals)

    if out_csv is None:
        out_csv = in_dir / f"ecospace_forpub_zoop_skill_{scenario}_ZC.csv"

    _save_csv(out, out_csv, decimals)
    print(f"Saved manuscript table: {out_csv}")
    return out

def build_supplement_skill_table(
    scenario: str,
    in_dir: str | Path,
    *,
    out_csv: str | Path | None = None,
    include_all: bool = True,
    decimals: int | None = 2,
    passes: tuple[str, ...] = ("ZC", "ZS"),
    total_name_in_source: str = "Total",
    total_name_fmt: str = "{pass_name}-TOTAL",
) -> pd.DataFrame:
    """
    Build supplement table:
        season, group, N, r, RMSE, MAE, WSS

    Stacks ZC and ZS files and keeps individual groups.
    Renames source 'Total' rows to e.g. ZC-TOTAL / ZS-TOTAL.
    """
    in_dir = Path(in_dir)

    rows = []
    for pass_name in passes:
        files = sorted(in_dir.glob(f"ecospace_zoop_performance_{scenario}_{pass_name}_*.csv"))
        for f in files:
            _, _, season = _parse_filename(f)
            if (not include_all) and season == "All":
                continue

            df = pd.read_csv(f).copy()
            df.insert(0, "season", season)
            df["group"] = df["Group"].astype(str)
            df.loc[df["group"] == total_name_in_source, "group"] = total_name_fmt.format(pass_name=pass_name)
            df = df[["season", "group", "N", "r", "RMSE", "MAE", "WSS"]]
            rows.append(df)

    if not rows:
        raise ValueError(f"No matching ZC/ZS seasonal files found for scenario {scenario}.")

    out = pd.concat(rows, ignore_index=True)

    # Keep seasons in the desired order; within season, put totals first, then alphabetical
    season_order = {s: i for i, s in enumerate(SEASON_ORDER)}
    out["_season_order"] = out["season"].map(season_order).fillna(999)
    out["_group_order"] = out["group"].str.contains("TOTAL", na=False).map({True: 0, False: 1})
    out = (
        out.sort_values(["_season_order", "_group_order", "group"])
           .drop(columns=["_season_order", "_group_order"])
           .reset_index(drop=True)
    )

    out = _round_numeric(out, decimals)

    if out_csv is None:
        out_csv = in_dir / f"ecospace_forpub_zoop_skill_{scenario}_ALLGROUPS.csv"

    _save_csv(out, out_csv, decimals)
    print(f"Saved supplement table: {out_csv}")
    return out

def export_forpub_zoop_skill_tables(
    scenario: str,
    in_dir: str | Path,
    *,
    include_all: bool = True,
    decimals: int | None = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
      1) manuscript table from ZC total rows
      2) supplement table from ZC + ZS seasonal files

    Defaults now include:
      - include_all=True
      - decimals=2
    """
    ms = build_manuscript_skill_table(
        scenario=scenario,
        in_dir=in_dir,
        include_all=include_all,
        decimals=decimals,
    )
    supp = build_supplement_skill_table(
        scenario=scenario,
        in_dir=in_dir,
        include_all=include_all,
        decimals=decimals,
    )
    return ms, supp

# =============================================================================
# WORKFLOW DRIVERS
# =============================================================================

def anomaly_comparisons(
    cfg5: Eval5Config,
    groups: Optional[List[str]] = None,
    *,
    include_total: bool = True,
    seasons: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Create ecosim-style model-vs-obs anomaly panels for Ecospace matched outputs.

    To mirror the Ecosim workflow, this runs *two passes* when possible:
      1) Crustaceans (ZC*), with Total = sum(ZC*)
      2) Soft-bodied (ZS2–ZS4), with Total = sum(ZS2–ZS4)

    """
    print("[5d] Building anomaly comparisons (Model vs Obs)…")

    # --- load matched table
    matched_csv = os.path.join(cfg5.EVALOUT_P, cfg5.ZOOP_CSV_MATCH)
    if not os.path.exists(matched_csv):
        raise FileNotFoundError(
            f"Matched CSV not found: {matched_csv}. Run 5b first or set RECOMPUTE_MATCH=True."
        )

    df_match = pd.read_csv(matched_csv)
    df_match = _exclude_stations(
        df_match,
        getattr(cfg5, "ANOM_EXCLUDE_STATIONS", []),
    )

    # Determine which groups are actually present in the matched file
    candidate = groups or cfg5.ZOOP_GROUPS
    groups_present = [g for g in candidate if g in df_match.columns and f"EWE-{g}" in df_match.columns]
    if not groups_present:
        raise ValueError(
            "No zoop group columns found in matched CSV. "
            "Expected columns like 'ZC1-EUP' and 'EWE-ZC1-EUP'."
        )

    # Split to match Ecosim passes (same definitions as ecosim_eval_5_zoop.py)
    crust_groups = [g for g in groups_present if str(g).startswith("ZC")]
    zs_groups = [g for g in groups_present if str(g).startswith(("ZS2", "ZS3", "ZS4"))]

    passes: list[tuple[str, list[str]]] = []
    if crust_groups:
        passes.append(("ZC", crust_groups))
    if zs_groups:
        passes.append(("ZS", zs_groups))

    # Optional: warn if you have groups present that won't be included in either pass (e.g., ZF1-ICT)
    extra_groups = [g for g in groups_present if g not in set(crust_groups + zs_groups)]
    if extra_groups:
        print(f"[WARN] Groups present but excluded from anomaly passes to match Ecosim definitions: {extra_groups}")

    # --- optional external NPGO series (spring mean only, lagged)
    npgo_ann = None
    if cfg5.NPGO_CSV and os.path.exists(cfg5.NPGO_CSV):
        try:
            npgo = pd.read_csv(cfg5.NPGO_CSV)
            npgo["date"] = pd.to_datetime(npgo["date"])
            npgo["year"] = npgo["date"].dt.year
            npgo["month"] = npgo["date"].dt.month
            spring = npgo[npgo["month"].isin([3, 4, 5])]
            npgo_ann = (
                spring.groupby("year")["npgo"]
                .mean()
                .shift(cfg5.LAG_YEARS)
                .dropna()
            )
        except Exception as e:
            print(f"[WARN] Failed to load NPGO series from {cfg5.NPGO_CSV}: {e}")
            npgo_ann = None

    # --- seasons to run
    if seasons is not None:
        seasons_to_run = seasons
    else:
        seasons_to_run = cfg5.SEASON_ORDER if cfg5.ANOM_ALL_SEASONS else [cfg5.ANOM_SEASON_CHOICE]

    out_panel_last: Optional[str] = None
    out_total_last: Optional[str] = None

    for pass_label, pass_groups in passes:

        # build paired long table (tow-level) for this pass (Total is computed from pass_groups)
        paired = build_paired_long_ecospace(df_match, groups=pass_groups)
        paired = paired.query("@cfg5.START_YEAR <= year <= @cfg5.END_YEAR").copy()

        plot_groups = list(pass_groups) + (["Total"] if include_total else [])
        if "Total" in paired["group"].unique() and "Total" not in plot_groups and include_total:
            plot_groups.append("Total")

        seasons_plus_all = ["All"] + list(seasons_to_run)

        if cfg5.MAKE_SCATTER and pass_label.lower().startswith("zc"):
            plot_scatter_total_by_season(
                paired,
                cfg5=cfg5,
                total_group="Total",  # in crustacean pass, Total == sum(ZC*)
                seasons=cfg5.SEASON_ORDER,
                label=f"zoop_{pass_label}",
                outname="scatter_tows_total4panel",
                log10=cfg5.SCATTER_LOG10,
                title_text="Total crustaceans",
            )

        if cfg5.MAKE_SCATTER and pass_label == "ZC":
            plot_scatter_anomalies_total_by_season(
                paired,
                cfg5=cfg5,
                seasons=cfg5.SEASON_ORDER,
                label=f"zoop_{pass_label}",
                outname=f"scatter_anoms_total4panel_{pass_label}",
                title_text="Total crustaceans",
            )

        if include_total and pass_label == "ZC":
            plot_anomaly_bars_total_by_season(
                paired,
                cfg5=cfg5,
                seasons=cfg5.SEASON_ORDER,
                label=f"zoop_{pass_label}",
                outname=f"anomaly_bars_total4panel_{pass_label}",
                title_text="Total crustaceans",
            )

        if pass_label == "ZC" and cfg5.PUB_TOTAL_PANEL:
            plot_pub_panel_total_single_season(
                paired,
                cfg5=cfg5,
                season=cfg5.ANOM_SEASON_CHOICE,
                total_group="Total",
                label=f"zoop_{pass_label}",
                outname="pub_total_single_season",
                title_text="Total crustaceans",
                log10_scatter=cfg5.SCATTER_LOG10,
            )

        for season in seasons_plus_all:
            season_filter = None if str(season).lower() == "all" else season
            # Annual anomalies
            annual = compute_anomalies_for_eval(
                paired,
                cfg5=cfg5,
                season=season_filter,
            )

            # Counts for labels (use Total by default)
            counts = counts_by_season_year_from_paired(
                paired,
                season=season_filter,
                group_for_counts="Total" if "Total" in paired["group"].unique() else None,
                unique_index=True,
                valid_pairs_only=True,
            )

            # Panel plot (Obs vs Model together)
            out_panel_last = plot_anomaly_panel_paired(
                annual,
                cfg5=cfg5,
                plot_groups=plot_groups,
                label=f"zoop_{pass_label}",
                season=season,
                plot_type=cfg5.ANOM_PLOT_TYPE,
                counts=counts,
                show_counts=cfg5.SHOW_COUNTS,
                npgo_ann=npgo_ann,
            )

            # Optional standalone Total plot (matches old behavior)
            if include_total and "Total" in plot_groups:
                years = sorted(annual["year"].unique().tolist())
                tot = annual[annual["group"] == "Total"].set_index("year").reindex(years)

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.axhline(0, lw=1, linestyle="--", alpha=0.6)
                ax.bar(np.arange(len(years)) - 0.18, tot["obs_anom"].values, width=0.36, label="Observed")
                ax.bar(np.arange(len(years)) + 0.18, tot["model_anom"].values, width=0.36, label="Model")
                ax.set_xticks(np.arange(len(years)))
                ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
                ax.set_title(f"Ecospace vs Obs annual anomalies — Total ({pass_label}) — {season}")
                ax.set_ylabel("Standardized anomaly")

                # annotate counts once (for Total)
                if cfg5.SHOW_COUNTS and not counts.empty:
                    cmap = dict(zip(counts["year"].astype(int), counts["n_tows"]))
                    ax.margins(y=0.15)
                    ymin, ymax = ax.get_ylim()
                    yrange = ymax - ymin if ymax > ymin else 1.0
                    for j, yr in enumerate(years):
                        n = cmap.get(int(yr))
                        if n is None:
                            continue
                        y_top = np.nanmax([tot["obs_anom"].values[j], tot["model_anom"].values[j]])
                        ax.text(j, y_top + yrange * 0.05, f"{int(n)}", ha="center", va="bottom", fontsize=9)

                ax.legend()
                fig.tight_layout()
                os.makedirs(cfg5.FIGSOUT_P, exist_ok=True)
                out_total_last = os.path.join(
                    cfg5.FIGSOUT_P,
                    f"ecospace_{cfg5.ecospace_code}_zoop_anomalies_total_{pass_label}_{season}.png",
                )

                fig.savefig(out_total_last, dpi=300)
                _maybe_show(fig, cfg5.SHOW_MODELONLY_ANOM_BARS or cfg5.SHOW_PLTS)


                print(f"[INFO] Saved total anomaly plot: {out_total_last}")

            # Skill metric summary (print + CSV), like Ecosim
            try:
                stats_df = skill_metrics_by_group(
                    annual,
                    groups=plot_groups,
                    log_or_anom=True,  # anomalies are dimensionless
                ).sort_values("WSS", ascending=False)
            except Exception as e:
                print(f"[WARN] Skill metrics failed for {pass_label}/{season}: {e}")
                stats_df = pd.DataFrame()

            if not stats_df.empty:
                print(f"\n[{pass_label}] {season} skill metrics (anomalies):")
                print(stats_df.to_string(index=False))

                out_stats = os.path.join(
                    cfg5.EVALOUT_P,
                    f"ecospace_zoop_performance_{cfg5.ecospace_code}_{pass_label}_{season}.csv",
                )
                stats_df.to_csv(out_stats, index=False)
                print(f"[INFO] Saved skill metrics: {out_stats}")

            # Scatter plots (tow-level + anomaly scatter)
            if cfg5.MAKE_SCATTER:
                paired_seas = paired.copy() if season_filter is None else paired.query("season == @season").copy()
                plot_scatter_matched_tows(
                    paired_seas,
                    cfg5=cfg5,
                    groups=plot_groups,
                    label=f"zoop_{pass_label}",
                    outname=f"scatter_tows_{pass_label}_{season}",
                    log10=cfg5.SCATTER_LOG10,
                    season=season,
                )
                plot_scatter_anomalies(
                    annual,
                    cfg5=cfg5,
                    groups=plot_groups,
                    label=f"zoop_{pass_label}",
                    outname=f"scatter_anoms_{pass_label}_{season}",
                    season=season,
                )


    return out_panel_last or "", out_total_last

def run_model_only_box_anomaly(cfg5: Optional[Eval5Config] = None) -> tuple[str, str, str]:
    """
    Convenience runner for model-only box anomaly workflow.

    Returns
    -------
    model_series_csv, anomaly_csv, figure_path
    """
    t0 = time.perf_counter()
    cfg5 = cfg5 or Eval5Config()

    verbose = bool(getattr(cfg, "ZP_MODELONLY_VERBOSE", True))
    plot_group = getattr(cfg, "ZP_MODELONLY_PLOT_GROUP", "Total")
    season = getattr(cfg, "ZP_MODELONLY_SEASON", None)

    print("[5m] ==================================================")
    print("[5m] Starting model-only anomaly time series workflow")
    print(f"[5m] Ecospace code: {cfg5.ecospace_code}")
    print(f"[5m] Plot group:    {plot_group}")
    print(f"[5m] Season:        {season if season is not None else 'All seasons'}")
    print(f"[5m] Verbose:       {verbose}")

    annual_anom, model_ts_named, cells = compute_model_only_box_anomalies(cfg5)

    series_csv = os.path.join(
        cfg5.EVALOUT_P,
        getattr(cfg, "ZP_MODELONLY_SERIES_CSV", f"ecospace_zoop_model_box_series_{cfg5.ecospace_code}.csv")
    )
    anom_csv = os.path.join(
        cfg5.EVALOUT_P,
        getattr(cfg, "ZP_MODELONLY_ANOM_CSV", f"ecospace_zoop_model_box_anom_{cfg5.ecospace_code}.csv")
    )
    cells_csv = os.path.join(
        cfg5.EVALOUT_P,
        f"ecospace_zoop_model_box_cells_{cfg5.ecospace_code}.csv"
    )

    if verbose:
        print("[5m] Writing outputs ...")

    model_ts_named.to_csv(series_csv, index=False)
    annual_anom.to_csv(anom_csv, index=False)
    cells.to_csv(cells_csv, index=False)

    if verbose:
        print(f"[5m]   model series rows x cols: {model_ts_named.shape}")
        print(f"[5m]   annual anomaly rows x cols: {annual_anom.shape}")
        print(f"[5m]   selected cells rows x cols: {cells.shape}")

    plot_groups = list(model_ts_named.columns)
    plot_groups = [c for c in plot_groups if c not in ("date", "year", "month", "season")]

    fig_paths = []
    for plot_group in plot_groups:
        fig_path = plot_model_only_anomaly_bars(
            annual_anom,
            cfg5=cfg5,
            group=plot_group,
            season=getattr(cfg, "ZP_MODELONLY_SEASON", None),
            label="zoop_model_box",
            outname="model_box_anomaly_bars",
            title_text=getattr(cfg, "ZP_MODELONLY_LABELS", {}).get(plot_group, plot_group),
        )
        fig_paths.append(fig_path)

    print(f"[5m] Saved {len(fig_paths)} model-only figure(s).")

    print(f"[5m] Saved model-only box time series: {series_csv}")
    print(f"[5m] Saved model-only annual anomalies: {anom_csv}")
    print(f"[5m] Saved selected box cells: {cells_csv}")
    print(f"[5m] Saved model-only figure: {fig_path}")
    print(f"[5m] Workflow complete in {_fmt_elapsed(t0)}")

    panel_path = plot_model_only_anomaly_panel(
        annual_anom,
        cfg5=cfg5,
        top_group="Total",
        bottom_group="ZC2-AMP",
        season=getattr(cfg, "ZP_MODELONLY_SEASON", None),
        label="zoop_model_box",
        outname="model_box_anomaly_panel",
    )

    print(f"[5m] Saved {len(fig_paths)} model-only figure(s).")
    print(f"[5m] Saved model-only panel figure: {panel_path}")
    print("[5m] ==================================================")

    return series_csv, anom_csv, fig_path

def run_zoop_eval(
    recompute_match: Optional[bool] = None,
    make_viz: Optional[bool] = None,
    make_anom: Optional[bool] = None,
    make_modelonly_box: Optional[bool] = None,
    groups: Optional[List[str]] = None,
) -> None:

    cfg5 = Eval5Config()

    if recompute_match is not None:
        cfg5.RECOMPUTE_MATCH = recompute_match
    if make_viz is not None:
        cfg5.MAKE_VIZ = make_viz
    if make_anom is not None:
        cfg5.MAKE_ANOM = make_anom
    if make_modelonly_box is not None:
        cfg5.MODELONLY_DO = make_modelonly_box

    # ---------------------------------------------------------
    #  obs-vs-model workflow
    # ---------------------------------------------------------
    need_obs_model_path = bool(cfg5.MAKE_VIZ or cfg5.MAKE_ANOM or cfg5.PUB_TOTAL_PANEL)

    if need_obs_model_path:
        matched = match_zoop_to_ecospace(cfg5)

        if cfg5.MAKE_VIZ:
            visualize_and_stats(matched, cfg5)

        if cfg5.MAKE_ANOM:
            if cfg5.COMPARE_ANOM_SUMMARIES:
                compare_anomaly_obs_summaries(cfg5, groups=groups)
            anomaly_comparisons(cfg5, groups=groups)

    # ---------------------------------------------------------
    # New model-only box workflow
    # ---------------------------------------------------------
    if cfg5.MODELONLY_DO:
        run_model_only_box_anomaly(cfg5)

    print("[5] DONE.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Ecospace zooplankton evaluation workflow."
    )
    parser.add_argument(
        "--recompute-match",
        action="store_true",
        help="Force recomputation of obs–model match",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip making obs-vs-model figures",
    )
    parser.add_argument(
        "--no-anom",
        action="store_true",
        help="Skip obs-vs-model anomaly calculations",
    )
    parser.add_argument(
        "--model-box",
        action="store_true",
        help="Run model-only box anomaly workflow",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Subset of zoop groups to include in obs-vs-model anomaly comparisons",
    )
    parser.add_argument(
        "--export-tables",
        action="store_true",
        help="Also export manuscript and supplement skill tables after the main run",
    )
    parser.add_argument(
        "--table-scenario",
        default=SCENARIO,
        help="Scenario code used in ecospace_zoop_performance_<scenario>_*.csv filenames",
    )
    parser.add_argument(
        "--table-in-dir",
        default=TAB_P,
        help="Directory containing ecospace_zoop_performance_<scenario>_*.csv files",
    )
    parser.add_argument(
        "--exclude-all",
        action="store_true",
        help="Exclude the All-season summary rows/files when exporting tables",
    )
    parser.add_argument(
        "--table-decimals",
        type=int,
        default=2,
        help="Number of decimal places for exported table values; use -1 for no rounding",
    )

    args = parser.parse_args()

    run_zoop_eval(
        recompute_match=True if args.recompute_match else None,
        make_viz=not args.no_viz,
        make_anom=not args.no_anom,
        make_modelonly_box=True if args.model_box else None,
        groups=args.groups,
    )

    if args.export_tables:
        decimals = None if args.table_decimals < 0 else args.table_decimals
        export_forpub_zoop_skill_tables(
            scenario=args.table_scenario,
            in_dir=Path(args.table_in_dir),
            include_all=not args.exclude_all,
            decimals=decimals,
        )
