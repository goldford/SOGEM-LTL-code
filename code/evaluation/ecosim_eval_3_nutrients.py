"""ecosim_eval_3_nutrients.py

Infer a surface-layer dissolved inorganic nitrogen inventory (g N m-2) for Ecosim
by closing an N budget with biomass pools.

Core idea
---------
- N_bound(t): derived from Ecosim biomasses (g C m-2) using C->N multipliers
- N_free(t):  max(0, N_total_init_used - N_bound(t))

Optional
--------
- apply a nutrient-supply multiplier (seasonal/monthly/3-day)
- overlay observational climatology (integrated over the evaluation layer)
- overlay Ecospace climatologies ("box" and/or "matched")

Created by: G. Oldford
Last edit: 2026-02-15
"""

from __future__ import annotations

import os
import importlib
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import ecosim_eval_config as cfg

import matplotlib
if not bool(getattr(cfg, "N_SHOW_PLOT", False)):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SCENARIO = cfg.SCENARIO
ECOSIM_F_PREPPED_SINGLERUN = cfg.ECOSIM_F_PREPPED_SINGLERUN
OUTPUT_DIR_EVAL = cfg.OUTPUT_DIR_EVAL
OUTPUT_DIR_FIGS = cfg.OUTPUT_DIR_FIGS

# Optional obs overlay
N_DATA_REF = getattr(cfg, "NUTRIENTS_F_PREPPED", None)
OBS_AVG_TYPE = getattr(cfg, "OBS_AVG_TYPE", "mean")  # mean|median

# Groups that "bind" N (numeric IDs; Ecosim output columns are strings)
N_BOUND_GROUPS = list(getattr(cfg, "N_BOUND_GROUPS", []))

# Initial inventory terms (g N m-2). These are added in ecosim_eval_config_units_v2.py.
N_FREE_INIT_GNM2 = float(getattr(cfg, "N_FREE_INIT_GNM2", np.nan))
N_BOUND_INIT_GNM2 = float(getattr(cfg, "N_BOUND_INIT_GNM2", np.nan))
N_TOTAL_INIT_GNM2 = float(getattr(cfg, "N_TOTAL_INIT_GNM2", np.nan))

N_BOUND_INIT_MODE = str(getattr(cfg, "N_BOUND_INIT_MODE", "ecopath")).lower()
N_FREE_INIT_MODE = str(getattr(cfg, "N_FREE_INIT_MODE", "config")).lower()

# Layer definition for obs integration
ZMIN = float(getattr(cfg, "N_EVAL_ZMIN_M", 0.1))
ZMAX = float(getattr(cfg, "N_EVAL_ZMAX_M", 20.0))
MMOL_TO_GN = float(getattr(cfg, "MMOL_TO_GN", 14.0 / 1000.0))

# C->N multipliers
N_C_TO_N_LIVING = float(getattr(cfg, "N_C_TO_N_LIVING", 0.176))
N_C_TO_N_BY_GROUP: Dict[int, float] = dict(getattr(cfg, "N_C_TO_N_BY_GROUP", {}) or {})

# Flux multiplier controls
USE_N_MULT = bool(getattr(cfg, "USE_N_MULT", False))
N_MULT_TYPE = str(getattr(cfg, "N_MULT_TYPE", "none")).lower()
SEASON_MAP = getattr(cfg, "SEASON_MAP", {})
SEASONAL_N_FLUX_MULT = getattr(cfg, "SEASONAL_N_FLUX_MULT", {})
MONTHLY_N_FLUX_MULT = getattr(cfg, "MONTHLY_N_FLUX_MULT", {})
EWE_NUTR_LOADING_FILE = getattr(cfg, "EWE_NUTR_LOADING_FILE", None)
N_FLUX_APPLY_MODE = str(getattr(cfg, "N_FLUX_APPLY_MODE", "scale_total")).lower()

# Exclude any spin-up window
START_FULL = getattr(cfg, "START_FULL", None)
END_FULL = getattr(cfg, "END_FULL", None)

# Output
OUT_CSV = getattr(cfg, "ECOSIM_F_W_NUTRIENTS", os.path.join(OUTPUT_DIR_EVAL, f"ecosim_{SCENARIO}_nutrients.csv"))

# Plot toggles
N_SHOW_PLOT = bool(getattr(cfg, "N_SHOW_PLOT", False))
N_PLOT_INCLUDE_OBS = bool(getattr(cfg, "N_PLOT_INCLUDE_OBS", False))

# Ecospace overlay on this Ecosim plot (optional)
ES_OVERLAY_ENABLE = bool(getattr(cfg, "ES_OVERLAY_ENABLE", False))
ES_OVERLAY_CONFIG_MODULE = str(getattr(cfg, "ES_OVERLAY_CONFIG_MODULE", "ecospace_eval_config"))
ES_OVERLAY_SERIES = getattr(cfg, "ES_OVERLAY_SERIES", ["box"])  # list or str: 'matched', 'box'
ES_OVERLAY_MATCHED_CSV = getattr(cfg, "ES_OVERLAY_MATCHED_CSV", None)
ES_OVERLAY_BOX_CSV = getattr(cfg, "ES_OVERLAY_BOX_CSV", None)
ES_OVERLAY_SPATIAL_REDUCER = str(getattr(cfg, "ES_OVERLAY_SPATIAL_REDUCER", "mean")).lower()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------



def _compute_skill_metrics(
    df: pd.DataFrame,
    *,
    obs_col: str = "avg_obs",
    model_col: str = "avg_model",
) -> pd.DataFrame:
    """
    Compute standard paired skill metrics from a paired table.
    Expects one row per paired comparison unit (e.g., year x biweekly).
    """
    d = df[[obs_col, model_col]].copy()
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    d[model_col] = pd.to_numeric(d[model_col], errors="coerce")
    d = d.dropna(subset=[obs_col, model_col]).copy()

    if d.empty:
        return pd.DataFrame([{
            "n": 0,
            "bias": np.nan,
            "obs_std": np.nan,
            "model_std": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "r": np.nan,
            "WSS": np.nan,
        }])

    obs = d[obs_col].to_numpy(dtype=float)
    mod = d[model_col].to_numpy(dtype=float)
    resid = mod - obs

    n = len(d)
    bias = float(np.mean(resid))
    obs_std = float(np.std(obs, ddof=1)) if n > 1 else np.nan
    model_std = float(np.std(mod, ddof=1)) if n > 1 else np.nan
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r = float(np.corrcoef(obs, mod)[0, 1]) if n > 1 else np.nan

    obs_mean = float(np.mean(obs))
    denom = np.sum((np.abs(mod - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    wss = float(1.0 - np.sum((mod - obs) ** 2) / denom) if denom > 0 else np.nan

    return pd.DataFrame([{
        "n": n,
        "bias": bias,
        "obs_std": obs_std,
        "model_std": model_std,
        "MAE": mae,
        "RMSE": rmse,
        "r": r,
        "WSS": wss,
    }])


def _plot_skill_scatter(
    df: pd.DataFrame,
    *,
    obs_col: str = "avg_obs",
    model_col: str = "avg_model",
    out_png: str,
    title: str,
    stats_df: pd.DataFrame | None = None,
):
    """
    Scatter plot from the same paired table used for skill stats.
    """
    d = df[[obs_col, model_col]].copy()
    if "year" in df.columns:
        d["year"] = df["year"]
    if "biweekly" in df.columns:
        d["biweekly"] = df["biweekly"]

    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    d[model_col] = pd.to_numeric(d[model_col], errors="coerce")
    d = d.dropna(subset=[obs_col, model_col]).copy()

    if d.empty:
        print(f"[skill scatter] No paired data available for {title}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(d[obs_col], d[model_col], alpha=0.7)

    lo = np.nanmin([d[obs_col].min(), d[model_col].min()])
    hi = np.nanmax([d[obs_col].max(), d[model_col].max()])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_xlabel("Observed N (g N m$^{-2}$)")
    ax.set_ylabel("Modelled N (g N m$^{-2}$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if stats_df is not None and not stats_df.empty:
        s = stats_df.iloc[0]
        txt = (
            f"n = {int(s['n'])}\n"
            f"r = {s['r']:.2f}\n"
            f"RMSE = {s['RMSE']:.2f}\n"
            f"MAE = {s['MAE']:.2f}\n"
            f"WSS = {s['WSS']:.2f}"
        )
        ax.text(
            0.98, 0.02, txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def derive_season_from_month(month: int) -> str:
    return SEASON_MAP.get(month, "Winter")


def group_c_to_n(group_id: int) -> float:
    """Return gN per gC for a numeric Ecosim group id."""
    return float(N_C_TO_N_BY_GROUP.get(int(group_id), N_C_TO_N_LIVING))


def _compute_multiplier(mdf: pd.DataFrame) -> pd.Series:
    if not USE_N_MULT:
        return pd.Series(1.0, index=mdf.index)

    if N_MULT_TYPE == "monthly":
        return mdf["month"].map(MONTHLY_N_FLUX_MULT).astype(float).fillna(1.0)

    if N_MULT_TYPE == "seasonal":
        return mdf["season"].map(SEASONAL_N_FLUX_MULT).astype(float).fillna(1.0)

    if N_MULT_TYPE == "3day":
        if not EWE_NUTR_LOADING_FILE:
            raise ValueError("N_MULT_TYPE='3day' but EWE_NUTR_LOADING_FILE is not set in config.")
        forc = pd.read_csv(EWE_NUTR_LOADING_FILE)
        forc = forc.rename(columns={
            "dayofyear": "day_of_year",
            "stdfilter_varmixing_m": "multiplier",
        })
        dfm = mdf.merge(
            forc[["year", "day_of_year", "multiplier"]],
            on=["year", "day_of_year"],
            how="left",
        )
        return pd.to_numeric(dfm["multiplier"], errors="coerce").fillna(1.0)

    # default (unknown type)
    return pd.Series(1.0, index=mdf.index)


def _t0_bound_from_series(df: pd.DataFrame) -> float:
    """Bound N at the first available timestep in the *series* (1D: earliest date)."""
    d = df.dropna(subset=["date", "model_N_bound_gN_m2"]).sort_values("date")
    if d.empty:
        raise ValueError("No non-NaN bound values to compute t0.")
    return float(d.iloc[0]["model_N_bound_gN_m2"])


def _umolL_profile_to_gNm2(depth_m: np.ndarray, conc_umol_L: np.ndarray, *, zmin: float, zmax: float) -> float:
    """
    Integrate a single vertical profile of concentration (umol/L == mmol/m3) over zmin..zmax
    and return g N m-2.
    """
    # Clean/sort
    ok = np.isfinite(depth_m) & np.isfinite(conc_umol_L)
    depth = depth_m[ok].astype(float)
    conc = conc_umol_L[ok].astype(float)

    if depth.size < 2:
        return np.nan

    # enforce positive depths
    depth = np.clip(depth, 0.0, None)
    order = np.argsort(depth)
    depth = depth[order]
    conc = conc[order]

    # Restrict to zmin..zmax with endpoint interpolation
    if depth.max() < zmin or depth.min() > zmax:
        return np.nan

    # Build an integration grid including endpoints
    z_nodes = depth[(depth >= zmin) & (depth <= zmax)]
    if z_nodes.size == 0:
        # no interior nodes; rely on endpoints
        z_nodes = np.array([], dtype=float)

    # Add endpoints explicitly (interpolate within available depth range)
    z_grid = np.unique(np.concatenate([[zmin], z_nodes, [zmax]]))
    # interpolate conc onto z_grid (no extrapolation beyond min/max)
    zmin_clip = max(zmin, depth.min())
    zmax_clip = min(zmax, depth.max())
    if zmin_clip > zmax_clip:
        return np.nan
    z_grid = z_grid[(z_grid >= zmin_clip) & (z_grid <= zmax_clip)]
    if z_grid.size < 2:
        return np.nan

    conc_grid = np.interp(z_grid, depth, conc)  # conc in mmol/m3

    mmol_m2 = float(np.trapezoid(conc_grid, z_grid))  # mmol/m2
    return mmol_m2 * MMOL_TO_GN  # g N m-2


def _compute_obs_model_climatology_matched(
    obs_csv: str,
    mdf: pd.DataFrame,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-match cast-level observed inventories to the Ecosim model, then
    build paired year-biweekly tables and paired climatologies.

    Returns
    -------
    paired_ybw : DataFrame
        year, biweekly, avg_obs, avg_model
    obs_clima : DataFrame
        biweekly climatology for observations
    model_clima : DataFrame
        biweekly climatology for Ecosim sampled only on obs dates
    """
    obs = pd.read_csv(obs_csv)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date"]).copy()

    if "cast_nitrogen_int_gN_m2" not in obs.columns:
        raise ValueError(
            "Expected cast-level obs file with column 'cast_nitrogen_int_gN_m2'. "
            f"Got columns: {list(obs.columns)}"
        )

    # restrict to the model-evaluation window
    obs_date_min = getattr(cfg, "N_OBS_DATE_MIN", None)
    obs_date_max = getattr(cfg, "N_OBS_DATE_MAX", None)

    if obs_date_min is not None:
        obs = obs[obs["date"] >= pd.to_datetime(obs_date_min)]
    if obs_date_max is not None:
        obs = obs[obs["date"] <= pd.to_datetime(obs_date_max)]

    if obs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # collapse multiple casts on the same day to one basin-scale obs value
    obs_reducer_name = str(getattr(cfg, "N_OBS_DATE_REDUCER", "median")).lower()
    obs_reducer = np.nanmean if obs_reducer_name == "mean" else np.nanmedian

    obs_day = (
        obs.groupby("date", as_index=False)
           .agg(obs_gN_m2=("cast_nitrogen_int_gN_m2", lambda x: float(obs_reducer(x.to_numpy(dtype=float)))))
           .sort_values("date")
    )

    model_day = (
        mdf[["date", value_col]]
        .dropna()
        .sort_values("date")
        .rename(columns={value_col: "model_gN_m2"})
        .copy()
    )

    tol_days = int(getattr(cfg, "N_MATCH_TOL_DAYS", 3))
    paired = pd.merge_asof(
        obs_day,
        model_day,
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=tol_days),
    ).dropna(subset=["model_gN_m2"]).copy()

    if paired.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    paired["year"] = paired["date"].dt.year
    paired["day_of_year"] = paired["date"].dt.dayofyear
    biweek_max = int(getattr(cfg, "N_BIWEEK_MAX", 26))
    paired["biweekly"] = np.minimum(
        biweek_max,
        ((paired["day_of_year"] - 1) // 14 + 1)
    ).astype(int)

    # reduce within year x biweekly so each year gets equal weight
    paired_ybw = (
        paired.groupby(["year", "biweekly"], as_index=False)
              .agg(
                  avg_obs=("obs_gN_m2", "median" if OBS_AVG_TYPE == "median" else "mean"),
                  avg_model=("model_gN_m2", "mean"),
              )
    )

    # climatologies built from the SAME paired table
    g_obs = paired_ybw.groupby("biweekly")["avg_obs"]
    obs_clima = pd.DataFrame(index=sorted(paired_ybw["biweekly"].unique()))
    obs_clima.index.name = "biweekly"
    obs_clima["obs_center"] = g_obs.mean() if OBS_AVG_TYPE == "mean" else g_obs.median()
    obs_clima["obs_q10"] = g_obs.quantile(0.1)
    obs_clima["obs_q90"] = g_obs.quantile(0.9)

    g_mod = paired_ybw.groupby("biweekly")["avg_model"]
    model_clima = pd.DataFrame(index=sorted(paired_ybw["biweekly"].unique()))
    model_clima.index.name = "biweekly"
    model_clima["model_center"] = g_mod.mean()
    model_clima["model_q10"] = g_mod.quantile(0.1)
    model_clima["model_q90"] = g_mod.quantile(0.9)

    return paired_ybw, obs_clima, model_clima


def _compute_obs_climatology(obs_csv: str) -> pd.DataFrame:
    """
    Compute a biweekly climatology of observed dissolved N inventories (g N m-2).

    Supported inputs:
      1) strict cast-level file from prep1_nutrients_v3.py
         columns include: date, cast_nitrogen_int_gN_m2
      2) raw long-format sample file
         columns include: date, depth, nitrogen
    """
    obs = pd.read_csv(obs_csv)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date"]).copy()

    # Optional date filter: restrict obs to model-overlap window
    obs_date_min = getattr(cfg, "N_OBS_DATE_MIN", None)
    obs_date_max = getattr(cfg, "N_OBS_DATE_MAX", None)

    if obs_date_min is not None:
        obs = obs[obs["date"] >= pd.to_datetime(obs_date_min)]
    if obs_date_max is not None:
        obs = obs[obs["date"] <= pd.to_datetime(obs_date_max)]

    if obs.empty:
        return pd.DataFrame()

    obs["year"] = obs["date"].dt.year
    obs["day_of_year"] = obs["date"].dt.dayofyear
    biweek_max = int(getattr(cfg, "N_BIWEEK_MAX", 26))
    obs["biweekly"] = np.minimum(
        biweek_max,
        ((obs["day_of_year"] - 1) // 14 + 1)
    ).astype(int)

    reducer = np.mean if OBS_AVG_TYPE == "mean" else np.median

    # -------------------------------------------------
    # Preferred path: pre-screened cast-level inventory
    # -------------------------------------------------
    if "cast_nitrogen_int_gN_m2" in obs.columns:
        obs["cast_nitrogen_int_gN_m2"] = pd.to_numeric(
            obs["cast_nitrogen_int_gN_m2"], errors="coerce"
        )
        obs = obs.dropna(subset=["cast_nitrogen_int_gN_m2"])

        if obs.empty:
            return pd.DataFrame()

        yrbw = (
            obs.groupby(["year", "biweekly"], as_index=False)
               .agg(avg_obs=("cast_nitrogen_int_gN_m2", reducer))
        )

    # -------------------------------------------------
    # Fallback path: raw sample-level file
    # -------------------------------------------------
    else:
        if "nitrogen" not in obs.columns:
            raise ValueError(f"Obs file missing 'nitrogen' column: {obs_csv}")
        if "depth" not in obs.columns:
            raise ValueError(f"Obs file missing 'depth' column: {obs_csv}")

        obs["depth"] = pd.to_numeric(obs["depth"], errors="coerce")
        obs["nitrogen"] = pd.to_numeric(obs["nitrogen"], errors="coerce")
        obs.loc[obs["depth"] == 0, "depth"] = 0.1

        prof_keys = ["date"]
        if "ewe_row" in obs.columns and "ewe_col" in obs.columns:
            prof_keys += ["ewe_row", "ewe_col"]

        prof = []
        for key, d in obs.groupby(prof_keys):
            gNm2 = _umolL_profile_to_gNm2(
                depth_m=d["depth"].to_numpy(),
                conc_umol_L=d["nitrogen"].to_numpy(),
                zmin=ZMIN,
                zmax=ZMAX,
            )
            if not np.isfinite(gNm2):
                continue

            row = dict(zip(prof_keys, key if isinstance(key, tuple) else (key,)))
            row.update({
                "year": int(d["year"].iloc[0]),
                "biweekly": int(d["biweekly"].iloc[0]),
                "obs_gN_m2": float(gNm2),
            })
            prof.append(row)

        if not prof:
            return pd.DataFrame()

        prof_df = pd.DataFrame(prof)

        yrbw = (
            prof_df.groupby(["year", "biweekly"], as_index=False)
                   .agg(avg_obs=("obs_gN_m2", reducer))
        )

    g = yrbw.groupby("biweekly")["avg_obs"]
    clima = pd.DataFrame(index=sorted(yrbw["biweekly"].unique()))
    clima.index.name = "biweekly"
    clima["obs_center"] = g.mean() if OBS_AVG_TYPE == "mean" else g.median()
    clima["obs_q10"] = g.quantile(0.1)
    clima["obs_q90"] = g.quantile(0.9)
    return clima


def _compute_model_climatology(mdf: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # Reduce within (year, biweekly) first so each year counts equally
    yrbw = mdf.groupby(["year", "biweekly"], as_index=False).agg(avg_model=(value_col, "mean"))
    g = yrbw.groupby("biweekly")["avg_model"]

    clima = pd.DataFrame(index=sorted(yrbw["biweekly"].unique()))
    clima.index.name = "biweekly"
    clima["model_center"] = g.mean()
    clima["model_q10"] = g.quantile(0.1)
    clima["model_q90"] = g.quantile(0.9)
    return clima


def _reducer(name: str):
    name = (name or "mean").lower()
    if name == "median":
        return lambda a: float(np.nanmedian(np.asarray(a, dtype=float)))
    return lambda a: float(np.nanmean(np.asarray(a, dtype=float)))


def _compute_ecospace_climatology(series_kind: str) -> pd.DataFrame:
    """Return Ecospace biweekly climatology (center, q10, q90) for free N (g N m-2).

    We read the *_model_matched.csv or *_model_box.csv outputs from ecospace_eval_6a,
    infer bound/free N using ecospace_eval_config settings, then pool to a 1D
    (year,biweekly)->(biweekly) climatology.
    """
    escfg = importlib.import_module(ES_OVERLAY_CONFIG_MODULE)

    scenario = str(getattr(escfg, "ECOSPACE_SC", ""))
    evalout = str(getattr(escfg, "EVALOUT_P", ""))

    if series_kind == "matched":
        csv_path = ES_OVERLAY_MATCHED_CSV or os.path.join(evalout, f"ecospace_{scenario}_nutrients_biweek_model_matched.csv")
    elif series_kind == "box":
        csv_path = ES_OVERLAY_BOX_CSV or os.path.join(evalout, f"ecospace_{scenario}_nutrients_biweek_model_box.csv")
    else:
        raise ValueError("series_kind must be 'matched' or 'box'")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ecospace overlay CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Choose a time column (only needed for t0 anchoring)
    time_col = "model_time" if "model_time" in df.columns else ("date_rep" if "date_rep" in df.columns else None)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    model_groups = list(getattr(escfg, "NU_MODEL_GROUPS", []))
    c_to_n_default = float(getattr(escfg, "NU_C_TO_N_LIVING", 0.176))
    c_to_n_by = dict(getattr(escfg, "NU_C_TO_N_BY_GROUP", {}) or {})

    def c_to_n(code: str) -> float:
        return float(c_to_n_by.get(code, c_to_n_default))

    # Bound N(t)
    bound = np.zeros(len(df), dtype=float)
    for g in model_groups:
        if g not in df.columns:
            continue
        bound += pd.to_numeric(df[g], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(dtype=float) * c_to_n(g)

    df["es_model_N_bound_gN_m2"] = np.maximum(0.0, bound)

    # Init modes
    bound_init_ecopath = float(getattr(escfg, "NU_BOUND_INIT_GNM2", np.nan))
    free_init_cfg = float(getattr(escfg, "NU_FREE_INIT_GNM2", np.nan))

    bound_mode = str(getattr(escfg, "NU_BOUND_INIT_MODE", "ecopath")).lower()
    free_mode = str(getattr(escfg, "NU_FREE_INIT_MODE", "config")).lower()

    reducer = _reducer(ES_OVERLAY_SPATIAL_REDUCER)

    bound_init_used = bound_init_ecopath
    if bound_mode == "t0_series":
        # t0 per cell if available, else global first row
        if time_col is not None:
            df = df.sort_values(time_col)
        else:
            df = df.sort_values(["year", "biweekly"])

        if bool(getattr(escfg, "NU_T0_PER_SERIES", True)) and {"ewe_row","ewe_col"}.issubset(df.columns):
            firsts = df.groupby(["ewe_row","ewe_col"], as_index=False).first()
            bound_init_used = reducer(firsts["es_model_N_bound_gN_m2"].to_numpy(dtype=float))
        else:
            bound_init_used = float(df["es_model_N_bound_gN_m2"].iloc[0])

    if free_mode == "t0_preserve_total":
        total_ref = float(free_init_cfg + bound_init_ecopath)
        free_init_used = float(total_ref - bound_init_used)
    else:
        free_init_used = free_init_cfg

    total_init_used = float(free_init_used + bound_init_used)

    df["es_model_N_free_gN_m2"] = np.maximum(0.0, total_init_used - df["es_model_N_bound_gN_m2"])

    # Optional flux multiplier
    use_flux = bool(getattr(escfg, "NU_USE_FLUX_MULT", False))
    flux_col = str(getattr(escfg, "NU_FLUX_MULT_COL", "N_FLUX_MULT"))
    flux_apply = str(getattr(escfg, "NU_FLUX_APPLY_MODE", "scale_free")).lower()
    flux_fill = getattr(escfg, "NU_FLUX_MULT_FILL", None)

    if use_flux and flux_col in df.columns:
        mult = pd.to_numeric(df[flux_col], errors="coerce")
        if flux_fill is not None:
            mult = mult.fillna(float(flux_fill))
        if flux_apply == "scale_total":
            df["es_model_N_free_used_gN_m2"] = np.maximum(0.0, total_init_used * mult - df["es_model_N_bound_gN_m2"])
        else:
            df["es_model_N_free_used_gN_m2"] = np.maximum(0.0, df["es_model_N_free_gN_m2"] * mult)
    else:
        df["es_model_N_free_used_gN_m2"] = df["es_model_N_free_gN_m2"]

    if not {"year","biweekly"}.issubset(df.columns):
        raise ValueError("Ecospace overlay table missing year/biweekly columns")

    # Pool across space within each year+biweekly
    yrbw = df.groupby(["year","biweekly"], as_index=False).agg(avg_es=("es_model_N_free_used_gN_m2", reducer))

    # Climatology across years
    g = yrbw.groupby("biweekly")["avg_es"]
    clima = pd.DataFrame(index=sorted(yrbw["biweekly"].unique()))
    clima.index.name = "biweekly"
    clima["es_center"] = g.mean()
    clima["es_q10"] = g.quantile(0.1)
    clima["es_q90"] = g.quantile(0.9)

    return clima


def _plot_overlay(
    obs_clima: pd.DataFrame | None,
    model_clima: pd.DataFrame,
    ecospace_climas: dict | None,
    *,
    out_png: str,
    save_pdf: bool = False,
) -> None:
    """Publication-style climatology plot with color- and style-safe encodings."""

    # Okabe–Ito palette (color-blind friendly)
    col_obs = "darkorange"       # orange
    col_ecosim = "blue"    # blue dashed;
    col_ecospace = "blue"  # blue; alt #56B4E9" light blue

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    # Model (Ecosim)
    x = model_clima.index.to_numpy()
    ax.fill_between(x, model_clima["model_q10"], model_clima["model_q90"], alpha=0.15, color=col_ecosim, linewidth=0)
    ax.plot(
        x,
        model_clima["model_center"],
        color=col_ecosim,
        linewidth=1.5,
        linestyle="--",
        marker="s",
        markersize=3.0,
        label="Ecosim (1D)",
    )

    # Obs
    if obs_clima is not None and not obs_clima.empty:
        xo = obs_clima.index.to_numpy()
        ax.fill_between(xo, obs_clima["obs_q10"], obs_clima["obs_q90"], alpha=0.15, color=col_obs, linewidth=0)
        ax.plot(
            xo,
            obs_clima["obs_center"],
            color=col_obs,
            linewidth=1.5,
            linestyle="-",
            marker="o",
            markersize=3,
            label="Observations",
        )

    # Ecospace overlays (optional)
    if ecospace_climas:
        for series_name, es_clima in ecospace_climas.items():
            if es_clima is None or es_clima.empty:
                continue

            xe = es_clima.index.to_numpy()
            ax.fill_between(xe, es_clima["es_q10"], es_clima["es_q90"], alpha=0.12, color=col_ecospace, linewidth=0)

            # Differentiate Ecospace series by line style + marker (works in grayscale)
            if str(series_name).lower() == "matched":
                ls, mk = "-", "D"
                lab = "Ecospace (matched)"
            elif str(series_name).lower() == "box":
                ls, mk = "-", "^"
                lab = "Ecospace (box)"
            else:
                ls, mk = "-", "^"
                lab = f"Ecospace ({series_name})"

            ax.plot(
                xe,
                es_clima["es_center"],
                color=col_ecospace,
                linewidth=1.5,
                linestyle=ls,
                marker=mk,
                markersize=3.0,
                label=lab,
            )

    # X ticks as month labels (biweekly axis)
    month_start_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # biweek_max = int(getattr(cfg, "N_BIWEEK_MAX", 26))
    # month_ticks = [((d - 1) // 14) + 1 for d in month_start_doy]
    # ax.set_xlim(1, biweek_max)
    # ax.set_xticks(month_ticks)
    # ax.set_xticklabels(month_labels)

    biweek_max = int(getattr(cfg, "N_BIWEEK_MAX", 26))
    ax.set_xlim(1, biweek_max)

    # Month labels placed at true month start positions in "biweek units"
    month_ticks = [1.0 + (d - 1.0) / 14.0 for d in month_start_doy]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)


    ax.set_xlabel("Time of year")
    ax.set_ylabel("Free dissolved N inventory (g N m$^{-2}$)")
    ax.set_title("Surface-layer dissolved inorganic N evaluation")

    ax.grid(True, alpha=0.3)
    # ax.legend(frameon=False)

    fig.savefig(out_png, dpi=300)

    if save_pdf:
        base, _ = os.path.splitext(out_png)
        fig.savefig(base + ".pdf")

    if N_SHOW_PLOT:
        plt.show()
    plt.close(fig)

def run_nutrient_eval() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    mdf : pd.DataFrame
        Full model table with inferred nutrient series.
    model_clima : pd.DataFrame
        Biweekly climatology of model_N_free_used_gN_m2.
    """

    # ------------------
    # Load model output
    # ------------------
    mdf = pd.read_csv(ECOSIM_F_PREPPED_SINGLERUN)
    mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce")
    mdf = mdf.dropna(subset=["date"]).copy()

    mdf["year"] = mdf["date"].dt.year
    mdf["month"] = mdf["date"].dt.month
    mdf["day_of_year"] = mdf["date"].dt.dayofyear
    biweek_max = int(getattr(cfg, "N_BIWEEK_MAX", 26))

    mdf["biweekly"] = np.minimum(biweek_max, ((mdf["day_of_year"] - 1) // 14 + 1)).astype(int)
    if "season" not in mdf.columns:
        mdf["season"] = mdf["month"].apply(derive_season_from_month)

    mdf["multiplier"] = _compute_multiplier(mdf).astype(float)

    # ------------------
    # Compute bound N(t)
    # ------------------
    group_cols = [str(g) for g in N_BOUND_GROUPS if str(g) in mdf.columns]
    if not group_cols:
        raise ValueError("No N_BOUND_GROUPS columns found in model output. Check N_BOUND_GROUPS vs CSV columns.")

    # Convert each group biomass to N and sum
    bound_terms = []
    for g in N_BOUND_GROUPS:
        col = str(g)
        if col not in mdf.columns:
            continue
        mult = group_c_to_n(g)
        x = pd.to_numeric(mdf[col], errors="coerce").clip(lower=0)
        bound_terms.append(x * mult)

    mdf["model_N_bound_gN_m2"] = np.sum(bound_terms, axis=0)

    # ------------------
        # Choose initialization anchor
    bound_init_ecopath = float(N_BOUND_INIT_GNM2)

    if N_BOUND_INIT_MODE == "t0_series":
        # Ecosim time=0 is typically one step after the Ecopath baseline; use the first model row
        t0 = mdf["date"].min()
        bound_init_used = float(mdf.loc[mdf["date"] == t0, "model_N_bound_gN_m2"].mean())
    elif N_BOUND_INIT_MODE == "ecopath":
        bound_init_used = bound_init_ecopath
    else:
        raise ValueError(f"Unrecognized N_BOUND_INIT_MODE: {N_BOUND_INIT_MODE}")

    # Total inventory reference is fixed by the configured initialization
    total_ref = float(N_TOTAL_INIT_GNM2)

    if N_FREE_INIT_MODE == "t0_preserve_total":
        # Adjust free so the total stays constant while using the chosen bound anchor
        free_init_used = max(0.0, total_ref - bound_init_used)
    elif N_FREE_INIT_MODE == "config":
        free_init_used = float(N_FREE_INIT_GNM2)
    else:
        raise ValueError(f"Unrecognized N_FREE_INIT_MODE: {N_FREE_INIT_MODE}")

    total_init_used = float(free_init_used + bound_init_used)

    # Free inventory implied by the bound pool time series (inventory closure)
    mdf["model_N_free_gN_m2"] = total_init_used - mdf["model_N_bound_gN_m2"]
    mdf.loc[mdf["model_N_free_gN_m2"] < 0, "model_N_free_gN_m2"] = 0.0

    # Record anchors for transparency/debugging
    mdf["model_N_total_init_used_gN_m2"] = total_init_used
    mdf["model_N_bound_init_used_gN_m2"] = bound_init_used
    mdf["model_N_free_init_used_gN_m2"] = free_init_used

    # ------------------
    # Apply multiplier (optional)
    # ------------------
    if USE_N_MULT:
        if N_FLUX_APPLY_MODE == "scale_free":
            mdf["model_N_free_used_gN_m2"] = mdf["model_N_free_gN_m2"] * mdf["multiplier"]
        elif N_FLUX_APPLY_MODE == "scale_total":
            mdf["model_N_free_used_gN_m2"] = (total_init_used * mdf["multiplier"]) - mdf["model_N_bound_gN_m2"]
        else:
            raise ValueError("N_FLUX_APPLY_MODE must be 'scale_free' or 'scale_total'")
        mdf.loc[mdf["model_N_free_used_gN_m2"] < 0, "model_N_free_used_gN_m2"] = 0.0
        value_col = "model_N_free_used_gN_m2"
    else:
        value_col = "model_N_free_gN_m2"

    # ------------------
    # Save full series
    # ------------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    mdf.to_csv(OUT_CSV, index=False)
    print(f"[ecosim nutrients] Saved → {OUT_CSV}")

    # Exclude spin-up period for climatology
    if START_FULL is not None:
        mdf = mdf[mdf["date"] >= pd.to_datetime(START_FULL)]
    if END_FULL is not None:
        mdf = mdf[mdf["date"] <= pd.to_datetime(END_FULL)]

    # ------------------
    # Climatology
    # ------------------
    # Full-model climatology retained for reference/debugging
    model_clima_full = _compute_model_climatology(mdf, value_col=value_col)

    paired_ybw = None
    obs_clima = None
    model_clima_plot = model_clima_full
    model_clima = model_clima_full

    if N_PLOT_INCLUDE_OBS and N_DATA_REF and bool(getattr(cfg, "N_MATCH_OBS_IN_TIME", True)):
        paired_ybw, obs_clima, model_clima_plot = _compute_obs_model_climatology_matched(
            N_DATA_REF,
            mdf,
            value_col=value_col,
        )

        if bool(getattr(cfg, "N_WRITE_MATCHED_CSV", True)) and paired_ybw is not None and not paired_ybw.empty:
            out_pair = os.path.join(
                OUTPUT_DIR_EVAL,
                f"ecosim_{SCENARIO}_nutrients_obs_model_paired_biweekly.csv"
            )
            paired_ybw.to_csv(out_pair, index=False)
            print(f"[ecosim nutrients] Paired obs-model table saved → {out_pair}")

    # keep the plot-generation guard
    if N_SHOW_PLOT:
        out_png = os.path.join(
            OUTPUT_DIR_FIGS,
            f"ecosim_{SCENARIO}_nutrient_clima_overlay_gNm2.png"
        )
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        ecospace_climas = None
        if ES_OVERLAY_ENABLE:
            ecospace_climas = {}
            kinds = ES_OVERLAY_SERIES
            if isinstance(kinds, str):
                kinds = [kinds]
            kinds = [str(k).lower() for k in kinds]
            if "matched" in kinds:
                try:
                    ecospace_climas["matched"] = _compute_ecospace_climatology("matched")
                except Exception as e:
                    print(f"[ecospace overlay] matched failed: {e}")
                    ecospace_climas["matched"] = None
            if "box" in kinds:
                try:
                    ecospace_climas["box"] = _compute_ecospace_climatology("box")
                except Exception as e:
                    print(f"[ecospace overlay] box failed: {e}")
                    ecospace_climas["box"] = None

        _plot_overlay(obs_clima, model_clima_plot, ecospace_climas, out_png=out_png)
        print(f"[ecosim nutrients] Plot saved → {out_png}")

    out_skill_csv = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecosim_{SCENARIO}_nutrient_skill.csv"
    )
    out_pairs_csv = os.path.join(
        OUTPUT_DIR_EVAL,
        f"ecosim_{SCENARIO}_nutrient_skill_pairs.csv"
    )
    out_scatter_png = os.path.join(
        OUTPUT_DIR_FIGS,
        f"ecosim_{SCENARIO}_nutrient_scatter.png"
    )

    stats = _compute_skill_metrics(
        paired_ybw,
        obs_col="avg_obs",
        model_col="avg_model",
    )

    paired_ybw.to_csv(out_pairs_csv, index=False)
    stats.to_csv(out_skill_csv, index=False)

    _plot_skill_scatter(
        paired_ybw,
        obs_col="avg_obs",
        model_col="avg_model",
        out_png=out_scatter_png,
        title=f"Ecosim nutrient skill",
        stats_df=stats,
    )

    print(f"[ecosim nutrients] Skill stats saved → {out_skill_csv}")
    print(f"[ecosim nutrients] Paired table saved → {out_pairs_csv}")
    print(f"[ecosim nutrients] Scatter saved → {out_scatter_png}")


    return mdf, model_clima


if __name__ == "__main__":
    run_nutrient_eval()
