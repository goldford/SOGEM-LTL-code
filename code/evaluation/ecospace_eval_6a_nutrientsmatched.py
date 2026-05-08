"""\

ecospace_eval_6a_nutrientsmatched.py

workflow
---------------------------------------------------------------
Ecospace is 2-D (no depth), but nutrient observations are vertical profiles.
To avoid overweighting casts with more sampled depths and to reduce compute,
this script:

1) Filters obs years (config option; default 2015–2018)
2) station or cast-level depth integration
   over [NU_ZMIN_M, NU_ZMAX_M] using trapezoids
   (linear between depths), after adding endpoints. Using 20 m here given sampling
3) Pools casts to biweekly bins by year and optionally by grid cell.
4) Matches each pooled bin to nearest Ecospace time.
5) Extracts model biomass for ALL groups at matched (time,row,col).

Input
-----
CSV cfg.NU_F_PREPPED must contain columns:
- date
- ewe_row, ewe_col
- depth (m)
- nitrogen (umol/L)

Outputs
-------
- ecospace_<SC>_nutrients_cast_depthint.csv
- ecospace_<SC>_nutrients_biweek_obs.csv
- ecospace_<SC>_nutrients_cast_model_matched.csv
- ecospace_<SC>_nutrients_biweek_model_matched.csv

Notes
-----
- We store BOTH cast inventory (umol/m^2) and cast depth-average (umol/L).
  The pooled column written as avg_nitrogen_obs is chosen by cfg.NU_OBS_VALUE_MODE.
- We do NOT compute model_N_free here; do that in 6b.

- to avoid issues, obs file uses 0.1 m as the shallowest sample (not 0.0 m). If NU_ZMIN_M=0.0
and NU_REQUIRE_FULL_COVERAGE=True with no surface padding tolerance, ALL casts
fail the surface-coverage check and are dropped.
Fix: set NU_ZMIN_M=0.1.

"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

import ecospace_eval_config as cfg


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Match tolerance for model time (days). None = always take nearest.
TIME_TOL_DAYS = getattr(cfg, "NU_TIME_TOL_DAYS", 7)

# Restrict obs years (END exclusive). Default: 2015–2018 inclusive.
NU_OBS_YR_ST = getattr(cfg, "NU_OBS_YR_ST", 2015)
NU_OBS_YR_EN = getattr(cfg, "NU_OBS_YR_EN", 2019)

# Pooling: keep row/col in bins so matching stays spatial.
NU_POOL_BY_CELL = getattr(cfg, "NU_POOL_BY_CELL", True)

# Depth integration window
NU_ZMIN_M = float(getattr(cfg, "NU_ZMIN_M", 0.0))
NU_ZMAX_M = float(getattr(cfg, "NU_ZMAX_M", 20.0))

# How strict is "coverage"?
NU_REQUIRE_FULL_COVERAGE = bool(getattr(cfg, "NU_REQUIRE_FULL_COVERAGE", True))

# Padding tolerances for endpoints (m). If a cast is missing the exact endpoint but is
# within tolerance, we pad the endpoint with the nearest measured concentration.
NU_SURFACE_PAD_TOL_M = float(getattr(cfg, "NU_SURFACE_PAD_TOL_M", 0.11))
NU_DEEP_PAD_TOL_M = float(getattr(cfg, "NU_DEEP_PAD_TOL_M", 0.0))

# If beyond padding tolerance, allow extrapolation? (uses nearest-endpoint value; still constant)
NU_DEPTH_EXTRAPOLATE = bool(getattr(cfg, "NU_DEPTH_EXTRAPOLATE", False))

# Allow single-depth casts (constant profile) — usually False
NU_ALLOW_SINGLE_DEPTH = bool(getattr(cfg, "NU_ALLOW_SINGLE_DEPTH", False))

# Which cast metric becomes avg_nitrogen_obs in pooled bins:
#   - "integral" -> mmol/m^2 inventory over [zmin,zmax]
#   - "average"  -> mmol/L depth-average over [zmin,zmax]
NU_OBS_VALUE_MODE = str(getattr(cfg, "NU_OBS_VALUE_MODE", "integral")).lower()

# Aggregation within bins (across casts)
NU_OBS_BIN_AVG_TYPE = getattr(cfg, "NU_OBS_BIN_AVG_TYPE", getattr(cfg, "NU_OBS_AVG_TYPE", "mean"))

# Optional cast-quality filters
NU_MIN_DEPTHS_PER_CAST = getattr(cfg, "NU_MIN_DEPTHS_PER_CAST", None)
NU_MIN_MAXDEPTH_M = getattr(cfg, "NU_MIN_MAXDEPTH_M", None)


# Paths
NUTRIENTS_CSV = cfg.NU_F_PREPPED

SCENARIO = cfg.ECOSPACE_SC
ECOSPACE_OUT_PATH = cfg.NC_PATH_OUT
ECOSPACE_CODE = cfg.ECOSPACE_SC
FILENM_STRT_YR = cfg.ECOSPACE_RN_STR_YR
FILENM_END_YR = cfg.ECOSPACE_RN_END_YR

ECOSPACE_NC = rf"{ECOSPACE_OUT_PATH}/{ECOSPACE_CODE}_{FILENM_STRT_YR}-{FILENM_END_YR}.nc"

OUTPUT_DIR_EVAL = cfg.EVALOUT_P

OUT_CAST_CSV = rf"{OUTPUT_DIR_EVAL}/ecospace_{SCENARIO}_nutrients_cast_depthint.csv"
OUT_OBS_BIWEEK_CSV = rf"{OUTPUT_DIR_EVAL}/ecospace_{SCENARIO}_nutrients_biweek_obs.csv"
OUT_MATCHED_CAST_CSV = rf"{OUTPUT_DIR_EVAL}/ecospace_{SCENARIO}_nutrients_cast_model_matched.csv"
OUT_MATCHED_BIWEEK_CSV = rf"{OUTPUT_DIR_EVAL}/ecospace_{SCENARIO}_nutrients_biweek_model_matched.csv"
OUT_BOX_BIWEEK_CSV = rf"{OUTPUT_DIR_EVAL}/ecospace_{SCENARIO}_nutrients_biweek_model_box.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_model_box_biweekly(
    ds: xr.Dataset,
    *,
    years: range,
    biweek_max: int,
    box: tuple[int, int, int, int],
    stride: int,
    vars_to_get: list[str],
) -> pd.DataFrame:
    time_indexer = pd.Index(pd.to_datetime(ds["time"].values))

    rmin, rmax, cmin, cmax = box
    rows = list(range(rmin, rmax + 1, stride))
    cols = list(range(cmin, cmax + 1, stride))

    out_rows: list[dict] = []

    for year in years:
        for bw in range(1, biweek_max + 1):
            # midpoint day-of-year for biweek bin (1..26): 7, 21, 35, ...
            mid_doy = (bw - 1) * 14 + 7
            date_rep = pd.Timestamp(datetime(year, 1, 1) + timedelta(days=mid_doy - 1))

            t_idx = int(time_indexer.get_indexer([date_rep], method="nearest")[0])
            model_time = pd.to_datetime(ds["time"].isel(time=t_idx).values)

            for rr in rows:
                for cc in cols:
                    rr0, cc0 = rr - 1, cc - 1  # convert to 0-based
                    row = {
                        "year": year,
                        "biweekly": bw,
                        "ewe_row": rr,
                        "ewe_col": cc,
                        "date_rep": date_rep,
                        "model_time": model_time,
                        # obs columns intentionally absent/NaN in this file
                    }

                    for v in vars_to_get:
                        if v in ds.data_vars and (0 <= rr0 < ds.dims["row"]) and (0 <= cc0 < ds.dims["col"]):
                            row[v] = float(ds[v].isel(time=t_idx, row=rr0, col=cc0).values)
                        else:
                            row[v] = np.nan

                    out_rows.append(row)

    return pd.DataFrame(out_rows)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _agg_stat(series: pd.Series, how: str) -> float:
    how = str(how).lower()
    if how == "mean":
        return float(series.mean())
    if how == "median":
        return float(series.median())
    raise ValueError(f"Unsupported aggregation: {how!r} (use 'mean' or 'median')")


def model_vars_to_extract(ds: xr.Dataset) -> list[str]:
    """Choose model vars to extract (biomass groups).

    We intentionally extract *all* biomass groups so 6b can change bound-group logic
    without re-running matching.
    """
    if hasattr(cfg, "NU_EXTRACT_GRPS"):
        base = list(getattr(cfg, "NU_EXTRACT_GRPS"))
    elif hasattr(cfg, "ES_GROUPS_RELB"):
        base = sorted(list(getattr(cfg, "ES_GROUPS_RELB")))
    else:
        base = list(getattr(cfg, "NU_INCLUDE_GRPS", []))

    for v in ("DE1-POC", "DE2-DOC"):
        if v in ds.data_vars and v not in base:
            base.append(v)

    # Allow non-biomass extras (e.g., spatiotemporal nutrient-flux multipliers loaded into the NC)
    extra_vars = list(getattr(cfg, "NU_EXTRACT_EXTRA_VARS", []))
    for v in extra_vars:
        if v in ds.data_vars and v not in base:
            base.append(v)

        # Include any nutrient-flux forcing vars that were injected into the NC in script 1a
    nutr_flux = getattr(cfg, "NUTR_FLUX_ASC", {}) or {}
    for v in nutr_flux.keys():
        if v in ds.data_vars and v not in base:
            base.append(v)

    return _unique_keep_order(base)


def nearest_time_join(
    df: pd.DataFrame,
    *,
    date_col: str,
    model_times: pd.DatetimeIndex,
    tol_days: int | None,
) -> pd.DataFrame:
    """Add nearest model_time to each row based on date_col."""
    tdf = pd.DataFrame({"model_time": model_times}).sort_values("model_time")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col)

    merged = pd.merge_asof(
        out,
        tdf,
        left_on=date_col,
        right_on="model_time",
        direction="nearest",
    )

    if tol_days is not None:
        too_far = (merged["model_time"] - merged[date_col]).abs() > pd.Timedelta(days=tol_days)
        merged.loc[too_far, "model_time"] = pd.NaT

    return merged


# -----------------------------------------------------------------------------
# Depth integration (cast-level)
# -----------------------------------------------------------------------------


def _profile_integrate(depth_m: np.ndarray, conc_mmol_L: np.ndarray) -> dict:
    """Integrate one profile over [NU_ZMIN_M, NU_ZMAX_M] using trapezoids.

    Returns dict with inventory (mmol/m^2), depth-average (mmol/L), and flags.

    Endpoint handling:
      - If the shallowest/deepest sample is within PAD_TOL of the endpoint,
        pad the endpoint with the nearest measured concentration.
      - If beyond PAD_TOL, only allow if NU_DEPTH_EXTRAPOLATE=True.

    Note: np.interp extrapolates using constant endpoint values.
    """

    zmin, zmax = NU_ZMIN_M, NU_ZMAX_M
    if not (zmax > zmin):
        return {"ok": False}

    # Clean + sort
    m = np.isfinite(depth_m) & np.isfinite(conc_mmol_L)
    depth = np.asarray(depth_m[m], dtype=float)
    conc = np.asarray(conc_mmol_L[m], dtype=float)

    if depth.size == 0:
        return {"ok": False}

    # sort by depth
    order = np.argsort(depth)
    depth = depth[order]
    conc = conc[order]

    # merge duplicates (average conc at identical depths)
    if depth.size != np.unique(depth).size:
        df = pd.DataFrame({"depth": depth, "conc": conc})
        df = df.groupby("depth", as_index=False)["conc"].mean()
        depth = df["depth"].to_numpy(dtype=float)
        conc = df["conc"].to_numpy(dtype=float)

    n_depths = int(depth.size)
    min_d = float(depth.min())
    max_d = float(depth.max())

    # Determine whether endpoints are effectively "covered" (or close enough to pad)
    surface_gap = max(0.0, min_d - zmin)
    deep_gap = max(0.0, zmax - max_d)

    pad_surface = surface_gap > 0 and surface_gap <= NU_SURFACE_PAD_TOL_M
    pad_deep = deep_gap > 0 and deep_gap <= NU_DEEP_PAD_TOL_M

    covers_surface = (min_d <= zmin) or pad_surface
    covers_deep = (max_d >= zmax) or pad_deep

    if NU_REQUIRE_FULL_COVERAGE and (not covers_surface or not covers_deep):
        return {
            "ok": False,
            "n_depths": n_depths,
            "min_depth": min_d,
            "max_depth": max_d,
        }

    if n_depths == 1:
        if not NU_ALLOW_SINGLE_DEPTH:
            return {"ok": False, "n_depths": n_depths, "min_depth": min_d, "max_depth": max_d}
        c0_L = float(conc[0])
        c0_m3 = c0_L #
        inv = c0_m3 * (zmax - zmin)
        avg_L = (inv / (zmax - zmin)) / 1000.0
        return {
            "ok": True,
            "n_depths": n_depths,
            "min_depth": min_d,
            "max_depth": max_d,
            "pad_surface": True,
            "pad_deep": True,
            "extrap_surface": True,
            "extrap_deep": True,
            "inv_mmol_m2": float(inv),
            "avg_mmol_L": float(avg_L),
        }

    # If we are missing endpoints beyond pad tolerances, only proceed if extrapolation enabled
    need_extrap_surface = surface_gap > NU_SURFACE_PAD_TOL_M
    need_extrap_deep = deep_gap > NU_DEEP_PAD_TOL_M

    if (need_extrap_surface or need_extrap_deep) and (not NU_DEPTH_EXTRAPOLATE):
        return {
            "ok": False,
            "n_depths": n_depths,
            "min_depth": min_d,
            "max_depth": max_d,
        }

    # Keep interior points within window
    mask = (depth >= zmin) & (depth <= zmax)
    depth_in = depth[mask]
    conc_in = conc[mask]

    def interp(z: float) -> float:
        return float(np.interp(z, depth, conc))

    # Add endpoints if missing
    if depth_in.size == 0 or depth_in[0] > zmin:
        depth_in = np.insert(depth_in, 0, zmin)
        conc_in = np.insert(conc_in, 0, interp(zmin))

    if depth_in[-1] < zmax:
        depth_in = np.append(depth_in, zmax)
        conc_in = np.append(conc_in, interp(zmax))

    # Integrate (umol/L -> mmol/m^3; equivalent!) then dz -> mmol/m^2
    conc_m3 = conc_in
    inv = float(np.trapezoid(conc_m3, depth_in))
    avg_L = (inv / (zmax - zmin)) / 1000.0

    return {
        "ok": True,
        "n_depths": n_depths,
        "min_depth": min_d,
        "max_depth": max_d,
        "pad_surface": bool(pad_surface),
        "pad_deep": bool(pad_deep),
        "extrap_surface": bool(need_extrap_surface),
        "extrap_deep": bool(need_extrap_deep),
        "inv_mmol_m2": float(inv),
        "avg_mmol_L": float(avg_L),
    }


def depth_integrate_to_casts(obs: pd.DataFrame) -> pd.DataFrame:
    """Convert depth samples into one integrated value per cast (date,row,col)."""

    required = {"date", "ewe_row", "ewe_col", "depth", "nitrogen"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Obs file missing required columns for depth integration: {sorted(missing)}")

    d = obs.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
    d["nitrogen"] = pd.to_numeric(d["nitrogen"], errors="coerce")

    # Defensive: treat exact 0 m as 0.1 m if present
    d.loc[d["depth"] == 0, "depth"] = 0.1

    out_rows: list[dict] = []

    for (date, rr, cc), g in d.groupby(["date", "ewe_row", "ewe_col"], dropna=False):
        prof = _profile_integrate(g["depth"].to_numpy(), g["nitrogen"].to_numpy())
        if not prof.get("ok", False):
            continue

        out_rows.append(
            {
                "date": date,
                "ewe_row": int(rr),
                "ewe_col": int(cc),
                "n_depths": int(prof["n_depths"]),
                "min_depth": float(prof["min_depth"]),
                "max_depth": float(prof["max_depth"]),
                "pad_surface": bool(prof.get("pad_surface", False)),
                "pad_deep": bool(prof.get("pad_deep", False)),
                "extrap_surface": bool(prof.get("extrap_surface", False)),
                "extrap_deep": bool(prof.get("extrap_deep", False)),
                "cast_nitrogen_int_mmol_m2": float(prof["inv_mmol_m2"]),
                "cast_nitrogen_avg_mmol_L": float(prof["avg_mmol_L"]),
            }
        )

    cast = pd.DataFrame(out_rows)

    # Optional cast filters
    if NU_MIN_DEPTHS_PER_CAST is not None and not cast.empty:
        cast = cast[cast["n_depths"] >= int(NU_MIN_DEPTHS_PER_CAST)]

    if NU_MIN_MAXDEPTH_M is not None and not cast.empty:
        cast = cast[cast["max_depth"] >= float(NU_MIN_MAXDEPTH_M)]

    return cast


def prepare_casts_for_matching(cast: pd.DataFrame) -> pd.DataFrame:
    """Preserve one row per depth-integrated cast for nearest-time model matching.

    This keeps the evaluation closer to the original observation support than the
    pooled biweekly workflow, while still carrying year/biweekly metadata forward
    so downstream summaries can aggregate upward if desired.
    """

    c = cast.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")

    need_cols = {"cast_nitrogen_int_mmol_m2", "cast_nitrogen_avg_mmol_L"}
    miss = need_cols - set(c.columns)
    if miss:
        raise ValueError(f"Cast table is missing required columns: {sorted(miss)}")

    c = c.dropna(subset=["date", "ewe_row", "ewe_col", "cast_nitrogen_int_mmol_m2", "cast_nitrogen_avg_mmol_L"]).copy()

    c["year"] = c["date"].dt.year
    c["day_of_year"] = c["date"].dt.dayofyear
    c["biweekly"] = ((c["day_of_year"] - 1) // 14 + 1).astype(int)

    c = c[(c["year"] >= int(NU_OBS_YR_ST)) & (c["year"] < int(NU_OBS_YR_EN))].copy()

    c["date_rep"] = c["date"]
    c["zmin_m"] = NU_ZMIN_M
    c["zmax_m"] = NU_ZMAX_M
    c["n_casts"] = 1
    c["mean_n_depths"] = c["n_depths"]
    c["mean_max_depth"] = c["max_depth"]
    c["frac_pad_surface"] = c["pad_surface"].astype(float)
    c["frac_pad_deep"] = c["pad_deep"].astype(float)
    c["frac_extrap_surface"] = c["extrap_surface"].astype(float)
    c["frac_extrap_deep"] = c["extrap_deep"].astype(float)

    c["avg_nitrogen_mmol_m2_obs"] = c["cast_nitrogen_int_mmol_m2"]
    c["avg_nitrogen_mmol_L_obs"] = c["cast_nitrogen_avg_mmol_L"]
    c["avg_nitrogen_gN_m2_obs"] = c["avg_nitrogen_mmol_m2_obs"] * (14.0067 / 1000.0)
    c["avg_nitrogen_umol_L_obs"] = c["avg_nitrogen_mmol_L_obs"] * 1000.0

    mode = str(getattr(cfg, "NU_OBS_VALUE_MODE", "integral")).lower()
    if mode not in {"integral", "average"}:
        raise ValueError(f"NU_OBS_VALUE_MODE must be 'integral' or 'average', got: {mode!r}")

    if mode == "integral":
        c["avg_nitrogen_primary_mmol_m2_obs"] = c["avg_nitrogen_mmol_m2_obs"]
    else:
        c["avg_nitrogen_primary_mmol_L_obs"] = c["avg_nitrogen_mmol_L_obs"]

    return c


# -----------------------------------------------------------------------------
# Pool casts to biweekly bins
# -----------------------------------------------------------------------------


def pool_casts_to_biweekly(cast: pd.DataFrame) -> pd.DataFrame:
    """Pool cast-level values to biweekly bins, optionally by cell.

    We always compute and output BOTH:
      - avg_nitrogen_mmol_m2_obs  (inventory over [zmin,zmax])
      - avg_nitrogen_mmol_L_obs   (depth-average over [zmin,zmax])

    For convenience, we also include a "primary" column name that depends on
    cfg.NU_OBS_VALUE_MODE:
      - avg_nitrogen_<UNITS>_obs
    (i.e., the primary already has units in the name).

    for backward-compatible alias, add it in 6b (recommended) rather than here.
    """

    c = cast.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")

    # Both cast-level metrics must exist
    # should be umol??
    need_cols = {"cast_nitrogen_int_mmol_m2", "cast_nitrogen_avg_mmol_L"}
    miss = need_cols - set(c.columns)
    if miss:
        raise ValueError(f"Cast table is missing required columns: {sorted(miss)}")

    #
    c = c.dropna(subset=["date", "ewe_row", "ewe_col", "cast_nitrogen_int_mmol_m2", "cast_nitrogen_avg_mmol_L"]).copy()

    c["year"] = c["date"].dt.year
    c["day_of_year"] = c["date"].dt.dayofyear
    c["biweekly"] = ((c["day_of_year"] - 1) // 14 + 1).astype(int)

    # Filter years
    c = c[(c["year"] >= int(NU_OBS_YR_ST)) & (c["year"] < int(NU_OBS_YR_EN))]

    if NU_POOL_BY_CELL:
        gcols = ["year", "biweekly", "ewe_row", "ewe_col"]
    else:
        gcols = ["year", "biweekly"]

    def _median_datetime(series: pd.Series) -> pd.Timestamp:
        s = pd.to_datetime(series, errors="coerce").dropna()
        if s.empty:
            return pd.NaT
        vals = s.astype("int64").to_numpy()  # avoids Series.view deprecation
        return pd.to_datetime(int(np.median(vals)))

    pooled = (
        c.groupby(gcols, as_index=False)
        .agg(
            date_rep=("date", _median_datetime),
            # Both obs representations (always present)
            avg_nitrogen_mmol_m2_obs=("cast_nitrogen_int_mmol_m2", lambda s: _agg_stat(s.dropna(), NU_OBS_BIN_AVG_TYPE)),
            avg_nitrogen_mmol_L_obs=("cast_nitrogen_avg_mmol_L", lambda s: _agg_stat(s.dropna(), NU_OBS_BIN_AVG_TYPE)),
            n_casts=("cast_nitrogen_int_mmol_m2", "count"),
            mean_n_depths=("n_depths", "mean"),
            mean_max_depth=("max_depth", "mean"),
            frac_pad_surface=("pad_surface", "mean"),
            frac_pad_deep=("pad_deep", "mean"),
            frac_extrap_surface=("extrap_surface", "mean"),
            frac_extrap_deep=("extrap_deep", "mean"),
        )
    )

    pooled["zmin_m"] = NU_ZMIN_M
    pooled["zmax_m"] = NU_ZMAX_M

    # Select primary obs column name with units in the name
    mode = str(getattr(cfg, "NU_OBS_VALUE_MODE", "integral")).lower()
    if mode not in {"integral", "average"}:
        raise ValueError(f"NU_OBS_VALUE_MODE must be 'integral' or 'average', got: {mode!r}")

    if mode == "integral":
        pooled["avg_nitrogen_primary_mmol_m2_obs"] = pooled["avg_nitrogen_mmol_m2_obs"]
        pooled["avg_nitrogen_gN_m2_obs"] = pooled["avg_nitrogen_mmol_m2_obs"] * (14.0067 / 1000.0)
    else:
        pooled["avg_nitrogen_primary_mmol_L_obs"] = pooled["avg_nitrogen_mmol_L_obs"]
    pooled["avg_nitrogen_umol_L_obs"] = pooled["avg_nitrogen_mmol_L_obs"] * 1000.0

    return pooled


# -----------------------------------------------------------------------------
# Model extraction at bins
# -----------------------------------------------------------------------------


def extract_model_at_bins(ds: xr.Dataset, bins: pd.DataFrame, vars_to_get: list[str]) -> pd.DataFrame:
    """Extract model vars at (model_time,row,col) for each binned obs row.

    IMPORTANT: ewe_row/ewe_col in pipeline are 1-based. We subtract 1 here.
    """

    time_indexer = pd.Index(pd.to_datetime(ds["time"].values))
    out_rows: list[dict[str, float]] = []

    for r in bins.itertuples(index=False):
        if pd.isna(r.model_time):
            out_rows.append({"model_t_idx": np.nan, **{v: np.nan for v in vars_to_get}})
            continue

        t_idx = int(time_indexer.get_indexer([pd.to_datetime(r.model_time)], method="nearest")[0])

        # If NU_POOL_BY_CELL=False, still keep t_idx for non-spatial lookups
        if not NU_POOL_BY_CELL:
            out_rows.append({"model_t_idx": t_idx, **{v: np.nan for v in vars_to_get}})
            continue

        rr = int(r.ewe_row) - 1
        cc = int(r.ewe_col) - 1

        out = {"model_t_idx": t_idx}
        for v in vars_to_get:
            if v not in ds.data_vars:
                out[v] = np.nan
                continue

            if not (0 <= rr < ds.dims["row"] and 0 <= cc < ds.dims["col"]):
                out[v] = np.nan
                continue

            try:
                out[v] = float(ds[v].isel(time=t_idx, row=rr, col=cc).values)
            except Exception:
                out[v] = np.nan

        out_rows.append(out)

    return pd.concat([bins.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_nut_matched() -> None:
    _ensure_dir(OUTPUT_DIR_EVAL)

    # Load obs
    obs = pd.read_csv(NUTRIENTS_CSV)

    required = {"date", "ewe_row", "ewe_col", "depth", "nitrogen"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Nutrient obs file is missing required columns: {sorted(missing)}")

    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date", "ewe_row", "ewe_col", "depth", "nitrogen"]).copy()

    obs["ewe_row"] = obs["ewe_row"].astype(int)
    obs["ewe_col"] = obs["ewe_col"].astype(int)

    # Depth-integrate to cast-level
    cast = depth_integrate_to_casts(obs)
    cast.to_csv(OUT_CAST_CSV, index=False)
    print(f"Wrote cast-level depth-integrated obs: {OUT_CAST_CSV} ({len(cast):,} rows)")

    # Preserve cast-level rows for closer per-sample matching
    cast_match_obs = prepare_casts_for_matching(cast)

    # Pool casts to biweekly bins
    obs_bins = pool_casts_to_biweekly(cast)
    obs_bins.to_csv(OUT_OBS_BIWEEK_CSV, index=False)
    print(f"Wrote pooled biweekly obs: {OUT_OBS_BIWEEK_CSV} ({len(obs_bins):,} rows)")

    print('Writing cast-level matched data...')

    # Open model
    ds = xr.open_dataset(ECOSPACE_NC)

    # Ensure datetime time coordinate
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))

    model_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    vars_to_get = model_vars_to_extract(ds)

    # Match depth-integrated casts directly to nearest model time
    casts2 = nearest_time_join(cast_match_obs, date_col="date", model_times=model_times, tol_days=TIME_TOL_DAYS)
    matched_cast = extract_model_at_bins(ds, casts2, vars_to_get)
    matched_cast.to_csv(OUT_MATCHED_CAST_CSV, index=False)
    print(f"Wrote cast-level matched table: {OUT_MATCHED_CAST_CSV} ({len(matched_cast):,} rows)")

    print('Writing biweekly matched data...')

    # Match bins to nearest model time using date_rep
    bins2 = nearest_time_join(obs_bins, date_col="date_rep", model_times=model_times, tol_days=TIME_TOL_DAYS)

    # Extract model biomass vars
    matched = extract_model_at_bins(ds, bins2, vars_to_get)
    matched.to_csv(OUT_MATCHED_BIWEEK_CSV, index=False)
    print(f"Wrote biweekly matched table: {OUT_MATCHED_BIWEEK_CSV} ({len(matched):,} rows)")

    series = list(getattr(cfg, "NU_MODEL_SERIES", ["matched"]))

    if "box" in series:
        box = tuple(getattr(cfg, "NU_MODEL_SAMPLE_BOX"))
        stride = int(getattr(cfg, "NU_MODEL_SAMPLE_STRIDE", 1))
        yr_st = int(getattr(cfg, "NU_MODEL_SAMPLE_YR_ST", NU_OBS_YR_ST))
        yr_en = int(getattr(cfg, "NU_MODEL_SAMPLE_YR_EN", NU_OBS_YR_EN))
        biweek_max = int(getattr(cfg, "NU_BIWEEK_MAX", 26))

        box_df = extract_model_box_biweekly(
            ds,
            years=range(yr_st, yr_en),
            biweek_max=biweek_max,
            box=box,
            stride=stride,
            vars_to_get=vars_to_get,
        )
        box_df.to_csv(OUT_BOX_BIWEEK_CSV, index=False)
        print(f"Wrote biweekly box-sampled model table: {OUT_BOX_BIWEEK_CSV} ({len(box_df):,} rows)")



if __name__ == "__main__":
    run_nut_matched()
