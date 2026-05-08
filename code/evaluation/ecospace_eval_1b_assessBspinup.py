
import os
from typing import Dict, List

import xarray as xr
import numpy as np
import pandas as pd

import ecospace_eval_config as cfg

# ============================================================
# CONFIG (edit in ecospace_eval_config.py to keep things tidy)
# ============================================================

# Path to Ecospace NetCDF
ECOSPACE_NC_FILENAME = getattr(cfg, "NC_FILENAME", None)
ECOSPACE_NC_PATH_OUT = getattr(cfg, "NC_PATH_OUT", None)

# If  prefer an alternative naming,  can instead define in cfg:
#   ECOSPACE_SC, ECOSPACE_RN_STR_YR, ECOSPACE_RN_END_YR
# and the helper below will try to reconstruct.

def build_ecospace_nc_path() -> str:
    """Infer Ecospace NetCDF path using ecospace_eval_config."""
    # Direct (preferred)
    if ECOSPACE_NC_FILENAME and ECOSPACE_NC_PATH_OUT:
        pf = os.path.join(ECOSPACE_NC_PATH_OUT, ECOSPACE_NC_FILENAME)
        if os.path.exists(pf):
            return pf

    # Fallback: reconstruct from scenario + years if present
    scen = getattr(cfg, "ECOSPACE_SC", None)
    y0 = getattr(cfg, "ECOSPACE_RN_STR_YR", None)
    y1 = getattr(cfg, "ECOSPACE_RN_END_YR", None)
    if scen and y0 and y1 and ECOSPACE_NC_PATH_OUT:
        pf = os.path.join(ECOSPACE_NC_PATH_OUT, f"{scen}_{y0}-{y1}.nc")
        if os.path.exists(pf):
            return pf

    raise FileNotFoundError(
        "Could not locate Ecospace NetCDF. "
        "Check NC_FILENAME / NC_PATH_OUT / ECOSPACE_SC in ecospace_eval_config.py."
    )

# --- Periods (match Ecosim spin-up/full-run logic) ---

# Define these in ecospace_eval_config.py, e.g.:
#   ES_START_SPINUP = "1978-01-02"
#   ES_END_SPINUP   = "1982-12-31"
#   ES_START_FULL   = "1983-01-01"
#   ES_END_FULL     = "2018-12-30"
ES_START_SPINUP = pd.to_datetime(getattr(cfg, "ES_START_SPINUP", "1978-01-02"))
ES_END_SPINUP   = pd.to_datetime(getattr(cfg, "ES_END_SPINUP",   "1982-12-31"))
ES_START_FULL   = pd.to_datetime(getattr(cfg, "ES_START_FULL",   "1983-01-01"))
ES_END_FULL     = pd.to_datetime(getattr(cfg, "ES_END_FULL",     "2018-12-30"))

# --- Groups and Ecopath initial values ---

# List of group variable names in the Ecospace NetCDF to evaluate.
# e.g.:
#   ES_GROUPS_RELB = ["PP1-DIA", "PP2-NAN", "PP3-PIC"]
ES_GROUPS_RELB: List[str] = getattr(
    cfg,
    "ES_GROUPS_RELB",
    ["PP1-DIA", "PP2-NAN", "PP3-PIC"]  # fallback; override in cfg
)

# Mapping from group variable name -> Ecopath initial biomass (same units as Ecospace)
# e.g.:
#   ES_GROUPS_ECOPATH_B = {
#       "PP1-DIA": 2.31,
#       "PP2-NAN": 1.50,
#       "PP3-PIC": 0.35,
#   }
ES_GROUPS_ECOPATH_B: Dict[str, float] = getattr(
    cfg,
    "ES_GROUPS_ECOPATH_B",
    {}
)

# --- Output CSV (optional) ---
ES_PP_MULT_CSV = getattr(
    cfg,
    "ES_PP_MULT_CSV",
    os.path.join(getattr(cfg, "EVALOUT_P", ".."), "ecospace_PP_multipliers_2D.csv")
)

# --- Masking options ---

# Name of depth/bathymetry variable in the Ecospace NC, if present.
ES_DEPTH_VAR = getattr(cfg, "ES_DEPTH_VAR", "depth")

# If True, restrict averages to cells with depth > 0.
ES_MASK_POSITIVE_DEPTH = getattr(cfg, "ES_MASK_POSITIVE_DEPTH", True)


# ============================================================
# Core utilities
# ============================================================

def get_mask(ds: xr.Dataset) -> xr.DataArray:
    """
    Construct a 2D boolean mask for valid wet cells.

    Priority:
      1) if ES_DEPTH_VAR exists → depth > 0
      2) else if first group exists → any non-NaN at first timestep
      3) else → all True (no mask)
    """
    # Case 1: depth-based mask
    if ES_DEPTH_VAR in ds:
        dep = ds[ES_DEPTH_VAR]
        # Broadcast/handle shape flexibly; expect [row, col] or [time,row,col]
        if "time" in dep.dims:
            dep2d = dep.isel(time=0)
        else:
            dep2d = dep
        mask = dep2d > 0
        return mask.astype(bool)

    # Case 2: infer from first group
    for g in ES_GROUPS_RELB:
        if g in ds:
            da = ds[g]
            if "time" in da.dims:
                mask = ~np.isnan(da.isel(time=0))
            else:
                mask = ~np.isnan(da)
            return mask.astype(bool)

    # Case 3: fallback all True
    # Need dims; use lat/lon or row/col if present
    if ("row" in ds.dims) and ("col" in ds.dims):
        return xr.DataArray(np.ones((ds.dims["row"], ds.dims["col"]), dtype=bool),
                            dims=("row", "col"))
    raise ValueError("Could not infer spatial mask: no depth var, no PP vars, no (row,col) dims.")


def spatial_mean_timeseries(ds: xr.Dataset, varname: str, mask: xr.DataArray) -> pd.Series:
    """
    Compute spatially averaged time series for varname over cells where mask is True.

    Returns:
        pandas.Series indexed by datetime64, name=varname
    """
    if varname not in ds:
        raise KeyError(f"Variable '{varname}' not found in Ecospace dataset.")

    da = ds[varname]

    if "time" not in da.dims:
        raise ValueError(f"Variable '{varname}' has no 'time' dimension; cannot form time series.")

    # Align mask dims (assume mask is [row, col])
    if not set(mask.dims).issubset(set(da.dims)):
        raise ValueError(f"Mask dims {mask.dims} not subset of {da.dims} for '{varname}'.")

    # Apply mask and average over spatial dims
    spatial_dims = [d for d in da.dims if d != "time"]
    da_masked = da.where(mask)
    ts = da_masked.mean(dim=spatial_dims, skipna=True)

    # Convert to pandas Series
    t = pd.to_datetime(ts["time"].values)
    return pd.Series(ts.values, index=t, name=varname)


def mean_over_period(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Mean of ts between [start, end], inclusive."""
    sel = ts[(ts.index >= start) & (ts.index <= end)]
    if sel.empty:
        return float("nan")
    return float(sel.mean())


# ============================================================
# Main calc
# ============================================================

def run_ecospace_RELB_eval_2D():
    ecospace_nc = build_ecospace_nc_path()
    print(f"Using Ecospace file: {ecospace_nc}")

    ds = xr.open_dataset(ecospace_nc)

    # Build a single global mask (wet cells)
    mask = get_mask(ds)

    print("Computing spatially averaged time series for groups:")
    for g in ES_GROUPS_RELB:
        print(f" - {g}")

    print("\nSpin-up period : ", ES_START_SPINUP.date(), "to", ES_END_SPINUP.date())
    print("Full-run period: ", ES_START_FULL.date(),   "to", ES_END_FULL.date())
    print("")

    rows = []
    print("Group\tInitial\tSpinupMean\tSpinupMult\tFullRunMean\tFullRunMult")

    for group in ES_GROUPS_RELB:

        if group not in ES_GROUPS_ECOPATH_B:
            print(f"Warning: Initial value for group {group} not found in ES_GROUPS_ECOPATH_B.")
            initial_val = float("nan")
        else:
            initial_val = float(ES_GROUPS_ECOPATH_B[group])

        try:
            ts = spatial_mean_timeseries(ds, group, mask)
        except Exception as e:
            print(f"Error building timeseries for {group}: {e}")
            rows.append({
                "Group": group,
                "Initial": initial_val,
                "SpinupMean": np.nan,
                "SpinupMult": np.nan,
                "FullRunMean": np.nan,
                "FullRunMult": np.nan,
                "Note": f"ERROR: {e}"
            })
            continue

        spinup_mean = mean_over_period(ts, ES_START_SPINUP, ES_END_SPINUP)
        fullrun_mean = mean_over_period(ts, ES_START_FULL, ES_END_FULL)

        spinup_mult = (initial_val / spinup_mean) if (initial_val not in [0, np.nan] and not np.isnan(spinup_mean)) else np.nan
        fullrun_mult = (initial_val / fullrun_mean) if (initial_val not in [0, np.nan] and not np.isnan(fullrun_mean)) else np.nan

        print(f"{group}\t"
              f"{initial_val:.4f}\t"
              f"{spinup_mean:.4f}\t"
              f"{spinup_mult:.4f}\t"
              f"{fullrun_mean:.4f}\t"
              f"{fullrun_mult:.4f}")

        rows.append({
            "Group": group,
            "Initial": initial_val,
            "SpinupMean": spinup_mean,
            "SpinupMult": spinup_mult,
            "FullRunMean": fullrun_mean,
            "FullRunMult": fullrun_mult,
        })

    # Optional: write CSV summary
    if rows:
        df_out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(ES_PP_MULT_CSV), exist_ok=True)
        df_out.to_csv(ES_PP_MULT_CSV, index=False)
        print(f"\nSaved Ecospace PP multiplier summary to: {ES_PP_MULT_CSV}")


if __name__ == "__main__":
    run_ecospace_RELB_eval_2D()