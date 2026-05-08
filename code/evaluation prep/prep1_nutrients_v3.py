"""
prep1_nutrients_v3_defensible_0to20_hardcoded.py

Created by: G Oldford (original prep1_nutrients.py, 2026)

Purpose
-------
Prepare nitrate+nitrite observations from multiple sources (e.g., CSOP + IOS),
match samples to the Ecospace grid, and compute a defensible 0–20 m (actually
0.1–20 m by default) depth-mean concentration and depth-integrated inventory.

This version is identical in method to the v2 revision, but it is configured
via a simple "USER CONFIG" block at the top of the file (no CLI arguments).

Method alignment
----------------
We compute 0–20 m metrics the SAME WAY as ecospace_eval_6a_nutrientsmatched.py:

1) Cast-level depth integration over [ZMIN, ZMAX] using trapezoids
   with endpoint padding/extrapolation controls.
2) Pool casts to biweekly bins by year (and optionally by grid cell).
3) Summarize an annual-cycle climatology and compute annual means from that
   climatology (equal weight per biweekly bin).

Units
-----
Input nitrogen is assumed to be nitrate+nitrite in umol/L (aka uM).
Numerically, umol/L == mmol/m^3, so we can integrate in "mmol/m^3" space and
obtain inventories in mmol/m^2.

Outputs include:
- cast inventory: mmol/m^2 and g N m^-2
- cast depth-mean: mmol/L and umol/L
- biweekly pooled values and climatology plots

Notes
-----
- OBS_YEAR_END is END-EXCLUSIVE (2019 means through 2018), matching later scripts.
- If you want a winter baseline (pre-bloom), use the printed winter statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# =============================================================================
# USER CONFIG (edit these values)
# =============================================================================

# --- Inputs
BASE_P = r"C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/29. Oceanographic Atmospheric/"
GRID_P = r"C:/Users/Greig/Documents/github/Ecosystem-Model-Data-Framework/data/basemap"
CSOP_CSV = rf"{BASE_P}\\CSOP Nutrients\\MODIFIED\\CitSci_Nutrients_2015-2023_justEwEDomain.csv"  # must include: date,latitude,longitude,depth,no3
IOS_CSV = rf"{BASE_P}\\IOS Rosette Bottle Data\\MODIFIED\\IOS_BOT_1978to2019_JustEwEArea.csv"             # must include: time OR date, latitude, longitude, depth, NTRZAAZ1
ECOSPACE_GRID_CSV = rf"{GRID_P}\\Ecospace_grid_20210208_rowscols.csv"  # must include: latitude, longitude, EWE_row, EWE_col (or lat/lon)

# --- Outputs
OUT_DIR = Path(r".//..//..//..//data/evaluation")
FIG_DIR = Path(r"..//..//..//..//figs")

# --- Observation years to summarize (END-EXCLUSIVE)
OBS_YEAR_START = 2012
OBS_YEAR_END = 2019  # 2019 means through 2018

# --- Depth window (m)
ZMIN_M = 0.1
ZMAX_M = 20.0

# --- Endpoint handling (same intent as ecospace_eval_6a_nutrientsmatched.py)
REQUIRE_FULL_COVERAGE = True
ALLOW_SINGLE_DEPTH = False
DEPTH_EXTRAPOLATE = False
SURFACE_PAD_TOL_M = 0.11  # permits 0 m observations to represent 0.1 m window start
DEEP_PAD_TOL_M = 0.0      # strict deep endpoint; set to ~0.25 if many casts stop at 19.8–20.0 m

# --- Biweekly pooling
BIWEEK_DAYS = 14
POOL_BY_CELL = False      # True -> compute per (year,biweek,row,col) before climatology
BIN_AGG = "median"         # "median" or "mean" within each biweekly bin

# --- Winter baseline definition (biweekly bins; 1..26)
WINTER_BINS = {1, 2, 3, 4, 5, 6, 24, 25, 26}

# =============================================================================

# Conversion constants
G_PER_MOL_N = 14.0067
MMOL_TO_MOL = 1.0 / 1000.0


@dataclass(frozen=True)
class IntegratorCfg:
    zmin_m: float
    zmax_m: float
    require_full_coverage: bool
    allow_single_depth: bool
    depth_extrapolate: bool
    surface_pad_tol_m: float
    deep_pad_tol_m: float


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _read_csop(path: Path, zmin_m: float) -> pd.DataFrame:
    """Read CSOP CSV and return standardized columns."""
    df = pd.read_csv(path)
    sub = df[["date", "latitude", "longitude", "depth", "no3"]].copy()
    sub = sub.rename(columns={"no3": "nitrogen"})
    sub["dataset"] = "csop"
    sub.loc[sub["depth"] == 0, "depth"] = zmin_m
    return sub


def _read_ios(path: Path) -> pd.DataFrame:
    """Read IOS bottle CSV and return standardized columns."""
    df = pd.read_csv(path)

    # clean lat/lon formatting (remove leading quotes)
    df["latitude"] = df["latitude"].astype(str).str.replace("'", "", regex=False).astype(float)
    df["longitude"] = df["longitude"].astype(str).str.replace("'", "", regex=False).astype(float)

    # Convert time -> date if available
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        raise ValueError("IOS file must contain 'time' or 'date' column.")

    # NTRZAAZ1 used as nitrate+nitrite
    sub = df[["date", "latitude", "longitude", "depth", "NTRZAAZ1"]].copy()
    sub = sub.rename(columns={"NTRZAAZ1": "nitrogen"})
    sub["dataset"] = "ios"
    return sub


def _match_to_ecospace_grid(samples: pd.DataFrame, grid_csv: Path) -> pd.DataFrame:
    """Assign nearest Ecospace (row,col) to each sample using KDTree."""
    g = pd.read_csv(grid_csv).copy()
    if {"lat", "lon"}.issubset(g.columns):
        g = g.rename(columns={"lat": "latitude", "lon": "longitude"})

    required = {"latitude", "longitude", "EWE_row", "EWE_col"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Ecospace grid CSV missing columns: {sorted(missing)}")

    tree = cKDTree(g[["latitude", "longitude"]].values)
    coords = samples[["latitude", "longitude"]].values
    distances, indices = tree.query(coords, k=1)

    out = samples.copy()
    out["ewe_row"] = g.iloc[indices]["EWE_row"].to_numpy()
    out["ewe_col"] = g.iloc[indices]["EWE_col"].to_numpy()
    out["ewe_lat"] = g.iloc[indices]["latitude"].to_numpy()
    out["ewe_lon"] = g.iloc[indices]["longitude"].to_numpy()
    out["ewe_dist_km"] = distances * 111.0  # rough: 1 degree ~ 111 km
    return out


# -----------------------------------------------------------------------------
# Depth integration (same intent as ecospace_eval_6a_nutrientsmatched.py)
# -----------------------------------------------------------------------------

def profile_integrate(depth_m: np.ndarray, conc_umol_L: np.ndarray, cfg: IntegratorCfg) -> dict:
    """Integrate one profile over [zmin,zmax] using trapezoids."""
    zmin, zmax = cfg.zmin_m, cfg.zmax_m
    if not (zmax > zmin):
        return {"ok": False}

    m = np.isfinite(depth_m) & np.isfinite(conc_umol_L)
    depth = np.asarray(depth_m[m], dtype=float)
    conc = np.asarray(conc_umol_L[m], dtype=float)
    if depth.size == 0:
        return {"ok": False}

    order = np.argsort(depth)
    depth = depth[order]
    conc = conc[order]

    # merge duplicate depths
    if depth.size != np.unique(depth).size:
        tmp = pd.DataFrame({"depth": depth, "conc": conc}).groupby("depth", as_index=False)["conc"].mean()
        depth = tmp["depth"].to_numpy(dtype=float)
        conc = tmp["conc"].to_numpy(dtype=float)

    n_depths = int(depth.size)
    min_d = float(depth.min())
    max_d = float(depth.max())

    surface_gap = max(0.0, min_d - zmin)
    deep_gap = max(0.0, zmax - max_d)

    pad_surface = surface_gap > 0 and surface_gap <= cfg.surface_pad_tol_m
    pad_deep = deep_gap > 0 and deep_gap <= cfg.deep_pad_tol_m

    covers_surface = (min_d <= zmin) or pad_surface
    covers_deep = (max_d >= zmax) or pad_deep

    if cfg.require_full_coverage and (not covers_surface or not covers_deep):
        return {"ok": False, "n_depths": n_depths, "min_depth": min_d, "max_depth": max_d}

    if n_depths == 1:
        if not cfg.allow_single_depth:
            return {"ok": False, "n_depths": n_depths, "min_depth": min_d, "max_depth": max_d}
        c0_m3 = float(conc[0])  # umol/L == mmol/m^3 numerically
        inv = c0_m3 * (zmax - zmin)
        avg_mmol_L = (inv / (zmax - zmin)) / 1000.0
        return {"ok": True, "inv_mmol_m2": float(inv), "avg_mmol_L": float(avg_mmol_L)}

    need_extrap_surface = surface_gap > cfg.surface_pad_tol_m
    need_extrap_deep = deep_gap > cfg.deep_pad_tol_m
    if (need_extrap_surface or need_extrap_deep) and (not cfg.depth_extrapolate):
        return {"ok": False, "n_depths": n_depths, "min_depth": min_d, "max_depth": max_d}

    mask = (depth >= zmin) & (depth <= zmax)
    depth_in = depth[mask]
    conc_in = conc[mask]

    def interp(z: float) -> float:
        return float(np.interp(z, depth, conc))  # constant extrap at ends

    if depth_in.size == 0 or depth_in[0] > zmin:
        depth_in = np.insert(depth_in, 0, zmin)
        conc_in = np.insert(conc_in, 0, interp(zmin))
    if depth_in[-1] < zmax:
        depth_in = np.append(depth_in, zmax)
        conc_in = np.append(conc_in, interp(zmax))

    # integrate "mmol/m^3" (numeric == umol/L) over meters -> mmol/m^2
    inv = float(np.trapezoid(conc_in, depth_in))
    avg_mmol_L = (inv / (zmax - zmin)) / 1000.0

    return {"ok": True, "inv_mmol_m2": float(inv), "avg_mmol_L": float(avg_mmol_L)}


def depth_integrate_to_casts(samples: pd.DataFrame, cfg: IntegratorCfg) -> pd.DataFrame:
    """Convert depth samples into one integrated value per (date,ewe_row,ewe_col)."""
    required = {"date", "ewe_row", "ewe_col", "depth", "nitrogen"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Samples missing required columns: {sorted(missing)}")

    d = samples.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
    d["nitrogen"] = pd.to_numeric(d["nitrogen"], errors="coerce")
    d = d.dropna(subset=["date", "ewe_row", "ewe_col", "depth", "nitrogen"]).copy()
    d.loc[d["depth"] == 0, "depth"] = cfg.zmin_m

    out_rows: list[dict] = []
    for (date, rr, cc), g in d.groupby(["date", "ewe_row", "ewe_col"], dropna=False):
        prof = profile_integrate(g["depth"].to_numpy(), g["nitrogen"].to_numpy(), cfg)
        if not prof.get("ok", False):
            continue

        inv_mmol_m2 = float(prof["inv_mmol_m2"])
        avg_mmol_L = float(prof["avg_mmol_L"])

        out_rows.append({
            "date": pd.to_datetime(date),
            "ewe_row": int(rr),
            "ewe_col": int(cc),
            "n_depths": int(len(g)),
            "min_depth": float(np.nanmin(g["depth"].to_numpy())),
            "max_depth": float(np.nanmax(g["depth"].to_numpy())),
            "cast_nitrogen_int_mmol_m2": inv_mmol_m2,
            "cast_nitrogen_int_gN_m2": inv_mmol_m2 * (G_PER_MOL_N * MMOL_TO_MOL),
            "cast_nitrogen_avg_mmol_L": avg_mmol_L,
            "cast_nitrogen_avg_umol_L": avg_mmol_L * 1000.0,
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    return out.sort_values(["date", "ewe_row", "ewe_col"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Biweekly pooling + climatology
# -----------------------------------------------------------------------------

def pool_to_biweekly(casts: pd.DataFrame, pool_by_cell: bool, agg: str) -> pd.DataFrame:
    """Pool cast-level values to (year,biweekly[,row,col])."""
    c = casts.copy()
    c["date"] = pd.to_datetime(c["date"])
    c["year"] = c["date"].dt.year
    c["day_of_year"] = c["date"].dt.dayofyear
    c["biweekly"] = ((c["day_of_year"] - 1) // BIWEEK_DAYS + 1).astype(int)

    # Filter years (end-exclusive)
    c = c[(c["year"] >= int(OBS_YEAR_START)) & (c["year"] < int(OBS_YEAR_END))].copy()

    gcols = ["year", "biweekly", "ewe_row", "ewe_col"] if pool_by_cell else ["year", "biweekly"]

    if agg not in {"median", "mean"}:
        raise ValueError("BIN_AGG must be 'median' or 'mean'")

    def _aggfun(x: pd.Series) -> float:
        return float(np.nanmedian(x.to_numpy())) if agg == "median" else float(np.nanmean(x.to_numpy()))

    pooled = c.groupby(gcols).agg(
        n_casts=("cast_nitrogen_int_mmol_m2", "size"),
        avg_nitrogen_int_mmol_m2=("cast_nitrogen_int_mmol_m2", _aggfun),
        avg_nitrogen_int_gN_m2=("cast_nitrogen_int_gN_m2", _aggfun),
        avg_nitrogen_umol_L=("cast_nitrogen_avg_umol_L", _aggfun),
    ).reset_index()

    return pooled


def climatology_from_biweekly(pooled: pd.DataFrame, value_col: str, complete_bins: bool = True) -> pd.DataFrame:
    """Biweekly climatology (mean & std across years) from pooled data."""
    clim = pooled.groupby("biweekly")[value_col].agg(["mean", "std", "count"]).reset_index()
    clim = clim.rename(columns={
        "mean": f"{value_col}_clim_mean",
        "std": f"{value_col}_clim_std",
        "count": "n_year_bins",
    })

    if complete_bins:
        all_bins = pd.DataFrame({"biweekly": np.arange(1, 27, dtype=int)})
        clim = all_bins.merge(clim, on="biweekly", how="left")

    return clim.sort_values("biweekly").reset_index(drop=True)


def annual_mean_from_climatology(clim: pd.DataFrame, clim_mean_col: str) -> float:
    """Equal-weight mean across biweekly climatology bins present."""
    return float(np.nanmean(clim[clim_mean_col].to_numpy(dtype=float)))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_climatology(clim: pd.DataFrame,
                     mean_col: str,
                     std_col: str,
                     ylabel: str,
                     title: str,
                     out_png: Path) -> None:
    x = clim["biweekly"].to_numpy(dtype=float)
    mu = clim[mean_col].to_numpy(dtype=float)
    sd = clim[std_col].to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, mu, marker="o")
    if np.isfinite(sd).any():
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.2)
    plt.xlabel("Biweekly index (1..26)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _assert_paths() -> None:
    bad = []
    for name, val in {
        "CSOP_CSV": CSOP_CSV,
        "IOS_CSV": IOS_CSV,
        "ECOSPACE_GRID_CSV": ECOSPACE_GRID_CSV,
        "OUT_DIR": str(OUT_DIR),
        "FIG_DIR": str(FIG_DIR),
    }.items():
        if "PATH_TO" in str(val):
            bad.append(name)
    if bad:
        raise ValueError(
            "Please edit USER CONFIG at the top of the script. Unset placeholder(s): "
            + ", ".join(bad)
        )


def main() -> None:
    _assert_paths()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = IntegratorCfg(
        zmin_m=float(ZMIN_M),
        zmax_m=float(ZMAX_M),
        require_full_coverage=bool(REQUIRE_FULL_COVERAGE),
        allow_single_depth=bool(ALLOW_SINGLE_DEPTH),
        depth_extrapolate=bool(DEPTH_EXTRAPOLATE),
        surface_pad_tol_m=float(SURFACE_PAD_TOL_M),
        deep_pad_tol_m=float(DEEP_PAD_TOL_M),
    )

    # --- Load + standardize
    csop = _read_csop(Path(CSOP_CSV), zmin_m=cfg.zmin_m)
    ios = _read_ios(Path(IOS_CSV))
    combined = pd.concat([csop, ios], ignore_index=True)

    # Coerce types + drop invalid
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
    combined["depth"] = pd.to_numeric(combined["depth"], errors="coerce")
    combined["nitrogen"] = pd.to_numeric(combined["nitrogen"], errors="coerce")
    combined = combined.dropna(subset=["date", "latitude", "longitude", "depth", "nitrogen"]).copy()

    # --- Match to Ecospace grid
    combined = _match_to_ecospace_grid(combined, Path(ECOSPACE_GRID_CSV))

    # --- Save sample-level combined (input to later evaluation scripts)
    out_samples = OUT_DIR / "nutrients_ios_csop_combined_sampled.csv"
    combined.to_csv(out_samples, index=False)
    print(f"[OK] Wrote sample-level combined file: {out_samples}")

    # --- Depth integrate to casts
    casts = depth_integrate_to_casts(combined, cfg)
    out_casts = OUT_DIR / "nutrients_ios_csop_cast_depthint_0p1to20m.csv"
    casts.to_csv(out_casts, index=False)
    print(f"[OK] Wrote cast-level depth-integrated file: {out_casts}")
    if casts.empty:
        print("[WARN] No casts passed the integration filters. Consider relaxing DEEP_PAD_TOL_M or REQUIRE_FULL_COVERAGE.")
        return

    # --- Pool to biweekly and build climatologies
    pooled = pool_to_biweekly(casts, pool_by_cell=POOL_BY_CELL, agg=BIN_AGG)
    out_biweek = OUT_DIR / f"nutrients_obs_biweekly_{OBS_YEAR_START}_{OBS_YEAR_END-1}_{BIN_AGG}.csv"
    pooled.to_csv(out_biweek, index=False)
    print(f"[OK] Wrote biweekly pooled file: {out_biweek}")

    clim_umol = climatology_from_biweekly(pooled, "avg_nitrogen_umol_L", complete_bins=True)
    clim_gN = climatology_from_biweekly(pooled, "avg_nitrogen_int_gN_m2", complete_bins=True)

    annual_umol = annual_mean_from_climatology(clim_umol, "avg_nitrogen_umol_L_clim_mean")
    annual_gN = annual_mean_from_climatology(clim_gN, "avg_nitrogen_int_gN_m2_clim_mean")

    winter_umol = float(np.nanmean(clim_umol.loc[clim_umol["biweekly"].isin(WINTER_BINS), "avg_nitrogen_umol_L_clim_mean"]))
    winter_gN = float(np.nanmean(clim_gN.loc[clim_gN["biweekly"].isin(WINTER_BINS), "avg_nitrogen_int_gN_m2_clim_mean"]))

    # --- Report
    summary_txt = OUT_DIR / f"nutrients_obs_summary_{OBS_YEAR_START}_{OBS_YEAR_END-1}.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Nutrients observation summary (0.1–20 m)\n")
        f.write(f"Years: {OBS_YEAR_START}–{OBS_YEAR_END-1} (end-exclusive input)\n")
        f.write(f"Depth window: {cfg.zmin_m}–{cfg.zmax_m} m\n")
        f.write(f"Biweekly agg: {BIN_AGG}\n")
        f.write(f"Pool-by-cell: {POOL_BY_CELL}\n")
        f.write("\n--- Annual mean (equal-weight across biweekly climatology bins) ---\n")
        f.write(f"Annual mean conc: {annual_umol:.2f} umol/L\n")
        f.write(f"Annual mean inventory: {annual_gN:.3f} g N m-2\n")
        f.write("\n--- Winter baseline (configured WINTER_BINS) ---\n")
        f.write(f"Winter mean conc: {winter_umol:.2f} umol/L\n")
        f.write(f"Winter mean inventory: {winter_gN:.3f} g N m-2\n")
        f.write("\nNote: If some biweekly bins are missing, annual means ignore NaNs.\n")

    print(f"[OK] Wrote summary: {summary_txt}")
    print(f"Annual mean (climatology): {annual_umol:.2f} umol/L, {annual_gN:.3f} g N m-2")
    print(f"Winter baseline: {winter_umol:.2f} umol/L, {winter_gN:.3f} g N m-2")

    # --- Plots
    plot_climatology(
        clim_umol,
        mean_col="avg_nitrogen_umol_L_clim_mean",
        std_col="avg_nitrogen_umol_L_clim_std",
        ylabel="Depth-mean nitrate+nitrite (umol/L)",
        title=f"Obs biweekly climatology (0.1–20 m) | {OBS_YEAR_START}–{OBS_YEAR_END-1}",
        out_png=FIG_DIR / f"obs_nutrients_climatology_umolL_{OBS_YEAR_START}_{OBS_YEAR_END-1}.png",
    )
    plot_climatology(
        clim_gN,
        mean_col="avg_nitrogen_int_gN_m2_clim_mean",
        std_col="avg_nitrogen_int_gN_m2_clim_std",
        ylabel="Depth-integrated nitrate+nitrite (g N m-2)",
        title=f"Obs biweekly climatology inventory (0.1–20 m) | {OBS_YEAR_START}–{OBS_YEAR_END-1}",
        out_png=FIG_DIR / f"obs_nutrients_climatology_gNm2_{OBS_YEAR_START}_{OBS_YEAR_END-1}.png",
    )

    print(f"[OK] Wrote figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
