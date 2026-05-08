
"""
Fit piece‑wise Gaussian thermal performance curves (TPCs) to zooplankton
abundance observations matched with model temperatures.

* Author: Greig Oldford
* Created: 2025‑07‑30

Workflow
========
1. Read a wide CSV that already contains zooplankton groups and temperature
   columns.
2. Reshape to long format (one row per observation × group).
3. Fit the Salish‑Sea‑style piece‑wise Gaussian TPC to each group using
   non‑linear least squares (SciPy).
4. Save parameter table and diagnostic plots.

Dependencies: pandas, numpy, scipy, matplotlib.
Install with e.g.:
    pip install pandas numpy scipy matplotlib

to test with args etc
in pycharm, click debug
in bottom left change debug config
put this in as param (eg)
--csv "C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/4. Zooplankton/Zoop_Perryetal_2021/MODIFIED/Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv" --compare Temp_0to10m Temp_30to40m Temp_150toBot
command line:
python fit_zooplankton_TPC.py --csv zoop_temp_wide.csv
       --compare Temp_0to10m Temp_30to40m Temp_150toBot

"""


from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('TkAgg')

###############################################################################
# CONFIGURATION – edit to match dataset
###############################################################################
CSV_PATH: pathlib.Path = pathlib.Path("C:/Users/Greig/Sync/6. SSMSP Model/Model Greig/Data/4. Zooplankton/Zoop_Perryetal_2021/MODIFIED/Zooplankton_B_C_gm2_EWEMODELGRP_Wide_NEMO3daymatch.csv")  # input file
ENV_COLUMN: str = "Temp_0to10m"      # Temp_150toBot, Temp_0to10m, Temp_30to40m default single‑depth run
ZOOP_GROUP_COLUMNS: List[str] = [
    "ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG",
    "ZC5-CSM", "ZS2-CTH", "ZS3-CHA", "ZS4-LAR",
]
OUTPUT_DIR: pathlib.Path = pathlib.Path("TPC_fits")
LOG_TRANSFORM: bool = True
PLOT_FITS: bool = True
BIN_WIDTH = 1

###############################################################################
# MODEL DEFINITION
###############################################################################

def gaussian_tpc(T: np.ndarray | float, Topt: float, a: float, b: float, c: float) -> np.ndarray:
    """Piece‑wise Gaussian TPC (Eq. 13)."""
    T = np.asarray(T)
    left = np.exp(-a * (T - Topt) ** 2)
    right = np.exp(-b * (T - Topt) ** 2)
    return c * np.where(T <= Topt, left, right)

###############################################################################
# FITTING UTILITIES
###############################################################################

@dataclass
class FitResult:
    group: str
    temp_col: str
    Topt: float
    a: float
    b: float
    c: float
    n: int
    rmse: float
    aic: float


def _initial_guess(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, float]:
    """Robust scalar starting values for (Topt, a, b, c).

    Converts any pandas/NumPy objects to plain Python floats so that
    `p0` is a homogeneous 1‑D numeric array, preventing the
    `ValueError: setting an array element with a sequence` raised by
    `scipy.optimize.curve_fit`.
    """
    peak_label = y.idxmax()
    Topt0 = float(x.loc[peak_label])
    c0 = float(y.max())
    a0 = b0 = 0.01  # gentle curvature
    return Topt0, a0, b0, c0



def _fit(sub: pd.DataFrame, temp_col: str) -> Tuple[np.ndarray, float]:
    # keep only rows where both T and abund are good numbers
    sub = sub.dropna(subset=[temp_col, "abund"]).copy()
    x = sub[temp_col].astype(float).values
    y_raw = sub["abund"].astype(float).values
    y = np.log1p(y_raw) if LOG_TRANSFORM else y_raw

    p0 = _initial_guess(sub[temp_col], sub["abund"])
    popt, _ = curve_fit(gaussian_tpc, x, y, p0=p0,
                        bounds=([0,0,0,0],[30,1,1,np.inf]), maxfev=10000)
    rmse = np.sqrt(np.mean((y - gaussian_tpc(x, *popt)) ** 2))
    return popt, rmse



def fit_group_multi(df_long: pd.DataFrame,
                    group: str,
                    temp_cols: List[str]) -> List[FitResult]:
    """
    Fit one zooplankton *group* to several temperature columns, returning
    a FitResult for each depth that has enough data.

    • Skips a (group, depth) combo unless it has at least `min_rows`
      non-NaN observations *after* binning.
    • Uses that same row count when computing AIC so the comparison
      matches single-depth behaviour.
    """
    min_rows = 5
    sub_all = df_long[df_long["group"] == group]
    if len(sub_all) < min_rows:
        print(f"[WARN] {group}: too few total points – skipping group.")
        return []

    fits: List[FitResult] = []
    k = 4  # free parameters in Gaussian TPC

    for Tcol in temp_cols:
        # keep only rows that have real values for this depth column
        sub = sub_all.dropna(subset=[Tcol, "abund"]).copy()
        if len(sub) < min_rows:
            print(f"[WARN] {group} – {Tcol}: < {min_rows} rows, skipped.")
            continue

        try:
            popt, rmse = _fit(sub, Tcol)
            n = len(sub)                      # rows actually fitted
            aic = n * np.log(rmse**2) + 2 * k
            fits.append(FitResult(group, Tcol, *popt, n, rmse, aic))
        except RuntimeError:
            print(f"[FAIL] {group} – {Tcol}: fit did not converge")

    return fits

# ---------------------------------------------------------------------
# OPTIONAL: degree-bin the data to dampen outliers / noise
# ---------------------------------------------------------------------
def degree_bin_df(long: pd.DataFrame,
                  temp_col: str,
                  bin_width: float = 1.0,
                  agg: str = "median",
                  min_n: int = 3) -> pd.DataFrame:
    """Collapse observations into ±½‑degree temperature bins.

    The returned DataFrame has **one numeric temperature column** – the
    mean temperature of observations inside each bin – which avoids the
    duplicate‑column issue that was causing Series→float conversion
    errors downstream.
    """
    # 1. create degree bins
    t_min, t_max = long[temp_col].min(), long[temp_col].max()
    bins = np.arange(np.floor(t_min), np.ceil(t_max) + bin_width, bin_width)
    long = long.copy()
    long["T_bin"] = pd.cut(long[temp_col], bins, labels=bins[:-1])

    # 2. aggregate within (group, T_bin)
    binned = (
        long.groupby(["group", "T_bin"], observed=True)
            .agg(abund=("abund", agg),
                 T_mean=(temp_col, "mean"),
                 n=("abund", "size"))
            .reset_index()
            .query("n >= @min_n")
            .drop(columns=["n", "T_bin"])     # drop bin label; keep mean temp
            .rename(columns={"T_mean": temp_col})
    )
    binned[temp_col] = binned[temp_col].astype(float)
    return binned



###############################################################################
# WORKFLOW
###############################################################################


def compare_depths(temp_columns: List[str]) -> None:
    """
    For each zooplankton group, fit the TPC to *each* temperature column
    in `temp_columns` and write out AIC / RMSE so you can pick a winner.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load & reshape once, keeping every candidate temp column
    df = pd.read_csv(CSV_PATH)
    long = (
        df.melt(id_vars=temp_columns + ["Year", "Month", "Day"],
                value_vars=ZOOP_GROUP_COLUMNS,
                var_name="group", value_name="abund")
          .dropna(subset=temp_columns + ["abund"])
    )

    # ▸ Optional noise-damping per temperature column
    bin_width = BIN_WIDTH  # set to 0 or None to disable binning
    if bin_width:
        frames = []
        for Tcol in temp_columns:
            # work on a fresh slice so we don't lose other temp columns
            slice_ = long[["group", "abund", Tcol]].copy()
            frames.append(
                degree_bin_df(slice_, Tcol, bin_width=bin_width, agg="median")
            )
        # merge all the individually-binned frames back together
        long = frames[0]
        for f in frames[1:]:
            long = long.merge(f, on=["group", "abund"], how="outer")

    # 2. Fit every group × temp column
    all_results: List[FitResult] = []
    for g in ZOOP_GROUP_COLUMNS:
        all_results.extend(fit_group_multi(long, g, temp_columns))

    if not all_results:
        print("[WARN] No successful fits.")
        return

    # 3. Collate metrics and flag the best (lowest AIC) per group
    res_df = pd.DataFrame([r.__dict__ for r in all_results])
    res_df["best"] = res_df.groupby("group")["aic"].transform(
        lambda s: s == s.min()
    )
    res_df.to_csv(OUTPUT_DIR / "TPC_compare_metrics.csv", index=False)
    print(f"[OK] Comparison table → {OUTPUT_DIR/'TPC_compare_metrics.csv'}")

    # 4. Quick ΔAIC bar-plots (omit with --no_plots)
    if PLOT_FITS:
        for g, gdf in res_df.groupby("group"):
            plt.figure(figsize=(6, 3))
            plt.bar(gdf.temp_col, gdf.aic - gdf.aic.min())
            plt.ylabel("ΔAIC")
            plt.title(g)
            plt.axhline(0, color="k", lw=0.8, ls="--")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"AIC_{g}.png", dpi=120)
            plt.close()

def _plot_fit(sub: pd.DataFrame, popt: np.ndarray, temp_col: str, group: str):
    """Scatter + fitted curve for a single group (single‑depth mode)."""
    T_min, T_max = sub[temp_col].min(), sub[temp_col].max()
    T_grid = np.linspace(T_min - 1, T_max + 1, 300)

    y_obs = np.log1p(sub["abund"]) if LOG_TRANSFORM else sub["abund"]
    y_fit = gaussian_tpc(T_grid, *popt)

    plt.figure(figsize=(5, 4))
    plt.scatter(sub[temp_col], y_obs, s=15, alpha=0.6, label="obs")
    plt.plot(T_grid, y_fit, lw=2, label="fit")
    plt.title(group)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("log(Abund+1)" if LOG_TRANSFORM else "Abundance")
    plt.legend(); plt.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR / f"TPC_{group}.png", dpi=150)
    plt.close()

###############################################################################
# SINGLE‑DEPTH (legacy) PIPELINE – unchanged
###############################################################################

def run_single_depth():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    long = df.melt(
        id_vars=[ENV_COLUMN, "Year", "Month", "Day"],
        value_vars=ZOOP_GROUP_COLUMNS,
        var_name="group", value_name="abund",
    ).dropna(subset=[ENV_COLUMN, "abund"])

    long = degree_bin_df(long, ENV_COLUMN, bin_width=BIN_WIDTH, agg="median")

    rows = []
    for g in ZOOP_GROUP_COLUMNS:
        sub = long[long.group == g]
        if len(sub) < 5:
            continue
        popt, rmse = _fit(sub, ENV_COLUMN)
        if PLOT_FITS:
            _plot_fit(sub, popt, ENV_COLUMN, g)
        n = len(sub); k=4; aic = n*np.log(rmse**2)+2*k
        rows.append(FitResult(g, ENV_COLUMN, *popt, n, rmse, aic))
    if rows:
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(OUTPUT_DIR/"TPC_parameters.csv", index=False)
        print(f"[OK] Saved parameter table → {OUTPUT_DIR/'TPC_parameters.csv'}")
    else:
        print("[WARN] No successful fits.")

###############################################################################
# CLI ENTRY
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gaussian TPC fitter with depth comparison")
    p.add_argument("--csv", default=str(CSV_PATH), help="Input CSV path")
    p.add_argument("--temp_col", default=ENV_COLUMN, help="Temperature column for single‑depth run")
    p.add_argument("--compare", nargs="+", help="List of temperature columns to compare")
    p.add_argument("--log", action="store_true", help="Fit to ln(abund+1)")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG diagnostics")
    args = p.parse_args()

    CSV_PATH = pathlib.Path(args.csv)
    LOG_TRANSFORM = args.log
    PLOT_FITS = not args.no_plots

    if args.compare:
        compare_depths(args.compare)
    else:
        ENV_COLUMN = args.temp_col
        run_single_depth()
