"""
Ecospace 2D → Seasonal Phytoplankton Comparison (vs McEwan)
By: G. Oldford (drafted with ChatGPT), 2025-08-09

Goal
====
Replicate the functionality of `ecosim_eval_2_assess_seasonal_phyto.py` but using
2D Ecospace NetCDF outputs, spatially masked to NSoG / CSoG domains from YAML,
then aggregated by season and grouped into Diatoms / Nano / Other, and finally
plotted as stacked bars next to the static McEwan seasonal estimates.

Dependencies
============
- xarray, numpy, pandas, matplotlib
- your project helpers: `read_sdomains` (from helpers.py)
- project config: `ecospace_eval_config as cfg` (mirrors your other scripts)

Notes
=====
- Seasons default to Winter (Dec–Feb), Spring (Mar–May), Summer (Jun–Aug).
  Adjust `SEASON_DEF` to exactly match the 1D script if different.
- Group mapping assumes Ecospace variables: PP1-DIA, PP2-NAN, PP3-PIC.
  Adjust `GROUP_MAP` if your names differ.
- Observations: hard-coded McEwan seasonal means for NSoG and CSoG (g C m^-2).
- Outputs: CSV of seasonal stats per domain and a PNG plot per domain.
"""

import os
from typing import Dict, List
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path
import matplotlib
matplotlib.use('TkAgg')

# --- Project helpers / config ---
from helpers import read_sdomains
import ecospace_eval_config as cfg

# ================================
# User-configurable bits
# ================================

# Seasons — match to your 1D script
SEASON_DEF = cfg.MW_SEASON_DEF
SEASONS_ORDER = ["Winter", "Spring", "Summer"]

# Ecospace phytoplankton groups → variable names in NetCDF
GROUP_MAP = cfg.MW_GROUP_MAP
GROUP_COLORS = cfg.MW_GROUP_COLORS

SHOW_PLTS = cfg.MW_SHOW_PLTS

# Domains (region keys must match polygons in your YAML)
DOMAIN_FP = cfg.MW_DOMAIN_FP
DOMAINS = cfg.MW_DOMAINS # CSoG, SSoG

# Observations (McEwan) — g C m^-2
OBS_MCEWAN = cfg.MW_OBS_MCEWAN
SPATIAL_REDUCTION = cfg.MW_SPATIAL_REDUCTION

# Date window (reuse from Ecosim eval if you want a comparable range)
# Fallback to all available if not found in cfg.
START_DATE = cfg.MW_START_DATE
END_DATE   = cfg.MW_END_DATE

# Output dirs
STATS_OUT = cfg.MW_STATS_OUT
FIGS_OUT  = cfg.MW_FIGS_OUT
SCENARIO  = cfg.ECOSPACE_SC

ECOSPACE_NC_F = cfg.NC_FILENAME
ECOSPACE_NC_P = cfg.NC_PATH_OUT
ECOSPACE_RN_STR_YR = cfg.ECOSPACE_RN_STR_YR
ECOSPACE_RN_END_YR = cfg.ECOSPACE_RN_END_YR

# added by GO Mar '26 to include ecosim
import ecosim_eval_config as cfg_sim

ECOSIM_FILE = cfg_sim.ECOSIM_F_PREPPED_SINGLERUN

# Fix "Other" to 19 if that is meant to be PP3-PIC
ECOSIM_GROUP_MAP = {
    "Diatoms": 17,
    "Nano": 18,
    "Other": 19,
}


# ================================
# Utilities
# ================================

def build_ecospace_nc_path() -> str:
    """Try both naming conventions used across your scripts."""
    if ECOSPACE_NC_P:
        return os.path.join(ECOSPACE_NC_P, ECOSPACE_NC_F)
    # Bloom-timing style
    base = ECOSPACE_NC_P
    code = SCENARIO
    y0   = ECOSPACE_RN_STR_YR
    y1   = ECOSPACE_RN_END_YR
    if base and code and y0 and y1:
        return os.path.join(base, f"{code}_{y0}-{y1}.nc")
    raise FileNotFoundError("Could not infer Ecospace NetCDF path from ecospace_eval_config.")


def month_to_season(m: int) -> str:
    for s, months in SEASON_DEF.items():
        if m in months:
            return s
    return "Other"


def make_domain_mask(ds: xr.Dataset, domain_yaml_path: str, domain_key: str, depth_var: str = "depth") -> xr.DataArray:
    """Build a boolean mask for a domain polygon intersected with valid depth (>0)."""
    sdomains = read_sdomains(domain_yaml_path)
    if domain_key not in sdomains:
        raise KeyError(f"Domain '{domain_key}' not found in YAML.")

    lat = ds["lat"].values
    lon = ds["lon"].values
    dep = ds[depth_var].values

    poly = Path(sdomains[domain_key])  # expects sequence of (lat, lon) like in your helpers
    pts = np.vstack((lat.flatten(), lon.flatten())).T
    region_mask = poly.contains_points(pts).reshape(lat.shape)
    valid = dep > 0
    mask = (region_mask & valid)

    return xr.DataArray(mask, dims=("row", "col"))


def extract_domain_timeseries(ds: xr.Dataset, mask: xr.DataArray, var: str) -> pd.Series:
    """Spatially reduce a 2D field under mask for every timestep → pandas Series."""
    da = ds[var]
    if SPATIAL_REDUCTION == "median":
        ts = da.where(mask).median(dim=("row", "col"), skipna=True)
    else:
        ts = da.where(mask).mean(dim=("row", "col"), skipna=True)
    # ts: time → value
    return pd.Series(ts.values, index=pd.to_datetime(ts["time"].values), name=var)


def seasonal_means(series: pd.Series) -> Dict[str, float]:
    """Average by season name using SEASON_DEF, return dict season → mean."""
    df = series.to_frame("val").reset_index(names="date")
    if START_DATE is not None:
        df = df[df["date"] >= pd.to_datetime(START_DATE)]
    if END_DATE is not None:
        df = df[df["date"] <= pd.to_datetime(END_DATE)]
    df["season"] = df["date"].dt.month.map(month_to_season)
    df = df[df["season"].isin(SEASONS_ORDER)]
    return df.groupby("season")["val"].mean().reindex(SEASONS_ORDER).fillna(0).to_dict()


def compute_domain_group_seasonal(ds: xr.Dataset, mask: xr.DataArray) -> pd.DataFrame:
    rows = []
    for grp, var in GROUP_MAP.items():
        if var not in ds.variables:
            print(f"[WARN] Variable '{var}' not found in dataset; '{grp}' set to 0.")
            seas = {s: 0.0 for s in SEASONS_ORDER}
        else:
            s = extract_domain_timeseries(ds, mask, var)
            seas = seasonal_means(s)
        for season, mean_val in seas.items():
            rows.append({"Group": grp, "Season": season, "ModelMean": float(mean_val)})
    return pd.DataFrame(rows)


def obs_seasonal_to_df(obs_dict: Dict[str, Dict[str, Dict[str, float]]], domain_key: str) -> pd.DataFrame:
    rows = []
    dom = obs_dict[domain_key]
    for grp in GROUP_MAP.keys():
        for season in SEASONS_ORDER:
            rows.append({
                "Group": grp,
                "Season": season,
                "ObsMean": float(dom[grp][season])
            })
    return pd.DataFrame(rows)


def merge_and_bias(df_model: pd.DataFrame, df_obs: pd.DataFrame) -> pd.DataFrame:
    df = df_model.merge(df_obs, on=["Group", "Season"], how="left")
    df["Bias"] = df["ModelMean"] - df["ObsMean"]
    return df


def plot_stacked_side_by_side(seasonal_stats_df: pd.DataFrame, domain_label: str, out_png: str) -> None:
    groups = list(GROUP_MAP.keys())

    # Build heights for model/obs per season
    model_heights = {g: [] for g in groups}
    obs_heights   = {g: [] for g in groups}

    for season in SEASONS_ORDER:
        for g in groups:
            sub = seasonal_stats_df[(seasonal_stats_df["Season"] == season) & (seasonal_stats_df["Group"] == g)]
            model_heights[g].append(float(sub["ModelMean"].iloc[0]) if not sub.empty else 0.0)
            obs_heights[g].append(float(sub["ObsMean"].iloc[0])   if not sub.empty else 0.0)

    # X layout: Mod./Obs. pairs
    bar_width = 0.35
    pair_spacing = 0.4
    group_spacing = 1.0
    x_pos = []
    for i in range(len(SEASONS_ORDER)):
        center = i * group_spacing
        x_pos.append(center - pair_spacing / 2)  # model
        x_pos.append(center + pair_spacing / 2)  # obs

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    # bottoms for stacking
    bottom_model = [0.0] * len(SEASONS_ORDER)
    bottom_obs   = [0.0] * len(SEASONS_ORDER)

    # Plot model (even idxs)
    for g in groups:
        vals = model_heights[g]
        ax.bar(x_pos[::2], vals, bar_width, bottom=bottom_model, label=f"{g} (Model)", color=GROUP_COLORS[g])
        bottom_model = [a + b for a, b in zip(bottom_model, vals)]

    # Plot obs (odd idxs)
    for g in groups:
        vals = obs_heights[g]
        ax.bar(x_pos[1::2], vals, bar_width, bottom=bottom_obs, label=f"{g} (Obs)",
               color=GROUP_COLORS[g], hatch='//', edgecolor='grey')
        bottom_obs = [a + b for a, b in zip(bottom_obs, vals)]

    # X ticks as Mod./Obs. with season centered beneath
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Mod.' if i % 2 == 0 else 'Obs.' for i in range(len(x_pos))], rotation=0)

    # one centered season label per pair
    season_y_offset = -0.085
    for i, season in enumerate(SEASONS_ORDER):
        mid_x = (x_pos[2 * i] + x_pos[2 * i + 1]) / 2
        ax.text(mid_x, season_y_offset, season, ha='center', va='top', transform=ax.get_xaxis_transform())

    # Legend with only 3 entries
    legend_patches = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in groups]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    ax.set_ylabel('Biomass (g C m$^{-2}$)')
    ax.set_title(f'{domain_label}: Ecospace vs McEwan Seasonal Phytoplankton ({SCENARIO})')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    if SHOW_PLTS:
        plt.show()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)

    plt.close()

# added by GO Mar '26 for adding ecosim
def compute_ecosim_group_seasonal() -> pd.DataFrame:
    df = pd.read_csv(ECOSIM_FILE)
    df["date"] = pd.to_datetime(df["date"])

    if START_DATE is not None:
        df = df[df["date"] >= pd.to_datetime(START_DATE)]
    if END_DATE is not None:
        df = df[df["date"] <= pd.to_datetime(END_DATE)]

    # Recompute season here so both scripts use the same definition
    df["season"] = df["date"].dt.month.map(month_to_season)
    df = df[df["season"].isin(SEASONS_ORDER)]

    rows = []
    for grp, col in ECOSIM_GROUP_MAP.items():
        means = (
            df.groupby("season")[str(col)]
              .mean()
              .reindex(SEASONS_ORDER)
              .fillna(0.0)
        )
        for season, val in means.items():
            rows.append({
                "Group": grp,
                "Season": season,
                "Source": "1-D",
                "Mean": float(val),
            })

    return pd.DataFrame(rows)


def ecospace_stats_to_long(df_model: pd.DataFrame) -> pd.DataFrame:
    out = df_model.rename(columns={"ModelMean": "Mean"}).copy()
    out["Source"] = "2-D"
    return out[["Group", "Season", "Source", "Mean"]]


def obs_seasonal_to_long(obs_dict, domain_key: str) -> pd.DataFrame:
    rows = []
    dom = obs_dict[domain_key]
    for grp in GROUP_MAP.keys():
        for season in SEASONS_ORDER:
            rows.append({
                "Group": grp,
                "Season": season,
                "Source": "Data",
                "Mean": float(dom[grp][season]),
            })
    return pd.DataFrame(rows)


def plot_stacked_three_source(df_long: pd.DataFrame, domain_label: str, out_png: str) -> None:
    groups = list(GROUP_MAP.keys())
    sources = ["1-D", "2-D", "Data"]

    source_hatch = {
        "1-D": "//",
        "2-D": "xx",
         "Data": ""
    }

    source_edge = {
        "1-D": "0.25",
        "2-D": "0.25",
        "Data": "black",
    }

    bar_width = 0.22
    group_spacing = 1
    offsets = {
        "1-D":  -0.28,
        "2-D":  0.00,
        "Data":    0.28,
    }

    centers = [i * group_spacing for i in range(len(SEASONS_ORDER))]
    x_pos = {
        src: [c + offsets[src] for c in centers]
        for src in sources
    }

    fig, ax = plt.subplots(figsize=(4, 4))

    for src in sources:
        bottoms = [0.0] * len(SEASONS_ORDER)
        for grp in groups:
            vals = []
            for season in SEASONS_ORDER:
                sub = df_long[
                    (df_long["Season"] == season) &
                    (df_long["Group"] == grp) &
                    (df_long["Source"] == src)
                ]
                vals.append(float(sub["Mean"].iloc[0]) if not sub.empty else 0.0)

            ax.bar(
                x_pos[src],
                vals,
                bar_width,
                bottom=bottoms,
                color=GROUP_COLORS[grp],
                hatch=source_hatch[src],
                edgecolor=source_edge[src],
                linewidth=0.8,
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

    xticks = []
    xticklabels = []
    for i in range(len(SEASONS_ORDER)):
        for src in sources:
            xticks.append(x_pos[src][i])
            xticklabels.append(src)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, ha="right", fontsize=7.5)

    for i, season in enumerate(SEASONS_ORDER):
        ax.text(
            centers[i], -0.10, season,
            ha="center", va="top",
            transform=ax.get_xaxis_transform()
        )

    group_handles = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="black", label=g)
        for g in groups
    ]
    source_handles = [
        Patch(facecolor="white", edgecolor="0.25", hatch=source_hatch[s], label=s)
        for s in sources
    ]

    # leg1 = ax.legend(handles=group_handles, loc="upper left", title="Group", fontsize=8)
    # ax.add_artist(leg1)
    # ax.legend(handles=source_handles, loc="upper right", title="Source", fontsize=8)

    ax.set_ylabel("Biomass (g C m$^{-2}$)")
    ax.set_title(f"{domain_label}: Seasonal phytoplankton biomass")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if SHOW_PLTS:
        plt.show()
    plt.close(fig)


# ================================
# Main
# ================================

def run_eval_2d_phyto() -> None:
    ecospace_nc = build_ecospace_nc_path()
    if not os.path.exists(ecospace_nc):
        raise FileNotFoundError(ecospace_nc)


    print(f"Using Ecospace file: {ecospace_nc}")
    print(f"Using domain YAML:  {DOMAIN_FP}")

    ds = xr.open_dataset(ecospace_nc)

    results_all = []

    df_ecosim = compute_ecosim_group_seasonal()

    for dom_label, dom_key in DOMAINS.items():
        mask = make_domain_mask(ds, DOMAIN_FP, dom_key)

        df_ecospace = compute_domain_group_seasonal(ds, mask)
        df_ecospace_long = ecospace_stats_to_long(df_ecospace)
        df_obs_long = obs_seasonal_to_long(OBS_MCEWAN, dom_label)

        combo = pd.concat(
            [
                df_ecospace_long.assign(Domain=dom_label),
                df_ecosim.assign(Domain=dom_label),  # repeated 1D reference
                df_obs_long.assign(Domain=dom_label),
            ],
            ignore_index=True,
        )

        out_png = os.path.join(FIGS_OUT, f"ecospace_ecosim_mcewan_phyto_{dom_label}.png")
        plot_stacked_three_source(combo, dom_label, out_png)

    for dom_label, dom_key in DOMAINS.items():
        print(f"Computing domain: {dom_label} ({dom_key}) …")
        mask = make_domain_mask(ds, DOMAIN_FP, dom_key)
        df_mod = compute_domain_group_seasonal(ds, mask)
        df_obs = obs_seasonal_to_df(OBS_MCEWAN, dom_label)
        stats = merge_and_bias(df_mod, df_obs)
        stats.insert(0, "Domain", dom_label)
        results_all.append(stats)

        # Plot
        out_png = os.path.join(FIGS_OUT, f"ecospace_{SCENARIO}_phyto_seasonal_mcewan_{dom_label}.png")
        plot_stacked_side_by_side(stats, dom_label, out_png)
        print(f"Saved figure: {out_png}")

    all_stats = pd.concat(results_all, ignore_index=True)
    os.makedirs(STATS_OUT, exist_ok=True)
    out_csv = os.path.join(STATS_OUT, f"ecospace_{SCENARIO}_phyto_seasonal_stats.csv")
    all_stats.to_csv(out_csv, index=False)
    print(f"Saved seasonal stats: {out_csv}")


if __name__ == "__main__":
    run_eval_2d_phyto()
