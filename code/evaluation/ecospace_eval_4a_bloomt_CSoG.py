"""
Analyzing Bloom Timing in Ecospace Outputs
by: G Oldford, 2023-2025
----------------------------------------------------------
Compares bloom timing from Ecospace model outputs against:
1. Satellite-derived bloom dates (Suchy et al. 2022)
2. C09 1D model outputs at S3 (Collins et al. 2009; C09 & Latournelle)
3. SSC 3D model outputs (SalishSeaCast v201905)

Outputs:
- CSV files of bloom timing
- Multiple comparative figures

Notes:
- to do: if recompute bloom timing is False, and user changes the Vars to analyse - does it work?
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')

from helpers_cartopy import (
    read_sdomains, find_nearest_point, find_bloom_doy,
    buildSortableString
)
import ecospace_eval_config as cfg


# ================================
# Configuration
# ================================
ECOSPACE_OUT_PATH = cfg.NC_PATH_OUT
STATS_OUT_PATH = cfg.EVALOUT_P
FIGS_OUT_P = cfg.FIGS_P
DOMAIN_CONFIG_PATH = cfg.DOMAIN_P
DOMAIN_FILE = cfg.BT_DOMAIN_FILE
SAT_MASK_PF = cfg.BT_SAT_MASK_PF
BT_MASK_REGNM = cfg.BT_MASK_REGNM

RECOMPUTE_BLOOM_TIMING_SAT = cfg.BT_RECOMPUTE_BLOOM_TIMING_SAT  # Set to True to force recomputation as needed, saves time
RECOMPUTE_BLOOM_TIMING_C09 = cfg.BT_RECOMPUTE_BLOOM_TIMING_C09

SCENARIO = cfg.ECOSPACE_SC
ECOSPACE_CODE = cfg.ECOSPACE_SC
FILENM_STRT_YR = cfg.ECOSPACE_RN_STR_YR
FILENM_END_YR = cfg.ECOSPACE_RN_END_YR
START_YEAR = cfg.BT_START_YEAR # analysis years (exclude spinup?)
END_YEAR = cfg.BT_END_YEAR

# if multiple vars listed here, it will sum across them when computing anomalies, bloom timing etc
#OK with run 96 this is first time model fit has been okay with all pp groups
# VARIABLES_TO_ANALYZE_SAT = ["PP1-DIA"] #for 88_2 - pp1-dai only, annual, keep janfeb
VARIABLES_TO_ANALYZE_SAT = cfg.BT_VARS_TO_ANALYZE_SAT
VARIABLES_TO_ANALYZE_C09 = cfg.BT_VARS_TO_ANALYZE_C09

ANNUAL_AVG_METHOD_SAT = cfg.BT_ANNUAL_AVG_METHOD_SAT # should average bloom compared against be from all years, or just one year
ANNUAL_AVG_METHOD_C09 = cfg.BT_ANNUAL_AVG_METHOD_C09 # annual or all (hard coded to be annual for Pctmax method)


# use mask c09 instead of pnt?
USE_SAT_MASK_CO9 = cfg.BT_USE_SAT_MASK_CO9
EXCLUDE_DEC_JAN_SAT = cfg.BT_EXCLUDE_DEC_JAN_SAT

C09_LOG_TRNSFRM = cfg.BT_LOG_TRANSFORM_C09
C09_USE_PCT_MAX = cfg.BT_USE_PCT_MAX_C09
C09_PCT_MAX = cfg.BT_PCT_MAX_C09
C09_PCT_MAX_WINDOW_DAYS = cfg.BT_PCT_MAX_WINDOW_DAYS_C09
EXCLUDE_DEC_JAN_C09 = cfg.BT_EXCLUDE_DEC_JAN_C09

# Bloom detection method
LOG_TRANSFORM = cfg.BT_LOG_TRANSFORM_SAT
MEAN_OR_MEDIAN = cfg.BT_MEAN_OR_MEDIAN_SAT
THRESHOLD_FACTOR = cfg.BT_THRESHOLD_FACTOR_SAT
SUB_THRESHOLD_FACTOR = cfg.BT_SUB_THRESHOLD_FACTOR_SAT

MIN_Y_TICK = cfg.BT_MIN_Y_TICK
CREATE_MASKS = cfg.BT_CREATE_MASKS # Set to True to regenerate masks
DO_NUTRIENTS = cfg.BT_DO_NUTRIENTS # script 9 visualises nutrients now

USEC09_MASK_FOR_SAT = cfg.BT_USEC09_MASK_FOR_SAT
C09_ROW = cfg.BT_C09_ROW
C09_COL = cfg.BT_C09_COL


# ================================
# Utility Functions
# ================================

def load_overlay_csv(path):
    if not os.path.exists(path):
        print(f"Overlay CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    if 'Bloom Date' in df.columns:
        df['Bloom Date'] = pd.to_datetime(df['Bloom Date'], errors='coerce')
    return df

def convert_doy_to_date(doys, start_year):
    return [datetime(start_year + i, 1, 1) + timedelta(days=doy - 1) for i, doy in enumerate(doys)]


def create_bloom_df(years, doys, label="avg"):
    dates = convert_doy_to_date(doys, years[0])
    return pd.DataFrame({
        'Year': years,
        'Bloom Date': dates,
        'Day of Year': doys,
        'Bloom Early Late': label if isinstance(label, list) else [label]*len(years)
    })


def classify_bloom_by_std(doys, mean_val, std_val):
    results = []
    for doy in doys:
        if doy <= (mean_val - std_val):
            results.append("early")
        elif doy >= (mean_val + std_val):
            results.append("late")
        elif doy <= (mean_val + std_val - 1) and doy >= (mean_val - std_val + 1):
            results.append("avg")
        else:
            results.append("cusp")
    # exploring 'cusp' method
    # for doy in doys:
    #     if doy + 4 <= (mean_val - std_val):
    #         results.append("early")
    #     elif doy - 4 >= (mean_val + std_val):
    #         results.append("late")
    #     elif doy + 4 <= (mean_val + std_val - 1) and doy - 4 >= (mean_val - std_val + 1):
    #         results.append("avg")
    #     else:
    #         results.append("cusp")
    return results


# ================================
# Mask Generation
# ================================

def generate_2d_mask(ds, domain_file_path, region_name=BT_MASK_REGNM, depth_var='depth'):
    lat = ds['lat'].values
    lon = ds['lon'].values
    dep = ds[depth_var].values

    sdomains = read_sdomains(domain_file_path)
    polygon_path = Path(sdomains[region_name])
    points = np.vstack((lat.flatten(), lon.flatten())).T
    region_mask = polygon_path.contains_points(points).reshape(lat.shape)
    depth_mask = dep > 0
    combined_mask = region_mask & depth_mask

    mask_ds = xr.Dataset({
        'mask': (('row', 'col'), combined_mask)
    }, coords={'lat': (('row', 'col'), lat), 'lon': (('row', 'col'), lon)})

    out_path = os.path.join(DOMAIN_CONFIG_PATH, '..//..//data//evaluation//suchy_ecospace_mask.nc')
    mask_ds.to_netcdf(out_path)
    print(f"Saved 2D mask to {out_path}")
    return mask_ds


def generate_1d_mask(ds, lat_target, lon_target, dx=0.01):
    lats = ds['lat'].values
    lons = ds['lon'].values
    deps = ds['depth'].values
    valid_mask = deps != 0
    return find_nearest_point(lon_target, lat_target, lons, lats, valid_mask, dx)


# ================================
# Bloom Observation Datasets
# ================================

def load_observation_bloom_dfs():
    # Suchy satellite data
    doy_sat = [100, 68, 50, 83, 115, 115, 100, 100, 92, 88, 92, 92, 55, 77]
    years_sat = list(range(2003, 2017))
    sat_df = create_bloom_df(years_sat, doy_sat, ["avg", "avg", "early", "avg",
                                                        "late", "late", "avg", "avg",
                                                        "avg", "avg", "avg", "avg",
                                                        "early", "avg"])

    # Gower satellite data
    doy_gower = [83, 68, 76, 85, 71, 56, 84, 49, 71, 58, 90, 97, 91, 69, 94, 66, 86]
    years_gower = list(range(2000, 2017))
    gower_df = create_bloom_df(years_gower, doy_gower, "na")
    gower_df.columns = [col + "_Gower" if col != 'Year' else 'Year_Gower' for col in gower_df.columns]

    # C09 1D model data
    doy_C09 = [94, 78, 81, 82, 87, 76, 75, 87, 99, 76,
                 81, 78, 77, 55, 86, 86, 67, 87, 77, 104,
                 66, 70, 92, 86, 81, 62, 88, 100, 90, 97,
                 104, 103, 98, 88, 86, 77, 88, 104, 77]
    years_C09 = list(range(1980, 2019))
    mean_C09 = np.mean(doy_C09)
    std_C09 = np.std(doy_C09)
    labels_C09 = classify_bloom_by_std(doy_C09, mean_C09, std_C09)
    C09_df = create_bloom_df(years_C09, doy_C09, labels_C09)
    C09_df.columns = [col + "_C09" if col != 'Year' else 'Year_C09' for col in C09_df.columns]

    return sat_df, gower_df, C09_df


# ================================
# Model Bloom Detection
# ================================

def load_ecospace_dataset():
    fname = f"ecospace_{ECOSPACE_CODE}_{FILENM_STRT_YR}-{FILENM_END_YR}.nc"
    path = os.path.join(ECOSPACE_OUT_PATH, fname)
    return xr.open_dataset(path)


def find_bloom_doy_pctmax(df, biomass_col,
                          window_days=6,
                          pct_of_max=0.95,
                          exclude_months=None):
    """
    Bloom timing = first date where time-smoothed biomass >= pct_of_max * yearly max
    of that smoothed series. Uses a time-based rolling window, so it works for daily,
    3-day, or slightly irregular time steps.
    Returns: lists (bloom_dates, bloom_doys)
    """
    work = df.copy()
    if exclude_months:
        work = work[~work['Date'].dt.month.isin(exclude_months)]

    if C09_LOG_TRNSFRM:
        work[biomass_col] = np.log(work[biomass_col] + 0.01)

    bloom_dates, bloom_doys = [], []

    for yr, grp in work.groupby('Year', sort=True):
        if grp.empty:
            bloom_dates.append(pd.NaT); bloom_doys.append(np.nan); continue

        grp = grp.sort_values('Date').reset_index(drop=True)
        ts = grp.set_index('Date')[biomass_col].sort_index()

        # time-based rolling mean (e.g., '6D')
        smoothed = ts.rolling(f'{int(window_days)}D', min_periods=1).mean()
        max_val  = smoothed.max()

        if pd.isna(max_val) or max_val <= 0:
            bloom_dates.append(pd.NaT); bloom_doys.append(np.nan); continue

        thresh = pct_of_max * max_val
        hits = smoothed.index[smoothed >= thresh]

        if len(hits) > 0:
            first_dt = pd.Timestamp(hits[0])
            bloom_dates.append(first_dt)
            bloom_doys.append(first_dt.dayofyear)
        else:
            bloom_dates.append(pd.NaT)
            bloom_doys.append(np.nan)

    return bloom_dates, bloom_doys


def compute_bloom_timing(ds, var_name, mask=None,
                         row=None, col=None,
                         bloom_early=68, bloom_late=108,
                         yr_strt=1980, yr_end=2018,
                         exclude_dec_jan=False,
                         avg_method="annual",
                         method="threshold",
                         window_days=6,
                         pct_of_max=0.95):

    years = range(yr_strt, yr_end + 1)
    values, timestamps = [], []
    print("computing bloom timing")

    for year in years:
        print(year)
        yearly_ds = ds.sel(time=str(year))

        if mask is not None:
            yearly_ds = yearly_ds.where(mask)
        elif row is not None and col is not None:
            yearly_ds = yearly_ds.isel(row=row, col=col)
        else:
            raise ValueError("Must provide either a 2D mask or specific row/col indices.")

        if isinstance(var_name, list):
            total = None
            for g in var_name:
                if g in yearly_ds:
                    total = yearly_ds[g] if total is None else total + yearly_ds[g]
                else:
                    print(f"Warning: {g} not found in dataset for year {year}")
            yearly_var = total
        else:
            yearly_var = yearly_ds[var_name]

        if LOG_TRANSFORM:
            yearly_var = np.log(yearly_var + 0.01)
        # else:
        #     yearly_var = yearly_ds[var_name]


        for ts in yearly_ds.time:
            val = yearly_var.sel(time=ts)
            value = np.nanmedian(val) if MEAN_OR_MEDIAN == "median" else np.nanmean(val)
            values.append(value)
            timestamps.append(pd.Timestamp(ts.values))

    df = pd.DataFrame({
        'Year': pd.to_datetime(timestamps).year,
        'Date': pd.to_datetime(timestamps),
        'Value': values
    })

    if method == "pct_max":
        # map your exclude_dec_jan flag to months if desired
        excl_mo = [1, 12] if exclude_dec_jan else None

        bloom_dates, bloom_doys = find_bloom_doy_pctmax(
            df, 'Value',
            window_days=window_days,
            pct_of_max=pct_of_max,
            exclude_months=[7, 8, 9, 10, 11, 12]  # Jan–June spring bloom
        )

        # Derive categories from early/late thresholds for downstream consistency
        bloom_categories = []
        for doy in bloom_doys:
            if np.isnan(doy):
                bloom_categories.append("na")
            elif doy <= bloom_early:
                bloom_categories.append("early")
            elif doy >= bloom_late:
                bloom_categories.append("late")
            else:
                bloom_categories.append("avg")

    else:
        bloom_dates, bloom_doys, bloom_categories = find_bloom_doy(
            df, 'Value',
            thrshld_fctr=THRESHOLD_FACTOR,
            sub_thrshld_fctr=SUB_THRESHOLD_FACTOR,
            average_from=avg_method,
            mean_or_median=MEAN_OR_MEDIAN,
            exclude_juntojan=exclude_dec_jan,
            bloom_early=bloom_early,
            bloom_late=bloom_late
        )

    return pd.DataFrame({
        'Year': years,
        'Bloom Date': bloom_dates,
        'Day of Year': bloom_doys,
        'Bloom Early Late': bloom_categories
    })


# ================================
# Plotting Functions
# ================================

def standardize_columns(df):
    for col in df.columns:
        if "Year" in col and col != "Year" and col != "Day of Year":
            df = df.rename(columns={col: "Year"})
            break  # Stop after the first match
    for col in df.columns:
        if "Day of Year" in col and col != "Year":
            df = df.rename(columns={col: "Day of Year_Obs"})
            break  # Stop after the first match
    return df


def _standardize_model_cols(df, doy_name):
    out = df.copy()
    for col in out.columns:
        if "Year" in col and col != "Year" and col != "Day of Year":
            out = out.rename(columns={col: "Year"})
            break
    for col in out.columns:
        if "Day of Year" in col and col != "Year" and col != doy_name:
            out = out.rename(columns={col: doy_name})
            break
    return out


def plot_bloom_comparison(
    df_model,
    df_obs,
    label_model="SOGEM-LTL 2D",
    label_obs="sat",
    title="Bloom Timing Comparison",
    filename="bloom_timing_plot.png",
    df_overlay=None,
    label_overlay="SOGEM-LTL 1-D",
):
    df_obs = standardize_columns(df_obs)
    df_model = _standardize_model_cols(df_model, "Day of Year_Model")

    if df_overlay is not None:
        df_overlay = _standardize_model_cols(df_overlay, "Day of Year_Overlay")

    # Limit all plotted series to the observation years
    plot_years = sorted(df_obs["Year"].dropna().astype(int).unique())

    df_model = df_model[df_model["Year"].isin(plot_years)].copy()

    if df_overlay is not None:
        df_overlay = df_overlay[df_overlay["Year"].isin(plot_years)].copy()

    if label_obs == 'Satellite':
        fig_w, fig_h = 6, 3.5
    else:
        fig_w, fig_h = 6, 3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # observations
    ax.errorbar(
        df_obs['Year'], df_obs['Day of Year_Obs'],
        yerr=4, fmt='s',
        markersize=2,
        label=label_obs,
        color='darkorange', capsize=3
    )
    ax.plot(df_obs['Year'], df_obs['Day of Year_Obs'],
            linewidth=1,
            linestyle='-',
            color='darkorange')

    # primary model (Ecospace)
    ax.errorbar(
        df_model['Year'], df_model['Day of Year_Model'],
        yerr=1.5, fmt='o', markersize=2,
        label=label_model, color='blue', capsize=3
    )
    ax.plot(df_model['Year'], df_model['Day of Year_Model'],
            linewidth=1,
            linestyle='-',
            color='blue')

    # optional overlay model (Ecosim)
    if df_overlay is not None and 'Day of Year_Overlay' in df_overlay.columns:
        ax.errorbar(
            df_overlay['Year'], df_overlay['Day of Year_Overlay'],
            yerr=1.5, fmt='^',
            markersize=2,
            label=label_overlay,
            color='blue', capsize=3
        )
        ax.plot(df_overlay['Year'],
                df_overlay['Day of Year_Overlay'],
                linestyle='--',
                linewidth=1,
                color='blue')

    mean = np.nanmean(df_obs['Day of Year_Obs'])
    std = np.nanstd(df_obs['Day of Year_Obs'])

    lower_bound = mean - std
    upper_bound = mean + std

    ax.axhspan(
        lower_bound,
        upper_bound,
        facecolor='grey',
        alpha=0.25,
        zorder=0
    )

    ax.axhline(y=upper_bound, linestyle='--', color='grey', zorder=1)
    ax.axhline(y=mean, linestyle='-', color='grey', zorder=1)
    ax.axhline(y=lower_bound, linestyle='--', color='grey', zorder=1)


    ymax = np.nanmax(df_obs['Day of Year_Obs'])
    if len(df_model) > 0:
        ymax = max(ymax, np.nanmax(df_model['Day of Year_Model']))
    if df_overlay is not None and 'Day of Year_Overlay' in df_overlay.columns and len(df_overlay) > 0:
        ymax = max(ymax, np.nanmax(df_overlay['Day of Year_Overlay']))

    ax.set_xlabel("Year")
    ax.set_ylabel("Day of Year")
    ax.set_title(title)
    ax.set_ylim([MIN_Y_TICK, ymax + 10])

    if len(plot_years) > 0:
        ax.set_xlim(min(plot_years) - 0.5, max(plot_years) + 0.5)
        ax.set_xticks(plot_years if len(plot_years) <= 20 else plot_years[::2])
        ax.set_xticklabels(plot_years if len(plot_years) <= 20 else plot_years[::2], rotation=45)

    ax.grid(True)
    # ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS_OUT_P + "//" + filename)
    print("saved: " + filename)
    plt.show()
    plt.close()


# ================================
# Evaluation Statistics
# ================================

def willmott1981(obs, mod):
    mod = np.asarray(mod)
    obs = np.asarray(obs)
    num = np.nansum((mod - obs) ** 2)
    obs_mean = np.nanmean(obs)
    dM = np.abs(mod - obs_mean)
    dO = np.abs(obs - obs_mean)
    den = np.nansum((dM + dO) ** 2)
    if den == 0:
        return np.nan
    else:
        return np.max([0, 1 - num / den])


def evaluate_model(obs, mod, label=""):
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    valid = ~np.isnan(obs) & ~np.isnan(mod)
    obs_valid = obs[valid]
    mod_valid = mod[valid]

    rmse = np.sqrt(mean_squared_error(obs_valid, mod_valid))
    mse = mean_squared_error(obs_valid, mod_valid)
    mae = mean_absolute_error(obs_valid, mod_valid)
    bias = np.mean(mod_valid - obs_valid)
    r = np.corrcoef(obs_valid, mod_valid)[0, 1] if len(obs_valid) > 1 else np.nan
    skill = willmott1981(obs_valid, mod_valid)
    std_obs = np.std(obs_valid)
    std_mod = np.std(mod_valid)

    return {
        "Label": label,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "Bias": bias,
        "R": r,
        "Willmott Skill": skill,
        "Obs StdDev": std_obs,
        "Model StdDev": std_mod
    }


def evaluate_bloom_categories(df_obs, df_mod, col_obs='Bloom Early Late', col_mod='Bloom Early Late'):
    df_obs = standardize_columns(df_obs)

    for col in df_obs.columns:
        if "Bloom Early Late" in col and col != "Bloom Early Late":
            df_obs = df_obs.rename(columns={col: "Bloom Early Late"})
            break  # Stop after the first match

    df = df_obs.merge(df_mod, on='Year', suffixes=('_obs', '_mod'))
    agree_count = (df[f'{col_obs}_obs'] == df[f'{col_mod}_mod']).sum()
    total = len(df)
    return agree_count, total


def evaluate_overlap_by_timing(df_obs, df_mod, obs_col='Day of Year', mod_col='Day of Year'):

    df_obs = standardize_columns(df_obs)
    for col in df_obs.columns: #standardise
        if "Day of Year" in col and col != "Day of Year":
            df_obs = df_obs.rename(columns={col: "Day of Year"})
            break  # Stop after the first match


    df = df_obs.merge(df_mod, on='Year', suffixes=('_obs', '_mod'))
    # Define bounds
    obs_low = df[f'{obs_col}_obs'] - 4
    obs_high = df[f'{obs_col}_obs'] + 4
    mod_low = df[f'{mod_col}_mod'] - 1.5
    mod_high = df[f'{mod_col}_mod'] + 1.5
    overlap = (mod_high >= obs_low) & (mod_low <= obs_high)
    return overlap.sum(), len(overlap)


# ================================
# Nutrient Proxy from Biomass
# ================================

def compute_nutrient_concentration(ds, vars_list, year=2005, include_only=None, exclude=None, mask=None):
    # Constants
    # N_free_init = 41.0   # g N m^-2 (96%)
    # N_bound_init = 1.54  # g N m^-2 (4%)
    N_free_init = 21.27   # g N m^-2 (50%)
    N_bound_init = 21.27  # g N m^-2 (50%)
    # Ecopath_Base_B_Tot_C =
    total_N = N_free_init + N_bound_init

    # Molar mass ratios (Redfield C:N is 106:16)
    C_to_N_ratio = 106 / 16

    # Apply inclusion/exclusion filters
    if include_only:
        vars_list = [var for var in vars_list if var in include_only]
    if exclude:
        vars_list = [var for var in vars_list if var not in exclude]

    # Subset one year
    year_ds = ds.sel(time=str(year))

    # Compute total biomass over time (assumed carbon units)
    total_biomass = None
    for var in vars_list:
        if var in year_ds:
            if total_biomass is None:
                total_biomass = year_ds[var]
            else:
                total_biomass += year_ds[var]

    if total_biomass is None:
        raise ValueError("None of the specified biomass variables found in dataset.")

    # Apply spatial mask if provided
    if mask is not None:
        total_biomass = total_biomass.where(mask)

    # Mean over space (depth integrated already assumed)
    biomass_time_series = total_biomass.mean(dim=["row", "col"], skipna=True)

    # Convert carbon to nitrogen (g C to g N)
    N_bound = biomass_time_series / C_to_N_ratio

    # Estimate remaining free N = total_N - N_bound
    N_free = total_N - N_bound
    N_free = N_free.where(N_free >= 0, 0)  # avoid negative nutrient concentrations

    # Prepare DataFrame
    df = pd.DataFrame({
        "Date": pd.to_datetime(year_ds.time.values),
        "Day of Year": pd.to_datetime(year_ds.time.values).dayofyear,
        "Year": year,
        "Total Biomass (C)": biomass_time_series.values,
        "N Bound (g m^-2)": N_bound.values,
        "N Free (g m^-2)": N_free.values
    })
    return df


def plot_nutrient_concentration(df_nutrient_all, ds=None, include_only=None, mask=None):
    plt.figure(figsize=(10, 5))

    # Climatological stats
    grouped = df_nutrient_all.groupby("Day of Year")
    mean_nutrient = grouped["N Free (g m^-2)"].mean()
    q10 = grouped["N Free (g m^-2)"].quantile(0.10)
    q20 = grouped["N Free (g m^-2)"].quantile(0.20)
    q30 = grouped["N Free (g m^-2)"].quantile(0.30)
    q40 = grouped["N Free (g m^-2)"].quantile(0.40)
    q60 = grouped["N Free (g m^-2)"].quantile(0.60)
    q70 = grouped["N Free (g m^-2)"].quantile(0.70)
    q80 = grouped["N Free (g m^-2)"].quantile(0.80)
    q90 = grouped["N Free (g m^-2)"].quantile(0.90)

    # Shaded quantile bands
    plt.fill_between(mean_nutrient.index, q10, q90, color="0.6", alpha=0.2, label="")
    plt.fill_between(mean_nutrient.index, q20, q80, color="0.5", alpha=0.2, label="")
    plt.fill_between(mean_nutrient.index, q30, q70, color="0.4", alpha=0.1, label="")
    plt.fill_between(mean_nutrient.index, q40, q60, color="0.4", alpha=0.1, label="")
    plt.plot(mean_nutrient.index, mean_nutrient, color="0.3", linewidth=2, label="Nutrients - Clim. Avg.")

    plt.xlabel("Day of Year")
    plt.ylabel("Nutrient Concentration (g N m$^{-2}$)")
    plt.title("Climatological Nutrient Concentration vs Total Biomass (All Years) -" + SCENARIO)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Add secondary axis for biomass
    if ds is not None and include_only is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        colors = plt.cm.tab10.colors

        for i, var in enumerate(include_only):
            biomasses = []
            for yr in df_nutrient_all['Year'].unique():
                try:
                    da = ds[var].sel(time=str(yr))
                    if mask is not None:
                        da = da.where(mask)
                    ts = da.mean(dim=["row", "col"], skipna=True)
                    biomasses.append(pd.DataFrame({
                        'Day of Year': pd.to_datetime(da['time'].values).dayofyear,
                        'Biomass': ts.values,
                        'Year': yr
                    }))
                except KeyError:
                    continue

            df_biomass = pd.concat(biomasses, ignore_index=True)
            grouped_biomass = df_biomass.groupby("Day of Year")
            mean_b = grouped_biomass['Biomass'].mean()
            q10_b = grouped_biomass['Biomass'].quantile(0.10)
            q90_b = grouped_biomass['Biomass'].quantile(0.90)

            ax2.fill_between(mean_b.index, q10_b, q90_b, color=colors[i % len(colors)], alpha=0.15)
            ax2.plot(mean_b.index, mean_b, color=colors[i % len(colors)], label=f"{var} Biomass")

        ax2.set_ylabel("Biomass (g C m$^{-2}$)")
        ax2.legend(loc="upper right")

    plt.show()
    plt.savefig('..//..//figs//' + "nutrient_concentration_climatology_" + SCENARIO + ".png")
    plt.close()


# ================================
# Entry Point
# ================================

def run_bt_eval() -> None:
    # Load observation datasets
    sat_df, gower_df, C09_df = load_observation_bloom_dfs()

    # Load Ecospace model outputs
    ds = load_ecospace_dataset()

    row_C09, col_C09 = C09_ROW, C09_COL

    # Generate masks if required
    if CREATE_MASKS:
        generate_2d_mask(ds, os.path.join(DOMAIN_CONFIG_PATH, DOMAIN_FILE))
        lat_1d, lon_1d = 49.1887166, -123.4966697
        idx_1d = generate_1d_mask(ds, lat_1d, lon_1d)
        print("1D index (row, col, depth):", idx_1d)

    mask_ds = xr.open_dataset(SAT_MASK_PF)

    bloom_csv_path_sat = cfg.BT_CSV_SUCHY_PF
    bloom_csv_path_C09 = cfg.BT_CSV_ALLEN_PF

    var_name_C09 = VARIABLES_TO_ANALYZE_C09

    if RECOMPUTE_BLOOM_TIMING_C09:
        mask = None

        # override option to use satellite 2d mask
        if USE_SAT_MASK_CO9:
            row_C09 = None; col_C09 = None
            mask = mask_ds['mask']

        # Choose method for C09
        method = "pct_max" if C09_USE_PCT_MAX else "threshold"

        bloom_df_C09 = compute_bloom_timing(
            ds, var_name_C09,
            mask=mask,
            row=row_C09, col=col_C09,
            bloom_early=C09_df['Day of Year_C09'].mean() - C09_df['Day of Year_C09'].std(),
            bloom_late=C09_df['Day of Year_C09'].mean() + C09_df['Day of Year_C09'].std(),
            yr_strt=1980, yr_end=2018,
            exclude_dec_jan=EXCLUDE_DEC_JAN_C09,
            avg_method=ANNUAL_AVG_METHOD_C09,
            method=method,
            window_days=C09_PCT_MAX_WINDOW_DAYS,
            pct_of_max=C09_PCT_MAX
        )
        bloom_df_C09.to_csv(bloom_csv_path_C09, index=False)
        print(f"Saved computed bloom timing for C09 1D to {bloom_csv_path_C09}")

    else:
        bloom_df_C09 = pd.read_csv(bloom_csv_path_C09)
        print(f"Loaded bloom timing from cached files: {bloom_csv_path_C09}")

    var_name_SAT = VARIABLES_TO_ANALYZE_SAT

    if RECOMPUTE_BLOOM_TIMING_SAT:
        if USEC09_MASK_FOR_SAT:
            bloom_df_sat = compute_bloom_timing(
                ds, var_name_SAT, mask=None,
                row=row_C09, col=col_C09,
                bloom_early=68, bloom_late=108,
                yr_strt=2003, yr_end=2016,
                exclude_dec_jan=EXCLUDE_DEC_JAN_SAT,
                avg_method=ANNUAL_AVG_METHOD_SAT
            )
        else:
            bloom_df_sat = compute_bloom_timing(
                ds, var_name_SAT, mask=mask_ds['mask'],
                row=None, col=None,
                bloom_early=68, bloom_late=108,
                yr_strt=2003, yr_end=2016,
                exclude_dec_jan=EXCLUDE_DEC_JAN_SAT,
                avg_method=ANNUAL_AVG_METHOD_SAT
            )
        bloom_df_sat.to_csv(bloom_csv_path_sat, index=False)
        print(f"Saved computed bloom timing for SSoG (sat) to {bloom_csv_path_sat}")

    else:
        bloom_df_sat = pd.read_csv(bloom_csv_path_sat)
        print(f"Loaded bloom timing from cached files: {bloom_csv_path_sat}")


    print('bloom doy mean, C09:')
    print(C09_df['Day of Year_C09'].mean())
    print('bloom doy early and late thresholds, CO9:')
    print(C09_df['Day of Year_C09'].mean() - C09_df['Day of Year_C09'].std())
    print(C09_df['Day of Year_C09'].mean() + C09_df['Day of Year_C09'].std())

    # Print summaries
    print("Loaded bloom timing observation data:")
    print("Satellite (sat et al):", sat_df.shape)
    print("Satellite (Gower et al):", gower_df.shape)
    print("C09:", C09_df.shape)
    print("Ecospace (Satellite):", bloom_df_sat.shape)
    print("Ecospace (C09):", bloom_df_C09.shape)

    ecosim_sat_df = None
    ecosim_c09_df = None

    if getattr(cfg, "BT_OVERLAY_ECOSIM", False):
        ecosim_sat_df = load_overlay_csv(cfg.BT_ECOSIM_SAT_CSV)
        ecosim_c09_df = load_overlay_csv(cfg.BT_ECOSIM_C09_CSV)

    # Plot comparison with C09 1D model
    plot_bloom_comparison(
        bloom_df_C09, C09_df,
        label_model="Ecospace", label_obs="C09 1D Model",
        title=f"Bloom Timing: {SCENARIO} vs C09",
        filename=f"ecospace_vs_C09_{SCENARIO}.png",
        df_overlay=ecosim_c09_df,
        label_overlay=cfg.BT_ECOSIM_LABEL,
    )

    plot_bloom_comparison(
        bloom_df_sat, sat_df,
        label_model="Ecospace", label_obs="Satellite",
        title=f"Bloom Timing: {SCENARIO} vs Satellite",
        filename=f"ecospace_vs_satell_{SCENARIO}.png",
        df_overlay=ecosim_sat_df,
        label_overlay=cfg.BT_ECOSIM_LABEL,
    )

    # Compute and print statistics
    stats_sat = evaluate_model(
        sat_df["Day of Year"], bloom_df_sat["Day of Year"], label="sat")
    stats_C09 = evaluate_model(
        C09_df["Day of Year_C09"], bloom_df_C09["Day of Year"], label="C09")

    print("\nEvaluation Statistics:")
    stats_out = [stats_sat, stats_C09]
    for stat in [stats_sat, stats_C09]:
        print(
            f"{stat['Label']}: R = {stat['R']:.3f}, "
            f"RMSE = {stat['RMSE']:.2f}, "
            f"MAE = {stat['MAE']:.2f}, "
            f"Bias = {stat['Bias']:.2f}, "
            f"Obs σ = {stat['Obs StdDev']:.2f}, "
            f"Model σ = {stat['Model StdDev']:.2f}, "
            f"Willmott = {stat['Willmott Skill']:.3f}")

    # write to csv
    stats_df = pd.DataFrame(stats_out).round(2)
    stats_csv_path = os.path.join(STATS_OUT_PATH, f"ecospace_bloom_timing_stats_{SCENARIO}.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved evaluation stats to {stats_csv_path}")

    # Additional categorical comparisons
    agree_cat_sat, total_cat_sat = evaluate_bloom_categories(sat_df, bloom_df_sat)
    agree_cat_C09, total_cat_C09 = evaluate_bloom_categories(C09_df, bloom_df_C09,
                                                                 col_obs='Bloom Early Late')
    print("\nCategorical Agreement:")
    print(f"Satellite: {agree_cat_sat}/{total_cat_sat} years agree in category")
    print(f"C09: {agree_cat_C09}/{total_cat_C09} years agree in category")

    # for cat. agrmnt to csv
    cat_stats = [
        {
            "Label": "sat",
            "Type": "Categorical Agreement",
            "Count": agree_cat_sat,
            "Total": total_cat_sat,
            "Proportion": agree_cat_sat / total_cat_sat if total_cat_sat > 0 else np.nan
        },
        {
            "Label": "C09",
            "Type": "Categorical Agreement",
            "Count": agree_cat_C09,
            "Total": total_cat_C09,
            "Proportion": agree_cat_C09 / total_cat_C09 if total_cat_C09 > 0 else np.nan
        }
    ]

    # Overlap by timing comparison
    overlap_sat, n_sat = evaluate_overlap_by_timing(sat_df, bloom_df_sat)
    overlap_C09, n_C09 = evaluate_overlap_by_timing(C09_df, bloom_df_C09,
                                                        obs_col='Day of Year')

    cat_stats.extend([
        {
            "Label": "sat",
            "Type": "Timing Window Overlap",
            "Count": overlap_sat,
            "Total": n_sat,
            "Proportion": overlap_sat / n_sat if n_sat > 0 else np.nan
        },
        {
            "Label": "C09",
            "Type": "Timing Window Overlap",
            "Count": overlap_C09,
            "Total": n_C09,
            "Proportion": overlap_C09 / n_C09 if n_C09 > 0 else np.nan
        }
    ])

    cat_stats_df = pd.DataFrame(cat_stats)
    cat_stats_df["Proportion"] = cat_stats_df["Proportion"].round(2)
    cat_stats_csv_path = os.path.join(STATS_OUT_PATH, f"ecospace_bloom_timing_agreement_{SCENARIO}.csv")
    cat_stats_df.to_csv(cat_stats_csv_path, index=False)
    print(f"Saved categorical/timing agreement stats to {cat_stats_csv_path}")

    if DO_NUTRIENTS:
        print("\nOverlap of Bloom Timing Windows:")
        print(f"Satellite: {overlap_sat}/{n_sat} years overlap within timing window")
        print(f"C09: {overlap_C09}/{n_C09} years overlap within timing window")

        # Example: Nutrient diagnostic for one year
        all_biomass_vars = [
            "NK1-COH", "NK2-CHI", "NK3-FOR",
            "ZF1-ICT", "ZC1-EUP", "ZC2-AMP",
            "ZC3-DEC", "ZC4-CLG", "ZC5-CSM",
            "ZS1-JEL", "ZS2-CTH", "ZS3-CHA",
            "ZS4-LAR", "PZ1-CIL","PZ2-DIN",
            "PZ3-HNF","PP1-DIA", "PP2-NAN",
            "PP3-PIC", "BAC-BA1"
        ]

        include_only = ["PP1-DIA", "PP2-NAN", "PP3-PIC"]  # example filter
        years_to_plot = list(range(1980, 2019))

        nutrient_dfs = []
        for yr in years_to_plot:
            print(yr)
            df = compute_nutrient_concentration(ds, all_biomass_vars, year=yr, include_only=include_only,
                                                mask=mask_ds['mask'])
            nutrient_dfs.append(df)

        df_nutrient_all = pd.concat(nutrient_dfs, ignore_index=True)
        plot_nutrient_concentration(df_nutrient_all, ds=ds, include_only=include_only, mask=mask_ds['mask'])
        print("Saved nutrient climatology plot for 1980–2018.")
        print("done nutr")


if __name__ == "__main__":
    run_bt_eval()




